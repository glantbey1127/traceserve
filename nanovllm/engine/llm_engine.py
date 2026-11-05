from __future__ import annotations

import atexit
import math
import multiprocessing as mp
from collections import OrderedDict
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Iterator, Protocol, Sequence as TypingSequence

from nanovllm.config import Config
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.observability import (
    FinishEvent, GenerationEvent, GenerationOutput, MetricsStore, RequestMetrics,
    RunnerOutput, StepMetrics, StepResult, TokenEvent, TraceWriter,
)
from nanovllm.sampling_params import SamplingParams


class Tokenizer(Protocol):
    eos_token_id: int | None

    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int], **kwargs) -> str: ...
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs) -> list[int]: ...


class Runner(Protocol):
    backend_info: dict

    def call(self, method_name: str, *args) -> RunnerOutput | None: ...


@dataclass(frozen=True, slots=True)
class ChatPrompt:
    messages: tuple[dict[str, str], ...]

    def __post_init__(self):
        if not self.messages:
            raise ValueError("messages must not be empty")
        for message in self.messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("each message must contain exactly role and content")
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError("supported chat roles are system, user, assistant")
            if not isinstance(message["content"], str):
                raise ValueError("message content must be a string")


Prompt = str | list[int] | ChatPrompt


class IncrementalDecoder:
    """Append-stable Qwen decoding, withholding incomplete trailing UTF-8 characters."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.text = ""

    def update(self, token_ids: list[int], *, final: bool = False) -> str:
        decoded = self.tokenizer.decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        stable = decoded if final else decoded.rstrip("\ufffd")
        if not stable.startswith(self.text):
            raise RuntimeError("tokenizer changed an emitted prefix; append-stable decoding is required")
        delta = stable[len(self.text):]
        self.text = stable
        return delta


@dataclass(slots=True)
class RequestRecord:
    seq: Sequence
    metrics: RequestMetrics
    decoder: IncrementalDecoder


class LLMEngine:
    """Synchronous, single-owner engine. Use AsyncEngine for concurrent HTTP clients."""

    def __init__(
        self,
        model: str | Config,
        *,
        model_runner: Runner | None = None,
        tokenizer: Tokenizer | None = None,
        clock: Callable[[], float] = perf_counter,
        **config_kwargs,
    ):
        if isinstance(model, Config):
            if config_kwargs:
                raise ValueError("pass either Config or model plus configuration keywords")
            self.config = model
        else:
            self.config = Config(model=model, **config_kwargs)
        config = self.config
        Sequence.block_size = config.kvcache_block_size
        self.clock = clock
        self.ps: list[mp.Process] = []
        self._closed = True
        self._failure: Exception | None = None
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True)
        self.tokenizer = tokenizer
        config.eos = -1 if tokenizer.eos_token_id is None else tokenizer.eos_token_id
        if model_runner is not None and config.num_kvcache_blocks < 1:
            raise ValueError("injected runner requires a positive num_kvcache_blocks")
        self.trace = None if config.trace_path is None else TraceWriter(config.trace_path)
        try:
            if model_runner is None:
                try:
                    from nanovllm.engine.model_runner import ModelRunner
                except ImportError as exc:
                    raise RuntimeError(
                        "GPU backend requires Linux/CUDA, torch, Triton, Transformers and FlashAttention; "
                        "install the gpu dependencies described in README"
                    ) from exc
                events = []
                context = mp.get_context("spawn")
                for rank in range(1, config.tensor_parallel_size):
                    event = context.Event()
                    process = context.Process(target=ModelRunner, args=(config, rank, event))
                    process.start()
                    self.ps.append(process)
                    events.append(event)
                self.model_runner = ModelRunner(config, 0, events)
            else:
                self.model_runner = model_runner
        except Exception:
            for process in self.ps:
                if process.is_alive():
                    process.terminate()
                process.join()
            if self.trace:
                self.trace.close()
            raise
        if config.num_kvcache_blocks < 1:
            raise ValueError("runner must configure a positive num_kvcache_blocks before engine startup")
        self.scheduler = Scheduler(config)
        self.metrics = MetricsStore(config.metrics_history_size, config.step_history_size)
        self._records: dict[str, RequestRecord] = {}
        self._seq_to_request: dict[int, str] = {}
        self._completed: OrderedDict[str, GenerationOutput] = OrderedDict()
        self._step_id = 0
        self._closed = False
        atexit.register(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("engine is closed")
        if self._failure is not None:
            raise RuntimeError("engine execution failed; restart the engine") from self._failure

    def add_request(
        self,
        prompt: Prompt,
        sampling_params: SamplingParams,
        *,
        request_id: str | None = None,
        arrived_at: float | None = None,
    ) -> str:
        self._ensure_open()
        now = self.clock()
        if arrived_at is None:
            arrived_at = now
        elif not math.isfinite(arrived_at) or arrived_at > now:
            raise ValueError("arrived_at must be a finite timestamp from this process's monotonic clock")
        if not isinstance(sampling_params, SamplingParams):
            raise TypeError("sampling_params must be SamplingParams")
        if isinstance(prompt, ChatPrompt):
            token_ids = self.tokenizer.apply_chat_template(
                list(prompt.messages), tokenize=True, add_generation_prompt=True,
            )
        elif isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        elif isinstance(prompt, list):
            if any(
                not isinstance(token, int) or isinstance(token, bool)
                or not 0 <= token < self.config.hf_config.vocab_size
                for token in prompt
            ):
                raise ValueError("prompt token IDs must be integers within the model vocabulary")
            token_ids = prompt
        else:
            raise TypeError("prompt must be text, a token ID list, or ChatPrompt")
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        total = len(token_ids) + sampling_params.max_tokens
        if total > self.config.max_model_len:
            raise ValueError(f"prompt plus max_tokens ({total}) exceeds max_model_len")
        blocks = (total - 1 + self.config.kvcache_block_size - 1) // self.config.kvcache_block_size
        if blocks > self.config.num_kvcache_blocks:
            raise ValueError("one request's maximum KV context cannot fit in the configured block pool")
        seq = Sequence(token_ids, sampling_params)
        if request_id is None:
            request_id = str(seq.seq_id)
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")
        if request_id in self._records or request_id in self._completed:
            raise ValueError(f"duplicate request_id: {request_id}")
        seq.request_id = request_id
        request_metrics = RequestMetrics(request_id, len(token_ids), arrived_at)
        self._records[request_id] = RequestRecord(seq, request_metrics, IncrementalDecoder(self.tokenizer))
        self._seq_to_request[seq.seq_id] = request_id
        self.metrics.active[request_id] = request_metrics
        self.scheduler.add(seq)
        if self.trace:
            self.trace.write("request_started", now, {
                "request_id": request_id, "seq_id": seq.seq_id,
                "prompt_tokens": len(token_ids), "max_tokens": sampling_params.max_tokens,
                "arrived_at": arrived_at,
            })
        return request_id

    def _finish(self, record: RequestRecord, timestamp: float, error: str | None = None) -> FinishEvent:
        seq = record.seq
        final_delta = record.decoder.update(seq.completion_token_ids, final=True)
        record.metrics.finished_at = timestamp
        record.metrics.finish_reason = seq.finish_reason
        output = GenerationOutput(
            seq.request_id, seq.seq_id, record.decoder.text, seq.completion_token_ids,
            seq.finish_reason, record.metrics.snapshot(), error,
        )
        self.metrics.finish(record.metrics)
        del self._records[seq.request_id]
        del self._seq_to_request[seq.seq_id]
        self._completed[seq.request_id] = output
        while len(self._completed) > self.config.metrics_history_size:
            self._completed.popitem(last=False)
        if self.trace:
            self.trace.write("request_finished", timestamp, output.to_dict())
        return FinishEvent(seq.request_id, output, final_delta)

    def cancel_request(
        self,
        request_id: str,
        *,
        error: str | None = None,
    ) -> FinishEvent | None:
        record = self._records.get(request_id)
        if record is None:
            return None
        self.scheduler.cancel(record.seq, failed=error is not None)
        return self._finish(record, self.clock(), error)

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def step_events(self) -> StepResult:
        self._ensure_open()
        if self.is_finished():
            return StepResult([], [], None, 0)
        started = self.clock()
        free_before = len(self.scheduler.block_manager.free_block_ids)
        seqs, is_prefill = self.scheduler.schedule()
        scheduled_at = self.clock()
        decision = self.scheduler.last_events
        preempted_request_ids = []
        for preemption in decision.preemptions:
            request_id = self._seq_to_request[preemption.seq_id]
            preempted_request_ids.append(request_id)
            self._records[request_id].metrics.preemptions += 1
            if self.trace:
                self.trace.write("preemption", scheduled_at, {
                    **asdict(preemption), "request_id": request_id,
                })
        for seq_id, hit_tokens in decision.cache_hits.items():
            self._records[self._seq_to_request[seq_id]].metrics.prefix_cached_tokens += hit_tokens
        scheduled = sum(seq.num_scheduled_tokens for seq in seqs)
        recomputed = {}
        before_outputs = {}
        for seq in seqs:
            record = self._records[seq.request_id]
            if record.metrics.first_scheduled_at is None:
                record.metrics.first_scheduled_at = scheduled_at
            end = seq.num_cached_tokens + seq.num_scheduled_tokens
            overlap = max(0, min(end, seq.computed_high_watermark) - seq.num_cached_tokens)
            recomputed[seq.seq_id] = overlap
            record.metrics.recomputed_tokens += overlap
            before_outputs[seq.seq_id] = seq.num_completion_tokens
        runner_started = self.clock()
        try:
            result = self.model_runner.call("run", seqs, is_prefill)
            if result is None or len(result.token_ids) != len(seqs):
                raise RuntimeError("runner returned an invalid number of sampled tokens")
            runner_finished = self.clock()
            self.scheduler.postprocess(seqs, result.token_ids, is_prefill)
        except Exception as exc:
            self._failure = exc
            for request_id in list(self._records):
                self.cancel_request(request_id, error=str(exc))
            if self.trace:
                self.trace.write("engine_error", self.clock(), {"message": str(exc)})
            raise RuntimeError("model execution failed; engine must be restarted") from exc
        token_ready = self.clock()
        events: list[GenerationEvent] = []
        completed = []
        produced = 0
        for seq in seqs:
            record = self._records[seq.request_id]
            if seq.num_completion_tokens > before_outputs[seq.seq_id]:
                record.metrics.token_times.append(token_ready)
                delta = record.decoder.update(seq.completion_token_ids)
                event = TokenEvent(
                    seq.request_id, seq.seq_id, seq.last_token,
                    seq.num_completion_tokens - 1, delta, token_ready,
                )
                events.append(event)
                produced += 1
                if self.trace:
                    self.trace.write("token", token_ready, asdict(event))
            if seq.is_finished:
                event = self._finish(record, token_ready)
                events.append(event)
                completed.append(event.output)
        self._step_id += 1
        metrics = StepMetrics(
            step_id=self._step_id,
            phase="prefill" if is_prefill else "decode",
            request_ids=[seq.request_id for seq in seqs],
            real_batch_size=len(seqs),
            model_rows=scheduled,
            graph_batch_size=result.graph_batch_size,
            padded_rows=0 if result.graph_batch_size is None else result.graph_batch_size - len(seqs),
            execution_mode=result.execution_mode,
            execution_reason=result.execution_reason,
            scheduled_tokens=scheduled,
            output_tokens=produced,
            recomputed_tokens=sum(recomputed.values()),
            prefix_cached_tokens=sum(decision.cache_hits.values()),
            preempted_request_ids=preempted_request_ids,
            free_blocks_before=free_before,
            free_blocks_after=len(self.scheduler.block_manager.free_block_ids),
            wall_ms=(self.clock() - started) * 1000,
            runner_wall_ms=(runner_finished - runner_started) * 1000,
        )
        self.metrics.record_step(metrics)
        if self.trace:
            self.trace.write("step", self.clock(), asdict(metrics))
        return StepResult(events, completed, metrics, scheduled if is_prefill else -len(seqs))

    def step(self):
        """Legacy nano-vLLM tuple API; streaming clients use step_events()."""
        result = self.step_events()
        return [(output.seq_id, output.token_ids) for output in result.completed], result.num_tokens

    def generate_stream(
        self,
        prompts: TypingSequence[Prompt],
        sampling_params: SamplingParams | TypingSequence[SamplingParams],
    ) -> Iterator[GenerationEvent]:
        if isinstance(prompts, str):
            raise TypeError("generate_stream expects a sequence of prompts, not a single string")
        params = (
            [sampling_params] * len(prompts)
            if isinstance(sampling_params, SamplingParams) else list(sampling_params)
        )
        if len(params) != len(prompts):
            raise ValueError("prompts and sampling_params must have the same length")
        request_ids: set[str] = set()
        try:
            for prompt, param in zip(prompts, params):
                request_ids.add(self.add_request(prompt, param))
            while any(request_id in self._records for request_id in request_ids):
                for event in self.step_events().events:
                    if event.request_id in request_ids:
                        yield event
        finally:
            for request_id in request_ids:
                self.cancel_request(request_id)

    def generate(
        self,
        prompts: TypingSequence[Prompt],
        sampling_params: SamplingParams | TypingSequence[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        outputs = []
        progress = None
        if use_tqdm:
            from tqdm import tqdm
            progress = tqdm(total=len(prompts), desc="Generating")
        try:
            for event in self.generate_stream(prompts, sampling_params):
                if isinstance(event, FinishEvent):
                    outputs.append(event.output)
                    if progress:
                        progress.update(1)
        finally:
            if progress:
                progress.close()
        return [output.to_dict() for output in sorted(outputs, key=lambda output: output.seq_id)]

    def get_request_metrics(self, request_id: str) -> dict | None:
        request = self.metrics.active.get(request_id) or self.metrics.completed.get(request_id)
        return None if request is None else request.snapshot()

    def get_output(self, request_id: str) -> GenerationOutput | None:
        return self._completed.get(request_id)

    def stats(self) -> dict:
        return {
            **self.metrics.summary(),
            "kv": {
                "total_blocks": len(self.scheduler.block_manager.blocks),
                "free_blocks": len(self.scheduler.block_manager.free_block_ids),
                "used_blocks": len(self.scheduler.block_manager.used_block_ids),
                "resident_requests": self.scheduler.resident_count,
            },
        }

    def reset_metrics(self, *, clear_trace: bool = False):
        if not self.is_finished():
            raise RuntimeError("metrics can only be reset while the engine is idle")
        self.metrics = MetricsStore(self.config.metrics_history_size, self.config.step_history_size)
        self._completed.clear()
        self._step_id = 0
        if clear_trace and self.trace:
            self.trace.close()
            self.trace = TraceWriter(self.config.trace_path)

    def clear_prefix_cache(self):
        self.scheduler.block_manager.clear_cache()
        if self.trace:
            self.trace.write("cache_cleared", self.clock(), {})

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            for request_id in list(self._records):
                self.cancel_request(request_id)
            self.model_runner.call("exit")
        finally:
            for process in self.ps:
                process.join()
            if self.trace:
                self.trace.close()

    exit = close
