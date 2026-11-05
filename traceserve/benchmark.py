from __future__ import annotations

import hashlib
import json
import platform
import random
from dataclasses import asdict, dataclass
from statistics import median
from time import perf_counter

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.observability import distribution
from nanovllm.sampling_params import SamplingParams


@dataclass(frozen=True, slots=True)
class Workload:
    num_requests: int = 256
    min_input_tokens: int = 100
    max_input_tokens: int = 1024
    min_output_tokens: int = 100
    max_output_tokens: int = 1024
    shared_prefix_tokens: int = 0
    seed: int = 0
    temperature: float = 0.0

    def __post_init__(self):
        for name in ("num_requests", "min_input_tokens", "max_input_tokens",
                     "min_output_tokens", "max_output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_input_tokens > self.max_input_tokens or self.min_output_tokens > self.max_output_tokens:
            raise ValueError("minimum token lengths must not exceed maximum lengths")
        if not 0 <= self.shared_prefix_tokens <= self.min_input_tokens:
            raise ValueError("shared prefix must fit in every prompt")
        SamplingParams(self.temperature, self.max_output_tokens, True)


def run_benchmark(
    engine: LLMEngine,
    workload: Workload,
    *,
    repeats: int = 1,
    warmup_requests: int = 1,
    cache_state: str = "cold",
) -> dict:
    if repeats < 1 or warmup_requests < 0:
        raise ValueError("repeats must be positive and warmup_requests nonnegative")
    if cache_state not in {"cold", "warm"}:
        raise ValueError("cache_state must be cold or warm")
    if not engine.is_finished():
        raise ValueError("benchmark requires an idle engine")
    if workload.max_input_tokens + workload.max_output_tokens > engine.config.max_model_len:
        raise ValueError("workload maximum input plus output exceeds the engine context limit")
    rng = random.Random(workload.seed)
    vocab_size = engine.config.hf_config.vocab_size
    prefix = [rng.randrange(vocab_size) for _ in range(workload.shared_prefix_tokens)]
    prompts = [
        prefix + [rng.randrange(vocab_size) for _ in range(
            rng.randint(workload.min_input_tokens, workload.max_input_tokens) - len(prefix)
        )]
        for _ in range(workload.num_requests)
    ]
    params = [
        SamplingParams(
            workload.temperature,
            rng.randint(workload.min_output_tokens, workload.max_output_tokens),
            True,
        )
        for _ in prompts
    ]
    fingerprint = hashlib.sha256(json.dumps({
        "prompts": prompts, "output_lengths": [param.max_tokens for param in params],
        "temperature": workload.temperature,
    }, separators=(",", ":")).encode()).hexdigest()
    for index in range(warmup_requests):
        engine.generate(
            [prompts[index % len(prompts)]],
            SamplingParams(temperature=workload.temperature, max_tokens=1, ignore_eos=True),
            use_tqdm=False,
        )
    runs = []
    for repeat in range(repeats):
        if cache_state == "cold":
            engine.clear_prefix_cache()
        engine.reset_metrics(clear_trace=repeat == 0)
        if engine.trace:
            engine.trace.write("benchmark_run_started", engine.clock(), {"run": repeat + 1})
        started = perf_counter()
        outputs = engine.generate(prompts, params, use_tqdm=False)
        elapsed = perf_counter() - started
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        input_tokens = sum(len(prompt) for prompt in prompts)
        successful = [output["metrics"] for output in outputs if output["finish_reason"] in {"stop", "length"}]
        latency = {}
        for key in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"):
            latency[key] = distribution([value[key] for value in successful if value[key] is not None])
        latency["engine_itl_ms"] = distribution([value for request in successful for value in request["engine_itl_ms"]])
        stats = engine.stats()
        runs.append({
            "run": repeat + 1,
            "elapsed_s": elapsed,
            "completed_requests": len(successful),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "target_output_tokens": sum(param.max_tokens for param in params),
            "output_throughput_tok_s": output_tokens / elapsed,
            "total_logical_throughput_tok_s": (input_tokens + output_tokens) / elapsed,
            "request_throughput_req_s": len(successful) / elapsed,
            "engine_latency": latency,
            "engine_counters": stats["totals"],
            "phase_throughput_tok_s": stats["phase_throughput_tok_s"],
            "kv": stats["kv"],
        })
    config = engine.config
    return {
        "schema_version": 1,
        "report_kind": "benchmark",
        "project": "TraceServe",
        "version": "0.1.0",
        "runner": type(engine.model_runner).__name__,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "backend": dict(engine.model_runner.backend_info),
        "timing_scope": (
            "warm synchronous generate() wall time; includes admission, scheduler, input preparation, "
            "GPU work, sampling, token transfer and text decoding; excludes startup, data construction and warmup"
        ),
        "latency_scope": "engine token IDs ready after postprocess; excludes HTTP delivery",
        "trace_enabled": engine.trace is not None,
        "cache_state": cache_state,
        "workload": asdict(workload),
        "workload_fingerprint": fingerprint,
        "configuration": {
            "model": config.model, "max_model_len": config.max_model_len,
            "num_kvcache_blocks": config.num_kvcache_blocks,
            "kvcache_block_size": config.kvcache_block_size,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs, "max_resident_seqs": config.max_resident_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "decode_reserve_blocks": config.decode_reserve_blocks,
            "max_consecutive_prefill_steps": config.max_consecutive_prefill_steps,
            "enforce_eager": config.enforce_eager,
            "graph_buckets": list(config.graph_policy.buckets),
            "max_graph_padding_ratio": config.max_graph_padding_ratio,
        },
        "runs": runs,
        "summary": {
            "elapsed_s": distribution([run["elapsed_s"] for run in runs]),
            "output_throughput_tok_s": distribution([run["output_throughput_tok_s"] for run in runs]),
            "median_output_throughput_tok_s": median(run["output_throughput_tok_s"] for run in runs),
        },
    }
