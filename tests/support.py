"""Test-only token/runner fixtures. These do not implement model inference."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from nanovllm.config import Config
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.observability import RunnerOutput


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds=0.01):
        self.now += seconds


class TestTokenizer:
    __test__ = False
    eos_token_id = 0

    def encode(self, text):
        return [ord(character) % 4095 + 1 for character in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token % 26 + 65) for token in token_ids if token != self.eos_token_id)

    def apply_chat_template(self, messages, **kwargs):
        text = "".join(f"{message['role']}: {message['content']}\n" for message in messages)
        return self.encode(text + "assistant: ")


class TestRunner:
    __test__ = False
    backend_info = {"type": "test_reference", "performance_representative": False}

    def __init__(self, config, *, clock=None, token=None, fail_on=None, delay=0):
        self.config = config
        self.clock = clock
        self.token = token
        self.fail_on = fail_on
        self.delay = delay
        self.calls = []
        self.threads = []
        self.closed = False
        self.entered = threading.Event()
        self.release = None

    def call(self, method_name, *args):
        self.threads.append(threading.get_ident())
        if method_name == "exit":
            self.closed = True
            return None
        assert method_name == "run"
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5), "test runner release timed out"
        if self.delay:
            time.sleep(self.delay)
        seqs, is_prefill = args
        self.calls.append({
            "phase": "prefill" if is_prefill else "decode",
            "request_ids": [seq.request_id for seq in seqs],
            "positions": [(seq.num_cached_tokens, seq.num_scheduled_tokens) for seq in seqs],
        })
        if self.fail_on == len(self.calls):
            raise RuntimeError("deliberate test runner failure")
        if self.clock is not None:
            self.clock.advance()
        selection = self.config.graph_policy.select(
            sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else len(seqs),
            is_prefill=is_prefill, enforce_eager=self.config.enforce_eager,
        )
        tokens = [
            self.token if self.token is not None else 65 + seq.num_completion_tokens
            for seq in seqs
        ]
        return RunnerOutput(tokens, selection.mode, selection.bucket, selection.reason)


def make_config(path, **kwargs):
    values = {
        "model": str(path), "hf_config": SimpleNamespace(vocab_size=4096, max_position_embeddings=4096),
        "num_kvcache_blocks": 64, "max_num_seqs": 8, "max_model_len": 4096,
    }
    values.update(kwargs)
    return Config(**values)


def make_engine(path, *, clock=None, runner_kwargs=None, **config_kwargs):
    config = make_config(path, **config_kwargs)
    runner = TestRunner(config, clock=clock, **(runner_kwargs or {}))
    kwargs = {} if clock is None else {"clock": clock}
    return LLMEngine(config, model_runner=runner, tokenizer=TestTokenizer(), **kwargs)
