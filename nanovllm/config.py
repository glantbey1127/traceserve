from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanovllm.graph import GraphPolicy


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_resident_seqs: int | None = None
    decode_reserve_blocks: int = 1
    max_consecutive_prefill_steps: int = 1
    cuda_graph_max_batch_size: int = 512
    graph_batch_sizes: tuple[int, ...] | None = None
    max_graph_padding_ratio: float = 0.25
    metrics_history_size: int = 1024
    step_history_size: int = 512
    trace_path: str | None = None
    graph_policy: GraphPolicy = field(init=False, repr=False)

    def __post_init__(self):
        self.model = str(Path(self.model).expanduser().resolve())
        if not Path(self.model).is_dir():
            raise ValueError(f"model must be an existing local directory: {self.model}")
        for name in (
            "max_num_batched_tokens", "max_num_seqs", "max_model_len", "tensor_parallel_size",
            "kvcache_block_size", "max_consecutive_prefill_steps", "cuda_graph_max_batch_size",
            "metrics_history_size", "step_history_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.tensor_parallel_size > 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")
        if self.kvcache_block_size % 256:
            raise ValueError("kvcache_block_size must be a multiple of 256")
        if not math.isfinite(self.gpu_memory_utilization) or not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if (
            not isinstance(self.num_kvcache_blocks, int) or isinstance(self.num_kvcache_blocks, bool)
            or (self.num_kvcache_blocks != -1 and self.num_kvcache_blocks < 1)
        ):
            raise ValueError("num_kvcache_blocks must be -1 (automatic) or positive")
        if (
            not isinstance(self.decode_reserve_blocks, int) or isinstance(self.decode_reserve_blocks, bool)
            or self.decode_reserve_blocks < 0
        ):
            raise ValueError("decode_reserve_blocks must be a nonnegative integer")
        if self.max_resident_seqs is None:
            self.max_resident_seqs = self.max_num_seqs
        if (
            not isinstance(self.max_resident_seqs, int) or isinstance(self.max_resident_seqs, bool)
            or self.max_resident_seqs < 1
        ):
            raise ValueError("max_resident_seqs must be positive")
        self.graph_policy = GraphPolicy(
            min(self.max_num_seqs, self.max_num_batched_tokens, self.cuda_graph_max_batch_size),
            self.graph_batch_sizes,
            self.max_graph_padding_ratio,
        )
        if self.hf_config is None:
            try:
                from transformers import AutoConfig
            except ImportError as exc:
                raise RuntimeError("GPU inference dependencies are missing; install traceserve[gpu]") from exc
            self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        if self.hf_config.vocab_size < 1 or self.hf_config.max_position_embeddings < 1:
            raise ValueError("model vocabulary and position limits must be positive")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
