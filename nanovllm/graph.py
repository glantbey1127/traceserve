from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class GraphSelection:
    mode: Literal["eager", "cuda_graph"]
    real_rows: int
    bucket: int | None
    reason: str

    @property
    def padded_rows(self) -> int:
        return 0 if self.bucket is None else self.bucket - self.real_rows


class GraphPolicy:
    """Owns graph-shape validation and selection; contains no torch dependency."""

    def __init__(
        self,
        max_batch_size: int,
        buckets: tuple[int, ...] | None = None,
        max_padding_ratio: float = 0.25,
    ):
        if not isinstance(max_batch_size, int) or isinstance(max_batch_size, bool) or max_batch_size < 1:
            raise ValueError("graph maximum batch size must be positive")
        if not math.isfinite(max_padding_ratio) or not 0 <= max_padding_ratio < 1:
            raise ValueError("max_graph_padding_ratio must be in [0, 1)")
        if buckets is None:
            values = [1, 2, 4, 8, *range(16, max_batch_size + 1, 16)]
            values = [value for value in values if value <= max_batch_size]
        else:
            values = list(buckets)
            if any(
                not isinstance(value, int) or isinstance(value, bool)
                or value < 1 or value > max_batch_size
                for value in values
            ):
                raise ValueError("custom graph buckets must be positive integers within the graph limit")
        self.max_batch_size = max_batch_size
        self.buckets = tuple(sorted(set([*values, max_batch_size])))
        self.max_padding_ratio = max_padding_ratio

    def select(
        self,
        real_rows: int,
        *,
        is_prefill: bool = False,
        enforce_eager: bool = False,
    ) -> GraphSelection:
        if real_rows < 1:
            raise ValueError("cannot execute an empty batch")
        if is_prefill:
            return GraphSelection("eager", real_rows, None, "prefill")
        if enforce_eager:
            return GraphSelection("eager", real_rows, None, "forced_eager")
        index = bisect_left(self.buckets, real_rows)
        if index == len(self.buckets):
            return GraphSelection("eager", real_rows, None, "batch_limit")
        bucket = self.buckets[index]
        if (bucket - real_rows) / bucket > self.max_padding_ratio:
            return GraphSelection("eager", real_rows, None, "padding_limit")
        return GraphSelection("cuda_graph", real_rows, bucket, "bucket_match")
