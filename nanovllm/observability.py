from __future__ import annotations

import json
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


def distribution(values: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(values)
    result: dict[str, float | int | None] = {"count": len(ordered)}
    if not ordered:
        return {**result, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    result["mean"] = sum(ordered) / len(ordered)
    for name, probability in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        position = (len(ordered) - 1) * probability
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        result[name] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    result["max"] = ordered[-1]
    return result


@dataclass(slots=True)
class RequestMetrics:
    request_id: str
    prompt_tokens: int
    arrived_at: float
    first_scheduled_at: float | None = None
    token_times: list[float] = field(default_factory=list)
    finished_at: float | None = None
    finish_reason: str | None = None
    prefix_cached_tokens: int = 0
    recomputed_tokens: int = 0
    preemptions: int = 0

    def snapshot(self) -> dict:
        times = self.token_times
        intervals = [(right - left) * 1000 for left, right in zip(times, times[1:])]
        return {
            "request_id": self.request_id,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": len(times),
            "finish_reason": self.finish_reason,
            "queue_ms": None if self.first_scheduled_at is None
            else (self.first_scheduled_at - self.arrived_at) * 1000,
            "engine_ttft_ms": None if not times else (times[0] - self.arrived_at) * 1000,
            "engine_tpot_ms": None if len(times) < 2
            else (times[-1] - times[0]) * 1000 / (len(times) - 1),
            "engine_itl_ms": intervals,
            "engine_e2e_ms": None if self.finished_at is None
            else (self.finished_at - self.arrived_at) * 1000,
            "prefix_cached_tokens": self.prefix_cached_tokens,
            "recomputed_tokens": self.recomputed_tokens,
            "preemptions": self.preemptions,
        }


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    request_id: str
    seq_id: int
    text: str
    token_ids: list[int]
    finish_reason: str
    metrics: dict
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: str
    seq_id: int
    token_id: int
    index: int
    delta: str
    timestamp: float
    type: Literal["token"] = "token"


@dataclass(frozen=True, slots=True)
class FinishEvent:
    request_id: str
    output: GenerationOutput
    delta: str = ""
    type: Literal["finished"] = "finished"


GenerationEvent = TokenEvent | FinishEvent


@dataclass(frozen=True, slots=True)
class RunnerOutput:
    token_ids: list[int]
    execution_mode: str
    graph_batch_size: int | None
    execution_reason: str


@dataclass(frozen=True, slots=True)
class StepMetrics:
    step_id: int
    phase: Literal["prefill", "decode"]
    request_ids: list[str]
    real_batch_size: int
    model_rows: int
    graph_batch_size: int | None
    padded_rows: int
    execution_mode: str
    execution_reason: str
    scheduled_tokens: int
    output_tokens: int
    recomputed_tokens: int
    prefix_cached_tokens: int
    preempted_request_ids: list[str]
    free_blocks_before: int
    free_blocks_after: int
    wall_ms: float
    runner_wall_ms: float


@dataclass(frozen=True, slots=True)
class StepResult:
    events: list[GenerationEvent]
    completed: list[GenerationOutput]
    metrics: StepMetrics | None
    num_tokens: int


class TraceWriter:
    def __init__(self, path: str):
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        self.file = target.open("w", encoding="utf-8")

    def write(self, kind: str, timestamp: float, data: dict):
        self.file.write(json.dumps({
            "schema_version": 1,
            "event": kind,
            "timestamp_s": timestamp,
            "data": data,
        }, ensure_ascii=False) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class MetricsStore:
    def __init__(self, request_limit: int, step_limit: int):
        self.request_limit = request_limit
        self.active: dict[str, RequestMetrics] = {}
        self.completed: OrderedDict[str, RequestMetrics] = OrderedDict()
        self.steps: deque[StepMetrics] = deque(maxlen=step_limit)
        self.totals = {
            "steps": 0, "prefill_tokens": 0, "decode_tokens": 0, "output_tokens": 0,
            "recomputed_tokens": 0, "prefix_cached_tokens": 0, "preemptions": 0,
            "graph_padded_rows": 0, "prefill_wall_ms": 0.0, "decode_wall_ms": 0.0,
            "finished_requests": 0, "cancelled_requests": 0, "failed_requests": 0,
        }

    def finish(self, request: RequestMetrics):
        self.active.pop(request.request_id)
        self.completed[request.request_id] = request
        while len(self.completed) > self.request_limit:
            self.completed.popitem(last=False)
        key = {
            "cancelled": "cancelled_requests", "error": "failed_requests",
        }.get(request.finish_reason, "finished_requests")
        self.totals[key] += 1

    def record_step(self, step: StepMetrics):
        self.steps.append(step)
        self.totals["steps"] += 1
        self.totals[step.phase + "_tokens"] += step.scheduled_tokens
        self.totals[step.phase + "_wall_ms"] += step.wall_ms
        self.totals["output_tokens"] += step.output_tokens
        self.totals["recomputed_tokens"] += step.recomputed_tokens
        self.totals["prefix_cached_tokens"] += step.prefix_cached_tokens
        self.totals["preemptions"] += len(step.preempted_request_ids)
        self.totals["graph_padded_rows"] += step.padded_rows

    def summary(self) -> dict:
        retained = [request.snapshot() for request in self.completed.values()]
        requests = [request for request in retained if request["finish_reason"] in {"stop", "length"}]
        result = {"totals": dict(self.totals), "active_requests": len(self.active),
                  "retained_completed_requests": len(retained)}
        result["latency_scope"] = "engine token IDs ready after postprocess; includes worker queue, excludes HTTP delivery"
        for source, target in (
            ("engine_ttft_ms", "engine_ttft_ms"),
            ("engine_tpot_ms", "engine_tpot_ms"),
            ("engine_e2e_ms", "engine_e2e_ms"),
        ):
            result[target] = distribution([request[source] for request in requests if request[source] is not None])
        result["engine_itl_ms"] = distribution([value for request in requests for value in request["engine_itl_ms"]])
        result["phase_throughput_tok_s"] = {
            phase: None if not self.totals[phase + "_wall_ms"]
            else self.totals[phase + "_tokens"] * 1000 / self.totals[phase + "_wall_ms"]
            for phase in ("prefill", "decode")
        }
        result["last_step"] = None if not self.steps else asdict(self.steps[-1])
        return result
