from __future__ import annotations

import json
import math
from dataclasses import fields
from pathlib import Path
from statistics import median

from nanovllm.observability import StepMetrics, distribution


def analyze_trace(path: str | Path, *, tpot_budget_ms: float | None = None) -> dict:
    if tpot_budget_ms is not None and (not math.isfinite(tpot_budget_ms) or tpot_budget_ms <= 0):
        raise ValueError("tpot_budget_ms must be finite and positive")
    steps: list[StepMetrics] = []
    requests: list[dict] = []
    started_requests = 0
    required = {field.name for field in fields(StepMetrics)}
    with Path(path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if (
                    not isinstance(record, dict) or record.get("schema_version") != 1
                    or not isinstance(record.get("event"), str) or not isinstance(record.get("data"), dict)
                ):
                    raise ValueError("expected a schema_version=1 trace event")
                data = record["data"]
                if record["event"] == "step":
                    if set(data) != required:
                        raise ValueError("step fields do not match the trace schema")
                    if data["phase"] not in {"prefill", "decode"}:
                        raise ValueError("invalid step phase")
                    for key in (
                        "step_id", "real_batch_size", "model_rows", "padded_rows", "scheduled_tokens",
                        "output_tokens", "recomputed_tokens", "prefix_cached_tokens",
                        "free_blocks_before", "free_blocks_after",
                    ):
                        if not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 0:
                            raise ValueError(f"invalid integer field: {key}")
                    for key in ("wall_ms", "runner_wall_ms"):
                        if not isinstance(data[key], (int, float)) or not math.isfinite(data[key]) or data[key] < 0:
                            raise ValueError(f"invalid duration: {key}")
                    if (
                        not isinstance(data["request_ids"], list)
                        or not isinstance(data["preempted_request_ids"], list)
                        or any(not isinstance(value, str) for value in
                               [*data["request_ids"], *data["preempted_request_ids"]])
                    ):
                        raise ValueError("request IDs must be lists")
                    bucket = data["graph_batch_size"]
                    if data["real_batch_size"] < 1 or len(data["request_ids"]) != data["real_batch_size"]:
                        raise ValueError("step must contain one request ID per real batch row")
                    if data["execution_mode"] == "cuda_graph":
                        valid_bucket = (
                            isinstance(bucket, int) and not isinstance(bucket, bool)
                            and bucket >= data["real_batch_size"]
                            and data["padded_rows"] == bucket - data["real_batch_size"]
                            and data["phase"] == "decode"
                        )
                    elif data["execution_mode"] == "eager":
                        valid_bucket = bucket is None and data["padded_rows"] == 0
                    else:
                        raise ValueError("invalid execution mode")
                    if not valid_bucket:
                        raise ValueError("invalid graph batch size")
                    steps.append(StepMetrics(**data))
                elif record["event"] == "request_started":
                    started_requests += 1
                elif record["event"] == "request_finished":
                    metrics = data["metrics"]
                    if not isinstance(metrics, dict):
                        raise ValueError("request metrics must be an object")
                    if metrics["finish_reason"] not in {"stop", "length", "cancelled", "error"}:
                        raise ValueError("invalid finish reason")
                    for key in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms"):
                        value = metrics[key]
                        if value is not None and (
                            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                        ):
                            raise ValueError(f"invalid request latency: {key}")
                    intervals = metrics["engine_itl_ms"]
                    if not isinstance(intervals, list) or any(
                        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                        for value in intervals
                    ):
                        raise ValueError("invalid ITL values")
                    requests.append(metrics)
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"invalid trace at line {line_number}: {exc}") from exc
    phase = {}
    for name in ("prefill", "decode"):
        selected = [step for step in steps if step.phase == name]
        tokens = sum(step.scheduled_tokens for step in selected)
        wall_ms = sum(step.wall_ms for step in selected)
        phase[name] = {
            "steps": len(selected), "scheduled_tokens": tokens, "step_wall_ms": wall_ms,
            "throughput_tok_s": None if wall_ms == 0 else tokens * 1000 / wall_ms,
        }
    successful = [request for request in requests if request["finish_reason"] in {"stop", "length"}]
    latency = {
        key: distribution([request[key] for request in successful if request[key] is not None])
        for key in ("engine_ttft_ms", "engine_tpot_ms", "engine_e2e_ms")
    }
    latency["engine_itl_ms"] = distribution([value for request in successful for value in request["engine_itl_ms"]])
    preemptions = sum(len(step.preempted_request_ids) for step in steps)
    recomputed = sum(step.recomputed_tokens for step in steps)
    padded = [step for step in steps if step.padded_rows > 0]
    findings = []
    if preemptions:
        findings.append({
            "kind": "kv_pressure",
            "evidence": {"preemptions": preemptions, "recomputed_tokens": recomputed},
            "source": "Scheduler.preempt / BlockManager.can_append",
            "interpretation": "capacity-driven preemption occurred; use per-request trace to inspect replayed positions",
        })
    if padded:
        findings.append({
            "kind": "graph_padding",
            "evidence": {
                "padded_steps": len(padded), "padded_rows": sum(step.padded_rows for step in padded),
                "max_padding_ratio": max(step.padded_rows / step.graph_batch_size for step in padded),
            },
            "source": "GraphPolicy.select / ModelRunner.run_model",
            "interpretation": "padding was executed; its presence alone does not prove a latency bottleneck",
        })
    tpots = [request["engine_tpot_ms"] for request in successful if request["engine_tpot_ms"] is not None]
    if tpot_budget_ms is not None and any(value > tpot_budget_ms for value in tpots):
        findings.append({
            "kind": "tpot_budget_exceeded",
            "evidence": {"budget_ms": tpot_budget_ms, "requests_over_budget": sum(v > tpot_budget_ms for v in tpots)},
            "interpretation": "engine-internal TPOT exceeded the chosen budget; this is not a client-side SLO measurement",
        })
    return {
        "schema_version": 1, "report_kind": "trace_analysis",
        "started_requests": started_requests, "finished_requests": len(requests),
        "successful_requests": len(successful),
        "cancelled_requests": sum(r["finish_reason"] == "cancelled" for r in requests),
        "failed_requests": sum(r["finish_reason"] == "error" for r in requests),
        "phases": phase, "engine_latency": latency,
        "prefix_cached_tokens": sum(step.prefix_cached_tokens for step in steps),
        "recomputed_tokens": recomputed, "preemptions": preemptions,
        "findings": findings,
        "scope": "trace counters and engine timestamps; no kernel-level or network-time inference",
    }


def compare_reports(baseline: dict, candidate: dict) -> dict:
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if (
            not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("report_kind") != "benchmark"
        ):
            raise ValueError(f"{label} is not a nonempty TraceServe benchmark report")
        try:
            if (
                not isinstance(report["runs"], list) or not report["runs"]
                or not isinstance(report["workload_fingerprint"], str)
                or not isinstance(report["backend"], dict)
                or report["cache_state"] not in {"cold", "warm"}
                or not isinstance(report["configuration"]["model"], str)
            ):
                raise ValueError("invalid report metadata")
            for run in report["runs"]:
                for key in ("elapsed_s", "output_throughput_tok_s"):
                    value = run[key]
                    if (
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0
                    ):
                        raise ValueError(f"{key} must be finite and positive")
                for key in ("engine_ttft_ms", "engine_tpot_ms"):
                    value = run["engine_latency"][key]["p50"]
                    if value is not None and (
                        not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 0
                    ):
                        raise ValueError(f"invalid {key} p50")
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"invalid {label} benchmark report: {exc}") from exc
    warnings = []
    for key in ("workload_fingerprint", "backend", "cache_state"):
        if baseline[key] != candidate[key]:
            warnings.append(f"{key} differs; treat comparison as descriptive")
    if baseline["configuration"]["model"] != candidate["configuration"]["model"]:
        warnings.append("model path differs; model equivalence has not been established")
    comparisons = {}
    for key in ("elapsed_s", "output_throughput_tok_s"):
        left = median(float(run[key]) for run in baseline["runs"])
        right = median(float(run[key]) for run in candidate["runs"])
        comparisons[key] = {"baseline_median": left, "candidate_median": right,
                            "change_percent": (right / left - 1) * 100}
    for key in ("engine_ttft_ms", "engine_tpot_ms"):
        left_values = [run["engine_latency"][key]["p50"] for run in baseline["runs"]
                       if run["engine_latency"][key]["p50"] is not None]
        right_values = [run["engine_latency"][key]["p50"] for run in candidate["runs"]
                        if run["engine_latency"][key]["p50"] is not None]
        comparisons[key] = {
            "baseline_median_of_p50": median(left_values) if left_values else None,
            "candidate_median_of_p50": median(right_values) if right_values else None,
        }
    return {"schema_version": 1, "report_kind": "comparison", "comparisons": comparisons,
            "warnings": warnings, "causality": "descriptive comparison, not proof of an optimization's cause"}
