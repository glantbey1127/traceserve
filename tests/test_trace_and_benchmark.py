import copy
import json
import subprocess
import sys

import pytest

from nanovllm.sampling_params import SamplingParams
from traceserve.analysis import analyze_trace, compare_reports
from traceserve.benchmark import Workload, run_benchmark
from tests.support import FakeClock, make_engine


def test_trace_recomputations_and_latency_aggregate(tmp_path):
    path = tmp_path / "trace.jsonl"
    with make_engine(tmp_path, clock=FakeClock(), num_kvcache_blocks=4,
                     decode_reserve_blocks=0, trace_path=str(path)) as engine:
        engine.generate([[1] * 512, [2] * 300], SamplingParams(max_tokens=2), use_tqdm=False)
    report = analyze_trace(path, tpot_budget_ms=1)
    assert report["started_requests"] == report["finished_requests"] == 2
    assert report["preemptions"] == 1
    assert report["recomputed_tokens"] == 44
    assert report["prefix_cached_tokens"] == 256
    assert report["phases"]["prefill"]["scheduled_tokens"] == 857
    assert report["phases"]["decode"]["scheduled_tokens"] == 1
    assert report["engine_latency"]["engine_ttft_ms"]["count"] == 2
    assert {finding["kind"] for finding in report["findings"]} == {"kv_pressure", "tpot_budget_exceeded"}
    assert all(record["schema_version"] == 1 for record in map(json.loads, path.read_text().splitlines()))


def test_trace_cancelled_requests_are_excluded_from_success_latency(tmp_path):
    path = tmp_path / "cancel.jsonl"
    with make_engine(tmp_path, trace_path=str(path)) as engine:
        request = engine.add_request("x", SamplingParams(max_tokens=2))
        engine.cancel_request(request)
    report = analyze_trace(path)
    assert report["cancelled_requests"] == 1
    assert report["successful_requests"] == 0
    assert report["engine_latency"]["engine_e2e_ms"]["count"] == 0


@pytest.mark.parametrize("text", [
    "not-json\n", "[]\n",
    '{"schema_version":2,"event":"step","data":{}}\n',
    '{"schema_version":1,"event":"step","data":{}}\n',
    '{"schema_version":1,"event":"request_finished","data":{"metrics":{}}}\n',
])
def test_invalid_trace_has_line_number(tmp_path, text):
    path = tmp_path / "invalid.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        analyze_trace(path)


def test_trace_rejects_inconsistent_graph_metadata_at_input_boundary(tmp_path):
    path = tmp_path / "valid.jsonl"
    with make_engine(tmp_path, trace_path=str(path)) as engine:
        engine.generate(["x"], SamplingParams(max_tokens=2), use_tqdm=False)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    step = next(record for record in records if record["event"] == "step")
    step["data"].update(execution_mode="cuda_graph", graph_batch_size=None, padded_rows=1)
    path.write_text(json.dumps(step) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        analyze_trace(path)


@pytest.mark.parametrize("invalid", [[], None, "not a report", {"schema_version": 1}])
def test_compare_rejects_unknown_report_shapes(invalid):
    with pytest.raises(ValueError, match="benchmark report"):
        compare_reports(invalid, {})


def test_benchmark_uses_actual_outputs_repeated_seed_and_excludes_warmup(tmp_path):
    path = tmp_path / "bench.jsonl"
    with make_engine(tmp_path, metrics_history_size=1, trace_path=str(path)) as engine:
        report = run_benchmark(engine, Workload(
            num_requests=5, min_input_tokens=260, max_input_tokens=270,
            min_output_tokens=2, max_output_tokens=3, shared_prefix_tokens=256, seed=19,
        ), repeats=2, warmup_requests=2)
    assert report["backend"]["performance_representative"] is False
    assert report["report_kind"] == "benchmark"
    assert len(report["workload_fingerprint"]) == 64
    for run in report["runs"]:
        assert run["completed_requests"] == 5
        assert run["output_tokens"] == run["target_output_tokens"]
        assert run["engine_counters"]["finished_requests"] == 5
        assert run["engine_latency"]["engine_ttft_ms"]["count"] == 5
    trace = analyze_trace(path)
    assert trace["started_requests"] == 10
    assert trace["finished_requests"] == 10
    same = compare_reports(report, report)
    assert not same["warnings"]
    assert same["comparisons"]["output_throughput_tok_s"]["change_percent"] == 0

    candidate = copy.deepcopy(report)
    candidate["workload_fingerprint"] = "different"
    candidate["runs"][0]["output_throughput_tok_s"] *= 2
    assert compare_reports(report, candidate)["warnings"]


def test_cli_without_gpu_dependencies_and_error_exit(tmp_path):
    help_result = subprocess.run(
        [sys.executable, "-m", "traceserve", "--help"], capture_output=True, text=True, check=False,
    )
    assert help_result.returncode == 0
    assert "benchmark" in help_result.stdout
    trace = tmp_path / "empty.jsonl"
    trace.write_text("", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "traceserve", "analyze", str(trace)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout)["finished_requests"] == 0
    failed = subprocess.run(
        [sys.executable, "-m", "traceserve", "analyze", str(tmp_path / "missing")],
        capture_output=True, text=True, check=False,
    )
    assert failed.returncode == 1
    assert "traceserve:" in failed.stderr


def test_invalid_workload_and_report(tmp_path):
    with pytest.raises(ValueError):
        Workload(num_requests=0)
    with pytest.raises(ValueError):
        Workload(min_input_tokens=100, max_input_tokens=90)
    with pytest.raises(ValueError):
        Workload(shared_prefix_tokens=200)
    with pytest.raises(ValueError):
        compare_reports({}, {})
    with make_engine(tmp_path) as engine:
        with pytest.raises(ValueError, match="repeats"):
            run_benchmark(engine, Workload(), repeats=0)
