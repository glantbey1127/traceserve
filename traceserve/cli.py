from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def add_engine_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model", required=True, help="local Qwen3 checkpoint directory")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-resident-seqs", type=int)
    parser.add_argument("--max-batched-tokens", type=int, default=16384)
    parser.add_argument("--decode-reserve-blocks", type=int, default=1)
    parser.add_argument("--max-consecutive-prefill-steps", type=int, default=1)
    parser.add_argument("--memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-blocks", type=int, default=-1)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--cuda-graph-max-batch-size", type=int, default=512)
    parser.add_argument("--graph-buckets", help="comma-separated graph sizes; maximum is always added")
    parser.add_argument("--max-graph-padding-ratio", type=float, default=0.25)
    parser.add_argument("--trace", help="optional JSONL trace path")


def engine_config(args):
    from nanovllm.config import Config
    buckets = None if args.graph_buckets is None else tuple(int(value) for value in args.graph_buckets.split(","))
    return Config(
        model=args.model, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
        max_resident_seqs=args.max_resident_seqs, max_num_batched_tokens=args.max_batched_tokens,
        decode_reserve_blocks=args.decode_reserve_blocks,
        max_consecutive_prefill_steps=args.max_consecutive_prefill_steps,
        gpu_memory_utilization=args.memory_utilization, tensor_parallel_size=args.tp,
        enforce_eager=args.eager, num_kvcache_blocks=args.kv_blocks,
        cuda_graph_max_batch_size=args.cuda_graph_max_batch_size,
        graph_batch_sizes=buckets, max_graph_padding_ratio=args.max_graph_padding_ratio,
        trace_path=args.trace,
    )


def write_json(value: dict | list, output: str | None):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    if output is None:
        print(text)
    else:
        path = Path(output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(str(path.resolve()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traceserve", description="Observable nano-vLLM derivative")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="local generation")
    add_engine_arguments(generate)
    generate.add_argument("--prompt", action="append", required=True)
    generate.add_argument("--max-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.6)
    generate.add_argument("--ignore-eos", action="store_true")
    generate.add_argument("--stream", action="store_true", help="emit structured JSONL token/finish events")
    benchmark = commands.add_parser("benchmark", help="offline engine benchmark")
    add_engine_arguments(benchmark)
    benchmark.add_argument("--num-requests", type=int, default=256)
    benchmark.add_argument("--min-input-tokens", type=int, default=100)
    benchmark.add_argument("--max-input-tokens", type=int, default=1024)
    benchmark.add_argument("--min-output-tokens", type=int, default=100)
    benchmark.add_argument("--max-output-tokens", type=int, default=1024)
    benchmark.add_argument("--shared-prefix-tokens", type=int, default=0)
    benchmark.add_argument("--seed", type=int, default=0)
    benchmark.add_argument("--temperature", type=float, default=0)
    benchmark.add_argument("--repeats", type=int, default=1)
    benchmark.add_argument("--warmup-requests", type=int, default=1)
    benchmark.add_argument("--cache-state", choices=["cold", "warm"], default="cold")
    benchmark.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze", help="aggregate a JSONL trace without GPU dependencies")
    analyze.add_argument("trace")
    analyze.add_argument("--tpot-budget-ms", type=float)
    analyze.add_argument("--output")
    compare = commands.add_parser("compare", help="descriptive comparison of two benchmark reports")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--output")
    serve = commands.add_parser("serve", help="start a local HTTP/SSE server")
    add_engine_arguments(serve)
    serve.add_argument("--model-name", default="traceserve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--max-pending-requests", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            from traceserve.analysis import analyze_trace
            write_json(analyze_trace(args.trace, tpot_budget_ms=args.tpot_budget_ms), args.output)
        elif args.command == "compare":
            from traceserve.analysis import compare_reports
            baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
            write_json(compare_reports(baseline, candidate), args.output)
        elif args.command == "serve":
            import uvicorn
            from nanovllm.llm import LLM
            from traceserve.server import create_app
            config = engine_config(args)
            app = create_app(lambda: LLM(config), args.model_name, max_pending_requests=args.max_pending_requests)
            uvicorn.run(app, host=args.host, port=args.port, workers=1)
        else:
            from nanovllm.llm import LLM
            config = engine_config(args)
            with LLM(config) as engine:
                if args.command == "generate":
                    from nanovllm.sampling_params import SamplingParams
                    params = SamplingParams(args.temperature, args.max_tokens, args.ignore_eos)
                    if args.stream:
                        for event in engine.generate_stream(args.prompt, params):
                            print(json.dumps(asdict(event), ensure_ascii=False), flush=True)
                    else:
                        write_json(engine.generate(args.prompt, params, use_tqdm=False), None)
                else:
                    from traceserve.benchmark import Workload, run_benchmark
                    workload = Workload(
                        args.num_requests, args.min_input_tokens, args.max_input_tokens,
                        args.min_output_tokens, args.max_output_tokens, args.shared_prefix_tokens,
                        args.seed, args.temperature,
                    )
                    report = run_benchmark(engine, workload, repeats=args.repeats,
                                           warmup_requests=args.warmup_requests, cache_state=args.cache_state)
                    write_json(report, args.output)
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, TypeError, RuntimeError, OSError, ImportError, KeyError) as exc:
        print(f"traceserve: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
