import argparse
from contextlib import closing

from traceserve import FinishEvent, LLM, SamplingParams, TokenEvent


def main():
    parser = argparse.ArgumentParser(description="Real Qwen3 batch and streaming inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain why KV cache helps decoding.")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    with LLM(
        args.model, max_model_len=2048, max_num_seqs=20, max_resident_seqs=8,
        max_num_batched_tokens=2048, trace_path="artifacts/example.jsonl",
    ) as engine:
        output = engine.generate([args.prompt], params, use_tqdm=False)[0]
        print("Batch:", output["text"])
        print("Metrics:", output["metrics"])
        with closing(engine.generate_stream([args.prompt], params)) as stream:
            for event in stream:
                if isinstance(event, TokenEvent):
                    print(event.delta, end="", flush=True)
                elif isinstance(event, FinishEvent):
                    print(event.delta, end="", flush=True)
                    print(f"\nFinished: {event.output.finish_reason}")


if __name__ == "__main__":
    main()
