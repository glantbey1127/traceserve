import argparse
import asyncio
from contextlib import aclosing

from traceserve import FinishEvent, LLM, SamplingParams, TokenEvent
from traceserve.async_engine import AsyncEngine


async def main():
    parser = argparse.ArgumentParser(description="Concurrent requests with one real inference owner")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    service = AsyncEngine(lambda: LLM(
        args.model, max_model_len=2048, max_num_seqs=20, max_num_batched_tokens=2048,
    ))
    await service.start()

    async def consume(prompt):
        async with aclosing(service.generate(prompt, SamplingParams(temperature=0, max_tokens=32))) as events:
            async for event in events:
                if isinstance(event, TokenEvent):
                    print(f"{event.request_id}: {event.delta!r}")
                elif isinstance(event, FinishEvent):
                    print(f"{event.request_id} done: {event.output.finish_reason}, final={event.delta!r}")

    try:
        await asyncio.gather(
            consume("What is prefill?"),
            consume("What is decode?"),
        )
    finally:
        await service.aclose()


if __name__ == "__main__":
    asyncio.run(main())
