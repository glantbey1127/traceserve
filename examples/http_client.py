import argparse
import json
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description="Text-only SSE client; uses the Python standard library")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="traceserve")
    parser.add_argument("--prompt", default="Explain KV cache in three sentences.")
    args = parser.parse_args()
    payload = {
        "model": args.model, "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": 64, "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
    }
    request = Request(
        args.base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=120) as response:
        for line in response:
            text = line.decode("utf-8").strip()
            if not text.startswith("data: "):
                continue
            frame = text[6:]
            if frame == "[DONE]":
                break
            chunk = json.loads(frame)
            if "error" in chunk:
                raise RuntimeError(chunk["error"]["message"])
            if chunk["choices"]:
                print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
            if "usage" in chunk:
                print("\nUsage:", chunk["usage"])
    print()


if __name__ == "__main__":
    main()
