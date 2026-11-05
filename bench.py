"""Benchmark entry point; implementation lives in traceserve.benchmark."""
import sys

from traceserve.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
