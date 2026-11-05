"""Generation entry point; see examples/ for complete Python examples."""
import sys

from traceserve.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["generate", *sys.argv[1:]]))
