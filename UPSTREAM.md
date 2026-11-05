# Upstream provenance

TraceServe is a derivative of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
The starting revision is **bb823b3e06983d71485a8e1f23715ebd87d98ef8**.

The original MIT license and Copyright (c) 2025 Xingkai Yu are preserved in LICENSE.
The original model, attention kernels, tensor-parallel layers and weight-loading code remain attributed to
that project. TraceServe does not claim those implementations as original work.

TraceServe adds KV-aware admission and phase fairness, bounded graph-padding selection, incremental
request events, cancellation, request/step metrics, JSONL traces, an async single-owner serving worker,
HTTP/SSE endpoints, CLI commands, benchmark reports, and offline trace analysis.
It also adds greedy temperature-zero sampling, non-mutating float32 RMSNorm,
checkpoint-presence and TP-dimension checks, and explicit GPU-buffer cleanup on exit.

The local repository was created from the pinned public source files via the GitHub connector.
No upstream Git history is fabricated. No GPU performance results are included.
