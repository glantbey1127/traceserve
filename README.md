# TraceServe

[![CPU CI](https://github.com/glantbey1127/traceserve/actions/workflows/tests.yml/badge.svg)](https://github.com/glantbey1127/traceserve/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**面向 KV 容量调度、CUDA Graph 选择与请求可观测性的轻量 LLM 推理引擎。**

TraceServe 基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的固定版本改造，保留真实的 PyTorch / Triton / FlashAttention 执行链，支持单机 Qwen3 Dense 模型推理，并提供 Python、命令行与 HTTP/SSE 接口。

项目围绕三个工程问题展开：新请求的 Prefill 如何给 Decode 留出 KV 余量；Decode batch 怎样选择合适的 Graph 桶；发生排队、抢占与重算时，怎样用请求和 step 指标解释执行过程。


## 从原项目改了什么

| 模块 | 原有能力 | TraceServe 的改动 |
| --- | --- | --- |
| 调度 | Prefill 优先、KV 不足时抢占重算 | 独立驻留请求上限、接纳新请求时预留 Decode 空闲块、限制连续 Prefill 步数 |
| CUDA Graph | Decode 按固定 batch 桶 replay | 桶覆盖实际配置上限，支持自定义桶；补齐比例过大时回退 eager，记录选择原因 |
| 请求生命周期 | 阻塞式批量生成 | 结构化 Token / Finish 事件、主动取消、流关闭后的资源回收、异步多请求适配 |
| 可观测性 | 总耗时与吞吐打印 | 请求 TTFT / ITL / TPOT、排队耗时、重算量、前缀命中量、抢占与 Graph 补齐追踪 |
| 服务与工具 | Python 示例与简单 benchmark | HTTP / SSE、健康与指标接口、CLI、可重复构造负载的 benchmark、trace 分析与报告对比 |
| 兼容性修复 | 主要面向半精度推理 | 增加 temperature=0 贪心采样，修复 FP32 RMSNorm 输入别名修改问题 |

## 执行链与目录

~~~mermaid
flowchart TD
    Client["Python / CLI / HTTP"] --> Entry["LLMEngine：请求与 token 事件"]
    Entry --> Scheduler["Scheduler：接纳、Prefill / Decode、抢占"]
    Scheduler <--> KV["BlockManager：前缀共享与块引用"]
    Scheduler --> Runner["ModelRunner：输入准备与模型执行"]
    Runner --> Policy["GraphPolicy：Graph 桶或 eager"]
    Policy --> Model["Qwen3 → LMHead → Sampler"]
    Model --> Events["后处理 → TokenEvent / FinishEvent"]
    Events --> Observe["指标 / JSONL trace / SSE"]
~~~

~~~text
traceserve/
├── nanovllm/
│   ├── engine/           
│   ├── layers/          
│   ├── models/qwen3.py   
│   ├── utils/loader.py   
│   ├── graph.py          
│   └── observability.py  
├── traceserve/          
├── tests/               
├── examples/                     
├── .github/workflows/       
└── LICENSE              
~~~

## 环境与安装

先获取源码：

~~~bash
git clone https://github.com/glantbey1127/traceserve.git
cd traceserve
~~~

真实推理使用 **Linux + NVIDIA CUDA GPU**。

支持 Python 3.11–3.12。建立虚拟环境后，先按照 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/) 安装与本机 CUDA 匹配、且满足 torch>=2.4,<2.9 的版本。

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel packaging ninja
~~~

完成 PyTorch 安装后，检查 CUDA，再安装其余 GPU 依赖。FlashAttention 放在 PyTorch 之后构建，避免构建环境找不到 torch：

~~~bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install "transformers>=4.51,<5" "safetensors>=0.4,<1" "triton>=3,<3.5"
python -m pip install "flash-attn>=2.7,<3" --no-build-isolation
python -m pip install -e ".[gpu,http]"
~~~

Windows / 无 GPU 环境可以先验证控制层和接口。在安装了 Python 3.11 或 3.12 的 PowerShell 中运行：

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -e .
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu"
~~~

Linux / macOS 的 CPU 测试环境可运行：

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -e .
python -m pytest -q -m "not gpu"
~~~

## Python：批量与流式生成

~~~python
from traceserve import LLM, SamplingParams, TokenEvent, FinishEvent

with LLM(
    "/models/Qwen3-0.6B",
    max_model_len=2048,
    max_num_seqs=20,
    max_resident_seqs=8,
    max_num_batched_tokens=2048,
    decode_reserve_blocks=1,
    max_consecutive_prefill_steps=1,
    max_graph_padding_ratio=0.25,
    trace_path="artifacts/generation.jsonl",
) as llm:
    params = SamplingParams(temperature=0, max_tokens=64)

    outputs = llm.generate(["Explain KV cache briefly."], params, use_tqdm=False)
    print(outputs[0]["text"])
    print(outputs[0]["metrics"])

    stream = llm.generate_stream(["What is continuous batching?"], params)
    try:
        for event in stream:
            if isinstance(event, TokenEvent):
                print(event.delta, end="", flush=True)
            elif isinstance(event, FinishEvent):
                print(event.delta, end="", flush=True)
                print("\n", event.output.finish_reason)
    finally:
        stream.close()
~~~

## 命令行

~~~bash
traceserve generate --model /models/Qwen3-0.6B \
  --prompt "Explain paged KV cache." --max-tokens 64 --temperature 0 --stream

traceserve serve --model /models/Qwen3-0.6B \
  --model-name traceserve --max-num-seqs 20 --max-resident-seqs 8 \
  --max-model-len 2048 --max-batched-tokens 2048 \
  --trace artifacts/server.jsonl
~~~


~~~bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"traceserve","messages":[{"role":"user","content":"Explain KV cache."}],"max_tokens":64,"temperature":0,"stream":true}'
~~~

实现了常用 OpenAI 风格接口的**明确子集**：单条文本 / token ID completion，纯文本 system / user / assistant chat，temperature、max_tokens、ignore_eos、stream 和流式 usage。其他字段会显式拒绝，不声称完整协议兼容。接口、错误码及取消语义见 [docs/api.md](docs/api.md)。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | /health | 引擎就绪与失败状态 |
| GET | /v1/models | 服务模型名称 |
| GET | /metrics | 请求、调度和 KV 的 JSON 指标 |
| POST | /v1/completions | 文本或 token ID 生成 |
| POST | /v1/chat/completions | 纯文本聊天及 SSE |
| DELETE | /v1/requests/{id} | 请求取消 |

## 关键配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| max_num_seqs | 512 | 一次模型调用的请求数上限；CLI 默认 32 |
| max_resident_seqs | 与 max_num_seqs 相同 | 持有 KV 块的请求数上限，包含未完成 Prefill 的请求 |
| max_num_batched_tokens | 16384 | 单步实际计算的 token 预算 |
| decode_reserve_blocks | 1 | 接纳额外 Prefill 时预留的空闲块数 |
| max_consecutive_prefill_steps | 1 | 有 Running 请求时最多连续执行的 Prefill 步数 |
| max_graph_padding_ratio | 0.25 | 补齐行占图桶的比例上限；超出则 eager |
| graph_batch_sizes | 自动生成 | 自定义 Graph 桶；仍会补入配置上限 |
| temperature | 1.0 | SamplingParams 默认值；0 表示贪心；CLI / HTTP 的默认值见各接口 |
| trace_path | None | 可选 JSONL trace 输出路径 |

图桶实际最大值取 max_num_seqs、max_num_batched_tokens 和 cuda_graph_max_batch_size 的最小值。完整参数可查 Config 与 traceserve --help。

## 测量与分析

~~~bash
traceserve benchmark --model /models/Qwen3-0.6B \
  --num-requests 64 --min-input-tokens 256 --max-input-tokens 768 \
  --min-output-tokens 16 --max-output-tokens 64 \
  --shared-prefix-tokens 256 --temperature 0 --seed 7 \
  --max-model-len 2048 --max-batched-tokens 2048 \
  --repeats 3 --warmup-requests 2 --cache-state cold \
  --trace artifacts/run.jsonl --output artifacts/run.json

traceserve analyze artifacts/run.jsonl --tpot-budget-ms 100 --output artifacts/analysis.json
traceserve compare artifacts/baseline.json artifacts/run.json --output artifacts/comparison.json
~~~

## 测试与代码入口

~~~bash
python -m pytest -q -m "not gpu"
python -m ruff check .
python -m compileall -q nanovllm traceserve tests examples
~~~


- [架构与源码阅读路线](docs/architecture.md)
- [接口说明](docs/api.md)
- [指标与测量口径](docs/metrics.md)
- [来源与改造边界](UPSTREAM.md)
- [当前验证状态](PROJECT_STATUS.md)
- [GitHub 首次上传与后续更新完整流程](docs/github-publish.md)

