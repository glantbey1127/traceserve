# 验证状态

截至 2026-11-05，TraceServe 的本次实现与 CPU / API 验证已完成。真实 GPU 推理路径需要在有 CUDA 和本地模型的环境继续做冒烟验证；本目录没有性能实验结果。

## 已实现

- 保留 nano-vLLM 的真实 Qwen3 Dense 模型、Attention / Triton KV 写入、张量并行层和权重加载链。
- KV 接纳时预留 Decode 空闲块，独立限制驻留请求数，限制连续 Prefill 步数；Decode 同样遵守 token 预算。
- 完整 Graph 桶上限覆盖、自定义桶与按补齐比例回退 eager。
- Python 批量生成、Token / Finish 事件、主动取消、生成器关闭回收、请求和 step 指标、JSONL trace。
- 单线程持有推理引擎的 AsyncEngine，并发请求、接纳上限、执行失败终止、关闭流程。
- HTTP completions / chat、SSE、usage、健康状态、JSON 指标、主动取消和客户端断连后的资源回收。
- generate / benchmark / analyze / compare / serve CLI，以及完整示例、架构与指标说明。
- temperature=0 贪心、FP32 RMSNorm 输入别名修复；模型类型与 TP 维度检查、无权重目录报错、关闭时清理模型 / KV / Graph 缓冲区。
- 构建配置、源码包清单和 CPU CI 工作流。

公开接口均有实际实现。tests/support.py 中的 TestRunner 只用于 CPU 控制逻辑测试，不是可选的推理后端。原层实现中的抽象基类和类型 Protocol 是接口定义，具体模型路径均有对应实现。

## 实际执行的检查

| 检查 | 结果 |
| --- | --- |
| python -m pytest -q | **90 passed，2 skipped** |
| python -m ruff check . | 通过 |
| python -m compileall -q nanovllm traceserve tests examples | 通过 |
| editable 安装与 traceserve 命令入口 | 通过 |
| python -m build --no-isolation | 源码包与 wheel 构建通过 |
| 将 wheel 安装到独立目录，再从该目录导入与执行 CLI help | 通过，已核对导入路径来自安装目录 |
| wheel 中的模型 / runner / loader / 服务模块及原 MIT 许可证 | 已检查 |
| 源码包内的文档、示例、测试支持文件、CI 与来源说明 | 已检查 |

CPU 用例覆盖前缀块引用计数、空闲缓存复用、chunked Prefill、驻留数限制、公平轮换、Decode token 预算、KV 抢占与重算、EOS / 贪心参数、取消、生成器关闭、异步并发、异常终止、HTTP / SSE、ASGI 断连回收、报告解析与命令行错误。

当前有一条第三方弃用提示：Starlette TestClient 使用的 AnyIO BlockingPortal 别名已弃用。该提示不影响本次用例通过。

## 验证环境

- Windows，Python 3.12.14。
- NumPy 2.3.5，xxhash 3.6.0，pytest 8.4.2。
- FastAPI 0.115.12，Pydantic 2.13.5，Starlette 0.46.2，HTTPX 0.28.1。
- Ruff 0.11.13，setuptools 80.9.0。
- 没有安装 PyTorch / CUDA / Triton / FlashAttention，没有本地 GPU 模型。
- requirements-dev.txt 固定主要 CPU 开发依赖；GPU 环境需另外按本机 CUDA 安装。

## 尚未经过硬件验证的部分

2 项 GPU 用例因没有设置 TRACESERVE_TEST_MODEL 而跳过：

1. 真实 Qwen3 eager / CUDA Graph 贪心输出一致性，以及流式取消回收。
2. 使用真实 PyTorch 运行 FP32 RMSNorm 输入不被修改的检查。

真实模型加载、CUDA / Triton / FlashAttention 执行、显存预算、GPU 图捕获和多卡 TP 均未在当前主机运行。这些路径保留完整实现和可选测试，但 CPU 用例不能代替 GPU 验证。

## 支持边界

- Linux + NVIDIA CUDA，dense Qwen3，float16 / bfloat16；单机单引擎实例。
- TP 要求注意力头、KV 头、中间层维度和词表维度能够被 TP 数整除。多卡通信沿用上游固定本地地址与共享内存方案，未经真实多卡验证。
- HTTP 是纯文本、单候选的 OpenAI 风格接口子集；未知或未支持字段明确拒绝。
- 同步 LLM 只允许单调用者；并发通过 AsyncEngine 或 HTTP。
- CUDA 调用期间的取消在下一调度边界生效；执行失败后需要重启。
- 未实现训练、MoE、多模态、量化、LoRA、投机解码、跨机器调度或生产级多租户服务，也没有为这些能力提供空壳入口。
- 本次没有运行吞吐 / 延迟实验，没有生成或宣称任何性能提升百分比。
