# 指标与测量口径

## 请求延迟

所有引擎时间使用同一个进程的单调时钟 perf_counter。同步 add_request 开始时计到达时间；HTTP 请求在 AsyncEngine.submit 入队时计到达时间，包含 worker 命令队列等待，不包含 HTTP JSON 解析前的时间。

每步模型调用采样结果转回 CPU 并完成 scheduler.postprocess 后，记录 token-ready 时间。流式文本解码、Python / HTTP 格式化和网络传输发生在该时刻之后。因此这些字段全部明确命名为 engine_*，不宣称客户端延迟。

假设一条请求到达时刻为 a，首次调度为 s，生成 M 个 token，其 token-ready 时刻为 t1 ... tM，结束为 f：

| 指标 | 定义 |
| --- | --- |
| queue_ms | (s - a) × 1000 |
| engine_ttft_ms | (t1 - a) × 1000 |
| engine_itl_ms | [(t2 - t1), ..., (tM - t(M-1))] × 1000 |
| engine_tpot_ms | (tM - t1) × 1000 / (M - 1)，M < 2 时 null |
| engine_e2e_ms | (f - a) × 1000 |

特殊 token 即使解码成空文本仍计为 token。未生成 token 的取消请求 TTFT / TPOT 为 null。累计完成、取消、失败分别计数；默认延迟分布只取正常 stop / length 结束的请求。

百分位数使用排序后的线性插值。stats() 的延迟分布覆盖最近保留的正常完成请求，不等价于整个服务历史；累计 totals 永不因历史淘汰而减少。离线 benchmark 从本轮完整输出单独汇总，不受 metrics_history_size 限制。

## Step 与 KV

- scheduled_tokens：本步真实计算的逻辑 token 数；Prefill 为所有 chunk 长度之和，Decode 为实际 batch 数。
- output_tokens：本步真正附加到请求的生成 token 数。未完成 chunk 的临时采样被丢弃，不算输出。
- recomputed_tokens：本步计算区间与该请求已计算高水位的重叠；前缀缓存直接复用的 token 不算重算。
- prefix_cached_tokens：一次接纳或恢复时复用的完整前缀块 token 数之和。重复恢复可能多次累计同一逻辑前缀，不表示唯一 token 个数。
- preempted_request_ids：本次调度因 Decode KV 不足而释放的请求。
- real_batch_size：真实请求数。
- model_rows：本步真实输入 token 行数，Prefill 不等于请求数。
- graph_batch_size / padded_rows：图桶大小与补齐行数，eager 时为 null / 0。
- execution_reason：prefill、forced_eager、batch_limit、padding_limit、bucket_match。
- free_blocks_before / after：调度前与该步后处理后的 KV 空闲块数；“空闲”不代表块内 KV 字节已清零，可能保留缓存。
- runner_wall_ms：输入准备、模型执行、采样和结果回 CPU 的 CPU 墙钟耗时。
- wall_ms：该 step 的调度到事件构造耗时；含文本解码，不含随后向 trace 写入 step 记录的时间。

phase_throughput_tok_s = 该阶段累计 scheduled_tokens / 该阶段累计 step 墙钟秒数。这里用总量除总时间，不能对各步 tokens/s 直接取算术平均。该值不是 attention kernel 吞吐，也不是全流程输出吞吐。

## 离线 benchmark

run_benchmark 先使用独立 random.Random(seed) 构造 token ID 输入与输出长度，记录 SHA256 指纹。不会修改模型采样 RNG 的 seed；temperature>0 时负载相同不意味着生成结果完全一致。需要确定性比较时优先用 temperature=0。

- input_tokens 是每条请求原始输入长度之和，不因命中缓存而减少。
- output_tokens 来自实际返回结果；ignore_eos=true 使合成负载按指定长度结束。
- output_throughput_tok_s = 实际输出 token / 整次 generate 墙钟秒数。
- total_logical_throughput_tok_s = (原始输入 + 实际输出) / 同一墙钟秒数。
- request_throughput_req_s = 正常完成请求 / 同一墙钟秒数。
- 内部 phase 吞吐使用真实被调度的 token，其中包含 KV 重算；与逻辑总吞吐分母、分子都不同。

模型加载、显存峰值预热、Graph 捕获、负载构造和显式 warmup 均在 benchmark 计时之外。计时包含请求接纳、调度、输入准备、GPU 计算、采样、D2H、文本解码和开启 trace 时的写盘。

cache_state=cold：每轮开始前清除前缀元数据，保留已加载模型与已捕获图。它不表示冷启动。
cache_state=warm：不清除前缀缓存；第一轮仅继承 warmup / 原有工作负载状态，不保证所有请求前缀都已命中。后续轮次可从前一轮缓存受益。报告中如实记录该条件。

benchmark trace 会在 warmup 后重新打开；第一轮以前的记录被截断，从而不会把预热请求混入分析。多轮 trace 包含 benchmark_run_started，analyze 当前按整个文件汇总。独立轮次的精确统计在 benchmark JSON 的 runs 数组中。

compare 会检查负载指纹、后端、cache_state 和模型路径差异并提示。相同路径不保证 checkpoint 内容相同；要做严格对比还需固定模型文件、软件版本、GPU 状态与实验条件。输出仅为描述性结果，不代表因果证明。

## Trace schema v1

每行一个 JSON 对象：schema_version=1、event、timestamp_s、data。事件类型包括 request_started、token、request_finished、preemption、step、engine_error、cache_cleared、benchmark_run_started。

request_started 不记录原始 prompt 内容；token / request_finished 包含生成文本和 token ID，开启追踪会写入这些输出。默认不开启 trace。

analyze 对输入字段、时间数值和版本进行校验，并给出错误行号。它只推断 trace 直接支持的现象，无法代替 CUDA profiler、NCCL tracing 或真实 HTTP 负载测试。
