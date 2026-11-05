# 架构与源码路线

## 数据流

~~~text
CLI / Python / HTTP
        |
  AsyncEngine（仅异步服务）
  接收命令队列；专用线程持有 LLM
        |
  LLMEngine.add_request
        | tokenization / 参数与上下文检查 / RequestMetrics
  Scheduler.schedule
        | resident cap / admission reserve / prefill fairness / preemption
  BlockManager
        | block allocation / prefix hash / reference counting
  ModelRunner.run
        | prepare_prefill 或 prepare_decode
        | GraphPolicy.select -> eager 或 CUDA Graph replay
        | Qwen3 -> RMSNorm -> Attention / MLP -> LMHead -> Sampler
  Scheduler.postprocess
        | hash completed blocks / update cached prefix / append token / release finished requests
  LLMEngine.step_events
        | token-ready timestamp / incremental text / TokenEvent / FinishEvent / StepMetrics
  Python iterator 或 HTTP SSE
~~~

引擎层没有用 HTTP 请求处理函数直接触碰 GPU。AsyncEngine 把提交、取消、关闭命令传给一个线程；模型构造、step、KV 修改、退出都在该线程执行。事件通过 event loop 的线程安全投递函数送回客户端。

服务有 max_pending_requests 上限，用 429 拒绝过量请求。它包含运行中和等待中的请求，与 max_resident_seqs 不同；后者只限制拥有 KV 块的请求。

## 核心改造

### 1. KV 容量感知调度

代码：nanovllm/engine/scheduler.py、block_manager.py。

max_num_seqs 限制一次模型调用的请求数；max_resident_seqs 限制跨步仍持有 KV 的总请求数，包括尚未完成 chunked Prefill 的请求。

接纳一个新的 Prefill 时，如果已有驻留请求，要求分配完成后仍有 decode_reserve_blocks 个空闲块。第一条请求不应用预留，避免单请求无法启动。预留不会创建物理隔离池，Decode 和后续请求都按当前真实空闲量判断。

存在 Running 请求时，最多连续执行 max_consecutive_prefill_steps 次 Prefill，然后尝试 Decode。Prefill 与 Decode 分开执行；没有实现混合 batch 或优先级队列。

Decode 遇到块不足，仍沿用上游的尾部请求抢占。被抢占请求回到 Waiting，其 token 序列保留，KV 引用释放；已完成完整块的前缀可能从缓存重新命中，不完整块通过 Prefill 重算。

computed_high_watermark 记录该请求历史上实际计算到的最大逻辑位置。新一轮 Prefill 的计算区间与旧高水位重叠的 token 计为 recomputed_tokens。pending 的新 token 从未计算过，不应算作重算。

### 2. Graph 策略

代码：nanovllm/graph.py、engine/model_runner.py。

默认桶包含不超过上限的 1、2、4、8 和 16 的倍数，并额外加入实际上限；上限取 max_num_seqs、max_num_batched_tokens 和 cuda_graph_max_batch_size 的最小值。max_num_seqs=20 且 token 预算足够时，17 条 Decode 请求可以使用 20 行的桶。

补齐比例定义为 (bucket - real_batch) / bucket。超过阈值时使用 eager。选择结果包含原因，写入 step trace。所有桶在初始化时捕获；选择策略减少不必要的 replay 补齐，不会消除已经捕获的图本身占用的资源。

静态 slot_mapping 每步先填 -1，context_lens 先归零，再写真实请求，保证补齐行不会向 KV cache 写入有效 token。block_tables 只在 context_lens 描述的范围内有效。

### 3. 请求生命周期与可观测性

代码：engine/llm_engine.py、observability.py、traceserve/async_engine.py。

每个请求一个 RequestRecord，包含 Sequence、RequestMetrics 和增量 decoder。流式生成与批量生成复用 step_events，不维护第二套调度实现。

正常结束、主动取消、生成器关闭和模型执行错误都有对应终止事件。取消只能在模型调用之间生效，不能撤回已经发出的 token 或打断正在执行的 CUDA kernel。模型执行失败后引擎拒绝继续使用，需要重启。

已完成请求和 step 历史有容量上限；累计计数不受历史淘汰影响。详细逐 token 时间保留到请求结束；长输出会有相应 CPU 内存开销。

## 推荐精读顺序

1. config.py、sampling_params.py：配置边界、容量和采样约束。
2. engine/sequence.py：请求状态、KV 对应逻辑位置、序列化。
3. engine/block_manager.py：哈希链、缓存引用、free / used 状态。
4. engine/scheduler.py：接纳、分块 Prefill、公平轮换、抢占、取消。
5. engine/model_runner.py：峰值预热、KV 预算、张量准备、图捕获与 replay。
6. models/qwen3.py：Decoder、注意力、MLP 和最终残差。
7. layers/attention.py、linear.py、embed_head.py：KV 读写和张量并行。
8. utils/loader.py：checkpoint 权重合并和各 rank 的加载。
9. graph.py、observability.py、engine/llm_engine.py：策略、时间口径和流式生命周期。
10. traceserve/async_engine.py、server.py：异步服务如何保持引擎单一所有者。
11. benchmark.py、analysis.py：负载、计时、归因边界。
12. tests/：将代码行为写成可执行检查。

## 适用范围

保留上游单机 TP 通信实现，真实多卡路径未经本工作环境验证。默认使用固定本地进程组地址和共享内存命名，同一机器同时启动多个 TP 引擎实例不在支持范围内。不要在一个进程内同时构造多个真实 GPU 引擎。

没有实现训练、跨机器通信、KV CPU offload、量化、投机解码、LoRA 热切换、多模态编码器、MoE 或任意 checkpoint 架构适配。项目所有公开的新接口均有实际实现；这些未包含的能力没有以空函数形式占位。
