# 接口说明

## 同步引擎

导出：from traceserve import LLM, Config, SamplingParams, ChatPrompt。

~~~python
LLM(model: str | Config, **config_kwargs)
SamplingParams(temperature=1.0, max_tokens=64, ignore_eos=False)
~~~

model 是本地 checkpoint 目录。temperature=0 表示贪心，正值使用 softmax 后的指数随机数采样；不支持 top_p、top_k 或 stop 字符串。max_tokens 必须是正整数。

提示词可为 str、单条 list[int]，或 ChatPrompt(tuple[dict])。ChatPrompt 的每条消息恰有 role / content，role 支持 system、user、assistant，content 为文本。由本地 tokenizer 的 chat template 编码。

| 方法 | 返回值 / 行为 |
| --- | --- |
| add_request(prompt, params, request_id=None, arrived_at=None) | 返回请求 ID；立即验证非空输入、token ID、上下文与单请求 KV 容量 |
| step_events() | StepResult：events、completed、metrics、num_tokens |
| step() | 保留旧版 (完成的 seq_id / token_ids 列表, 有符号 token 计数) |
| generate(prompts, params, use_tqdm=True) | 按输入顺序返回输出 dict 列表；params 可为公共参数或等长参数列表 |
| generate_stream(prompts, params) | 多请求事件迭代器；关闭后取消该迭代器尚未完成的请求 |
| cancel_request(request_id) | 未完成请求返回 FinishEvent；未知或已完成返回 None |
| get_request_metrics(request_id) | 活跃或最近保留请求的指标；未知 / 已淘汰返回 None |
| get_output(request_id) | 最近保留的 GenerationOutput；未知 / 已淘汰返回 None |
| stats() | 累计计数、最近保留请求的延迟分布、最近一步、KV 状态 |
| reset_metrics(clear_trace=False) | 仅空闲时允许；清空指标历史和完成输出，可选截断 trace |
| clear_prefix_cache() | 仅无使用中 KV 块时允许；清空前缀索引 |
| close() / exit() | 取消未完成请求、关闭 runner 和 trace；可重复调用 |

generate 的 prompts 是“多条请求的集合”。单条 token 提示词写作 [[1, 2, 3]]，不要把 [1, 2, 3] 直接当多请求输入。

GenerationOutput 的字段为 request_id、seq_id、text、token_ids、finish_reason、metrics、error。finish_reason 为 stop / length / cancelled / error。

TokenEvent 包含 request_id、seq_id、token_id、index、delta、timestamp、type="token"。FinishEvent 包含 request_id、output、delta、type="finished"。delta 可能为空，最后的 FinishEvent.delta 需要一并输出，避免遗漏 tokenizer 的最终缓冲文本。

结构化流示例见 examples/generate.py。不要同时从多个调用者消费同一个同步引擎，也不要把同步生成直接运行在异步 HTTP event loop 上。

## 异步引擎

~~~python
from traceserve.async_engine import AsyncEngine
service = AsyncEngine(lambda: LLM("/models/Qwen3-0.6B"), max_pending_requests=128)
await service.start()
stream = await service.submit("Hello", SamplingParams(max_tokens=16))
try:
    async for event in stream:
        ...
finally:
    await stream.aclose()
await service.aclose()
~~~

上面的省略号仅表示调用者自己的事件处理；完整可运行示例在 examples/async_generate.py。

- submit 在接纳检查完成后才返回，请求参数错误不会伪装成已启动的 SSE。
- stream.accepted 给出 request_id 和 prompt_tokens。
- service.generate(prompt, params) 是自动关闭 stream 的异步生成器。
- service.cancel(request_id) 返回是否找到活跃请求，表示已发出取消命令；实际回收发生在下一步前。
- stream.aclose() 停止消费并请求取消；在模型调用期间不能同步抢占 CUDA kernel。
- service.aclose() 关闭 worker，未完成请求得到 cancelled 终止事件。
- RequestRejected 包含 status_code 和 error_type。
- 所有 service / stream 操作在创建 service 的同一 asyncio loop 上调用。

## HTTP

create_app(factory, model_name="traceserve", max_pending_requests=128) 返回实际 FastAPI 应用。factory 在专用推理线程内创建 LLM，生命周期结束时关闭。

| 方法与路径 | 说明 |
| --- | --- |
| GET /health | ready 时 200；执行失败后 503 |
| GET /v1/models | 返回唯一服务模型的名称 |
| GET /metrics | JSON 指标快照，不是 Prometheus 文本格式 |
| POST /v1/completions | 单条字符串或 token ID 列表 completion |
| POST /v1/chat/completions | 纯文本消息对话 |
| DELETE /v1/requests/{id} | 返回 200 + cancellation_requested，未知活跃 ID 返回 404 |

DELETE 的响应只确认排队的取消命令，不代表此时 GPU 已停止计算。请求 ID 在普通响应或首个 SSE chunk 中返回。

公共请求字段：

| 字段 | 规则 |
| --- | --- |
| model | 必填，必须与服务 model_name 一致 |
| max_tokens | 默认 64，正整数；prompt + max_tokens 不超过 max_model_len |
| temperature | 默认 0.6，有限非负数；0 为贪心 |
| ignore_eos | 默认 false |
| stream | 默认 false |
| n | 只接受 1 |
| stream_options | 可选 {"include_usage": true} |

completions 必须额外传 prompt: str | list[int]。chat 必须额外传 messages: [{"role": "...", "content": "..."}]。拒绝未知字段以及工具调用、图片、stop、logprobs、top_p 等未支持功能。

非流式返回 choices、usage 和模型 / 请求元信息。流式使用 text/event-stream，每帧为 data: JSON 加空行，以 data: [DONE] 结束；聊天先发 assistant role，再发 content delta。最后发送 finish_reason，include_usage=true 时再发 choices=[] 的 usage 帧。

| 状态码 | 含义 |
| --- | --- |
| 400 | 请求格式、参数、prompt 或上下文非法，或使用未支持字段 |
| 404 | 模型名称错误，或取消的活跃 ID 不存在 |
| 409 | 非流式请求被主动取消 |
| 429 | 等待和运行请求总量达到 max_pending_requests |
| 500 | 非流式请求执行失败 |
| 503 | 引擎不就绪或已经失败 |

SSE 头已发出后，运行错误 / 主动取消通过 error 帧传达，再结束数据流；HTTP 状态码不能回退修改。客户端断开 SSE 时自动请求取消，已经发生的计算和已发送 token 不会撤回。

## 兼容与边界

这是常用 OpenAI 风格请求 / 响应的子集；没有实现完整 SDK 协议、批量 prompt、多候选、工具调用、Responses API、文件上传或 embedding。所有未支持字段显式报错，避免默默忽略。

GPU 执行异常后健康状态变为 failed，需要重启服务。单机 TP 沿用上游进程组和共享内存通信；本项目未实现故障容错集群。
