from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Callable, Literal
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from nanovllm.engine.llm_engine import ChatPrompt, LLMEngine
from nanovllm.observability import FinishEvent, TokenEvent
from nanovllm.sampling_params import SamplingParams
from traceserve.async_engine import AsyncEngine, RequestRejected


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    include_usage: bool = False


class RequestOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str
    max_tokens: int = 64
    temperature: float = 0.6
    stream: bool = False
    ignore_eos: bool = False
    n: Literal[1] = 1
    stream_options: StreamOptions | None = None


class CompletionRequest(RequestOptions):
    prompt: str | list[int]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: str
    content: str


class ChatRequest(RequestOptions):
    messages: list[Message]


def encode_sse(value: dict | str) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return f"data: {text}\n\n"


def create_app(
    factory: Callable[[], LLMEngine],
    model_name: str = "traceserve",
    *,
    max_pending_requests: int = 128,
) -> FastAPI:
    service = AsyncEngine(factory, max_pending_requests)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.start()
        app.state.engine = service
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(title="TraceServe", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestRejected)
    async def rejected_handler(request: Request, exc: RequestRejected):
        return JSONResponse(
            {"error": {"message": str(exc), "type": exc.error_type, "code": exc.error_type}},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        message = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        return JSONResponse(
            {"error": {"message": message, "type": "invalid_request", "code": "invalid_request"}},
            status_code=400,
        )

    @app.get("/health")
    async def health():
        ready = service.state == "ready"
        return JSONResponse({"status": service.state}, status_code=200 if ready else 503)

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": 0, "owned_by": "local"}
        ]}

    @app.get("/metrics")
    async def metrics():
        return service.stats()

    @app.delete("/v1/requests/{request_id}")
    async def cancel_request(request_id: str):
        if not service.cancel(request_id):
            raise RequestRejected("active request not found", 404, "not_found")
        return {"id": request_id, "status": "cancellation_requested"}

    async def respond(options: RequestOptions, prompt, *, chat: bool) -> Response:
        if options.model != model_name:
            raise RequestRejected(f"unknown model; served model is {model_name}", 404, "model_not_found")
        try:
            params = SamplingParams(
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                ignore_eos=options.ignore_eos,
            )
        except ValueError as exc:
            raise RequestRejected(str(exc)) from exc
        request_id = f"{'chatcmpl' if chat else 'cmpl'}-{uuid4().hex}"
        handle = await service.submit(prompt, params, request_id=request_id)
        created = int(time.time())
        if options.stream:
            async def chunks():
                base = {"id": request_id, "created": created, "model": model_name,
                        "object": "chat.completion.chunk" if chat else "text_completion"}

                def choice(content: str | None, reason: str | None = None, *, role: bool = False):
                    item = {"index": 0, "finish_reason": reason}
                    if chat:
                        item["delta"] = {"role": "assistant", "content": ""} if role else (
                            {} if content is None else {"content": content}
                        )
                    else:
                        item.update(text=content or "", logprobs=None)
                    return {**base, "choices": [item]}

                try:
                    if chat:
                        yield encode_sse(choice("", role=True))
                    async for event in handle:
                        if isinstance(event, TokenEvent):
                            if event.delta:
                                yield encode_sse(choice(event.delta))
                        elif isinstance(event, FinishEvent):
                            output = event.output
                            if output.error or output.finish_reason == "cancelled":
                                yield encode_sse({"error": {
                                    "message": output.error or "request cancelled",
                                    "type": "engine_error" if output.error else "request_cancelled",
                                }})
                                break
                            if event.delta:
                                yield encode_sse(choice(event.delta))
                            yield encode_sse(choice(None, output.finish_reason))
                            if options.stream_options and options.stream_options.include_usage:
                                yield encode_sse({**base, "choices": [], "usage": {
                                    "prompt_tokens": output.metrics["prompt_tokens"],
                                    "completion_tokens": len(output.token_ids),
                                    "total_tokens": output.metrics["prompt_tokens"] + len(output.token_ids),
                                }})
                except RequestRejected as exc:
                    yield encode_sse({"error": {"message": str(exc), "type": exc.error_type}})
                finally:
                    await handle.aclose()
                yield encode_sse("[DONE]")
            return StreamingResponse(
                chunks(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        try:
            async for event in handle:
                if isinstance(event, FinishEvent):
                    output = event.output
                    if output.error:
                        raise RequestRejected("model execution failed", 500, "engine_error")
                    if output.finish_reason == "cancelled":
                        raise RequestRejected("request cancelled", 409, "request_cancelled")
                    item = {"index": 0, "finish_reason": output.finish_reason}
                    if chat:
                        item["message"] = {"role": "assistant", "content": output.text}
                    else:
                        item.update(text=output.text, logprobs=None)
                    return JSONResponse({
                        "id": request_id,
                        "object": "chat.completion" if chat else "text_completion",
                        "created": created,
                        "model": model_name,
                        "choices": [item],
                        "usage": {
                            "prompt_tokens": output.metrics["prompt_tokens"],
                            "completion_tokens": len(output.token_ids),
                            "total_tokens": output.metrics["prompt_tokens"] + len(output.token_ids),
                        },
                    })
        finally:
            await handle.aclose()
        raise RuntimeError("request stream ended without a terminal event")

    @app.post("/v1/completions")
    async def completions(options: CompletionRequest) -> Response:
        return await respond(options, options.prompt, chat=False)

    @app.post("/v1/chat/completions")
    async def chat_completions(options: ChatRequest) -> Response:
        try:
            prompt = ChatPrompt(tuple(message.model_dump() for message in options.messages))
        except ValueError as exc:
            raise RequestRejected(str(exc)) from exc
        return await respond(options, prompt, chat=True)

    return app
