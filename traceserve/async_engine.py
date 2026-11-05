from __future__ import annotations

import asyncio
import queue
import threading
from dataclasses import dataclass
from time import perf_counter
from typing import Callable
from uuid import uuid4

from nanovllm.engine.llm_engine import LLMEngine, Prompt
from nanovllm.observability import FinishEvent, GenerationEvent
from nanovllm.sampling_params import SamplingParams


class RequestRejected(Exception):
    def __init__(self, message: str, status_code: int = 400, error_type: str = "invalid_request"):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type


@dataclass(frozen=True, slots=True)
class Accepted:
    request_id: str
    prompt_tokens: int


@dataclass(frozen=True, slots=True)
class Rejected:
    error: RequestRejected


@dataclass(slots=True)
class RequestStream:
    request_id: str
    owner: AsyncEngine
    events: asyncio.Queue
    accepted: Accepted | None = None
    terminal: bool = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> GenerationEvent:
        if self.terminal:
            raise StopAsyncIteration
        event = await self.events.get()
        if isinstance(event, Rejected):
            self.terminal = True
            raise event.error
        if isinstance(event, FinishEvent):
            self.terminal = True
        return event

    async def aclose(self):
        if not self.terminal:
            self.terminal = True
            self.owner.cancel(self.request_id)


@dataclass(frozen=True, slots=True)
class Submit:
    handle: RequestStream
    prompt: Prompt
    params: SamplingParams
    arrived_at: float


@dataclass(frozen=True, slots=True)
class Cancel:
    request_id: str


@dataclass(frozen=True, slots=True)
class Shutdown:
    """Stop command; it carries no payload."""


class AsyncEngine:
    """All mutable engine/GPU operations run on one dedicated worker thread."""

    def __init__(self, factory: Callable[[], LLMEngine], max_pending_requests: int = 128):
        if max_pending_requests < 1:
            raise ValueError("max_pending_requests must be positive")
        self.factory = factory
        self.max_pending_requests = max_pending_requests
        self._commands: queue.Queue = queue.Queue()
        self._inflight: dict[str, RequestStream] = {}
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._startup_error: Exception | None = None
        self._state = "new"
        self._snapshot: dict = {}

    @property
    def state(self) -> str:
        return self._state

    def stats(self) -> dict:
        return {"state": self._state, "pending_requests": len(self._inflight), **self._snapshot}

    async def start(self):
        if self._state != "new":
            raise RuntimeError("AsyncEngine.start() may only be called once")
        self._loop = asyncio.get_running_loop()
        self._state = "starting"
        self._thread = threading.Thread(target=self._work, name="traceserve-worker", daemon=True)
        self._thread.start()
        await asyncio.to_thread(self._ready.wait)
        if self._startup_error is not None:
            raise RuntimeError("inference worker startup failed") from self._startup_error

    def _deliver(self, handle: RequestStream, event):
        # This callback and all submit/cancel calls run on the owning asyncio loop.
        if isinstance(event, (FinishEvent, Rejected)):
            self._inflight.pop(handle.request_id, None)
        handle.events.put_nowait(event)

    def _publish(self, handle: RequestStream, event):
        self._loop.call_soon_threadsafe(self._deliver, handle, event)

    async def submit(
        self,
        prompt: Prompt,
        params: SamplingParams,
        *,
        request_id: str | None = None,
    ) -> RequestStream:
        if self._state != "ready":
            raise RequestRejected("inference worker is not ready", 503, "engine_unavailable")
        if len(self._inflight) >= self.max_pending_requests:
            raise RequestRejected("pending request limit reached", 429, "queue_full")
        request_id = uuid4().hex if request_id is None else request_id
        if request_id in self._inflight:
            raise RequestRejected("duplicate active request ID")
        handle = RequestStream(request_id, self, asyncio.Queue())
        self._inflight[request_id] = handle
        self._commands.put(Submit(handle, prompt, params, perf_counter()))
        try:
            first = await handle.events.get()
        except asyncio.CancelledError:
            await handle.aclose()
            raise
        if isinstance(first, Rejected):
            handle.terminal = True
            raise first.error
        handle.accepted = first
        return handle

    def cancel(self, request_id: str) -> bool:
        if request_id not in self._inflight:
            return False
        self._commands.put(Cancel(request_id))
        return True

    async def generate(self, prompt: Prompt, params: SamplingParams):
        stream = await self.submit(prompt, params)
        try:
            async for event in stream:
                yield event
        finally:
            await stream.aclose()

    async def aclose(self):
        if self._state == "closed":
            return
        if self._thread is None:
            self._state = "closed"
            return
        self._state = "closing"
        self._commands.put(Shutdown())
        await asyncio.to_thread(self._thread.join)
        self._state = "closed"

    def _work(self):
        engine = None
        handles: dict[str, RequestStream] = {}
        broken = False
        try:
            try:
                engine = self.factory()
                self._snapshot = engine.stats()
                self._state = "ready"
            except Exception as exc:
                self._startup_error = exc
                self._state = "failed"
                return
            finally:
                self._ready.set()
            while True:
                commands = []
                if engine.is_finished():
                    commands.append(self._commands.get())
                while len(commands) < 64:
                    try:
                        commands.append(self._commands.get_nowait())
                    except queue.Empty:
                        break
                for command in commands:
                    if isinstance(command, Shutdown):
                        for request_id, handle in list(handles.items()):
                            event = engine.cancel_request(request_id)
                            if event is not None:
                                self._publish(handle, event)
                        # Commands already drained behind Shutdown must also receive a terminal response.
                        for pending in commands[commands.index(command) + 1:]:
                            if isinstance(pending, Submit):
                                self._publish(pending.handle, Rejected(
                                    RequestRejected("server is shutting down", 503, "engine_unavailable")
                                ))
                        self._snapshot = engine.stats()
                        return
                    if isinstance(command, Submit):
                        if broken:
                            self._publish(command.handle, Rejected(
                                RequestRejected("model execution failed", 503, "engine_unavailable")
                            ))
                            continue
                        try:
                            request_id = engine.add_request(
                                command.prompt, command.params,
                                request_id=command.handle.request_id,
                                arrived_at=command.arrived_at,
                            )
                        except (ValueError, TypeError) as exc:
                            self._publish(command.handle, Rejected(RequestRejected(str(exc))))
                        else:
                            handles[request_id] = command.handle
                            metrics = engine.get_request_metrics(request_id)
                            self._publish(command.handle, Accepted(request_id, metrics["prompt_tokens"]))
                    elif isinstance(command, Cancel):
                        handle = handles.pop(command.request_id, None)
                        if handle is not None:
                            event = engine.cancel_request(command.request_id)
                            if event is not None:
                                self._publish(handle, event)
                if not engine.is_finished() and not broken:
                    try:
                        result = engine.step_events()
                        for event in result.events:
                            handle = handles.get(event.request_id)
                            if handle is not None:
                                self._publish(handle, event)
                                if isinstance(event, FinishEvent):
                                    handles.pop(event.request_id)
                    except Exception as exc:
                        broken = True
                        self._state = "failed"
                        for request_id, handle in list(handles.items()):
                            output = engine.get_output(request_id)
                            if output is None:
                                event = engine.cancel_request(request_id, error=str(exc))
                            else:
                                event = FinishEvent(request_id, output)
                            if event is not None:
                                self._publish(handle, event)
                        handles.clear()
                self._snapshot = engine.stats()
        finally:
            if engine is not None:
                engine.close()
            self._ready.set()
