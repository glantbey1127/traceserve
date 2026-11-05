import asyncio
import json
import threading

import pytest
from fastapi.testclient import TestClient

from nanovllm.observability import FinishEvent, TokenEvent
from nanovllm.sampling_params import SamplingParams
from traceserve.async_engine import AsyncEngine, RequestRejected
from traceserve.server import create_app
from tests.support import make_engine


def test_async_concurrency_runs_engine_on_one_owner_thread(tmp_path):
    created = []

    def factory():
        engine = make_engine(tmp_path, runner_kwargs={"delay": 0.002})
        created.append(engine)
        return engine

    async def exercise():
        service = AsyncEngine(factory)
        await service.start()

        async def consume(prompt):
            return [event async for event in service.generate(prompt, SamplingParams(max_tokens=3))]

        results = await asyncio.gather(consume("a"), consume("b"), consume("c"))
        assert all(len([event for event in events if isinstance(event, TokenEvent)]) == 3 for events in results)
        assert all(isinstance(events[-1], FinishEvent) for events in results)
        await service.aclose()
        assert service.state == "closed"

    asyncio.run(asyncio.wait_for(exercise(), 10))
    runner = created[0].model_runner
    assert runner.closed
    assert len(set(runner.threads)) == 1
    assert runner.threads[0] != threading.get_ident()


def test_async_backpressure_and_cancellation(tmp_path):
    created = []

    def factory():
        engine = make_engine(tmp_path)
        engine.model_runner.release = threading.Event()
        created.append(engine)
        return engine

    async def exercise():
        service = AsyncEngine(factory, max_pending_requests=1)
        await service.start()
        handle = await service.submit("a", SamplingParams(max_tokens=50), request_id="cancel-me")
        try:
            with pytest.raises(RequestRejected) as error:
                await service.submit("b", SamplingParams(max_tokens=1))
            assert error.value.status_code == 429
            assert service.cancel(handle.request_id)
        finally:
            created[0].model_runner.release.set()
        events = [event async for event in handle]
        assert isinstance(events[-1], FinishEvent)
        assert events[-1].output.finish_reason == "cancelled"
        await service.aclose()
        assert created[0].stats()["kv"]["used_blocks"] == 0
        assert not service.cancel("not-active")

    asyncio.run(asyncio.wait_for(exercise(), 10))


def test_async_validation_failure_and_startup_failure(tmp_path):
    async def exercise():
        service = AsyncEngine(lambda: make_engine(tmp_path))
        await service.start()
        with pytest.raises(RequestRejected) as error:
            await service.submit([], SamplingParams(max_tokens=1))
        assert error.value.status_code == 400
        assert service.stats()["pending_requests"] == 0
        await service.aclose()

        def fail():
            raise RuntimeError("test startup failure")

        service = AsyncEngine(fail)
        with pytest.raises(RuntimeError, match="startup"):
            await service.start()
        assert service.state == "failed"
        await service.aclose()

    asyncio.run(asyncio.wait_for(exercise(), 10))


def test_async_execution_failure_delivers_terminal_to_every_request(tmp_path):
    async def exercise():
        service = AsyncEngine(lambda: make_engine(tmp_path, runner_kwargs={"fail_on": 1}))
        await service.start()
        handles = await asyncio.gather(*[
            service.submit("x", SamplingParams(max_tokens=2), request_id=str(index))
            for index in range(3)
        ], return_exceptions=True)
        for handle in handles:
            if isinstance(handle, RequestRejected):
                assert handle.status_code == 503
            else:
                events = [event async for event in handle]
                assert events[-1].output.finish_reason == "error"
        assert service.state == "failed"
        with pytest.raises(RequestRejected) as error:
            await service.submit("x", SamplingParams(max_tokens=1))
        assert error.value.status_code == 503
        await service.aclose()

    asyncio.run(asyncio.wait_for(exercise(), 10))


@pytest.fixture
def client(tmp_path):
    app = create_app(lambda: make_engine(tmp_path), "test-model")
    with TestClient(app) as client:
        yield client


def test_http_nonstream_completion_chat_health_and_metrics(client):
    assert client.get("/health").json() == {"status": "ready"}
    assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"
    completion = client.post("/v1/completions", json={
        "model": "test-model", "prompt": [1, 2], "max_tokens": 3, "temperature": 0,
    })
    assert completion.status_code == 200
    body = completion.json()
    assert body["choices"][0]["text"] == "NOP"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    chat = client.post("/v1/chat/completions", json={
        "model": "test-model", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2,
    })
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"] == {"role": "assistant", "content": "NO"}
    metrics = client.get("/metrics").json()
    assert metrics["totals"]["finished_requests"] == 2
    assert metrics["kv"]["used_blocks"] == 0
    assert client.delete("/v1/requests/unknown").status_code == 404


@pytest.mark.parametrize("chat", [False, True])
def test_sse_content_finish_usage_and_done(client, chat):
    route = "/v1/chat/completions" if chat else "/v1/completions"
    payload = {
        "model": "test-model", "max_tokens": 3, "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(messages=[{"role": "user", "content": "hi"}]) if chat else payload.update(prompt="hi")
    with client.stream("POST", route, json=payload) as response:
        assert response.status_code == 200
        frames = [line[6:] for line in response.iter_lines() if line.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    chunks = [json.loads(frame) for frame in frames[:-1]]
    if chat:
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
        text = "".join(
            chunk["choices"][0]["delta"].get("content", "")
            for chunk in chunks if chunk["choices"]
        )
    else:
        text = "".join(chunk["choices"][0]["text"] for chunk in chunks if chunk["choices"])
    assert text == "NOP"
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] == 3


@pytest.mark.parametrize("payload, status", [
    ({"model": "wrong", "prompt": "x", "max_tokens": 1}, 404),
    ({"model": "test-model", "prompt": "", "max_tokens": 1}, 400),
    ({"model": "test-model", "prompt": [True], "max_tokens": 1}, 400),
    ({"model": "test-model", "prompt": "x", "max_tokens": 0}, 400),
    ({"model": "test-model", "prompt": "x", "temperature": -1}, 400),
    ({"model": "test-model", "prompt": "x", "n": 2}, 400),
    ({"model": "test-model", "prompt": "x", "stop": ["!"]}, 400),
    ({"model": "test-model", "prompt": "x", "max_tokens": "2"}, 400),
])
def test_http_rejects_invalid_or_unsupported_input_before_streaming(client, payload, status):
    payload["stream"] = True
    response = client.post("/v1/completions", json=payload)
    assert response.status_code == status
    assert "error" in response.json()


def test_chat_role_validation_and_runtime_error(tmp_path, client):
    response = client.post("/v1/chat/completions", json={
        "model": "test-model", "messages": [{"role": "tool", "content": "x"}], "max_tokens": 1,
    })
    assert response.status_code == 400
    with TestClient(create_app(lambda: make_engine(tmp_path, runner_kwargs={"fail_on": 1}), "failed")) as broken:
        response = broken.post("/v1/completions", json={"model": "failed", "prompt": "x", "max_tokens": 1})
        assert response.status_code == 500
        assert broken.get("/health").status_code == 503


def test_asgi_stream_disconnect_cancels_request_and_releases_kv(tmp_path):
    async def exercise():
        app = create_app(lambda: make_engine(tmp_path, runner_kwargs={"delay": 0.005}), "test-model")
        disconnected = asyncio.Event()
        first_receive = True
        body = json.dumps({
            "model": "test-model", "prompt": "x", "max_tokens": 50, "stream": True,
        }).encode()

        async def receive():
            nonlocal first_receive
            if first_receive:
                first_receive = False
                return {"type": "http.request", "body": body, "more_body": False}
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body" and b"data: " in message.get("body", b""):
                disconnected.set()

        scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": "POST", "scheme": "http",
            "path": "/v1/completions", "raw_path": b"/v1/completions",
            "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
        }
        async with app.router.lifespan_context(app):
            await app(scope, receive, send)
            assert disconnected.is_set()
            for _ in range(200):
                stats = app.state.engine.stats()
                if stats["totals"]["cancelled_requests"] == 1:
                    break
                await asyncio.sleep(0.005)
            assert stats["totals"]["cancelled_requests"] == 1
            assert stats["totals"]["output_tokens"] < 50
            assert stats["kv"]["used_blocks"] == 0

    asyncio.run(asyncio.wait_for(exercise(), 10))
