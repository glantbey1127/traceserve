import random

import pytest

from nanovllm.engine.llm_engine import ChatPrompt, IncrementalDecoder
from nanovllm.observability import TokenEvent
from nanovllm.sampling_params import SamplingParams
from tests.support import FakeClock, make_engine


def test_generate_order_and_request_metrics(tmp_path):
    clock = FakeClock()
    with make_engine(tmp_path, clock=clock) as engine:
        outputs = engine.generate(["short", "other"], [
            SamplingParams(max_tokens=3, temperature=0),
            SamplingParams(max_tokens=1, temperature=0),
        ], use_tqdm=False)
        assert [len(output["token_ids"]) for output in outputs] == [3, 1]
        assert outputs[0]["text"] == "NOP"
        assert outputs[0]["finish_reason"] == "length"
        metrics = outputs[0]["metrics"]
        assert metrics["engine_ttft_ms"] == pytest.approx(10)
        assert metrics["engine_tpot_ms"] == pytest.approx(10)
        assert metrics["engine_itl_ms"] == pytest.approx([10, 10])
        assert metrics["engine_e2e_ms"] == pytest.approx(30)
        assert outputs[1]["metrics"]["engine_tpot_ms"] is None
        assert engine.stats()["totals"]["output_tokens"] == 4
        assert engine.stats()["kv"]["used_blocks"] == 0


def test_chunked_prefill_only_emits_after_last_chunk(tmp_path):
    with make_engine(tmp_path, max_num_batched_tokens=100) as engine:
        request_id = engine.add_request([1] * 250, SamplingParams(max_tokens=2))
        for expected_cached in (100, 200):
            step = engine.step_events()
            assert not step.events
            assert engine._records[request_id].seq.num_cached_tokens == expected_cached
        step = engine.step_events()
        assert len([event for event in step.events if isinstance(event, TokenEvent)]) == 1
        assert step.metrics.scheduled_tokens == 50
        final = engine.step_events()
        assert final.completed[0].token_ids == [65, 66]
        assert engine.is_finished()


def test_legacy_step_and_empty_step(tmp_path):
    with make_engine(tmp_path) as engine:
        request_id = engine.add_request("a", SamplingParams(max_tokens=1))
        results, signed_tokens = engine.step()
        assert results[0][1] == [65]
        assert signed_tokens == 1
        assert engine.get_output(request_id).finish_reason == "length"
        assert engine.step() == ([], 0)


def test_cancel_waiting_partial_and_running_releases_all_blocks(tmp_path):
    with make_engine(tmp_path, max_num_batched_tokens=100) as engine:
        waiting = engine.add_request("a", SamplingParams(max_tokens=3))
        assert engine.cancel_request(waiting).output.finish_reason == "cancelled"
        partial = engine.add_request([1] * 300, SamplingParams(max_tokens=3))
        engine.step_events()
        assert engine.scheduler.resident_count == 1
        assert engine.cancel_request(partial).output.finish_reason == "cancelled"
        running = engine.add_request("b", SamplingParams(max_tokens=3))
        engine.step_events()
        assert engine.cancel_request(running).output.token_ids == [65]
        assert engine.cancel_request("unknown") is None
        assert engine.stats()["kv"]["free_blocks"] == 64
        assert engine.is_finished()


def test_stream_close_cancels_all_its_requests(tmp_path):
    with make_engine(tmp_path, max_num_seqs=1) as engine:
        stream = engine.generate_stream(["first", "second"], SamplingParams(max_tokens=20))
        assert isinstance(next(stream), TokenEvent)
        stream.close()
        assert engine.is_finished()
        assert engine.stats()["totals"]["cancelled_requests"] == 2
        assert engine.stats()["kv"]["used_blocks"] == 0


def test_batch_admission_error_cleans_previously_admitted_requests(tmp_path):
    with make_engine(tmp_path) as engine:
        with pytest.raises(ValueError, match="at least one token"):
            list(engine.generate_stream(["valid", ""], SamplingParams(max_tokens=3)))
        assert engine.is_finished()
        assert not engine.metrics.active


@pytest.mark.parametrize("prompt", ["", [], [-1], [4096], [True], [1.5], {"text": "x"}])
def test_invalid_prompt_boundary(tmp_path, prompt):
    with make_engine(tmp_path) as engine:
        with pytest.raises((ValueError, TypeError)):
            engine.add_request(prompt, SamplingParams(max_tokens=1))
        assert engine.is_finished()


def test_request_ids_context_and_pool_capacity(tmp_path):
    with make_engine(tmp_path, max_model_len=512, num_kvcache_blocks=1) as engine:
        engine.add_request("a", SamplingParams(max_tokens=1), request_id="same")
        with pytest.raises(ValueError, match="duplicate"):
            engine.add_request("a", SamplingParams(max_tokens=1), request_id="same")
        with pytest.raises(ValueError, match="max_model_len"):
            engine.add_request([1] * 512, SamplingParams(max_tokens=1))
        with pytest.raises(ValueError, match="block pool"):
            engine.add_request([1] * 256, SamplingParams(max_tokens=2))
        with pytest.raises(RuntimeError, match="idle"):
            engine.reset_metrics()


def test_eos_and_ignore_eos(tmp_path):
    with make_engine(tmp_path, runner_kwargs={"token": 0}) as engine:
        outputs = engine.generate(["a", "b"], [
            SamplingParams(max_tokens=3), SamplingParams(max_tokens=3, ignore_eos=True),
        ], use_tqdm=False)
        assert outputs[0]["finish_reason"] == "stop"
        assert outputs[0]["token_ids"] == [0]
        assert outputs[1]["finish_reason"] == "length"
        assert outputs[1]["token_ids"] == [0, 0, 0]
        assert outputs[1]["text"] == ""


def test_engine_failure_finishes_all_requests_and_rejects_future_work(tmp_path):
    with make_engine(tmp_path, runner_kwargs={"fail_on": 1}) as engine:
        for request_id in ("first", "second"):
            engine.add_request("abc", SamplingParams(max_tokens=2), request_id=request_id)
        with pytest.raises(RuntimeError, match="restart"):
            engine.step_events()
        assert engine.is_finished()
        assert engine.stats()["kv"]["used_blocks"] == 0
        assert engine.stats()["totals"]["failed_requests"] == 2
        assert engine.get_output("first").error == "deliberate test runner failure"
        with pytest.raises(RuntimeError, match="restart"):
            engine.add_request("new", SamplingParams(max_tokens=1))


def test_small_history_does_not_lose_generate_results(tmp_path):
    with make_engine(tmp_path, metrics_history_size=1) as engine:
        outputs = engine.generate(["a", "b", "c"], SamplingParams(max_tokens=2), use_tqdm=False)
        assert len(outputs) == 3
        assert engine.stats()["retained_completed_requests"] == 1
        assert engine.stats()["totals"]["finished_requests"] == 3


def test_chat_template_and_invalid_roles(tmp_path):
    with make_engine(tmp_path) as engine:
        outputs = engine.generate(
            [ChatPrompt(({"role": "user", "content": "hello"},))],
            SamplingParams(max_tokens=1), use_tqdm=False,
        )
        assert outputs[0]["metrics"]["prompt_tokens"] > 5
    with pytest.raises(ValueError, match="roles"):
        ChatPrompt(({"role": "tool", "content": "x"},))
    with pytest.raises(ValueError, match="empty"):
        ChatPrompt(())


def test_incremental_decode_withholds_partial_utf8_and_flushes():
    class ByteTokenizer:
        def decode(self, ids, **kwargs):
            return bytes(ids).decode("utf-8", errors="replace")

    decoder = IncrementalDecoder(ByteTokenizer())
    assert decoder.update([0xE4]) == ""
    assert decoder.update([0xE4, 0xB8]) == ""
    assert decoder.update([0xE4, 0xB8, 0xAD]) == "中"
    assert decoder.update([0xE4, 0xB8, 0xAD, 0xF0]) == ""
    assert decoder.update([0xE4, 0xB8, 0xAD, 0xF0], final=True) == "\ufffd"


def test_resident_limit_and_prefill_decode_fairness(tmp_path):
    with make_engine(
        tmp_path, max_num_seqs=8, max_resident_seqs=2, max_num_batched_tokens=100,
    ) as engine:
        for _ in range(4):
            engine.add_request([1] * 250, SamplingParams(max_tokens=3))
        phases = []
        for _ in range(100):
            if engine.is_finished():
                break
            step = engine.step_events()
            phases.append(step.metrics.phase)
            assert engine.scheduler.resident_count <= 2
        assert engine.is_finished()
        assert "decode" in phases

    with make_engine(tmp_path, max_num_seqs=1) as engine:
        engine.add_request("a", SamplingParams(max_tokens=4), request_id="running")
        assert engine.step_events().metrics.phase == "prefill"
        engine.add_request("b", SamplingParams(max_tokens=4), request_id="waiting")
        assert engine.step_events().metrics.phase == "decode"


def test_decode_respects_small_token_budget(tmp_path):
    with make_engine(tmp_path, max_num_batched_tokens=2, max_num_seqs=8) as engine:
        for index in range(5):
            engine.add_request([index + 1], SamplingParams(max_tokens=3))
        while not engine.is_finished():
            result = engine.step_events()
            assert result.metrics.scheduled_tokens <= 2


@pytest.mark.parametrize("reserve, expected_recomputed, expected_preemptions", [(0, 44, 1), (1, 0, 0)])
def test_kv_admission_headroom_avoids_recomputation(tmp_path, reserve, expected_recomputed, expected_preemptions):
    with make_engine(tmp_path, num_kvcache_blocks=4, decode_reserve_blocks=reserve) as engine:
        outputs = engine.generate(
            [[1] * 512, [2] * 300], SamplingParams(max_tokens=2), use_tqdm=False,
        )
        assert [output["token_ids"] for output in outputs] == [[65, 66], [65, 66]]
        totals = engine.stats()["totals"]
        assert totals["recomputed_tokens"] == expected_recomputed
        assert totals["preemptions"] == expected_preemptions
        assert totals["prefill_tokens"] == (857 if reserve == 0 else 812)


def test_scheduler_forward_progress_across_mixed_lengths_and_small_pools(tmp_path):
    rng = random.Random(82)
    for pool_size in (3, 4, 6):
        with make_engine(
            tmp_path, num_kvcache_blocks=pool_size, max_num_seqs=3,
            max_resident_seqs=3, max_num_batched_tokens=97, decode_reserve_blocks=0,
        ) as engine:
            for index in range(9):
                length = rng.randint(1, min(650, pool_size * 256 - 4))
                engine.add_request([index + 1] * length, SamplingParams(max_tokens=4))
            for _ in range(300):
                if engine.is_finished():
                    break
                engine.step_events()
                manager = engine.scheduler.block_manager
                assert len(manager.used_block_ids) + len(manager.free_block_ids) == pool_size
                assert set(manager.free_block_ids).isdisjoint(manager.used_block_ids)
                assert all(manager.blocks[block].ref_count > 0 for block in manager.used_block_ids)
            assert engine.is_finished()
            assert engine.stats()["totals"]["finished_requests"] == 9
