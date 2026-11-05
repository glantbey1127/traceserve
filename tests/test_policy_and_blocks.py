import pickle

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.graph import GraphPolicy
from nanovllm.sampling_params import SamplingParams
from tests.support import make_config


@pytest.mark.parametrize("maximum", [1, 3, 7, 17, 20, 513])
def test_graph_buckets_cover_exact_maximum(maximum):
    policy = GraphPolicy(maximum)
    assert policy.buckets[-1] == maximum
    assert policy.select(maximum).bucket == maximum
    assert all(0 < bucket <= maximum for bucket in policy.buckets)


def test_graph_padding_fallback_and_custom_buckets():
    policy = GraphPolicy(32, (32, 4, 4, 1), max_padding_ratio=0.25)
    assert policy.buckets == (1, 4, 32)
    assert policy.select(17).reason == "padding_limit"
    assert policy.select(24).bucket == 32
    assert policy.select(33).reason == "batch_limit"
    assert policy.select(2, enforce_eager=True).reason == "forced_eager"
    assert policy.select(100, is_prefill=True).reason == "prefill"
    with pytest.raises(ValueError):
        policy.select(0)


@pytest.mark.parametrize("kwargs", [
    {"max_batch_size": 0}, {"max_batch_size": 20, "buckets": (21,)},
    {"max_batch_size": 20, "buckets": (True,)},
    {"max_batch_size": 20, "max_padding_ratio": 1},
    {"max_batch_size": 20, "max_padding_ratio": float("nan")},
])
def test_invalid_graph_configuration(kwargs):
    with pytest.raises(ValueError):
        GraphPolicy(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_num_seqs": 0}, {"max_model_len": -1}, {"num_kvcache_blocks": 0},
    {"kvcache_block_size": 128}, {"gpu_memory_utilization": 1},
    {"max_resident_seqs": 0}, {"tensor_parallel_size": 9},
    {"decode_reserve_blocks": -1}, {"max_consecutive_prefill_steps": 0},
    {"num_kvcache_blocks": True}, {"num_kvcache_blocks": 2.5},
    {"max_resident_seqs": True}, {"decode_reserve_blocks": True},
])
def test_invalid_engine_configuration(tmp_path, kwargs):
    with pytest.raises(ValueError):
        make_config(tmp_path, **kwargs)


def test_missing_model_and_position_clamp(tmp_path):
    with pytest.raises(ValueError, match="local directory"):
        make_config(tmp_path / "missing")
    config = make_config(tmp_path, max_model_len=8192)
    assert config.max_model_len == 4096
    assert config.max_resident_seqs == config.max_num_seqs
    limited = make_config(tmp_path, max_num_batched_tokens=3)
    assert limited.graph_policy.max_batch_size == 3


@pytest.mark.parametrize("kwargs", [
    {"temperature": -1}, {"temperature": float("nan")},
    {"temperature": float("inf")}, {"max_tokens": 0}, {"max_tokens": True},
])
def test_invalid_sampling(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_shared_prefix_references_and_free_cache_retention():
    Sequence.block_size = 256
    manager = BlockManager(4, 256)
    first = Sequence([1] * 300)
    manager.allocate(first, manager.can_allocate(first))
    first.num_scheduled_tokens = 300
    manager.hash_blocks(first)
    prefix_block = first.block_table[0]

    second = Sequence([1] * 256 + [2] * 44)
    assert manager.can_allocate(second) == 1
    manager.allocate(second, 1)
    assert second.block_table[0] == prefix_block
    assert manager.blocks[prefix_block].ref_count == 2
    assert len(manager.used_block_ids) == 3
    with pytest.raises(RuntimeError, match="idle"):
        manager.clear_cache()

    manager.deallocate(first)
    assert manager.blocks[prefix_block].ref_count == 1
    manager.deallocate(second)
    assert not manager.used_block_ids
    assert manager.blocks[prefix_block].hash != -1

    third = Sequence([1] * 300)
    assert manager.can_allocate(third) == 1
    manager.allocate(third, 1)
    assert third.num_cached_tokens == 256
    manager.deallocate(third)
    manager.clear_cache()
    assert manager.can_allocate(Sequence([1] * 300)) == 0


def test_hash_chain_requires_the_same_previous_blocks():
    manager = BlockManager(6, 256)
    first = Sequence([1] * 256 + [2] * 256 + [3])
    manager.allocate(first, 0)
    first.num_scheduled_tokens = 513
    manager.hash_blocks(first)
    manager.deallocate(first)
    assert manager.can_allocate(Sequence([9] * 256 + [2] * 256 + [3])) == 0
    assert manager.can_allocate(Sequence([1] * 256 + [2] * 256 + [4])) == 2
    # The final block is never reused, even if it happens to be full.
    assert manager.can_allocate(Sequence([1] * 256 + [2] * 256)) == 1


def test_sequence_worker_serialization_for_prefill_and_decode():
    sequence = Sequence([1, 2, 3], SamplingParams(max_tokens=2))
    sequence.num_scheduled_tokens = 3
    sequence.block_table = [7]
    restored = pickle.loads(pickle.dumps(sequence))
    assert restored.token_ids == [1, 2, 3]
    assert restored.block_table == [7]
    sequence.is_prefill = False
    sequence.append_token(65)
    restored = pickle.loads(pickle.dumps(sequence))
    assert restored.last_token == 65
    assert restored.num_tokens == 4
