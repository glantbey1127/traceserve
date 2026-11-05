"""Opt-in real CUDA tests. No model is downloaded and no result is fabricated."""
import os

import pytest

from nanovllm.observability import TokenEvent
from nanovllm.sampling_params import SamplingParams


@pytest.mark.gpu
def test_real_qwen3_graph_eager_stream_and_cancel():
    model = os.environ.get("TRACESERVE_TEST_MODEL")
    if not model:
        pytest.skip("set TRACESERVE_TEST_MODEL to a local dense Qwen3 checkpoint")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from nanovllm.llm import LLM

    outputs = []
    for eager in (True, False):
        with LLM(
            model, max_model_len=256, max_num_seqs=20, max_resident_seqs=4,
            max_num_batched_tokens=256, enforce_eager=eager, decode_reserve_blocks=1,
            gpu_memory_utilization=0.8,
        ) as engine:
            result = engine.generate(["Hello", "What is 1+1?"], SamplingParams(
                temperature=0, max_tokens=3, ignore_eos=True,
            ), use_tqdm=False)
            outputs.append([output["token_ids"] for output in result])
            stream = engine.generate_stream(["Hello"], SamplingParams(max_tokens=20))
            assert isinstance(next(stream), TokenEvent)
            stream.close()
            assert engine.stats()["kv"]["used_blocks"] == 0
    assert outputs[0] == outputs[1]


@pytest.mark.gpu
def test_rmsnorm_float32_residual_is_not_mutated():
    if not os.environ.get("TRACESERVE_TEST_MODEL"):
        pytest.skip("opt-in GPU dependency environment required")
    torch = pytest.importorskip("torch")
    from nanovllm.layers.layernorm import RMSNorm

    layer = RMSNorm(4)
    layer.weight.data.fill_(1)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    residual = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    before_value, before_residual = value.clone(), residual.clone()
    normalized, summed = layer(value, residual)
    expected = before_value + before_residual
    assert torch.equal(value, before_value)
    assert torch.equal(residual, before_residual)
    assert torch.allclose(summed, expected)
    assert torch.allclose(normalized, expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + layer.eps))
