"""Compare the gfx908 mHC path with the accepted FP32/BF16 decomposition."""
import pytest
import torch

from vllm import ir
import vllm.kernels.vllm_c  # Register the production normalization provider.
from vllm.model_executor.kernels.mhc import gfx908
from vllm.model_executor.kernels.mhc.torch import mhc_pre_torch, mhc_post_torch


@pytest.fixture(autouse=True, params=['native', 'vllm_c'])
def norm_provider(request):
    with ir.ops.rms_norm.set_priority([request.param]):
        yield


@pytest.fixture(autouse=True, params=[(False, False), (True, False),
                                     (False, True), (True, True)])
def decode_fastpaths(request, monkeypatch):
    project, norm = request.param
    monkeypatch.setattr(gfx908, 'PROJECT_DECODE', project)
    monkeypatch.setattr(gfx908, 'FUSE_PRE_NORM', norm)


@pytest.mark.parametrize("tokens", [1, 2, 4, 16, 128, 2048])
@pytest.mark.parametrize("norm", [False, True])
def test_pre_and_layer_transition(tokens, norm):
    torch.manual_seed(170 + tokens)
    residual = torch.randn(tokens, 4, 4096, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, 16384, device="cuda") * 0.01
    scale = torch.tensor([0.2, 0.5, 0.7], device="cuda")
    base = torch.randn(24, device="cuda")
    weight = torch.randn(4096, device="cuda", dtype=torch.bfloat16) if norm else None
    args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    expected = mhc_pre_torch(residual, *args)
    if norm:
        expected = (*expected[:2], ir.ops.rms_norm(expected[2], weight, 1e-6))
    actual = gfx908.pre(residual, *args, norm_weight=weight, norm_eps=1e-6)
    for a, b in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
    # BF16 rounding ties can differ by one representable step after an FP32
    # reduction. Use a norm error bound as well as an elementwise bound.
    error = (actual[2].float() - expected[2].float()).norm() / expected[2].float().norm()
    assert error < 0.001, error
    torch.testing.assert_close(actual[2], expected[2], rtol=0.016, atol=0.001)
    x = torch.randn_like(expected[2])
    ref_res = mhc_post_torch(x, residual, expected[0], expected[1])
    out = gfx908.post(x, residual, expected[0], expected[1])
    assert (out.float() - ref_res.float()).norm() / ref_res.float().norm() < 0.001
    got = gfx908.fused_post_pre(x, residual, expected[0], expected[1], *args,
        norm_weight=weight, norm_eps=1e-6)
    ref = mhc_pre_torch(ref_res, *args)
    if norm:
        ref = (*ref[:2], ir.ops.rms_norm(ref[2], weight, 1e-6))
    for a, b in zip(got[1:3], ref[:2]):
        torch.testing.assert_close(a, b, rtol=0.002, atol=0.0002)
    assert (got[3].float() - ref[2].float()).norm() / ref[2].float().norm() < 0.002


@pytest.mark.parametrize("magnitude", [1.0, 100.0])
def test_noncontiguous_and_extreme(magnitude):
    torch.manual_seed(51)
    storage = torch.randn(4, 4, 8192, device="cuda", dtype=torch.bfloat16) * magnitude
    residual = storage[..., ::2]
    assert not residual.is_contiguous()
    fn = torch.randn(24, 16384, device="cuda") * 0.01
    scale = torch.tensor([0.2, 0.5, 0.7], device="cuda")
    base = torch.randn(24, device="cuda") * magnitude
    args = (fn, scale, base, 1e-6, 1e-6, 1e-6, 2.0, 20)
    expected = mhc_pre_torch(residual.contiguous(), *args)
    actual = gfx908.pre(residual, *args)
    for a, b in zip(actual[:2], expected[:2]):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=3e-6, atol=3e-7)
    assert (actual[2].float() - expected[2].float()).norm() / expected[2].float().norm() < 0.001
