import importlib.util

import json

import pathlib

import sys

import traceback

import types

import torch

def load(name, relative):
    import importlib
    return importlib.import_module(relative.removesuffix('.py').replace('/', '.'))


def check(name, fn):
    try:
        detail = fn()
        torch.cuda.synchronize()
        result = {'test': name, 'status': 'pass', 'detail': detail}
    except Exception as exc:
        result = {'test': name, 'status': 'fail', 'error': type(exc).__name__, 'detail': str(exc)[-3000:]}
    results.append(result)
    print(json.dumps(result), flush=True)

def config_probe():
    m = load('review_config', 'vllm/transformers_utils/configs/glm5_next.py')
    c = m.Glm5NextConfig(**json.loads(pathlib.Path('/model/config.json').read_text()))
    t = c.text_config
    return {k: getattr(t, k) for k in ('linear_head_dim', 'linear_num_heads', 'linear_conv_kernel_dim', 'linear_lower_bound', 'mla_nope')}

def sparse_probe(dtype, splits):
    m = load('review_sparse', 'vllm/v1/attention/ops/triton_mla_sparse_kernel.py')
    T, H, D, N, K = 4, 8, 512, 2048, 128
    q = torch.randn(T, H, D, device='cuda', dtype=dtype)
    kv = torch.randn(N, 1, D, device='cuda', dtype=dtype)
    idx = torch.stack([torch.randperm(N, device='cuda')[:K].int() for _ in range(T)]).view(T, 1, K)
    out = m.triton_mla_sparse_attention(q, kv, idx, sm_scale=D**-0.5, num_kv_splits=splits)
    ref = torch.stack([torch.softmax(q[t].float() @ kv[idx[t, 0].long(), 0].float().T * D**-0.5, dim=-1) @ kv[idx[t, 0].long(), 0].float() for t in range(T)])
    assert torch.isfinite(out).all()
    error = (out.float() - ref).abs().max().item()
    relative = error / ref.abs().max().item()
    assert relative < 0.02, relative
    return {'dtype': str(dtype), 'splits': splits, 'shape': list(out.shape), 'max_abs_error': error, 'relative_max_error': relative}

def kda_probe():
    package = types.ModuleType('review_kda')
    package.__path__ = ['/unused']
    sys.modules['review_kda'] = package
    m = load('review_kda.kernels', 'vllm/models/glm5next/amd/ops/third_party/kda/kernels.py')
    T, H, D = 2, 8, 128
    qkv = torch.randn(T, 3 * H * D, dtype=torch.bfloat16, device='cuda')
    q, k, v = [x.view(1, T, H, D) for x in qkv.split(H * D, dim=-1)]
    g = torch.randn(1, T, H, D, dtype=torch.bfloat16, device='cuda')
    beta = torch.randn(1, T, H, dtype=torch.bfloat16, device='cuda')
    a = torch.randn(H, dtype=torch.float32, device='cuda') * 0.5
    bias = torch.randn(H * D, dtype=torch.float32, device='cuda') * 0.1
    state = torch.randn(T + 1, H, D, D, dtype=torch.float32, device='cuda')
    original = state.clone()
    slots = torch.tensor([1, 2], dtype=torch.int32, device='cuda')
    out, _ = m.fused_recurrent_kda(q, k, v, g, beta, initial_state=state, cu_seqlens=torch.tensor([0, 1, 2], dtype=torch.int32, device='cuda'), ssm_state_indices=slots, a_log=a, g_bias=bias, use_qk_l2norm_in_kernel=True, sigmoid_beta=True, compute_gate=True, lower_bound=-5.0)
    expected = []
    for t in range(T):
        qf, kf, vf = (x[0, t].float() for x in (q, k, v))
        qf = qf / torch.sqrt(qf.square().sum(-1, keepdim=True) + 1e-6) * D**-0.5
        kf = kf / torch.sqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
        gate = -5.0 * torch.sigmoid(a.exp()[:, None] * (g[0, t].float() + bias.view(H, D)))
        s = original[t + 1] * gate.exp()[:, None, :]
        u = beta[0, t].float().sigmoid()[:, None] * (vf - torch.einsum('hvk,hk->hv', s, kf))
        s = s + u[:, :, None] * kf[:, None, :]
        expected.append(torch.einsum('hvk,hk->hv', s, qf))
        torch.testing.assert_close(state[t + 1], s, rtol=1e-5, atol=1e-5)
    ref = torch.stack(expected)
    torch.testing.assert_close(out[0].float(), ref, rtol=1e-2, atol=1e-3)
    return {'shape': list(out.shape), 'max_abs_error': (out[0].float() - ref).abs().max().item(), 'strided_qkv': not q.is_contiguous(), 'state_checked': True}

import pytest

@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('splits', [1, 4])
def test_sparse_reference(dtype, splits):
    torch.manual_seed(17)
    sparse_probe(dtype, splits)

def test_local_config():
    assert config_probe()['linear_num_heads'] == 64

def test_kda_original_reference():
    torch.manual_seed(17)
    kda_probe()
