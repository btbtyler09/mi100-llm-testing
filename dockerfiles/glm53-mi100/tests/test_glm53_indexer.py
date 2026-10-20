import importlib.util
import json
import pathlib
import sys

import torch
from vllm import envs


def load(name, relative):
    import importlib
    return importlib.import_module(relative.removesuffix('.py').replace('/', '.'))


load('vllm.v1.attention.ops.fp8_sm80', 'vllm/v1/attention/ops/fp8_sm80.py')
for name, value in [('VLLM_INDEXER_LOGITS_MAXNREG', 0), ('VLLM_INDEXER_LOGITS_FACTOR_K_SCALE', False)]:
    if not hasattr(envs, name):
        setattr(envs, name, value)
        print('PROBE_DEFAULT', name, value, flush=True)
m = load('review_mqa', 'vllm/v1/attention/ops/mqa_logits_triton.py')
torch.manual_seed(29)
print('DEVICE', torch.cuda.get_device_properties(0).gcnArchName, flush=True)


def check(name, fn):
    try:
        detail = fn()
        torch.cuda.synchronize()
        print(json.dumps({'test': name, 'status': 'pass', 'detail': detail}), flush=True)
    except Exception as exc:
        print(json.dumps({'test': name, 'status': 'fail', 'error': type(exc).__name__, 'detail': str(exc)[-3500:]}), flush=True)


def compare(out, reference):
    valid = torch.isfinite(reference)
    assert torch.equal(torch.isfinite(out), valid)
    err = (out[valid] - reference[valid]).abs().max().item()
    rel = err / reference[valid].abs().max().item()
    assert rel < 0.005, rel
    return {'shape': list(out.shape), 'relative_max_error': rel, 'mask_matches': True}


def prefill():
    M, H, D, N = 4, 32, 128, 256
    q = torch.randn(M, H, D, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(N, D, device='cuda').to(torch.float8_e4m3fn)
    scale = torch.rand(N, device='cuda') + 0.1
    weight = torch.randn(M, H, device='cuda')
    start = torch.tensor([0, 3, 8, 17], dtype=torch.int32, device='cuda')
    end = torch.tensor([256, 187, 94, 51], dtype=torch.int32, device='cuda')
    out = m.fp8_mqa_logits_triton(q, (k, scale), weight, start, end)
    reference = ((torch.einsum('mhd,nd->mhn', q.float(), k.float()) * scale).relu() * weight[:, :, None]).sum(1)
    pos = torch.arange(N, device='cuda')
    reference.masked_fill_((pos[None, :] < start[:, None]) | (pos[None, :] >= end[:, None]), -torch.inf)
    return compare(out, reference)


def paged():
    B, H, D, BS, NB = 2, 32, 128, 64, 4
    q = torch.randn(B, 1, H, D, device='cuda').to(torch.float8_e4m3fn)
    k = torch.randn(NB, BS, D, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.rand(NB, BS, device='cuda') + 0.1
    cache = torch.empty(NB, BS * (D + 4), dtype=torch.uint8, device='cuda')
    cache[:, :BS * D] = k.view(torch.uint8).view(NB, BS * D)
    cache[:, BS * D:] = scales.view(torch.uint8).view(NB, BS * 4)
    cache = cache.view(NB, BS, 1, D + 4)
    tables = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device='cuda')
    lengths = torch.tensor([85, 127], dtype=torch.int32, device='cuda')
    weights = torch.randn(B, H, device='cuda')
    out = m.fp8_paged_mqa_logits_triton(q, cache, weights, lengths, tables, max_model_len=128)
    reference = torch.full((B, 128), -torch.inf, device='cuda')
    for b in range(B):
        length = int(lengths[b])
        keys = k.float()[tables[b].long()].reshape(-1, D)[:length]
        scale = scales[tables[b].long()].reshape(-1)[:length]
        reference[b, :length] = ((q[b, 0].float() @ keys.T * scale).relu() * weights[b, :, None]).sum(0)
    return compare(out, reference)


def test_prefill_logits():
    prefill()
def test_paged_logits():
    paged()


def test_unshuffled_cache_gather_preserves_fp8_bytes_and_scales():
    from vllm.v1.attention.ops.glm53_gfx908 import gather_indexer_cache
    torch.manual_seed(29)
    blocks, block_size, dim = 7, 64, 128
    keys = torch.randn(blocks, block_size, dim, device='cuda').to(torch.float8_e4m3fn)
    scales = torch.rand(blocks, block_size, device='cuda')
    raw = torch.empty(blocks, block_size * (dim + 4), dtype=torch.uint8, device='cuda')
    raw[:, :block_size * dim] = keys.view(torch.uint8).reshape(blocks, -1)
    raw[:, block_size * dim:] = scales.view(torch.uint8).reshape(blocks, -1)
    cache = raw.view(blocks, block_size, dim + 4)
    table = torch.tensor([[4, 1], [5, 2]], dtype=torch.int32, device='cuda')
    cumulative = torch.tensor([0, 69, 152], dtype=torch.int32, device='cuda')
    token_to_seq = torch.cat([torch.zeros(69, dtype=torch.int32, device='cuda'),
                              torch.ones(83, dtype=torch.int32, device='cuda')])
    values_out = torch.empty(152, dim, dtype=torch.float8_e4m3fn, device='cuda')
    scales_out = torch.empty(152, 4, dtype=torch.uint8, device='cuda')
    gather_indexer_cache(cache, values_out, scales_out, table, cumulative, token_to_seq)
    expected = torch.cat([keys.view(torch.uint8)[table[0].long()].reshape(-1, dim)[:69],
                          keys.view(torch.uint8)[table[1].long()].reshape(-1, dim)[:83]])
    expected_scales = torch.cat([scales[table[0].long()].flatten()[:69],
                                 scales[table[1].long()].flatten()[:83]])
    assert torch.equal(values_out.view(torch.uint8), expected)
    assert torch.equal(scales_out.view(torch.float32).flatten(), expected_scales)


def test_hadamard_fp8_encoding_exact_bytes():
    from vllm.models.glm5next.nvidia.ops.kpool_compress import fwht128_quant_fp8
    torch.manual_seed(23)
    q = torch.randn(37, 128, dtype=torch.bfloat16, device='cuda')
    actual, scale = fwht128_quant_fp8(q)
    x = q.cpu().float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        groups = x.reshape(37, -1, 2, stride)
        a, b = groups[:, :, 0].clone(), groups[:, :, 1].clone()
        x = torch.stack((a + b, a - b), dim=2).reshape(37, 128)
    x = (x * (128 ** -0.5)).to(torch.bfloat16).float()
    expected_scale = torch.exp2(torch.ceil(torch.log2(
        x.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0)))
    expected = (x / expected_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    assert torch.equal(actual.view(torch.uint8).cpu(), expected.view(torch.uint8))
    torch.testing.assert_close(scale.cpu(), expected_scale, rtol=0, atol=0)
