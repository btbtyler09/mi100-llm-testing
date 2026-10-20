"""Numerical checks at the deployed sparse-attention and chunk boundaries."""
import pytest
import torch

from vllm.models.glm5next.amd.ops.third_party.kda import chunk_kda_with_fused_gate
from vllm.v1.attention.ops.triton_mla_sparse_kernel import triton_mla_sparse_attention


@pytest.mark.parametrize("seed", [47, 83])
@pytest.mark.parametrize("splits", [1, 4])
@torch.inference_mode()
def test_sparse_actual_topk_with_padding_and_strides(seed, splits):
    torch.manual_seed(seed)
    tokens, heads, dim, keys, topk = 3, 8, 512, 4096, 2048
    q = torch.randn(tokens, heads * 2, dim, device="cuda", dtype=torch.bfloat16)[:, ::2]
    kv = torch.randn(keys, 2, dim, device="cuda", dtype=torch.bfloat16)[:, :1]
    indices = torch.full((tokens, 1, topk), -1, device="cuda", dtype=torch.int32)
    valid_counts = [3, 641, topk]
    for t, count in enumerate(valid_counts):
        indices[t, 0, :count] = torch.randperm(keys, device="cuda")[:count].int()
    output = triton_mla_sparse_attention(q, kv, indices, sm_scale=dim ** -0.5,
                                         num_kv_splits=splits)
    references = []
    for t, count in enumerate(valid_counts):
        selected = kv[indices[t, 0, :count].long(), 0].float()
        references.append(torch.softmax(q[t].float() @ selected.T * dim ** -0.5,
                                        dim=-1) @ selected)
    reference = torch.stack(references)
    assert torch.isfinite(output).all()
    relative = (output.float() - reference).abs().amax() / reference.abs().amax()
    assert relative.item() < 0.02, relative.item()


@pytest.mark.parametrize("boundary", [3, 4, 5, 63, 64, 65, 639, 640, 641, 2048])
@torch.inference_mode()
def test_kda_partition_preserves_output_and_final_state(boundary):
    torch.manual_seed(101 + boundary)
    length, heads, dim = boundary + 5, 8, 128  # TP8 local head count.
    q, k, v, gate = [torch.randn(1, length, heads, dim, device="cuda",
                                dtype=torch.bfloat16) for _ in range(4)]
    beta = torch.randn(1, length, heads, device="cuda").sigmoid()
    a = torch.randn(heads, device="cuda") * 0.2
    bias = torch.randn(heads * dim, device="cuda") * 0.1
    initial = torch.randn(1, heads, dim, dim, device="cuda") * 0.1

    def run(start, end, state):
        return chunk_kda_with_fused_gate(
            q[:, start:end].contiguous(), k[:, start:end].contiguous(),
            v[:, start:end].clone(), gate[:, start:end].contiguous(),
            beta[:, start:end].contiguous(), a, bias, initial_state=state.clone(),
            output_final_state=True, use_qk_l2norm_in_kernel=True, safe_gate=True,
            lower_bound=-5.0,
            cu_seqlens=torch.tensor([0, end - start], device="cuda", dtype=torch.int32))

    complete, final = run(0, length, initial)
    first, state = run(0, boundary, initial)
    second, partitioned_final = run(boundary, length, state)
    partitioned = torch.cat([first, second], dim=1)
    torch.testing.assert_close(partitioned.float(), complete.float(), rtol=0.03, atol=0.004)
    torch.testing.assert_close(partitioned_final, final, rtol=0.03, atol=0.004)


@torch.inference_mode()
def test_mhc_gfx908_normalization_and_uniform_sinkhorn():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.mhc import MHCPreOp, MHCPostOp

    torch.manual_seed(37)
    residual = torch.randn(3, 4, 4096, device="cuda", dtype=torch.bfloat16)
    fn = torch.zeros(24, 16384, device="cuda", dtype=torch.float32)
    scale = torch.ones(3, device="cuda", dtype=torch.float32)
    base = torch.zeros(24, device="cuda", dtype=torch.float32)
    norm = torch.linspace(0.5, 1.5, 4096, device="cuda", dtype=torch.bfloat16)
    epsilon = 1e-6
    with set_current_vllm_config(VllmConfig()):
        post, comb, layer_input = MHCPreOp().forward_hip(
            residual, fn, scale, base, epsilon, epsilon, epsilon, 2.0, 20,
            norm_weight=norm, norm_eps=epsilon)
    # Zero mixing logits give pre=0.5+eps, post=1 and uniform residual mixing.
    torch.testing.assert_close(post, torch.ones_like(post), rtol=0, atol=0)
    torch.testing.assert_close(comb, torch.full_like(comb, 0.25), rtol=0, atol=1e-6)
    pre = ((0.5 + epsilon) * residual.float().sum(dim=1)).to(torch.bfloat16).float()
    expected = (pre * torch.rsqrt(pre.square().mean(-1, keepdim=True) + epsilon)
                * norm.float()).to(torch.bfloat16)
    torch.testing.assert_close(layer_input, expected, rtol=0.01, atol=0.015625)
    with set_current_vllm_config(VllmConfig()):
        actual = MHCPostOp().forward_hip(layer_input, residual, post, comb)
    expected_post = (residual.float().mean(dim=1, keepdim=True)
                     + layer_input.float().unsqueeze(1)).expand_as(residual).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected_post, rtol=0.01, atol=0.015625)
