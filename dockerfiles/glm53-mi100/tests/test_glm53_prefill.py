"""KDA chunk output/state must agree with the independent recurrent equation."""
import pytest
import torch

from vllm.models.glm5next.amd.ops.third_party.kda import chunk_kda_with_fused_gate
from test_glm53_kda_recurrent import D, H, LOWER_BOUND, naive_recurrent_kda


@pytest.mark.parametrize("length", [3, 4, 5, 63, 64, 65])
@torch.inference_mode()
def test_chunk_prefill_reference_and_state(length):
    torch.manual_seed(31 + length)
    q, k, v, gate = [torch.randn(1, length, H, D, dtype=torch.bfloat16,
                                device="cuda") for _ in range(4)]
    raw_beta = torch.randn(1, length, H, dtype=torch.bfloat16, device="cuda")
    a = torch.randn(H, device="cuda") * 0.2
    bias = torch.randn(H * D, device="cuda") * 0.1
    state = torch.randn(1, H, D, D, device="cuda") * 0.1
    # The chunk implementation reuses the V buffer for its output.
    ref, final_ref = naive_recurrent_kda(q[0], k[0], v[0], gate[0], raw_beta[0],
                                        a, bias.view(H, D), state[0])
    out, final = chunk_kda_with_fused_gate(
        q, k, v, gate, raw_beta.float().sigmoid(), a, bias,
        initial_state=state.clone(), output_final_state=True,
        use_qk_l2norm_in_kernel=True, safe_gate=True, lower_bound=LOWER_BOUND,
        cu_seqlens=torch.tensor([0, length], dtype=torch.int32, device="cuda"),
    )
    # Chunk kernels round intermediate activations to BF16.
    torch.testing.assert_close(out[0].float(), ref, rtol=0.03, atol=0.004)
    torch.testing.assert_close(final[0], final_ref, rtol=0.03, atol=0.004)


@pytest.mark.parametrize("lengths", [(63, 64, 65), (1280,), (1, 65)])
@torch.inference_mode()
def test_precomputed_chunk_indices_match_fallback(lengths, monkeypatch):
    from vllm.models.glm5next.amd.ops.third_party.kda import kernels
    from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    torch.manual_seed(925)
    length=sum(lengths)
    q,k,v,g=[torch.randn(1,length,H,D,device="cuda",dtype=torch.bfloat16)*.1 for _ in range(4)]
    beta=torch.rand(1,length,H,device="cuda")
    a=torch.zeros(H,device="cuda");bias=torch.zeros(H*D,device="cuda")
    state=torch.randn(len(lengths),H,D,D,device="cuda")*.1
    cu_cpu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
    cu=cu_cpu.cuda(); indices=prepare_chunk_indices(cu_cpu,FLA_CHUNK_SIZE).cuda()
    def call(chunk_indices=None):
        return chunk_kda_with_fused_gate(q.clone(),k.clone(),v.clone(),g.clone(),beta,a,bias,
            initial_state=state.clone(),output_final_state=True,use_qk_l2norm_in_kernel=True,
            safe_gate=True,lower_bound=LOWER_BOUND,cu_seqlens=cu,chunk_indices=chunk_indices)
    expected=call()
    def forbidden(*args,**kwargs):raise AssertionError("Candidate synchronized to prepare indices")
    monkeypatch.setattr(kernels,"prepare_chunk_indices",forbidden)
    actual=call(indices)
    for x,y in zip(expected,actual):torch.testing.assert_close(x,y,rtol=0,atol=0)
