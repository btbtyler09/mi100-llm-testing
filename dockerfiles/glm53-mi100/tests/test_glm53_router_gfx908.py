"""Conservative FP32 router projection and unchanged corrected top-8 selection."""
import json
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from vllm.models.glm5next.nvidia.gfx908_router import router_gemv


def compare(x, weight, bias):
    reference = torch.mm(x, weight.T, out_dtype=torch.float32)
    result = router_gemv(x, weight)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)
    error = (result-reference).norm() / reference.norm()
    assert error < 2e-6, error
    # Bias affects selection only. Original sigmoid values supply the weights.
    expected_scores, actual_scores = reference.sigmoid(), result.sigmoid()
    expected_ids = torch.topk(expected_scores + bias, 8, dim=-1).indices
    actual_ids = torch.topk(actual_scores + bias, 8, dim=-1).indices
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    a = torch.gather(actual_scores, 1, actual_ids)
    b = torch.gather(expected_scores, 1, expected_ids)
    torch.testing.assert_close(a/a.sum(-1,keepdim=True)*2.5,
                               b/b.sum(-1,keepdim=True)*2.5, rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize('tokens', [1, 2, 4])
@pytest.mark.parametrize('strided', [False, True])
def test_projection_and_routing(tokens, strided):
    torch.manual_seed(330 + tokens)
    x = torch.randn(tokens, 8192 if strided else 4096, device='cuda', dtype=torch.bfloat16)*0.1
    if strided:
        x = x[:, ::2]
    weight = torch.randn(288, 4096, device='cuda', dtype=torch.bfloat16)*0.1
    bias = torch.randn(288, device='cuda')*0.01
    compare(x, weight, bias)
    # Equal rows retain the existing topk operation's tie behavior.
    weight[:12] = weight[0].clone()
    bias[:12] = 1.0
    compare(x, weight, bias)


def test_actual_checkpoint_router_weights():
    model = Path('/model')
    index_path = model/'model.safetensors.index.json'
    if not index_path.exists():
        pytest.skip('Actual checkpoint is not mounted')
    index = json.loads(index_path.read_text())['weight_map']
    names = sorted(k for k in index if k.endswith('.mlp.gate.weight') and '.layers.45.' not in k)
    assert len(names) == 42
    torch.manual_seed(904)
    for name in names:
        with safe_open(model/index[name], framework='pt', device='cpu') as f:
            weight = f.get_tensor(name).to(device='cuda', dtype=torch.bfloat16)
        bias_name = name.removesuffix('weight')+'e_score_correction_bias'
        with safe_open(model/index[bias_name], framework='pt', device='cpu') as f:
            bias = f.get_tensor(bias_name).to(device='cuda', dtype=torch.float32)
        for tokens in [1, 2, 4]:
            x = torch.randn(tokens, 4096, device='cuda', dtype=torch.bfloat16)*0.1
            compare(x, weight, bias)
