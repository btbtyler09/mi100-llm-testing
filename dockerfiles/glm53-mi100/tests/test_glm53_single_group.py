"""Preserve selected experts where unique and the provider's unsorted top-k contract.

The deployed ROCm torch.topk(sorted=False) changes order on identical calls,
and chooses different valid subsets for exact boundary ties. Repeatability
evidence is retained in selector-repeatability.json; imposing an ordered oracle
would reject the unchanged baseline itself.
"""
import json
from pathlib import Path
import pytest
import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.router import grouped_topk_router as router


def compare(monkeypatch, logits, bias):
    x=torch.zeros(logits.shape[0],4096,device=logits.device,dtype=torch.bfloat16)
    def call(): return router.grouped_topk(x,logits,8,True,1,1,'sigmoid',2.5,bias)
    with set_current_vllm_config(VllmConfig()):
        monkeypatch.setattr(router,'USE_GLM53_SINGLE_GROUP',False)
        rw,ri=call()
        monkeypatch.setattr(router,'USE_GLM53_SINGLE_GROUP',True)
        cw,ci=call()
    scores=logits.sigmoid()+bias
    sorted_scores=scores.sort(dim=-1,descending=True).values
    for row in range(logits.shape[0]):
        # Unique cutoff: exact set of expert IDs, preserving ID/weight pairing.
        if sorted_scores[row,7] > sorted_scores[row,8]:
            rs=ri[row].argsort();cs=ci[row].argsort()
            torch.testing.assert_close(ci[row,cs],ri[row,rs],rtol=0,atol=0)
            torch.testing.assert_close(cw[row,cs],rw[row,rs],rtol=3e-6,atol=3e-7)
        else:
            # Exact ties: both calls must select eight distinct maximal scores.
            for ids,weights in [(ri[row],rw[row]),(ci[row],cw[row])]:
                assert ids.unique().numel()==8
                actual=scores[row,ids.long()].sort(descending=True).values
                torch.testing.assert_close(actual,sorted_scores[row,:8],rtol=0,atol=0)
                unbiased=logits[row].sigmoid()[ids.long()]
                expected=unbiased/unbiased.sum()*2.5
                torch.testing.assert_close(weights,expected,rtol=3e-6,atol=3e-7)


@pytest.mark.parametrize('m',[1,2,4,8])
@pytest.mark.parametrize('kind',['random','equal','near_tie','extreme','strided'])
def test_selection_contract(monkeypatch,m,kind):
    torch.manual_seed(810+m)
    logits=torch.randn(m,576 if kind=='strided' else 288,device='cuda')
    if kind=='strided':logits=logits[:,::2]
    bias=torch.randn(288,device='cuda')*.1
    if kind=='equal':logits.zero_();bias.zero_()
    if kind=='near_tie':logits.zero_();bias.copy_(torch.arange(288,device='cuda')*1e-8)
    if kind=='extreme':logits.mul_(80)
    compare(monkeypatch,logits,bias)


def test_real_gate_matrices(monkeypatch):
    from vllm.models.glm5next.nvidia.gfx908_router import router_gemv
    root=Path('/model');index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    names=sorted(k for k in index if k.endswith('.mlp.gate.weight') and '.layers.45.' not in k)
    assert len(names)==42
    torch.manual_seed(480)
    for name in names:
        with safe_open(root/index[name],framework='pt',device='cpu') as f:
            weight=f.get_tensor(name).to(device='cuda',dtype=torch.bfloat16)
        bn=name.removesuffix('weight')+'e_score_correction_bias'
        with safe_open(root/index[bn],framework='pt',device='cpu') as f:
            bias=f.get_tensor(bn).to(device='cuda',dtype=torch.float32)
        for m in [1,2,4]:
            x=torch.randn(m,4096,device='cuda',dtype=torch.bfloat16)*.1
            compare(monkeypatch,router_gemv(x,weight),bias)
