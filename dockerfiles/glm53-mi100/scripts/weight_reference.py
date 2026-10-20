"""Independent checkpoint unpacking and GLM MLP reference, test only."""
import json
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path('/model')
INDEX = json.loads((ROOT / 'model.safetensors.index.json').read_text())['weight_map']


def tensor(name):
    with safe_open(ROOT / INDEX[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name).cuda()


def weight(prefix, packed=False):
    if not packed:
        return tensor(prefix + '.weight').float()
    p = tensor(prefix + '.weight_packed')
    scale = tensor(prefix + '.weight_scale').float()
    # compressed-tensors packs eight unsigned nibbles in each int32, with
    # the symmetric int4 zero point offset by 8. Group size is 128.
    shifts = torch.arange(8, device='cuda', dtype=torch.int32) * 4
    q = ((p.unsqueeze(-1) >> shifts) & 15).reshape(p.shape[0], -1).float() - 8
    assert q.shape[1] == scale.shape[1] * 128
    return q * scale.repeat_interleave(128, dim=1)


def mlp(x, prefix, packed=False, clamp=None):
    gate = x.float() @ weight(prefix + '.gate_proj', packed).T
    up = x.float() @ weight(prefix + '.up_proj', packed).T
    if clamp is not None:
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
    hidden = torch.nn.functional.silu(gate) * up
    return hidden @ weight(prefix + '.down_proj', packed).T


def reference(x, layer, router_dtype=None):
    prefix = f'model.language_model.layers.{layer}.mlp'
    if layer == 0:
        return mlp(x, prefix)
    logits = x.float() @ weight(prefix + '.gate').T
    if router_dtype != torch.float32:
        logits = logits.bfloat16()
    scores = logits.sigmoid().float()
    bias = tensor(prefix + '.gate.e_score_correction_bias').float()
    selected = (scores + bias).topk(8, dim=-1).indices
    weights = scores.gather(1, selected)
    weights = weights / weights.sum(dim=-1, keepdim=True) * 2.5
    result = mlp(x, prefix + '.shared_experts', clamp=10)
    for expert in selected.unique().tolist():
        rows, slots = torch.where(selected == expert)
        y = mlp(x[rows], prefix + f'.experts.{expert}', packed=True, clamp=10)
        result.index_add_(0, rows, y * weights[rows, slots, None])
    return result


def compare(actual, expected, label):
    delta = actual.float() - expected.float()
    relative_l2 = (delta.norm() / expected.float().norm().clamp_min(1e-9)).item()
    relative_max = (delta.abs().max() / expected.float().abs().max().clamp_min(1e-9)).item()
    print('REFERENCE_ERROR', label, relative_l2, relative_max, flush=True)
    assert relative_l2 < 0.02 and relative_max < 0.03, (label, relative_l2, relative_max)
