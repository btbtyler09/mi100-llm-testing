#!/usr/bin/env python3
"""Load the first four actual model layers, including dense, KDA and sparse MoE."""
import json
import os
import pathlib
import re
import sys

os.environ["VLLM_GFX908_MOE_HIP"] = "0"
os.environ["VLLM_GFX908_MOE_FP16_COMPUTE"] = "0"
os.environ["VLLM_GFX908_SHARED_AS_EXPERT"] = "0"

import torch
from safetensors import safe_open

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from verify_image import make_config
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.model_executor.model_loader.utils import (
    initialize_model, process_weights_after_loading,
)
from vllm.v1.worker.workspace import init_workspace_manager
from weight_reference import reference, compare


def selected_weights():
    root = pathlib.Path("/model")
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    selected = {}
    for name, shard in index.items():
        if name.startswith("model.visual."):
            continue
        match = re.search(r"\.layers\.(\d+)\.", name)
        if match and int(match[1]) >= 4:
            continue
        selected.setdefault(shard, []).append(name)
    for shard, names in selected.items():
        with safe_open(root / shard, framework="pt", device="cpu") as file:
            for name in names:
                yield name, file.get_tensor(name)


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    tuning = bool(os.environ.get('GLM53_MOE_TUNE_DIR'))
    token_counts = list(map(int, os.environ.get('GLM53_PROBE_TOKEN_COUNTS', '3').split(',')))
    assert token_counts and min(token_counts) > 0
    direct_calls = []
    tile_calls = []
    if os.environ.get('GLM53_GFX908_MOE_M2_TILE') == '1':
        from vllm.model_executor.layers.fused_moe.experts import triton_moe
        original_config = triton_moe.try_get_optimal_moe_config
        def observed_config(w1_shape, w2_shape, top_k, dtype, M, block_shape=None):
            value = original_config(w1_shape, w2_shape, top_k, dtype, M, block_shape)
            if world == 8 and M == 2 and tuple(w1_shape) == (288, 512, 2048):
                assert value == {'BLOCK_SIZE_M':16,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':64,
                    'GROUP_SIZE_M':1,'SPLIT_K':1,'num_warps':4,'num_stages':1},value
                tile_calls.append(M)
            return value
        triton_moe.try_get_optimal_moe_config = observed_config
    if os.environ.get('GLM53_GFX908_DIRECT_W4') == '1':
        from vllm.model_executor.layers.fused_moe import gfx908_direct_w4
        original_direct = gfx908_direct_w4.direct_w4
        def observed_direct(output, hidden, *args, **kwargs):
            direct_calls.append(hidden.shape[0])
            return original_direct(output, hidden, *args, **kwargs)
        gfx908_direct_w4.direct_w4 = observed_direct
    config = make_config(tp=world, context=max(8192 if tuning else 2048, max(token_counts)),
                         batch=max(2048 if tuning else 1024, max(token_counts)))
    text = config.model_config.hf_text_config
    text.num_hidden_layers = 4
    text.layer_types = text.layer_types[:4]
    text.mlp_layer_types = text.mlp_layer_types[:4]
    init_distributed_environment(world_size=world, rank=rank, local_rank=local_rank,
        distributed_init_method='env://' if world > 1 else "tcp://127.0.0.1:29571", backend="nccl")
    with set_current_vllm_config(config):
        from vllm.distributed.parallel_state import set_custom_all_reduce
        set_custom_all_reduce(False)
        initialize_model_parallel(tensor_model_parallel_size=world)
    collective = torch.tensor([rank + 1.], device='cuda')
    torch.distributed.all_reduce(collective)
    assert collective.item() == world * (world + 1) / 2
    init_workspace_manager(torch.device("cuda"))
    torch.set_default_dtype(torch.bfloat16)
    with set_current_vllm_config(config), torch.device("cuda"):
        model = initialize_model(config)
    print("FOUR_LAYERS_CREATED", torch.cuda.memory_allocated(), flush=True)
    with set_current_vllm_config(config):
        loaded = model.load_weights(selected_weights())
        print("LOADED", len(loaded), "parameters", torch.cuda.memory_allocated(), flush=True)
        print("UNTRACKED_PARAMETERS", sorted(set(dict(model.named_parameters())) - loaded), flush=True)
        process_weights_after_loading(model, config.model_config, torch.device("cuda"))
        print("PROCESSED", torch.cuda.memory_allocated(), flush=True)
        for i in [0, 3]:
            layer = model.language_model.model.layers[i]
            for tokens in token_counts:
                torch.manual_seed(73 + i)
                x = torch.randn(tokens, 4096, dtype=torch.float32, device="cpu").cuda().bfloat16() * 0.1
                with set_forward_context(None, config, num_tokens=x.shape[0]):
                    y = layer.mlp(x)
                assert y.shape == x.shape and torch.isfinite(y).all(), (i, y)
                print("MLP_EXECUTED", i, y.float().abs().max().item(), flush=True)
                if rank == 0:
                    expected = reference(x, i, getattr(layer.mlp, 'router_dtype', None))
                    compare(y, expected, f'TP{world}-layer{i}-M{tokens}')
                torch.distributed.barrier()
        expert = model.language_model.model.layers[3].mlp.experts.routed_experts
        print("EXPERT_BACKEND", type(expert.quant_method).__name__,
              getattr(expert.quant_method, "wna16_backend", None), flush=True)
        for name, p in expert.named_parameters():
            print("EXPERT_PARAM", name, tuple(p.shape), p.dtype, flush=True)
    print("ACTUAL_WEIGHT_REFERENCE_PASS", rank, world, flush=True)
    if os.environ.get('GLM53_GFX908_DIRECT_W4') == '1':
        direct_sizes = (1,) if os.environ.get('GLM53_GFX908_MOE_M2_TILE') == '1' else (1, 2)
        expected = [n for n in token_counts if n in direct_sizes] if world == 8 else []
        assert direct_calls == expected, (rank, direct_calls, expected)
        print('DIRECT_W4_DISPATCH_PASS', rank, direct_calls, flush=True)
    if os.environ.get('GLM53_GFX908_MOE_M2_TILE') == '1':
        assert tile_calls == ([2] if world == 8 and 2 in token_counts else []), (rank,tile_calls)
        print('M2_TILE_DISPATCH_PASS', rank, tile_calls, flush=True)
    if directory := os.environ.get('GLM53_MOE_TUNE_DIR'):
        from tune_moe_graph import tune
        with set_current_vllm_config(config):
            tune(model.language_model.model.layers[3].mlp, config, directory)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
