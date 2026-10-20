#!/usr/bin/env python3
"""Check the native model and runtime configuration without loading weights."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path


def make_config(tp=8, context=8192, batch=2048):
    from vllm.engine.arg_utils import EngineArgs
    return EngineArgs(
        model="/model", tokenizer="/model", tensor_parallel_size=tp,
        dtype="bfloat16", quantization="compressed-tensors", model_impl="vllm",
        moe_backend="triton", attention_backend="TRITON_MLA_SPARSE",
        language_model_only=True, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        max_model_len=context, max_num_batched_tokens=batch, max_num_seqs=1,
        distributed_executor_backend="mp", disable_custom_all_reduce=True,
        gpu_memory_utilization=0.90,
    ).create_engine_config()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()
    versions = {name: importlib.metadata.version(name) for name in
                ["vllm", "torch", "transformers", "compressed-tensors", "triton"]}
    lock = json.loads((Path(__file__).resolve().parents[1] / 'source-lock.json').read_text())
    assert versions == lock['packages'], versions
    for module in ["vllm.v1.kv_cache_interface", "vllm.v1.core.kv_cache_utils",
                   "vllm.v1.core.single_type_kv_cache_manager"]:
        importlib.import_module(module)
    print("VERSIONS", json.dumps(versions), flush=True)
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'overlay-manifest.json').read_text())
    installed = Path(importlib.import_module('vllm').__file__).resolve().parent.parent
    mismatches = [name for name, hashes in manifest.items()
                  if not (installed / name).is_file() or
                  hashlib.sha256((installed / name).read_bytes()).hexdigest() != hashes['overlay']]
    assert not mismatches, mismatches
    print('OVERLAY_FILES_VERIFIED', len(manifest), flush=True)
    if args.cpu_only:
        return
    config = make_config()
    from vllm.models.glm5next import Glm5NextForConditionalGeneration
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
    assert isinstance(config.model_config.hf_config, Glm5NextConfig)
    text = config.model_config.hf_text_config
    assert text.hidden_size == 4096 and text.num_hidden_layers == 45
    assert text.n_routed_experts == 288 and text.num_experts_per_tok == 8
    assert text.index_kpool == 4 and text.index_topk == 2048
    assert config.model_config.use_mla
    assert config.model_config.get_head_size() == 512
    dtypes = Glm5NextForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    import torch
    assert dtypes == (torch.bfloat16, torch.float32), dtypes
    print("NATIVE_CONFIG_PASS", config.model_config.architectures, dtypes, flush=True)


if __name__ == "__main__":
    main()
