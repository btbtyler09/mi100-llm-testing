#!/usr/bin/env python3
"""Exercise actual cache allocation and model-runner paths on a small fixture."""
import os
import faulthandler
import signal
faulthandler.register(signal.SIGUSR1, all_threads=True)
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_GFX908_MOE_HIP"] = "0"
os.environ["VLLM_GFX908_MOE_FP16_COMPUTE"] = "0"
os.environ["VLLM_GFX908_SHARED_AS_EXPERT"] = "0"

import torch
import glm53_fixture_loader
from vllm import LLM, SamplingParams


def main():
    torch.set_num_threads(4)
    tp = int(os.environ.get('GLM53_PROBE_TP', '1'))
    graphs = os.environ.get('GLM53_PROBE_GRAPHS') == '1'
    max_seqs = int(os.environ.get('GLM53_PROBE_MAX_SEQS', '1' if graphs else '2'))
    llm = LLM(
        model="/model", tokenizer="/model", hf_overrides=glm53_fixture_loader.four_layer_config,
        load_format="glm53-four-layer-test", model_impl="vllm",
        tensor_parallel_size=tp, distributed_executor_backend='uni' if tp == 1 else 'mp',
        dtype="bfloat16", quantization="compressed-tensors", moe_backend="triton",
        attention_backend="TRITON_MLA_SPARSE", language_model_only=True,
        enforce_eager=not graphs, enable_prefix_caching=False, enable_chunked_prefill=True,
        compilation_config=({"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                             "cudagraph_capture_sizes": list(range(1, max_seqs + 1))} if graphs else None),
        max_model_len=8192, max_num_batched_tokens=2048,
        max_num_seqs=max_seqs,
        gpu_memory_utilization=0.65, disable_custom_all_reduce=True,
    )
    # Truncated model outputs have no semantic meaning. These requests exercise
    # real execution, chunked prefill, pool tails and state lifecycle only.
    prompts = ["one two three " * 5, "one two three " * 1200]
    params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    first = llm.generate(prompts, params, use_tqdm=False)
    again = llm.generate([prompts[0]], params, use_tqdm=False)
    assert first[0].outputs[0].token_ids == again[0].outputs[0].token_ids
    for output in first:
        assert len(output.outputs[0].token_ids) == 8
        print("FIXTURE_OUTPUT", len(output.prompt_token_ids), output.outputs[0].token_ids, flush=True)
    print("FOUR_LAYER_ENGINE_PASS", "graphs", graphs, "tp", tp,
          "mhc", os.environ.get("GLM53_GFX908_MHC"),
          "project", os.environ.get("GLM53_GFX908_MHC_PROJECT"),
          "norm", os.environ.get("GLM53_GFX908_MHC_FUSE_NORM"),
          "router", os.environ.get("GLM53_GFX908_ROUTER"), flush=True)


if __name__ == "__main__":
    main()
