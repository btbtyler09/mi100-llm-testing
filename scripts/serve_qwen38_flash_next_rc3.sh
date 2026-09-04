#!/bin/bash
# Qwen3.8-Flash-Next (qwen4_exp, 180B MoE, W4 GPTQ GS32) on 4x MI100 (gfx908), TP4.
# Image btbtyler09/vllm-rocm-gfx908:v0.28.0rc3.dev-q38fn = vllm-gfx908 branch qwen38-flash-next
# @ e79c3b2f49 with every gfx908 HIP extension prebuilt and the validated decode-path env
# 0.90 GPU memory utilization: 0.92 OOMs on a batched-prefill transient (PLE short-conv, ~0.5 GB)
# at the c=64 tier of the mixed-corpus benchmark; 0.90 leaves ~150k KV tokens.
# defaults baked in (zero-copy PLE, W4A8 int8-dot GEMVs, fused router+top-k, W8A16 GDN/lm_head,
# fused GDN glue, shared expert fold). Requires: the 102 GB PLE n-gram table stays in host page
# cache (>= 160 GB host RAM recommended), model at $MODEL, HF cache at $HF_CACHE.
# Usage: MODEL=/path/to/Qwen3.8-Flash-Next-GPTQ-4bit scripts/serve_qwen38_flash_next_rc3.sh [extra vllm args]
set -euo pipefail
IMG=${IMG:-btbtyler09/vllm-rocm-gfx908:v0.28.0rc3.dev-q38fn}
MODEL=${MODEL:-/mnt/slow-storage/quant/Qwen3.8-Flash-Next-GPTQ-4bit}
HF_CACHE=${HF_CACHE:-$HOME/.cache/huggingface}
NAME=${NAME:-vllm-q38fn}
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network=host --cpuset-cpus="0-11" --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --device=/dev/kfd \
  --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 --env HF_HOME=/huggingface \
  -v "$HF_CACHE":/huggingface -v "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro \
  "$IMG" \
  vllm serve "$MODEL" --served-model-name qwen38-flash-next \
    --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 8192 --max-num-seqs 48 --hf-overrides '{"language_model_only": true}' "$@"
echo "started $NAME ($IMG); wait: until curl -sf localhost:8000/health; do sleep 15; done  (~9 min: weights 3-5 min + PLE prewarm + compile/capture)"
