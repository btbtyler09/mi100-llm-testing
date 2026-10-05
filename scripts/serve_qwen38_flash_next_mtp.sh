#!/bin/bash
# Qwen3.8-Flash-Next GPTQ-4bit on 4x MI100 (gfx908), TP=4, with the 2026-09-30 MTP decode work.
#   MODE=interactive (default): MTP K=3, W4 draft experts, small-M HC kernels. Best for 1-4 concurrent requests.
#   MODE=batch: MTP off, max-num-seqs 40. Best from ~8 concurrent requests up.
# Image: btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn-mtp (this branch baked in).
# Usage: MODEL=/path/to/Qwen3.8-Flash-Next-GPTQ-4bit [MODE=batch] scripts/gfx908/serve_qwen38_flash_next_mtp.sh [extra vllm args]
set -euo pipefail
IMG=${IMG:-btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn-mtp}
MODEL=${MODEL:?set MODEL to the Qwen3.8-Flash-Next-GPTQ-4bit directory}
NAME=${NAME:-qwen38-flash-next}; PORT=${PORT:-8000}; MODE=${MODE:-interactive}
if [ "$MODE" = interactive ]; then
  ENVS=(--env VLLM_GFX908_GDN_FUSED_SPEC=1 --env VLLM_GFX908_MTP_QUANT=w4 --env VLLM_GFX908_SMALLM_HC_W8=1)
  ARGS=(--speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'
        --max-num-seqs 16 --max-model-len ${MAX_LEN:-32768})
else
  ENVS=(); ARGS=(--max-num-seqs 40 --max-model-len ${MAX_LEN:-65536})
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --network=host --group-add=video --ipc=host --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --device=/dev/kfd --device=/dev/dri \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 "${ENVS[@]}" \
  -v "$(dirname "$MODEL")":"$(dirname "$MODEL")":ro \
  "$IMG" vllm serve "$MODEL" --served-model-name qwen3.8-flash-next --port "$PORT" \
    --tensor-parallel-size 4 --dtype bfloat16 --gpu-memory-utilization 0.90 --max-num-batched-tokens 8192 \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 "${ARGS[@]}" "$@"
echo "started $NAME ($MODE) on :$PORT; health: curl -sf localhost:$PORT/health (first boot ~10 min)"
