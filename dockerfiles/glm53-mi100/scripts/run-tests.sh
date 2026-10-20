#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VLLM_GFX908_MOE_HIP=0 VLLM_GFX908_MOE_FP16_COMPUTE=0 VLLM_GFX908_SHARED_AS_EXPERT=0
export VLLM_USE_V2_MODEL_RUNNER=1 GLM53_VALIDATE_LOGITS=1
export GLM53_GFX908_MHC=1 GLM53_GFX908_MHC_PROJECT=1 GLM53_GFX908_MHC_FUSE_NORM=1
export GLM53_GFX908_ROUTER=1 GLM53_GFX908_SINGLE_GROUP=1 GLM53_GFX908_DIRECT_W4=1
export GLM53_LINEAR_TOOL_TEMPLATE=1 OMP_NUM_THREADS=1
export GLM53_PROBE_TOKEN_COUNTS=1,2,3
python3 "$script_dir/verify_image.py"
HIP_VISIBLE_DEVICES=0 python3 -m pytest -q "$script_dir/../tests"
HIP_VISIBLE_DEVICES=0 python3 "$script_dir/probe_real_weights.py"
torchrun --nnodes=1 --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29572 "$script_dir/probe_real_weights.py"
HIP_VISIBLE_DEVICES=0 python3 "$script_dir/probe_four_layer_engine.py"
GLM53_PROBE_TP=8 python3 "$script_dir/probe_four_layer_engine.py"
GLM53_PROBE_TP=8 GLM53_PROBE_GRAPHS=1 python3 "$script_dir/probe_four_layer_engine.py"
