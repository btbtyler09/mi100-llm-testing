#!/bin/bash
# Halo (c=1) variant: same as serve_qwen38_flash_next_rc3.sh plus the checkpoint's 1-layer MTP head
# (mtp.* tensors are bf16 in the artifact) as the drafter. N=${MTP_N:-2} draft tokens.
# Gate on acceptance metrics from /metrics (spec-decode counters are TP-summed: divide by 4), not on
# "coherent output". GDN fused glue falls back to the stock path under spec decode.
set -euo pipefail
MTP_N=${MTP_N:-2}
exec "$(dirname "$0")/serve_qwen38_flash_next_rc3.sh" \
  --speculative-config "{\"method\": \"qwen4_exp_mtp\", \"num_speculative_tokens\": $MTP_N}" "$@"
