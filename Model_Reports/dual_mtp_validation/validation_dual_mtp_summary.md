# Pre-merge validation — GPTQ8 dual + MTP (gfx908)

**Date:** 2026-06-09 · **Hardware:** 4×MI100 (gfx908) TP4 · **Image:** `btbtyler09/vllm-rocm-gfx908:v0.20-dual-mtp`
(pure-Python overlay of branch `mi100-gptq8-dual-mtp` onto round-8 prod `v0.20.0rc1.dev`)
**Harness:** `BenchAndReport.py`, full 12-tier sonnet (real-text) + coherence-pre per arm.

Validation gate for PR `mi100-gptq8-dual-mtp` → `mi100-optimized`. The merge makes
GPTQ8 **dual** the gfx908 default (opt out with `VLLM_GFX908_GPTQ8=native`) and enables
MTP under CUDA graphs (NCCL fix + auto-bump `max_num_batched_tokens` for n≥3). This run
gates both default-changing behaviors on both GPTQ8 models on disk.

## Result: PASS (4/4 gating arms)

| arm | model | config | KV cache | coherence | verdict |
|---|---|---|---|---|---|
| 1 | 27B-8bit | dual (default) | 12.43 GiB / 23.07× | ✅ | dual engaged by default |
| 2 | 27B-8bit | dual + MTP n=2 | 10.03 GiB / 15.44× | ✅ | loads, runs |
| 3 | 35B-A3B-8bit | native | 17.47 GiB / 52.53× | ✅ | baseline |
| 4 | 35B-A3B-8bit | dual (default) | 17.47 GiB / 52.53× | ✅ | no KV tax on MoE |

### 1. Reconstruction is exact (27B dual NEW vs saved test-image dual)
Every tier within **±0.4%** — dropping curvedinf's C++ `q_gemm.cu` kernel (dual uses the
Triton W8A16 path, never the C++ kernel) changed nothing. The clean pure-Python branch
reproduces the test image bit-for-bit in throughput.

### 2. MTP n=2 reproduces (27B dual+MTP2 NEW vs saved)
Within **±1.6%** (MTP acceptance variance). Confirms the auto-bump code is a no-op at n=2
(only fires at n≥3), so n=2 behavior is unchanged.

### 3. 35B-A3B dual default-on is safe — and marginally positive
dual vs native: **+0.1…+1.0% every tier, never negative.** dual is a near-no-op on the MoE
footprint because experts route through `fused_moe`, not `GPTQLinearMethod`; only the small
dense linears get a repacked copy (identical KV, tiny consistent prefill win). Default-on is
safe on MoE models. (35B-A3B is fast: 100 tok/s single-user, 1390 tok/s c=128.)

## Non-blocking finding: 35B-A3B MTP doesn't load (upstream gap)
Arm 5 (35B-A3B dual + MTP n=2) failed at weight load:
`KeyError: layers.0.mlp.experts.w2_weight` in `qwen3_5_mtp.py:load_weights`. The 35B-A3B's
MTP draft layer is itself MoE, and vLLM's `Qwen3_5MTP.load_weights` can't map the fused
expert weights. **Reproduced identically on the clean prod image (no overlay)** — pre-existing
upstream limitation, not a regression (our only `qwen3_5_mtp.py` change is the +7-line
`get_top_tokens`; `load_weights` is byte-identical to baseline). The merge does not claim MTP
on MoE models; MTP enablement targets models whose drafter loads (the dense 27B). Enabling
MoE-MTP is a separate, larger upstream weight-mapping task.

## Deferred
- **pre-bind GEMM** (`VLLM_GFX908_PREBIND_GEMM`) ships default-off; not measured here. Run a
  27B dual ± pre-bind A/B later to decide whether to flip its default.

Raw per-arm reports + result JSONs in this directory. Branch: `btbtyler09/vllm-gfx908`
`mi100-gptq8-dual-mtp`. AITER: `btbtyler09/aiter-gfx908` `mi100-optimized`.
