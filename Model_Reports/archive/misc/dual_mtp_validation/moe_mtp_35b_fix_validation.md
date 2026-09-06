# MoE MTP fix — 35B-A3B-GPTQ-8bit full 12-tier validation

**Date:** 2026-06-09 · **Hardware:** 4×MI100 (gfx908) TP4 · **Image:** `btbtyler09/vllm-rocm-gfx908:v0.20-dual-mtp` (pure-Python overlay incl. MoE MTP fix commit `5b9b8ec91`)
**Harness:** BenchAndReport full 12-tier sonnet (2048-in/512-out) + coherence-pre · MTP n=2 vs non-MTP (`results_val_35b_dual.json`, dual default).

## Fix
35B-A3B + MTP failed to load (`KeyError: layers.0.mlp.experts.w2_weight`): MTP draft layer is unquantized in the checkpoint but vLLM built it quantized (GPTQ `dynamic` map omits `mtp.*`). Fix = build the whole MTP draft layer unquantized. See memory `project_moe_mtp_gptq_fix`.

## Result: MTP n=2 helps at nearly every tier (incl. high concurrency)

| Tier | c | tok/s base | tok/s MTP2 | Δ tok/s | TPOT base (ms) | TPOT MTP2 (ms) | Δ TPOT | accept % | acc-len |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Single User Latency | 1 | 100.3 | 116.6 | **+16.2%** | 9.58 | 8.09 | -15.6% | 65.69 | 2.31 |
| Short Context Throughput | 16 | 508.2 | 525.8 | **+3.5%** | 21.68 | 16.49 | -23.9% | 68.17 | 2.36 |
| Long Context (16K) | 4 | 170.8 | 120.9 | **-29.2%** | 18.33 | 25.75 | +40.5% | 64.41 | 2.29 |
| Decode Stress Test | 1 | 105.8 | 137.8 | **+30.2%** | 9.43 | 7.19 | -23.8% | 69.72 | 2.39 |
| Mixed Traffic | 8 | 347.5 | 483.2 | **+39.0%** | 20.44 | 14.62 | -28.5% | 68.06 | 2.36 |
| Concurrency Scaling (c=2) | 2 | 166.0 | 167.2 | **+0.7%** | 11.39 | 11.30 | -0.8% | 67.05 | 2.34 |
| Concurrency Scaling (c=4) | 4 | 266.1 | 295.6 | **+11.1%** | 13.31 | 12.08 | -9.2% | 70.14 | 2.40 |
| Concurrency Scaling (c=8) | 8 | 349.8 | 517.4 | **+47.9%** | 20.03 | 13.55 | -32.4% | 70.29 | 2.41 |
| Concurrency Scaling (c=16) | 16 | 586.3 | 780.2 | **+33.1%** | 23.62 | 16.81 | -28.8% | 68.74 | 2.37 |
| Concurrency Scaling (c=32) | 32 | 863.3 | 1008.4 | **+16.8%** | 28.16 | 21.59 | -23.3% | 67.04 | 2.34 |
| Concurrency Scaling (c=64) | 64 | 1169.0 | 1306.2 | **+11.7%** | 40.93 | 32.56 | -20.4% | 67.95 | 2.36 |
| Concurrency Scaling (c=128) | 128 | 1390.6 | 1470.5 | **+5.7%** | 63.11 | 53.28 | -15.6% | 68.49 | 2.37 |

## Read
- Sparse MoE (only ~3B active) → draft-verify compute is cheap vs memory-bound decode, so MTP wins even at saturation. Contrast the **dense 27B**, where c≥64 regressed.
- Best: Decode Stress c=1 **+30.2%**, Mixed Traffic **+39.0%**, c=8 **+47.9%**; high-conc still positive (c=64 **+11.7%**, c=128 **+5.7%**).
- Acceptance stable **65–70%**, accept-length **~2.3–2.4** all tiers (trained draft layer).
- Single regression: **Long Context 16K c=4 (−29.2%)** — long-seq drafting penalty.
- Coherence-pre clean. 27B + MTP n=2 unchanged (no dense-path regression).

Raw: `results_val_35b_dual_mtp2_fixed.json` (this dir). Baseline: `results_val_35b_dual.json`.
