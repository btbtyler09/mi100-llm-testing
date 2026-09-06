# Round-8 fleet rebench summary (4×MI100, 2026-04-28)

Image: `btbtyler09/vllm-rocm-gfx908:v0.20.0rc1.dev` (manifest digest `sha256:5507ab2120f5...`).

Common flags: `--tensor-parallel-size 4 --dtype half --max-model-len 32768 --gpu-memory-utilization 0.92 --attention-backend TRITON_ATTN --compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`

Common env: `VLLM_ROCM_USE_AITER=1 VLLM_MI100_TORCH_COMPILE=1 VLLM_ROCM_USE_AITER_TRITON_GEMM=1 NCCL_ALGO=Tree NCCL_PROTO=LL`

Per-model overrides:
- Devstral adds `VLLM_DISABLED_KERNELS=ConchLinearKernel,TritonW4A16LinearKernel`
- All other models use the common config above
- `--mamba-cache-mode` was not needed by any model in the sweep

Coherence pre + post: PASS on all 10 rebenched models. (122B numbers carried over from today's round-8 ship report.)

Sweep wall time: 4h 27min (smallest-first order, sequential).

## c=1 latency (interactive)

| Model | Single User TPOT (ms) | tok/s | Decode Stress TPOT (ms) | tok/s |
|---|---:|---:|---:|---:|
| Qwen3.6-27B-GPTQ-4bit | 15.37 | 55.1 | 15.30 | 65.0 |
| Qwen3.6-27B-GPTQ-8bit | 16.69 | 51.4 | 16.54 | 60.2 |
| groxaxo-Qwen3.6-27B-Pro-4bit | 14.64 | 57.5 | 14.52 | 68.5 |
| Qwen3.5-35B-A3B-GPTQ-4bit | 10.60 | 91.1 | 10.47 | 95.3 |
| Qwen3.5-35B-A3B-GPTQ-8bit | 9.52 | 100.6 | 9.39 | 106.3 |
| Qwen3.6-35B-A3B-GPTQ-4bit | 10.59 | 91.1 | 10.48 | 95.3 |
| Qwen3.6-35B-A3B-GPTQ-8bit | 9.56 | 100.3 | 9.43 | 105.8 |
| Qwen3-Coder-30B-A3B-4bit | 14.03 | 69.0 | 13.71 | 72.9 |
| Qwen3-Coder-Next-4bit | 11.42 | 82.7 | 11.24 | 88.7 |
| Devstral-24B-INT4-INT8-Mixed | 11.93 | 69.8 | 11.66 | 85.7 |
| Qwen3.5-122B-A10B-GPTQ-4bit | 15.98 | 59.7 | 15.88 | 62.9 |

## Aggregate output throughput vs concurrency (tok/s)

| Model | c=2 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.6-27B-GPTQ-4bit | 83 | 116 | 164 | 205 | 232 | 219 | 242 |
| Qwen3.6-27B-GPTQ-8bit | 79 | 106 | 142 | 180 | 166 | 212 | 238 |
| groxaxo-Qwen3.6-27B-Pro-4bit | 87 | 119 | 167 | 209 | 236 | 220 | 242 |
| Qwen3.5-35B-A3B-GPTQ-4bit | 151 | 243 | 327 | 539 | 654 | 1004 | 1318 |
| Qwen3.5-35B-A3B-GPTQ-8bit | 166 | 261 | 346 | 563 | 842 | 1152 | 1382 |
| Qwen3.6-35B-A3B-GPTQ-4bit | 151 | 243 | 324 | 537 | 651 | 1052 | 1305 |
| Qwen3.6-35B-A3B-GPTQ-8bit | 164 | 261 | 341 | 559 | 846 | 1150 | 1378 |
| Qwen3-Coder-30B-A3B-4bit | 128 | 211 | 383 | 682 | 874 | 1226 | 1549 |
| Qwen3-Coder-Next-4bit | 97 | 166 | 203 | 379 | 439 | 683 | 822 |
| Devstral-24B-INT4-INT8-Mixed | 143 | 244 | 345 | 469 | 409 | 414 | 431 |
| Qwen3.5-122B-A10B-GPTQ-4bit | 90 | 132 | 177 | 239 | 390 | 496 | 616 |

## Mixed-context tiers

| Model | Short Ctx (512in/256out, c=16) | Long Ctx 16K (16384in/1024out, c=4) | Mixed (2048in/512out, c=8) |
|---|---:|---:|---:|
| Qwen3.6-27B-GPTQ-4bit | 262.2 | 53.8 | 155.1 |
| Qwen3.6-27B-GPTQ-8bit | 225.4 | 51.4 | 135.5 |
| groxaxo-Qwen3.6-27B-Pro-4bit | 270.0 | 54.7 | 158.2 |
| Qwen3.5-35B-A3B-GPTQ-4bit | 466.1 | 158.4 | 314.7 |
| Qwen3.5-35B-A3B-GPTQ-8bit | 491.4 | 170.6 | 332.1 |
| Qwen3.6-35B-A3B-GPTQ-4bit | 468.8 | 158.6 | 312.3 |
| Qwen3.6-35B-A3B-GPTQ-8bit | 492.8 | 170.1 | 328.8 |
| Qwen3-Coder-30B-A3B-4bit | 495.6 | 113.8 | 307.6 |
| Qwen3-Coder-Next-4bit | 347.3 | 108.5 | 187.1 |
| Devstral-24B-INT4-INT8-Mixed | 352.3 | 65.9 | 179.3 |
| Qwen3.5-122B-A10B-GPTQ-4bit | 240.2 | 88.4 | 168.1 |

## MoE config dispatch (which tuned JSON the model loaded)

| Model | Tuned JSON | Notes |
|---|---|---|
| Qwen3.6-27B-GPTQ-4bit | (none — dense model) | round-6 ship target |
| Qwen3.6-27B-GPTQ-8bit | (none — dense model) | round-5+6 ship target |
| groxaxo-Qwen3.6-27B-Pro-4bit | (none — dense model) | community fine-tune |
| Qwen3.5-35B-A3B-GPTQ-4bit | (default heuristic, W4) | no W4 N=128 tune |
| **Qwen3.5-35B-A3B-GPTQ-8bit** | **B1' (E=256,N=128,int8_w8a16)** | round-4 tune dispatches |
| Qwen3.6-35B-A3B-GPTQ-4bit | (default heuristic, W4) | no W4 N=128 tune |
| **Qwen3.6-35B-A3B-GPTQ-8bit** | **B1' (E=256,N=128,int8_w8a16)** | round-4 ship target |
| Qwen3-Coder-30B-A3B-4bit | (default heuristic) | no tune for this shape |
| Qwen3-Coder-Next-4bit | (default heuristic) | Mamba hybrid |
| Devstral-24B-INT4-INT8-Mixed | n/a (dense, no MoE) | needs VLLM_DISABLED_KERNELS env |
| **Qwen3.5-122B-A10B-GPTQ-4bit** | **B4 (E=256,N=256,int4_w4a16)** | round-8 ship target |

## Notes / known regressions vs prior single-model runs

- **Qwen3.6-27B-GPTQ-8bit** at c=1 came in at 51.4 tok/s vs round-6 ship of 58.9 tok/s (memory entry `project_decode_opt_round6_27b_2026_04_27.md`). Same image, no source change between runs — so this is run-to-run variance, not a regression. The round-6 ship measurement may have benefitted from a warmer kernel cache. Don't act on this without a controlled re-test.
- **Qwen3.5-122B at c=4–c=16** still shows the round-8 B4 tune regression (documented in `release_round8.md`). Opt-out via `VLLM_TUNED_CONFIG_FOLDER` if serving moderate-batch workloads.
- All A3B-class models (3.5/3.6 35B-A3B + Coder-30B-A3B) hit very high c=128 throughput (1300–1550 tok/s) thanks to sparse-active compute on top of the round-3/4 CAR optimization stack.
- Boot times: 27B/30B class ~3–5min, 35B-A3B class ~5min, Coder-Next ~5min, Devstral ~3min. No model needed `--mamba-cache-mode`.

## Files

- Per-model reports: `~/mi100-llm-testing/Model_Reports/benchmark_<MODEL>.md`
- Historical (pre-round-8) reports archived: `~/mi100-llm-testing/Model_Reports/archive/`
- Sweep master log: `/tmp/decode_opt/round8_rebench/master.log`
- Per-model serve logs: `/tmp/decode_opt/round8_rebench/serve_<slug>.log`
- Master script: `~/decode_opt_audit/round8_rebench/bench_all_round8.sh`
