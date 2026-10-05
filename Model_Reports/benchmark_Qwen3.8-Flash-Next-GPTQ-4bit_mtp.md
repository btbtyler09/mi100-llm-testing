# Qwen3.8-Flash-Next GPTQ-4bit with MTP speculative decoding — 4x MI100

**Hardware:** 4× AMD Instinct MI100 (gfx908), TP=4, 200 W power cap
**Image:** `btbtyler09/vllm-rocm-gfx908:v0.28.0rc10.dev-q38fn-mtp` (vllm-gfx908 branch [`qwen38-flash-next-mtp`](https://github.com/btbtyler09/vllm-gfx908/tree/qwen38-flash-next-mtp))
**Model:** [`btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit`](https://huggingface.co/btbtyler09/Qwen3.8-Flash-Next-GPTQ-4bit)
**Measured:** 2026-09-30, on the rc10 image with these changes applied as file overlays (the release image contains the same files, verified byte-identical).
**Measurement type:** paired A/B screens (below), not a full 12-tier BenchAndReport suite. For the batch config, the 12-tier rc9 report [`benchmark_Qwen3.8-Flash-Next-GPTQ-4bit.md`](benchmark_Qwen3.8-Flash-Next-GPTQ-4bit.md) still applies.

## Summary

The model's own MTP head (K=3 draft tokens per step) now makes single-user decode much faster:

| config | single-user decode, thinking off | thinking on | time per step | GSM8K (500 questions) |
|---|---|---|---|---|
| MTP off | 104.9 tok/s | 104.9 tok/s | 9.5 ms | — |
| MTP K=3 (this release) | **166.9 tok/s** | **153.8 tok/s** | 16.4 ms (≈2.7 tokens/step) | 489/500 (reference 485–491) |

MTP only pays at low concurrency. From about 6 concurrent requests up, MTP-off is faster, so the start script has two modes:

* `MODE=interactive` (default): MTP K=3, up to 16 sequences, 32k context. Use for 1–4 concurrent users.
* `MODE=batch`: MTP off, up to 40 sequences, 64k context. Use from ~8 concurrent requests up.

## Launch

```bash
MODEL=/path/to/Qwen3.8-Flash-Next-GPTQ-4bit scripts/serve_qwen38_flash_next_mtp.sh            # interactive (MTP K=3)
MODEL=/path/to/Qwen3.8-Flash-Next-GPTQ-4bit MODE=batch scripts/serve_qwen38_flash_next_mtp.sh # batch (MTP off)
```

Interactive mode sets `VLLM_GFX908_GDN_FUSED_SPEC=1 VLLM_GFX908_MTP_QUANT=w4 VLLM_GFX908_SMALLM_HC_W8=1` and
`--speculative-config '{"method":"qwen4_exp_mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'`.
Tool calls: `--tool-call-parser qwen3_xml`; reasoning: `--reasoning-parser qwen3`.

## What changed (each kept only if it won in paired boots)

| change | time per step (MTP K=3) | single-user tok/s (thinking off / on) |
|---|---|---|
| starting point (rc10, MTP K=3) | 24.3 ms | 115.2 / 105.9 |
| draft layer experts quantized to 4-bit at load | 23.5 ms | 118.6 / 109.1 |
| MTP verify step kept on the decode attention path (was falling to the prefill kernel) | 19.5 ms | 140.7 / 131.7 |
| upstream vLLM picks #58114, #55054, #55404 (no blocking host-to-device copies in the spec metadata) | 18.2 ms | 150.3 / 139.6 |
| one-pass small-batch kernels for the hyper-connection mixes | 16.4 ms | 166.2 / 155.3 |

A memory fix in the same work (no longer keeping unused vocabulary copies on the GPU) raised the KV cache from
219,738 to 228,733 tokens with MTP off, and made MTP K=3 fit at 32k context.

## Concurrency (aggregate output tok/s, 512-token outputs)

| concurrent requests | 1 | 2 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|
| MTP K=3 | 157 | 198 | 316 | 307 | 365 | 414 | 422 |
| MTP off | 101 | 126 | 226 | 306 | 399 | 481 | 622 |

KV cache at 32k context: 102,532 tokens with MTP K=3 (16 sequences), 280,776 with MTP off (40 sequences).

## MTP depth

Measured on 2026-09-30 before the speedups above (rc10 base), single user:

| draft tokens (K) | tokens/step | tok/s thinking off / on |
|---|---|---|
| off | 1.00 | 100.4 / 100.3 |
| 1 | 1.80 / 1.74 | 83.4 / 79.5 |
| 2 | — | 100.0 / 93.0 |
| 3 | — | 106.5 / 92.4 |

K=3 was the best depth on the base and is the one the speedups were tuned on. K=2 and K=4 were not re-measured on the final release.

## Method

* Single-user numbers: 8 real-text prompts, greedy, 768-token outputs, thinking off and on; tok/s from streamed token timestamps;
  tokens per step from the server's speculative-decoding counters.
* Every change: paired, alternated server boots (A-B-B-A), a log line proving the change was active in each boot, and a fixed keep rule
  (faster in both thinking modes with no overlap between boots, acceptance drop ≤ 0.02, 16-request throughput within 3%).
* Accuracy: GSM8K, first 500 questions, thinking on, temperature 0.6, 16 concurrent; 0 server faults.
* Full engineering write-up: [`qwen38_flash_next_mtp_release.md`](https://github.com/btbtyler09/vllm-gfx908/blob/qwen38-flash-next-mtp/docs/mi100_decode_opt/qwen38_flash_next_mtp_release.md).
