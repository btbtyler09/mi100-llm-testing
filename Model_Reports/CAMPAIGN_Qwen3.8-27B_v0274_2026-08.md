# Qwen3.8-27B Max-Inference Campaign — 2026-08-27

**Stack**: vllm-gfx908 `mi100-main-sync-2026-08-27` @ `ad768a2bc9` (995-commit upstream FF + gfx908 carries + DFlash fixes), aiter `mi100-aiter-sync-2026-08-27`, torch 2.12 base. Image: `btbtyler09/vllm-rocm-gfx908:v0.27.3rc2.dev`. Hardware: 4×MI100 TP4. Target: `Qwen3.8-27B-GPTQ-8bit` (W8A16 exllama, tuned), `--dtype half --attention-backend TRITON_ATTN --max-model-len 32768 --gpu-memory-utilization 0.92 --max-num-batched-tokens 8192`, `VLLM_MI100_TORCH_COMPILE=1`.

**Methodology**: every quoted number is a full 12-tier BenchAndReport run on real text (sonnet corpus; mixed-domain corpus for the supplementary section), accuracy-gated (GSM8K 5-shot 500Q + greedy-agreement). Spec-decode arms measured on real text only. Probes were used solely as binary debug signals.

## Headline results

| Config | Single-user tok/s (TPOT) | Decode-stress | c=8 | c=64 | Long-16K |
|---|---|---|---|---|---|
| no-spec control | 52.9 (17.6ms) | 56.8 (17.6ms) | 204.6 | **505.6** | **70.0** |
| MTP3 | 35.2 (26.8ms) | 38.5 | 132.2 | 224.0 | 50.7 |
| DFlash2 NS7 | 54.5 (16.9ms) | 67.8 (14.7ms) | 208.9 | 242.9 | 26.6 |
| **DFlash2 NS5** | **58.8 (15.5ms)** | **70.4 (14.1ms)** | **220.1** | 280.2 | 23.3 |
| DFlash2 NS3 | 59.1 (15.4ms) | 67.2 | 182.5 | 334.0 | 33.7 |

(sonnet 12-tier; full tables in `38control.md`, `38mtp3_sonnet.md`, `38dflash2_ns{3,5,7}_sonnet.md`)

- **Best honest single-stream 3.8 ever**: DFlash2 NS5 decode-stress **70.4 tok/s / 14.1ms**, single-user **58.8 / 15.5ms** (+11%); code-domain c=1 **67.5 tok/s / 12.3ms** (+~30%).
- **Ship guidance is concurrency-conditional**: DFlash2 NS5 for interactive/coding deployments (c≤16); no-spec for batch (c≥32) and long-context (16K) serving, where spec decode loses 34-52%.
- MTP3 is retired for real-text workloads (the old 82 tok/s report number was a random-token artifact).

## DFlash2-on-3.8: two silent bugs fixed, one honest limit found

1. **Drafter dtype overflow (fixed, `f6ea8d80b7`)**: vLLM forced the target's fp16 onto the bf16-trained drafter; its residual stream (≈65k absmax at L0) saturated fp16 → NaN drafts → 0% acceptance, silently. Fix: draft runs in its checkpoint dtype (bf16) with boundary casts and a bf16 draft KV cache.
2. **CandidateSelector inductor miscompile (fixed, `ad768a2bc9`)**: under `VLLM_MI100_TORCH_COMPILE=1` the compiled edge-scoring produced wrong candidates → acceptance 3.5→0.0, silently, output still coherent. Fix: selector runs eager; target + drafter stay compiled.
3. **The honest limit**: with both fixed, the DFlash2 *machinery* costs only 1.09× a plain step (profiled: 19.2ms step, 16.9ms of it the target's own verify GEMMs). But this drafter checkpoint's candidate chains die after ~1 token on natural text: it proposes only ~0.7 tok/step (acceptance-length ≈1.5) on prose/code/chat alike — the 3.5-4.6 acceptance lengths appear only on highly predictable text. **Breaking 80+ tok/s single-user on 3.8 requires a better-distilled drafter, not more serving work.** The stack is ready for a drop-in checkpoint. (3.6-27B + DFlash-v1 at 87 tok/s / 13.2ms keeps the fleet's interactive crown.)

Note when reading acceptance from `/metrics`: spec-decode counters are summed across TP ranks (4× on TP4).

## Accuracy gates (DFlash2 NS5)

- GSM8K 5-shot 500Q: **74.4 / 70.8** (flexible/strict) vs no-spec baseline 74.0 / 71.6 — within ±2.0 error band.
- Greedy-agreement vs saved reference: 15/20 exact / 0.845 mean — *better than the no-spec control's own self-agreement* (14/20 / 0.816; residual drift is reference-stack age, not spec decode).

## Mixed-domain corpus (new, `mi100-llm-testing/bench_corpus/`)

`--dataset mixed`: wiki/code/book/chat real text, exact-token-length prompt assembly, per-domain spec-decode acceptance diagnostic via `/metrics` deltas. Control mixed ≈ control sonnet (52.2 vs 52.9 single-user; 524.8 vs 505.6 c=64) — corpus calibration confirmed. Per-domain draft acceptance rates (of proposed tokens): code 92%, book 84%, sonnet 82%, wiki 78%, chat 68%. Sonnet remains this campaign's comparison corpus; mixed becomes standard next campaign.

**DFlash2 NS5 on mixed corpus** (vs control mixed): single-user 53.9 vs 52.2 (+3%), decode-stress **70.6 vs 56.5 (+25%)**, c=8 196.8 vs 200.8 (par), c=64 267.9 vs 524.8 (−49%), Long-16K 23.2 vs 72.7 (−68%) — the sonnet verdict generalizes across domains. Acceptance-by-domain on the ship config (this run): wiki 86%, code 83%, book 78%, chat 72% (rates of proposed tokens; proposal rate itself stays ~0.7/step, run-to-run domain ordering varies a few points). Full tables: `38control_mixed.md`, `38dflash2_ns5_mixed.md`.

## Everything else banked this campaign

- Upstream sync (995 commits) + torch 2.12 base: aggregate +23-42% at c=8-64 vs published v0.27 stack.
- CAR corruption root-caused & fixed (+15.8% c=64, default per concurrency guidance).
- CK W8A8 int8 route evaluated: compiles on gfx908, loses to tuned exllama at all M — parked with accuracy anchor (GSM8K 74.0/71.6).
- curvedinf claims audit: TPOT=8ms was NS15 acceptance-length division at c=8 on favorable text plus TG-metric flattery; his stack measured 25-46ms TPOT on identical hardware. Two of his fixes were real and are now upstreamed into our fork (drafter dtype pattern, CAR pool fixes).

## W4-GS32 + dual GPTQ4 dispatch (2026-08-28 addendum — new ship target)

`Qwen3.8-27B-GPTQ-4bit_foem` (GS32 symmetric) **beats the 8-bit on quality**: GSM8K **79.4 / 77.4** vs 74.0 / 71.6 (outside noise). Native W4 owned decode (86.1 tok/s decode-stress) but halved prefill (exllama fp16 reconstruction at prefill M, worse at GS32). Fix: the dual graph-safe M-dispatch extended to uint4b8 (`af8b4ff80c`, default ON `644e699fff`): decode keeps native exllama under cudagraphs; prefill M runs a repacked Triton W4A16 MFMA kernel (new gfx908 tiles) or dequant-once+hgemm above M=512. Kernel unit tests exact vs reference; GSM8K on the dual boot 79.4 / 76.6 (numerics clean).

| Tier (sonnet) | W8 control | W8+NS5 | **W4-dual control** | **W4-dual+NS5** |
|---|---|---|---|---|
| Single-user tok/s (TPOT) | 52.9 (17.6) | 58.8 (15.5) | 56.6 (16.3) | 57.9 (15.8) |
| Decode stress | 56.8 | 70.4 | 61.6 | **77.0 (12.9ms)** |
| c=16 | — | 280 | 384 | 301 |
| c=32 | — | 278 | 387 | **389** |
| c=64 | 505.6 | 280 | **569** | 395 |
| c=128 | 415 | 283 | **619** | 398 |
| Long-16K | 70.0 (40ms) | 23 | **79.0 (31ms)** | 24 |
| TTFT single (ms) | 668 | 775 | 710 | 772 |

**W4-dual sweeps W8 at every tier in its own mode, with +5 GSM8K — the 8-bit checkpoint is retired for 3.8.** Ship guidance (both on W4-dual, image `v0.27.3rc3.dev`):
- Interactive/coding (c≤16): + DFlash2 NS5 (`--speculative-config`), 12.9ms decode-stress, 77 tok/s
- Batch (c≥32) / long-context: no-spec, 569–619 tok/s aggregate, 79 tok/s @16K
- Decode-stress single-run variance measured ~±6% (86.1 native vs 77.0 dual on the identical decode kernel path); treat single-tier deltas under that as noise.

## Batch update — CORRECTED (2026-08-29): the 780 tok/s CAR suite was an outlier

The original CAR suite (`38w4dual_car_control_sonnet`: c=64 779.8, c=128 713.5) **does not replicate**: a same-image same-config replica plus four other batch suites all cluster at c=64 ≈ 573-585 and c=128 ≈ 620-628. That one suite was a machine-state outlier, and claims built on it (the "+37% CAR batch win") are retracted. **Replicated truth**: CAR at batch is neutral (c=64 ~575 vs 569 no-CAR; c=128 ~625 vs 619); CAR's real, replicated benefits are modest low-concurrency/16K gains (16K 79→83, single-user TTFT 710→562). Batch guidance: W4-dual no-spec, CAR optional. **Methodology rule adopted: any surprising suite delta (>15%) requires a replicate before entering this report** — four attribution suites were spent (checkpoint, dispatch band, rebuilt library all exonerated) before the anchor itself was retested. The real replicated batch progress from the campaign: c=64 505.6 → ~575 (+14%), c=128 415 → ~625 (+51%), both from the W4-dual dispatch.

**Roofline context** (`roofline.py`): even 780 tok/s is ~12% of the c=64 theoretical ceiling; c=1 no-spec is at 33% (roofline 185 tok/s). The remaining gaps are kernel-efficiency and AR/attention scaling, catalogued in `prof/w4dual_ns5/` with hardware-counter attribution (exllama W4 kernel: latency-bound, MemUnitBusy 9%).

## Overnight kernel campaign (2026-08-28, cont.)

Three shipped optimizations, each suite-validated on the W4-dual + DFlash2-NS5 arm:

1. **Pipelined exllama GPTQ4 k-loop** (`da20144113`): counters showed the kernel latency-bound (MemUnitBusy 9%); prefetching the next chunk's loads bought −9% (M=6) / −15% (M=1) at kernel level. Suite: decode-stress 77.0 → 85.1 tok/s.
2. **3D split-KV verify attention** (`c76fb82c8f`, default-on for gfx908 `f418143365`, floor 4096): the split-KV path was decode-only; verify blocks walked KV serially. Kernel: context-flat 0.135ms (7× at 8k). Suite: **Long-16K 24.4 → 31.4 tok/s (132 → 102ms TPOT)**, twice reproduced; floor keeps short tiers untouched.
3. **AITER AR+RMS fusion** (`fuse_allreduce_rms`, config flag): works on gfx908; suite: decode-stress → **91.2 tok/s (10.88ms)**, +2-4% c=8-64, single-user neutral.

**Best-of-campaign (suite-grade, sonnet):** single-user **68.5 tok/s / 13.1ms** (agrees across two suites), decode-stress **91.2 / 10.88ms**, Long-16K spec **31.4 / 102ms**, batch no-spec+CAR **779.8 (c=64) / 713.5 (c=128)**. GSM8K held at 79.x across every arm.

**Variance note:** suite-to-suite single-tier variance measured up to ±9% (single-user 62.5/68.45/68.50 across three same-code-path suites) — deltas under ~10% on a single tier need replication before being quoted; cross-suite agreement or multi-tier consistency is the bar this report uses. One suite was discarded outright for running with a debug overlay mounted.

**Methodology that found the wins:** roofline scoreboard (c=1 at 33% of physics, c=64 at 12%) → per-step kernel ledger from torch traces → rocprof counters → standalone hipcc microbench (60s/variant) → incremental `_C_stable` rebuild (~3 min via ROCm-path symlink) → probe → full suite. Kernel-level claims verified exact vs references before any serving test.

## Final ship: rc8 (2026-08-28 morning)

Fourth overnight find, via the 16K decode profile: the **DFlash drafter's auto-selected attention backend scanned the full context every draft step** (`_fwd_kernel` at 7.2ms/step; the drafter's sliding window is not honored on that path). A/B twin boots at 16K c=1: 51.2 → **12.5 ms/tok (4.1×)**. Shipped as a gfx908 default (`b97b9ea565`): DFlash draft backend → TRITON_ATTN, where unified attention + the 3D verify path run it context-flat.

**rc8 full suite (`38rc8_final_sonnet`, all defaults, no env/flags beyond the standard boot):**

| Tier | Campaign start | rc8 | Δ |
|---|---|---|---|
| Single-user | 52.9 tok/s (17.6ms) | **74.0 (12.06ms)** | **+40%** |
| Decode-stress | 56.8 (17.6ms) | 80.8 (12.28ms) | +42% |
| Long-16K (spec) | 24.4 (132ms) | **71.9 (40.7ms)** | **+195%** |
| c=16 | — | 306.8 | — |
| c=64 (spec / no-spec) | 505.6 | 400 / **~575** | +14% batch |
| c=128 (no-spec) | 415 | **~625** | +51% |

Single-user 12.06ms is the fleet's best interactive TPOT (previous crown: Qwen3.6+DFlash at 13.2ms). Image `v0.27.3rc8.dev` = `:latest`. Closing GSM8K on the exact shipped boot: **79.4 / 76.6** — every accuracy gate of the campaign passed.

## Quantized lm_head (2026-08-28 evening — cross-agent with the Quantizer session)

The roofline flagged the fp16 lm_head (0.64GB/rank read per step) as the largest removable floor item. The Quantizer session delivered `Qwen3.8-27B-GPTQ-4bit_foem-lmh8` by **grafting** an 8-bit GS128 GPTQ head (solved against the quantized body's own final-norm activations, 257k rows; holdout top-1 vs fp16 head 0.9966) onto the bit-identical shipped body.

Serving it surfaced three fixes, all committed: (1) GPTQModel `dynamic` patterns are checkpoint-relative but our matcher start-anchors under vLLM's wrapper prefixes ("language_model.lm_head") — overrides silently miss; prefix-agnostic patterns (`+:.*lm_head$`) are the idiom, and this mechanism retroactively explains the historical "mixed-bit crashes vLLM" precedent. (2) Arch-requested fp32 heads conflict with a quantized head — head_dtype now defaults to the model dtype for quantized-lm_head checkpoints (`bdb772b520`). (3) The DFlash drafter ties the target's lm_head — bf16 drafter hidden now casts at the boundary (`b0ba8821f9`).

**lmh8 full suite vs rc8**: single-user **78.0 tok/s (11.36ms)** (+5.5%), decode-stress **91.0 (10.90ms)** (+13%), short-ctx c=16 405, 16K holds; c≥64 −4-6% (at/near tier variance; lm_head amortizes at batch). **GSM8K 79.6 / 77.0 — the campaign's best strict-match.** Acceptance unchanged (3.82). The head delivered its roofline value (~0.66ms/step) almost exactly. Greedy baseline for the new checkpoint saved (`greedy_ref_lmh8.json`).

**Campaign totals from the start**: single-user 52.9 → **78.0 tok/s (17.6 → 11.36ms)**, decode-stress 56.8 → **91.0**, Long-16K spec 24.4 → **73.4**, batch c=64 505.6 → ~575 / c=128 415 → ~625 (replicated), GSM8K 74.0 → **79.6**.

## Follow-ups (not blocking ship)

- Unified-attention tile tuning for multi-row verify shapes (~+15% ceiling on any spec config; also helps MTP).
- Proper root-cause of the CandidateSelector inductor miscompile (upstream issue-worthy; workaround costs little).
- Watch for an official z-lab Qwen3.8 drafter; re-run the NS sweep drop-in when one lands.
- Upstream the DFlash draft-dtype fix to vLLM proper.

## Consolidation image `v0.27.4rc1.dev` — VALIDATED & SHIPPED `:latest` (2026-08-29)

Clean from-source build of branch HEAD `45eacfa4cc` (all 17 campaign commits, one toolchain — replaces the rc3→rc9 incremental-overlay chain). Full validation ladder, both ship configs, lmh8 checkpoint:

| Gate | Interactive (lmh8 + NS5, pure defaults) | Batch (no-spec + `CUSTOM_AR=1`) |
|---|---|---|
| Single-user | 77.8 tok/s (11.39ms) — matches rc9 | 62.0 (15.06ms), TTFT 566 |
| Decode-stress | **97.8 tok/s (10.14ms) — campaign best** | 66.6 (14.93ms) |
| c=64 / c=128 | 366 / 384 (spec, expected) | **574.9 / 625.5** — on replicated baseline |
| Long-16K | 71.8 (42.1ms) | 80.8 (30.1ms) |
| GSM8K 5-shot 500Q | **79.4 / 76.8** | **79.2 / 76.6** |
| Coherence / acceptance | clean / ~3.9 tok per draft (greedy Q&A) | clean (CAR corruption check) |

The decode-stress 97.8 (vs rc9's 91.0) is the BKN-128 exllama shape dispatch's (`45eacfa4cc`) first suite appearance — +7%, just outside the tier's ±6% variance band; treat as probable-real pending one replicate. Image pushed: `btbtyler09/vllm-rocm-gfx908:v0.27.4rc1.dev` = `:latest` (digest 7531b0d0).

**Final ship configs** (no env overrides needed beyond these):
- Interactive/coding: `:latest`, lmh8 checkpoint, `--speculative-config` DFlash2 NS5 — 78/98 tok/s single/decode-stress, 11.4/10.1ms
- Batch/long-context: `:latest`, lmh8, no spec, `VLLM_ROCM_USE_AITER_CUSTOM_AR=1` — 575/625 aggregate, 80.8 @16K, TTFT 566ms

## Overnight 2026-08-29: cudagraph memory is the spec-decode batch tax (config-only fix, +16% c=64)

Chasing "can we disable spec in flight" led to the real constraint. Attaching the DFlash drafter **halves the KV pool** (666,965 -> 262,398 tokens; 64 -> 29 concurrent requests at c=64), so the spec arm's batch throughput was queueing loss, not step latency (TPOT 86.8 vs 80.3 ms while throughput differed 2.36x). Cause is not the drafter's weights (0.90 GiB/rank): `adjust_cudagraph_sizes_for_spec_decode` rounds every capture size **up to a multiple of `1+num_speculative_tokens`** (6 at NS5), and a captured graph pins its activation arena at fixed addresses for the life of the server — 11.5 GiB/rank of it.

The fix is **one env var, no code and no capture trimming**: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` → **KV 528,860 (2.02x), 59 concurrent**, FULL graphs and default capture sizes retained. vLLM was pre-reserving an over-estimated graph arena; genuine spec graph memory is only ~13 GiB total, roughly a quarter of what was being held back.

**Suite-validated on the shipped interactive config** (`38ns5_noest_only_sonnet.md`): c=64 366 -> **422.9 (+16%)**, c=128 384 -> **440.7 (+15%)**, c=8/16/32 and single-user at par (226/307/391, 76.9), GSM8K **79.2/76.6** clean.

An 8-arm screen first picked a variant that also capped `cudagraph_capture_sizes` at 64 — **that was wrong and is retracted.** Capture sizes are TOKEN counts rounded up to multiples of `1+num_speculative_tokens`, so a cap of 64 covers only ~11 requests at NS5; a concurrency sweep at identical KV showed it costs **10-15% across c=8-16** (c=14: 400 vs 443) and buys no extra memory. Lesson: a screen sampling only c=1 and c=64 will mis-rank configs whose cost sits in the middle — sweep the band.

Negative results banked, all suite- or screen-grade: **PIECEWISE cudagraph mode is a trap** (c=1 -30%); **drafter `enforce_eager` frees zero memory** (target and drafter share one arena sized by the target's largest shape); **dynamic spec-token scheduling still isn't worth shipping** (batch lifts 245 -> 346 with the memory fix, still under plain-NS5's 426 and no-spec's 575); **P82 lossy acceptance is a wash** on every tier with GSM8K unmoved (79.4/76.8) — this drafter's chains die from wrong proposals, not strict rejection.

Batch champion unchanged (no-spec + CAR, 574.9/625.5). The two-config split stands, but the interactive arm now degrades far more gracefully under load.

## CLOSEOUT (2026-08-29): final ship = `v0.27.4rc2.dev` = `:latest` (digest 74933ffe)

rc2 = the clean rc1 build + fork commit `89ee11515f` (four env-gated spec-decode fixes). Final full validation on the closing interactive config — **lmh8 + DFlash2 NS5 + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` + `VLLM_GFX908_EAGLE_DROP_ATTN_ONLY=1`** (`38final_ns5_pcfix_sonnet.md`):

| gate | result |
|---|---|
| single-user / decode-stress | 76.58 (11.60ms) / 95.12 (10.41ms) |
| c=8 / c=16 / c=32 | 230.3 / 307.2 / 389.3 |
| c=64 / c=128 | **421.4 / 436.1** (+15% over pre-memory-fix) |
| Long-16K | 70.5 |
| GSM8K 5-shot 500Q | **79.6 / 77.0 — campaign best** |
| greedy agreement vs lmh8 ref | **16/20 exact / 0.880 mean** (control self-agreement was 14/20 / 0.816) |
| prefix caching under spec | RESTORED — 5712/6952 hits, repeat-prefix TTFT 2.42s -> 1.09s |

The prefix-cache fix costs nothing on the sonnet tiers (no shared prefixes there) and restores ~5x TTFT on repeated/shared-prefix workloads (multi-turn chat, agents) that spec decode was silently losing. Root cause was upstream: the eagle last-block-drop fallback flags ALL kv-cache groups on hybrid models — worth filing against vllm-project/vllm.

**Final ship configs (image `:latest` / `v0.27.4rc2.dev`, checkpoint `Qwen3.8-27B-GPTQ-4bit_foem-lmh8`):**
- **Interactive / coding / agents**: DFlash2 NS5 `--speculative-config` + env `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 VLLM_GFX908_EAGLE_DROP_ATTN_ONLY=1`
- **Batch / long-context**: no spec, `VLLM_ROCM_USE_AITER_CUSTOM_AR=1` (574.9 / 625.5 aggregate, 80.8 @16K)

Campaign final arc (all suite-grade): single-user 52.9 -> 76.6 (95.1 decode-stress), c=64 505.6 -> 575 (no-spec) / spec arm 280 -> 421, c=128 415 -> 625 / 436, GSM8K 74.0 -> 79.6. Fork branch `mi100-main-sync-2026-08-27` @ `89ee11515f`, pushed.
