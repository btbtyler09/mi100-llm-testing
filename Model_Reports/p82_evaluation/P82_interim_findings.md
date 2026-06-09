# Genesis P82 (lossy acceptance-threshold OR-clause) — interim findings

**Status: IN PROGRESS / not shippable yet.** Date: 2026-06-09 · 4×MI100 (gfx908) TP4
· Model: Qwen3.6-27B-GPTQ-8bit (dense) · Harness: BenchAndReport 12-tier sonnet.
Branch `mi100-genesis-p82` (commit `3b93afa95`), image `btbtyler09/vllm-rocm-gfx908:v0.20-p82-test`.

## What P82 is
SGLang-style acceptance-threshold OR-clause from Sandermage's "Genesis" patch set
(P82). Added to the V1 rejection sampler (both greedy + random kernels), env-gated
`VLLM_GFX908_MTP_ACCEPT_THRESHOLD` (default OFF). In addition to standard rejection
sampling, a draft token is **also accepted when the target assigns it probability
≥ threshold**. This is **LOSSY** — it deliberately emits draft tokens strict
sampling would reject — trading output exactness for higher acceptance length.
Hypothesis: rescue high-n MTP, where deep drafts rarely pass strict rejection
(n=5 rolled over vs n=3 under strict in the depth sweep).

Implementation note: at temperature>0 (deployment) the **random** kernel runs; the
P82 change there adds no new memory access (`accepted = accepted or target_prob ≥
thr`), so the kernel itself is memory-safe. The threshold operates on the
**post-top_k/top_p** distribution (apply_sampling_constraints runs first), so the
model's real generation params matter — all runs use temp=1.0/top_k=20/top_p=0.95
(generation_config.json), applied server-side via `--generation-config auto`.

## Headline result — P82 rescues n=5 (27B, c=1 is confound-free)
P82@0.1 (budget 16384) vs strict (budget 8192), n=5, full 12-tier:

| Tier | c | strict tok/s | P82@0.1 tok/s | Δ | acc-len s→p |
|---|---:|---:|---:|---:|---|
| Single User | 1 | 74.5 | 95.4 | **+28.1%** | 3.52→4.74 |
| Decode Stress | 1 | 87.1 | **127.0** | **+45.9%** | 3.42→5.04 |
| Short Context | 16 | 300.6 | 340.3 | +13.2% | 3.50→4.79 |
| Mixed Traffic | 8 | 213.4 | 248.5 | +16.4% | 3.57→4.73 |
| Conc c=2 | 2 | 91.1 | 114.0 | +25.1% | 3.49→4.60 |
| Conc c=4 | 4 | 130.9 | 157.9 | +20.6% | 3.56→4.63 |
| Conc c=8 | 8 | 215.6 | 251.5 | +16.7% | 3.52→4.60 |
| Conc c=16 | 16 | 273.8 | 303.4 | +10.8% | 3.42→4.61 |
| Conc c=32 | 32 | 330.0 | 346.6 | +5.0% | 3.36→4.57 |
| Conc c=64 | 64 | 350.6 | 361.7 | +3.2% | 3.39→4.57 |
| Conc c=128 | 128 | 361.7 | 374.3 | +3.5% | 3.48→4.60 |
| Long Context 16K | 4 | 60.6 | 64.0 | +5.7% | 3.53→4.67 |

**Every tier positive.** Acceptance-length jumps 3.5 → 4.6–5.0 — the mechanism.

### Does it beat the old (non-P82) MTP?
Old 27B best was the depth-sweep peak **n=3 strict ≈ 91 tok/s decode c=1** (curve
rolled over at n=5). P82@0.1 n=5 Decode Stress = **127** → **+40% over the old
peak**, ~2.2× the non-speculative dense-27B rate (~49–58 tok/s). c=1 numbers are
**confound-free** (budget irrelevant at c=1). **Caveat:** clean apples-to-apples
needs strict n=3 re-measured on this image (not yet done).

### Threshold sensitivity (partial — @0.1/@0.3 crashed before the fix)
| threshold | stable? | Single User c=1 |
|---|---|---:|
| strict | yes | 74.5 |
| @0.1 | crashed @8192 / **fixed @16384** | 96.5 (8192) / 95.4 (16384) |
| @0.3 | crashed @8192 | 89.8 (partial) |
| @0.5 | yes @8192 | 82.9 |

## The long-context crash (root-caused, fix identified — NOT yet committed)
Aggressive thresholds (@0.1/@0.3) crash the engine with a **HIP illegal memory
access** (async; surfaces at `num_accepted_tokens`/`_prepare_inputs` in
gpu_model_runner). Properties:
- **Cumulative** (~75 requests), at the **16K long-context tier**, threshold-gated
  (strict + @0.5 clean; @0.1/@0.3 die). Higher acceptance-length → trigger.
- **Mechanism = buffer sized to `max_num_batched_tokens` overflowing** when a
  chunked-prefill long prompt + n=5 drafts + high acceptance share a step. Same
  class as the earlier n≥3 cudagraph-buffer overflow.
- **Confirmed fix:** raising `max_num_batched_tokens` so long prompts don't cross a
  chunk boundary. At **16384**, P82@0.1 ran the **full 12-tier clean (457/0)**.

## 32K-context production-safety stress (input 31744 / out 512 / c=4 / 60 prompts / ignore-eos)
| arm | config | verdict |
|---|---|---|
| 1 | P82@0.1, **budget 32768** (=max_model_len) | ✅ **SURVIVED 60/60** |
| 2 | **shipped MTP n=2** (default budget) | ✅ **SURVIVED 60/60** |
| 3 | P82@0.1, budget 16384 | _in progress (expect crash: cliff moved 16K→32K)_ |

**Conclusions so far:** (a) `max_num_batched_tokens = max_model_len` makes
aggressive P82 safe at **full 32K context** — a viable production setting; (b) the
**already-shipped MTP n=2 is safe at 32K** (gap you flagged is closed); (c) a fixed
16384 budget only moves the cliff to 32K (arm 3 pending confirmation) — so the real
fix must be **context-aware** (bump budget to cover max input length), not a flat
constant.

## OPEN before P82 can ship (do NOT open the follow-up PR until these close)
1. **Commit the crash fix** in `rocm.py`: make the gfx908 MTP budget bump
   context-aware (cover max_model_len / max input length), not a flat 8192.
2. **Quality is UNVERIFIED.** P82 is lossy; the divergence proxy (temp-seeded
   token-match vs strict at real params) was derailed by the crash. Must measure
   the quality cost before any default-on consideration.
3. **n=2 / n=3 sweep** (requested): P82 likely lifts every depth and the optimal n
   may shift; lower n may dodge the crash and n=2 is the high-concurrency / dataset-
   gen production config. Run n∈{2,3,5} × {strict, @0.1, @0.3} with strict controls
   per depth on this image.
4. **35B-A3B** P82 characterization.
5. Then: durable final report → open the P82 follow-up PR (Task #10) with tables.

Raw JSONs in this dir (`results_p82_27b_n5_*.json`).
