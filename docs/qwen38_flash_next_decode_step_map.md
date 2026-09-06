# Flash-Next decode step map (gfx908, TP4, c=1)

Release: rc8 (`btbtyler09/vllm-rocm-gfx908:v0.28.0rc8.dev-q38fn`, vllm-gfx908 @ e1343cd539). The tables below were drawn on rc7 (382966cbdd) and still describe the kernel order; the rc8 deltas are listed in the section right below. Updated with each release.

One decode token, kernel by kernel, as the captured FULL graph launches it. Times are graph-timed cold per launch (3K context); bytes are per rank. Sources: agents/graph_branch, agents/qsa_glue, agents/ar_track, agents/spec_research, agents/isa_research. Step total measured in-server: 9.55 ms (kernel bodies ~5.9 ms; the rest is ~1,050 nodes of dispatch, all-reduce skew and the eager step boundary).

Per-rank bytes per token: HC mixes W8 636 MB, GDN int8 519, experts W4 365, lm_head 159, router bf16 126, QSA W4 78, misc ~30 = 1.91 GB.

## rc8 deltas (2026-09-06)

Measured step: 9.17-9.20 ms (rc7 9.55-9.61), c=1 105.7 tok/s decode-stress / 107-108 in probes; 12-tier and a 290 W halo in `Model_Reports/`.

| change | launches per step | where in the tables |
|---|---|---|
| HC-AR consumer fused into the HC combine (`VLLM_GFX908_HC_AR_FUSED`, split kernels, arrival counters allocated at model build) | -96 (the push all-reduce's consume launch folds into `hc_combine_norm` / `hc_combine`) | GDN/QSA node 1 and the MLP combine; the two "push AR" rows lose their consume half |
| W4A8 bf16 epilogue for the dense QSA slab GEMVs (`VLLM_GFX908_W4A8_BF16_EPILOGUE`) | -36 (the fp32→bf16 cast launch after each dense GEMV) | QSA qkv / index_qk / o_proj rows |
| PLE decode glue (`VLLM_GFX908_PLE_GLUE`, one HIP kernel, splitting op; compiled fallback body for prefill/mixed batches) | -22 (prologue row 3: 23 → 1) | step prologue |
| sampler radix passes merged | -1 (7 → 6), selection replay removed | epilogue sampler rows |
| fused push-AR producer (`VLLM_GFX908_PUSH_AR_FUSED_PRODUCER`) | 0 in practice: armed 96×, claimed 0× in the server; inert, under investigation | – |

Net about -155 launches: roughly 1,050 → ~900 graph nodes per step. Two rules learned on the way, both recorded in the fork docs: a custom op that replaces a vLLM splitting op must be registered as one, and persistent kernel state (counters, sentinel slots) must never be allocated lazily inside a graph capture.

## Where 9.55 ms goes

Kernel bodies alone sum to about 5.9 ms of graph-timed cold time. The rest of the measured step is what a ~1,050-node graph costs on this GPU: 1.4–1.8 µs of dispatch per node, launch tails, all-reduce skew across four ranks, and the eager pre-step metadata plus the logits gather at the boundary.

## Step prologue once per step

| # | launch | kind | bytes | us | what it does |
|---|---|---|---|---|---|
| 1 | embedding gather + repeat(1, hc) | aten ×2 | – | ~4 | Token embedding row, replicated into the 4 hyper-connection streams. |
| 2 | PLE input chain: cummax ×2, n-gram ids, zero-copy gather, 5 KB all-reduce, copy | mixed ×12 | host RAM | ~40 | Layer 1 only. Per-layer embedding rows read straight from the pinned checkpoint shards on the host (no device copy of the 95 GB table), then reduced across ranks. |
| 3 | PLE layer body: query norm, gate (dot·sign·sqrt·sigmoid), norm_conv, short-conv decode | eager ×16 | – | ~45 | Still eager glue; the largest un-fused block left outside the layer loop. |

## GDN layer × 36 · 17 launches · ~105 µs

Gated DeltaNet linear attention. Inputs: the four HC streams from the previous layer's MoE output and injection. Node 5 is the one independent branch in the whole layer; everything else is a chain.

| # | launch | kind | bytes | us | what it does |
|---|---|---|---|---|---|
| | **attention hyper-connection gfx908_hc_fused · W8 chain, 3 launches** | | | | |
| 1 | hc_combine_norm | HIP | – | ~5 | Combine the 4 streams with the previous block output and injection, RMSNorm. |
| 2 | hc_w8_gemv mix_down (+silu epilogue) | HIP GEMV | 3.3 MB | ~5 | int8 gs128 weights, the M ≤ 4 fused range. |
| 3 | hc_w8_gemv mix_up (+gate-mix epilogue) | HIP GEMV | 3.3 MB | ~5 | Produces the layer input and the residual gate in one pass. |
| | **GDN block qwen_gdn_linear_attn · gfx908_gdn_fused** | | | | |
| 4 | in_proj_qkvz · w8sw_gemv (swizzled int8) | HIP GEMV | 10.5 MB | 13.5 | 4096×2560 per rank; the M=1 path reads the MFMA-swizzled layout directly. |
| 5 | in_proj_ba · wvSplitK bf16 | HIP GEMV | 0.12 MB | 3.9 | 24 output rows (β, α). Independent of #4 (candidate for the merged launch in rc8). |
| 6 | gdn_fused decode | HIP | state | 5.6 | Conv1d state update, gated delta recurrence, gated RMSNorm, z copy: was 5 launches. |
| 7 | out_proj · w8sw_gemv | HIP GEMV | 3.9 MB | ~6 | 2560×1536 int8. |
| 8 | push all-reduce (5 KB) | xGMI | 5 KB ×3 | 5.3 | Each rank writes its partial into the peers' buffers and flips a sentinel; the consumer sums. Was 7.8 µs one-shot. |
| | **MLP hyper-connection same 3-launch W8 chain** | | | | |
| 9–11 | hc_combine_norm · mix_down · mix_up | HIP ×3 | 6.6 MB | ~15 |  |
| | **MoE block gfx908_router_topk · gfx908_w4a8 · shared expert folded as expert #E** | | | | |
| 12 | router_topk_fused | HIP | 2.6 MB | 6.5 | bf16 router GEMV (512×2560), softmax, top-10, plus the shared-expert gate: one launch with a last-arriving finalize. |
| 13 | w4a8_slab gate_up (prep fold) | HIP GEMV | 4.9 MB | 8.9 | Quantizes the activation to int8 per 32-group in LDS staging, then v_dot4_i32_i8 against 11 W4 expert slices (10 routed + shared). |
| 14 | _silu_mul_quant | Triton | – | ~3 | silu(gate)·up and re-quantize to Q8_1 for the down projection. |
| 15 | w4a8_rowlane down | HIP GEMV | 2.5 MB | ~5 | Per expert-row lanes; output partials per (token, expert). |
| 16 | _moe_reduce_weighted_sum | Triton | – | 2.0 | Weighted sum of the 11 partials (10 workgroups since the reduce-block change). |
| 17 | push all-reduce (5 KB) | xGMI | 5 KB ×3 | 5.3 | Block output goes to the next layer's combine. |

## QSA layer × 12 (every 4th) · ~25 launches · ~145 µs at 3K context

Sparse attention with a learned indexer: compressed key blocks are scored, the top 512 per query are selected, and only those are attended. Before the glue fusion this layer launched 45 kernels; the crossed-out group below is what one HIP kernel now does.

| # | launch | kind | bytes | us | what it does |
|---|---|---|---|---|---|
| | **attention hyper-connection 3 launches, as above** | | | | |
| 1–3 | hc_combine_norm · mix_down · mix_up | HIP ×3 | 6.6 MB | ~15 |  |
| | **projections dense W4A8 path: slab + fp32→bf16 cast** | | | | |
| 4–5 | qkv_proj · w4a8 slab + cast | HIP + Triton | 4.3 MB | ~9.5 | 2560 → 3584 (q, k, v, gate). |
| 6–7 | index_qk_proj · w4a8 slab + cast | HIP + Triton | 0.8 MB | ~6.5 | 2560 → 640 indexer q/k. |
| | **fused decode glue gfx908_qsa_glue · replaces 18 launches (norms, MRoPE, gathers, copies, cache writes)** | | | | |
| 8 | qsa_glue_pre (1 workgroup per token) | HIP | cache | 9.9 | Main q/k Gemma-norm + interleaved MRoPE; K/V written straight into the paged cache; indexer q norm + RoPE; group pooling + k norm + RoPE into the compressed key cache; raw-key ring and int64 position ring. Reproduces the compiled numerics (inductor drops the bf16 round trip between norm and RoPE). |
| | **indexer + selection** | | | | |
| 9 | _qsa_mqa_paged scorer | Triton | compressed K | 5.8–19 | Scores every visible compressed block (8192 columns under capture); cost grows with context. |
| 10 | qsa_topk_expand | HIP | – | 3.2–32 | Deterministic radix top-512 with the stable tie rule, expanded to token indices: was topKPerRowDecode + repair + expand. |
| | **attention** | | | | |
| 11 | _qsa_sparse_paged_gqa_splitk (64 splits) | Triton | selected KV | ~13 | Attends only the selected blocks. |
| 12 | _qsa_merge_splitk_gate | Triton | – | ~5.4 | Split-K merge with the sigmoid output gate and padded-row zeroing folded in. |
| 13–14 | o_proj · w4a8 slab + cast | HIP + Triton | ~1 MB | ~6.5 | 1536 → 2560. |
| 15 | push all-reduce (5 KB) | xGMI | 5 KB ×3 | 5.3 |  |
| | **MLP hyper-connection + MoE identical to GDN layer nodes 9–17** | | | | |
| 16–24 | hc ×3 · router_topk_fused · slab · silu_mul_quant · rowlane · reduce · push all-reduce | mixed ×9 | 16.6 MB | ~46 |  |

## Step epilogue once per step

| # | launch | kind | bytes | us | what it does |
|---|---|---|---|---|---|
| 1 | final mixer combine_and_mix | HIP ×3 | 6.6 MB | ~15 | Collapses the 4 HC streams to the hidden state. |
| 2 | graph replay ends → eager | boundary | – | skew | Rank 0 carries 0.2–0.3 ms more work than ranks 1–3 and pays it here; the in-graph logits path (rc8 arm) targets this. |
| 3 | lm_head · w8sw_gemv (int8, vocab shard) | HIP GEMV | 159 MB | ~150 | 62080×2560 per rank, one row. |
| 4 | logits all-gather (RCCL) | RCCL | 121→485 KB | 57 | The only RCCL collective left in the step; custom xGMI gather is gated for rc8. |
| 5 | sampler: radix top-k/top-p (7 launches), gumbel, combine | Triton ×11 | – | ~120 | Exact byte-radix select for k ≤ 64 replaced a full-vocab sort (343 → 99 µs at one row). |
| 6 | next-step metadata (eager): QSA/GDN builders, 1 custom all-reduce | host | – | ~200 | Rebuilt every step outside the graph; part of the ~1.2 ms boundary. |

## Reading the map

Sources: agents/graph_branch (per-layer DAG and node times), agents/qsa_glue (QSA inventory before/after), agents/ar_track (collectives, skew), agents/spec_research (byte ledger), agents/isa_research (dispatch cost). Times are graph-timed cold microbenches per launch at 3K context; in-server the same kernels sit inside a ~1,050-node graph whose dispatch and tails make up the difference to the measured 9.55 ms.
