# GLM-5.3-Flash W4A16 with native MTP K=1 — eight MI100 GPUs

**Model:** `voska/GLM-5.3-Flash-Uncensored-W4A16`, revision `fb58f613c212bbe804794c7a4d2d6deb47d17f6f`

**Hardware:** eight AMD Instinct MI100 32 GiB GPUs, gfx908, tensor parallelism 8; observed power cap 200 W per GPU

**Runtime:** pinned vLLM port with original INT4/group-128 expert weights, BF16 activations/main KV, FP32 recurrent state and the checkpoint's native prediction head

**Measurement:** short real-text coding fixture, returned-token streaming timings; separate functional API checks

## Current reproduction

**Measured October 20, 2026:** the replacement serving profile achieved **49.91 output tokens/s median**, exceeding the 44 tokens/s target. All five timed requests used 34 input and 128 returned tokens, following three warmups.

| Fresh boot | Context limit | Median decode tokens/s | Range | Median TTFT | Median end-to-end |
|---|---:|---:|---:|---:|---:|
| a1-off | 32,768 | **34.79** | 34.70–34.82 | 0.471 s | 4.119 s |
| b1-k1 | 32,768 | **52.09** | 49.85–52.14 | 0.447 s | 2.897 s |
| a2-off | 32,768 | **34.85** | 34.77–34.90 | 0.450 s | 4.092 s |
| b2-k1 | 32,768 | **51.01** | 50.80–51.21 | 0.483 s | 2.982 s |
| deployed-k1 | 387,200 | **49.91** | 48.24–50.01 | 0.463 s | 3.007 s |

The matched 32K comparison spans two fresh boots and ten measured requests per arm. Pooled sample medians were 34.79 tokens/s without MTP and 51.16 with K=1 (**+47.04%**). Both arms used the same configured graph sizes, batch/admission limits, memory budget, checkpoint and power caps. The final auto-fit row is a separate deployment run.

The first warmup in each fresh boot was retained: a1-off 34.83 tokens/s; b1-k1 50.79 tokens/s; a2-off 34.68 tokens/s; b2-k1 51.07 tokens/s; deployed-k1 49.97 tokens/s. Warmups are excluded from the measured medians.

**Validation:** 274 component tests passed with two expected skips. Actual-weight references passed at TP1/TP8, including one- and two-token direct-W4 dispatch. Four-layer engine fixtures passed at TP1 eager, TP8 eager and TP8 graph replay. The final full model passed code, cached-answer, parallel, tool and cancellation/reuse checks.

The final pair of simultaneous streams measured 38.34 / 38.33 tokens/s per call, with 3.312 seconds of output overlap. Three separate final-code chat responses passed six input cases each. Cache checks returned the correct answers with cached-token counts of 0, 640, 640.

During the final profile's warmups and timed requests, native engine counters reported 497 accepted draft tokens from 524 proposed (94.85%). Functional checks ran afterward and are excluded from these counter deltas.

Runtime/source commit: `14cff007ab4be481d478698ff7e4d98c9689d24a`. Local deployed image ID: `sha256:db537cb335a572812fe2226bc4be28069e1d6bad4ccd98625e04233af5440048`. This image ID is a local verification record, not a published registry image. A fresh clone of the pushed GitHub branch built successfully; all 91 runtime/build/test file hashes matched the tested image, including the 62 locked runtime overlay files. Raw measurements and validation logs were retained outside the repository; the only documentation added is the two Markdown performance reports.

The performance target is at least **44 output tokens/s median on the specified single-request fixture**. This is not a guaranteed rate for arbitrary prompts, longer generations or concurrent users.

## Build and launch

The [Dockerfile](../dockerfiles/glm53-mi100/Dockerfile) builds from:

```text
btbtyler09/vllm-rocm-gfx908@sha256:3e9d1e5a128413c6f038d6ae7a073b3ca99fda3483303cdfe1791385ef8ae013
```

It checks expected base-file hashes before installing the overlay, then verifies package versions and all 62 installed overlay hashes. No local predecessor image is required. Source/checkpoint pins and package versions are in [source-lock.json](../dockerfiles/glm53-mi100/source-lock.json); the native model donor is `wtdcode/vllm-backport@29e66dad461b11caae6039f510eaa7913f9dfb19`.

Supply the complete checkpoint locally. Metadata and shard presence are checked before launch; full weight-payload hashes are not recomputed. The checkpoint contains approximately 190.81 GB of weights and needs all eight GPUs in this configuration. Four-GPU serving is not qualified.

From the repository root:

```bash
export GLM53_MODEL_DIR=/absolute/path/to/GLM-5.3-Flash-Uncensored-W4A16
export GLM53_CACHE_DIR=/absolute/path/to/glm53-cache
export GLM53_IMAGE=glm53-flash-mi100:local
bash scripts/serve_glm53_flash_mtp.sh build
bash scripts/serve_glm53_flash_mtp.sh up
```

The [launcher](../scripts/serve_glm53_flash_mtp.sh) uses [Compose](../dockerfiles/glm53-mi100/compose.yaml) and [the example environment](../dockerfiles/glm53-mi100/serving.env.example). An optional `GLM53_ENV_FILE` can point to a customized environment outside the checkout. Shell variables override that file.

The API binds to `127.0.0.1:8000` by default. `GLM53_HOST`, `GLM53_PORT`, `GLM53_CONTAINER_NAME`, and `GLM53_RENDER0` through `GLM53_RENDER7` are configurable. The model directory is mounted read-only; the cache directory is persistent. The image's entrypoint already supplies `vllm serve`.

The launcher also supports `config`, `status`, `logs`, and `stop`. It never stops unrelated containers. Ensure all eight GPUs are available before starting this full model.

## Measured serving settings

| Setting | Value |
|---|---|
| Served model | `glm53-flash-uncensored` |
| Layers / parallelism | Complete 45-layer text model; TP8; multiprocessing executor |
| Speculation | `{"method":"mtp","num_speculative_tokens":1}` |
| Runner / quantization | V2; `compressed-tensors`; Triton MoE |
| Attention | `TRITON_MLA_SPARSE`; BF16 activations and main KV |
| Decode graphs | `FULL_DECODE_ONLY`, sizes `[1,2,4]`, compilation mode `0` |
| Scheduling | Two active sequences; batch budget 2,560; long-prefill threshold 1,280 |
| Prefix caching | Enabled; Mamba `align`; cached-token usage details enabled |
| Memory | Utilization `0.95`; automatic context `-1`; CPU offload `0` |
| CPU / collectives | One OpenMP thread; custom all-reduce disabled |
| API parsers | `glm47` reasoning and tools; automatic tool choice |

The current profile enables the gfx908 mHC, projection/norm fusion, router, single-group selector, direct packed-W4 expert path and linear tool-result renderer. Diagnostic finite-logit synchronization is enabled during correctness tests and disabled during timing. Unsupported expert/indexer options remain disabled.

The original native-MTP deployment reported 387,200 prompt-plus-output tokens of auto-fit capacity, versus 661,760 without MTP. Query `/v1/models` after startup for the capacity actually resolved on the current host. This is a shared allocation, not two independent full-context reservations. Long-context generation is not qualified by the short measurements here.

## Reproduce timing and functional checks

```bash
python3 scripts/benchmark_glm53_flash_mtp.py \
  --base-url http://127.0.0.1:8000 \
  --output /absolute/path/outside/checkout/glm53-k1.json \
  --warmups 3 --samples 5 --qualify --min-tps 44 \
  --label local-k1
```

The [benchmark client](../scripts/benchmark_glm53_flash_mtp.py) requires only Python's standard library and never controls the server. It refuses to overwrite an existing evidence file.

Timing uses a 34-token sorted-list merge prompt and exactly 128 returned tokens, temperature zero, seed 17, and a fresh cache salt for each request. Warmups are recorded separately. The first token-bearing event may contain multiple tokens under MTP:

```text
decode tokens/s = (returned tokens − tokens in first event)
                 / (last token-event time − first token-event time)
```

Rates count returned token IDs and are checked against API usage. TTFT and end-to-end latency are reported separately. The JSON retains each sample, stream events, warmups, model capacity and engine metrics. This timing fixture may include reasoning tokens; it is not a final-answer-only throughput measurement.

`--qualify` separately checks:

- Complete final-answer code against six sorted-list cases. It evaluates parsed chat content, not draft code inside raw reasoning; the generation budget is 512 tokens.
- Cold/shared-prefix answers, including a changed question. The approximately 1,571-token fixture permits a 640-token cache hit under K=1.
- Two simultaneous 128-token streams with measured output overlap, plus two concurrent final-code chat requests.
- A local tool-call/result round trip with separated reasoning/final content.
- Request cancellation followed by successful reuse.

To compare MTP off and K=1, use an identical explicit context for both arms, stop the service between boots, and change only the speculative configuration:

```bash
export GLM53_MAX_MODEL_LEN=32768
export GLM53_SPECULATIVE_CONFIG=null
bash scripts/serve_glm53_flash_mtp.sh up
# Run the benchmark with a unique output filename, then:
bash scripts/serve_glm53_flash_mtp.sh stop
export GLM53_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
bash scripts/serve_glm53_flash_mtp.sh up
```

Repeat the A/B pair with fresh processes, three warmups and five measured requests per arm. Record capacity and graph settings rather than comparing an arbitrary cold request with a warmed serving profile. Restore `GLM53_MAX_MODEL_LEN=-1` for the auto-fit deployment profile. Other GPU workloads must be stopped before the next model is started.

The component/reference suite is included in the image:

```bash
docker run --rm --network host --ipc host \
  --device /dev/kfd --device /dev/dri \
  -v "$GLM53_MODEL_DIR:/model:ro" \
  -v "$GLM53_CACHE_DIR:/cache" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HOME=/cache/huggingface -e TRITON_CACHE_DIR=/cache/triton \
  --entrypoint bash "$GLM53_IMAGE" \
  /opt/glm53-build/scripts/run-tests.sh
```

Run that suite with the serving model stopped. It covers existing GLM kernel/cache/renderer tests, actual-weight references at TP1/TP8, and four-layer engine fixtures in eager and graph modes. The reduced fixtures exercise implementation correctness; production loads all 45 layers.

## Earlier evidence and limits

The October 7 short serving-profile samples were 48.9703, 49.3089 and 49.4553 tokens/s, versus two MTP-off samples of 33.3980 and 33.4687. The +47.48% median difference was a small operational comparison with capacity/warmup differences. An initial cold, smaller-capacity MTP run measured 36.1668 tokens/s. These historical values are distinct from the current reproduction above.

See [performance history](archive/GLM-5.3-Flash/performance_history.md) for earlier optimizations and their tradeoffs. No full 12-tier benchmark, long-context qualification, power-efficiency study or broad model-quality evaluation is claimed. Free-form greedy text can vary on close alternatives even with MTP off; numerical references and fixed functional checks accompany timing.
