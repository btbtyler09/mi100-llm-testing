# mi100-llm-testing
This is a repository for documenting the setup and performance of MI100s in popular inference engines.

# vLLM

vLLM officially supports MI200 and MI300 series GPUs, but older cards like the MI100 (gfx908) are not officially supported. With some modifications it is possible to run vLLM on these GPUs. The MI100 lacks FP8/FP4 hardware and is incompatible with Composable Kernel (CK) ops, but Triton-based kernels work well.

**6/11/2026 Update — v0.21 + 439-commit AITER sync (Unified Attention fixed)**
* New `:latest` = `btbtyler09/vllm-rocm-gfx908:v0.21.0rc1.dev-aitersync` (vLLM v0.21.0rc1+mi100, AITER pinned to the 439-commit upstream sync `395f84533`).
* **The AITER Unified Attention (UA) state-corruption bug on gfx908 is fixed.** A ~1,200-request soak under the production config (MTP n=3 + P82) stays coherent pre+post — well past the old ~200-request failure threshold.
* On **dense GPTQ-8 models running MTP**, UA is now the *faster* backend: **+15% throughput on long-output dataset generation (4k in / 6k out), +6–8% interactive (c=1), +29% long-context (16K)**. The win is MTP-specific (UA ≈ Triton without MTP) and architecture-specific (MoE shows the opposite). Enable with `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` (and drop `--attention-backend`). `TRITON_ATTN` stays the default and is ~10% better for short-context high-throughput.
* Full eval: [`Model_Reports/ua_eval_27B_2026-06-11.md`](Model_Reports/ua_eval_27B_2026-06-11.md).

**4/26/2026 Update — Round-3 MI100 patches (custom ops + NCCL Tree+LL)**
* Qwen3.6 family rebenchmarked on the same v0.19.2rc1+mi100 image plus Round-3 patches: `5h` custom operators and `5j` NCCL Tree+LL all-reduce path.
* Notable wins on Qwen3.6-35B-A3B-GPTQ-8bit: peak aggregate throughput 1365.89 tok/s at c=128 with TPOT 10.87 ms at c=1.
* Reports use the `_v0.19_round3` suffix in `Model_Reports/`. Triton-only runs from 4/19 are retained for the models that haven't been rerun yet.

**4/20/2026 Update — v0.19 benchmark refresh**
* All models rebenchmarked on vLLM v0.19.2rc1+mi100 with ROCm 7.2.1.
* Attention backend: `--attention-backend TRITON_ATTN` (stable on gfx908).
* Compile + piecewise CUDA graphs enabled for improved decode throughput.
* New reports are under `Model_Reports/` with the `_v0.19_triton` suffix.

**3/11/2026 Update**
* vLLM v0.16.1 with AITER (AMD Inference and Training Extension for ROCm) support for gfx908.
* AITER provides Triton-based RoPE and attention kernels. CK-based ops (GEMM, MoE, Flash Attention, norms) are disabled on gfx908 since CK uses gfx90a+ instructions.
* Only one env var needed: `VLLM_ROCM_USE_AITER=1`. All other AITER flags are auto-configured for gfx908.
* ROCm 7.0, PyTorch 2.9.1, Triton 3.4.0.
* Tested with GPTQ quantized models (4-bit and 8-bit). Recommended quant providers on HuggingFace: jart25, QuantTrio, cpatonn, or my own (btbtyler09).

**Known issues:**
* AITER Unified Attention (UA) **was fixed in the 6/11/2026 `:latest` image** (439-commit AITER sync). It previously corrupted model state after ~200+ sustained requests on gfx908. UA is now stable and, on dense GPTQ-8 + MTP workloads, faster than Triton (see the 6/11 update). It remains opt-in (`VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`); `TRITON_ATTN` is still the default. On older images (≤ v0.19) UA must stay off.
* GPTQ models require `--dtype half` (float16). bfloat16 will cause errors.
* `HSA_OVERRIDE_GFX_VERSION` is no longer needed with native gfx908 support.

## Pull the prebuilt container from Docker Hub

```bash
docker pull btbtyler09/vllm-rocm-gfx908:latest
```

Start a container with GPU access:
* Specify render devices for your GPUs (renderD128 = GPU 0, incrementing from there).
* Mount your HuggingFace cache to avoid re-downloading models.
* `VLLM_ROCM_USE_AITER=1` enables AITER's Triton-based kernels for gfx908. All other AITER flags are auto-configured — CK ops and FP8/FP4 are disabled, Triton RoPE is enabled, and `TRITON_ATTN` is the default backend. Unified Attention is now stable (6/11 image) but stays opt-in via `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`.

```bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --device=/dev/dri/renderD131 \
  --env VLLM_USE_V1=1 \
  --env VLLM_ROCM_USE_AITER=1 \
  --env HF_HOME=/huggingface \
  -v /home/{user}/.cache/huggingface:/huggingface \
  btbtyler09/vllm-rocm-gfx908:latest \
  bash
```

Run a model (benchmark-ready server — this is the exact form used for the v0.19 benchmarks in `Model_Reports/`):
```bash
docker run -d --name mi100-bench \
  --network=host --cpuset-cpus="0-11" --group-add=video --ipc=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 --device=/dev/dri/renderD131 \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  --env HF_HOME=/huggingface \
  --env VLLM_ROCM_USE_AITER=1 \
  --env VLLM_MI100_TORCH_COMPILE=1 \
  -v ~/.cache/huggingface:/huggingface \
  -v /path/to/models:/models \
  btbtyler09/vllm-rocm-gfx908:latest \
  vllm serve /models/Qwen3.6-35B-A3B-GPTQ-4bit \
    --served-model-name qwen3.6-35b-4bit \
    --tensor-parallel-size 4 \
    --dtype half \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --attention-backend TRITON_ATTN \
    --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

**Docker flags:**
* `--network=host` + `--ipc=host` — required for vLLM's tensor-parallel all-reduce across GPUs.
* `--cpuset-cpus="0-11"` — pins the container to a NUMA-local CPU set, reducing cross-socket traffic during dispatch.
* `--device=/dev/kfd` + 4× `/dev/dri/renderD12{8..31}` — exposes the AMD kernel driver and all four MI100s. Drop devices to match your GPU count.
* `--cap-add=SYS_PTRACE` + `--security-opt seccomp=unconfined` — needed by ROCm's profiler/ptrace paths and a few kernel syscalls.
* `HSA_OVERRIDE_GFX_VERSION=9.0.8` — forces the ROCm runtime to report gfx908 for the MI100.
* `VLLM_ROCM_USE_AITER=1` — enables AITER's Triton RoPE and attention kernels; all other AITER flags are auto-configured off for gfx908 (CK ops, FP8/FP4, Unified Attention).
* `VLLM_MI100_TORCH_COMPILE=1` — custom flag (set by the `+mi100` image patches) that lets `torch.compile` run on gfx908 where stock vLLM would gate it off.

**vLLM serve flags:**
* `--tensor-parallel-size 4` — shard across 4 GPUs. Use 1/2/4 to match your hardware.
* `--dtype half` — fp16. Required for GPTQ on MI100 (no bfloat16 support in the kernels we use).
* `--max-model-len 32768` — KV-cache max context. Raise if you have memory headroom; lower for single-GPU runs.
* `--gpu-memory-utilization 0.92` — fraction of VRAM vLLM reserves. 0.75 is conservative; 0.92–0.94 is what the benchmarks used; 0.95+ risks OOM on the 122B model.
* `--attention-backend TRITON_ATTN` — the default backend on gfx908; best for short-context high-throughput. For **dense GPTQ-8 + MTP** workloads (dataset generation, interactive, long context), drop this flag and set `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` instead — Unified Attention is ~15% faster there on the 6/11 image (and the old corruption bug is fixed). On images ≤ v0.19, keep TRITON_ATTN and leave UA off.
* `--compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}'` — enables `torch.compile` (mode 3 = max autotune) with a full CUDA-graph for the decode path plus piecewise graphs for prefill. This is the main decode-throughput win in v0.19 vs. v0.16.

## Build from source

1. Pull the git repos for vLLM and AITER
2. Build the AITER MI100 image (includes ROCm 7.0, PyTorch, Triton):
```bash
cd aiter
DOCKER_BUILDKIT=1 docker build \
  -f Dockerfile.mi100 \
  -t aiter-mi100:latest .
```
3. Build the vLLM container on top of it:
```bash
cd vllm
DOCKER_BUILDKIT=1 docker build \
  --build-arg BASE_IMAGE=aiter-mi100:latest \
  -f docker/Dockerfile.mi100 \
  -t vllm-rocm-gfx908:latest .
```

## Benchmark Results

Performance benchmarks for quantized models running on 4x AMD Instinct MI100 GPUs (gfx908) via vLLM with AITER (compile+piecewise). The fleet charts below are a v0.19.2rc1+mi100 snapshot (TRITON_ATTN). The newest results — the v0.21 + AITER-sync UA-vs-Triton backend evaluation on Qwen3.6-27B-GPTQ-8bit — are in [`Model_Reports/ua_eval_27B_2026-06-11.md`](Model_Reports/ua_eval_27B_2026-06-11.md). Full interactive charts with legend toggle are in the [interactive dashboard](charts/benchmark_charts.html); detailed per-model reports are in [`Model_Reports/`](Model_Reports/).

### UA vs TRITON_ATTN backend — Qwen3.6-27B-GPTQ-8bit (MTP n=3 + P82, 6/11)
![UA vs TRITON_ATTN](charts/ua_vs_triton_27b.png)

Unified Attention wins interactive (c=1 +8%), decode (+6%), long-context (16K +29%), and **dataset generation (4k in / 6k out: +15%)**; TRITON_ATTN wins short-context batch (c=16 −10%); mid/high concurrency is a tie. Regenerate with `python generate_ua_chart.py`. The fleet charts below are the v0.19 TRITON snapshot.

### Single-User Prefill & Decode (c=1)
![Prefill & Decode Comparison](charts/pp_tg_comparison.png)

### Mixed Traffic (c=8, variable input lengths)
![Mixed Traffic Performance](charts/mixed_traffic.png)

### Concurrency Scaling
![Concurrency Scaling](charts/concurrency_scaling.png)

### Per-User Throughput vs Concurrency
![Per-User Scaling](charts/per_user_scaling.png)

**Models tested:**
* On v0.19_round3 (4/26): Qwen3.6-27B-A3B (GPTQ-4bit, GPTQ-8bit), Qwen3.6-35B-A3B (GPTQ-4bit, GPTQ-8bit)
* On v0.19_triton (4/19): Qwen3.5-9B, Devstral-Small-2-24B (Mixed-GPTQ), Qwen3-Coder-30B-A3B (GPTQ-4bit), Qwen3-Coder-Next (GPTQ-4bit), Qwen3.5-35B-A3B (GPTQ-4bit, GPTQ-8bit), Qwen3.5-122B-A10B (GPTQ-4bit)

To regenerate charts after running new benchmarks:
```bash
python generate_charts.py
```

## Supported Quantizations

GPTQ quantization works well in 4-bit and 8-bit. AWQ is also supported. GGUF models are not supported by vLLM on ROCm.

Pre-quantized models on HuggingFace:
* [btbtyler09/Llama-3.1-8B-Instruct-gptq-4bit](https://huggingface.co/btbtyler09/Llama-3.1-8B-Instruct-gptq-4bit)

## Docker Hub Tags

| Tag | vLLM Version | AITER | Notes |
|-----|-------------|-------|-------|
| `latest` / `v0.21.0rc1.dev-aitersync` | 0.21.0rc1.dev | Yes (439-commit sync, `395f84533`) | **Latest** — UA state-corruption fixed; UA faster than Triton for dense GPTQ-8 + MTP. TRITON_ATTN still default. ROCm 7.2.3 |
| `v0.21.0rc1.dev` | 0.21.0rc1.dev | Yes (pre-sync) | Historical — v0.21 upstream sync before the AITER UA fix |
| `v0.19.2rc1` | 0.19.2rc1 | Yes | TRITON_ATTN + compile+piecewise + Round-3 MI100 patches, ROCm 7.2.1 |
| `v0.16.1.dev` | 0.16.1.dev | Yes | Deprecated — earlier AITER Triton ops, UA-OFF fix |
