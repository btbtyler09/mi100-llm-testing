# mi100-llm-testing
This is a repository for documenting the setup and performance of MI100s in popular inference engines.

# vLLM

vLLM officially supports MI200 and MI300 series GPUs, but older cards like the MI100 (gfx908) are not officially supported. With some modifications it is possible to run vLLM on these GPUs. The MI100 lacks FP8/FP4 hardware and is incompatible with Composable Kernel (CK) ops, but Triton-based kernels work well.

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
* AITER Unified Attention is disabled on gfx908 — it corrupts model state after ~200+ sustained requests, causing degenerate repetitive output. The default Triton Attention backend is stable and performance-equivalent.
* GPTQ models require `--dtype half` (float16). bfloat16 will cause errors.
* `HSA_OVERRIDE_GFX_VERSION` is no longer needed with native gfx908 support.

## Pull the prebuilt container from Docker Hub

```bash
docker pull btbtyler09/vllm-rocm-gfx908:v0.19.2rc1
```

Start a container with GPU access:
* Specify render devices for your GPUs (renderD128 = GPU 0, incrementing from there).
* Mount your HuggingFace cache to avoid re-downloading models.
* `VLLM_ROCM_USE_AITER=1` enables AITER's Triton-based kernels for gfx908. All other AITER flags are auto-configured — CK ops, FP8/FP4, and Unified Attention are automatically disabled, while Triton RoPE is enabled. No other env vars are needed.

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
  btbtyler09/vllm-rocm-gfx908:v0.19.2rc1 \
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
  btbtyler09/vllm-rocm-gfx908:v0.19.2rc1 \
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
* `--attention-backend TRITON_ATTN` — the stable attention backend on gfx908. AITER's Unified Attention (UA) is known to corrupt model state after ~200 sustained requests on MI100 and must stay off.
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

Performance benchmarks for quantized models running on 4x AMD Instinct MI100 GPUs (gfx908) via vLLM v0.19.2rc1+mi100 with AITER (TRITON_ATTN, compile+piecewise). Full interactive charts with legend toggle are available in the [interactive dashboard](charts/benchmark_charts.html). Detailed per-model reports are in [`Model_Reports/`](Model_Reports/).

### Single-User Prefill & Decode (c=1)
![Prefill & Decode Comparison](charts/pp_tg_comparison.png)

### Mixed Traffic (c=8, variable input lengths)
![Mixed Traffic Performance](charts/mixed_traffic.png)

### Concurrency Scaling
![Concurrency Scaling](charts/concurrency_scaling.png)

### Per-User Throughput vs Concurrency
![Per-User Scaling](charts/per_user_scaling.png)

**Models tested:** Qwen3.5-9B, Devstral-Small-2-24B (Mixed-GPTQ), Qwen3-Coder-30B-A3B (GPTQ-4bit), Qwen3.5-35B-A3B (GPTQ-4bit, GPTQ-8bit), Qwen3.6-35B-A3B (GPTQ-4bit, GPTQ-8bit), Qwen3-Coder-Next (GPTQ-4bit), Qwen3.5-122B-A10B (GPTQ-4bit)

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
| `v0.19.2rc1` | 0.19.2rc1 | Yes | **Latest** — TRITON_ATTN + compile+piecewise, ROCm 7.2.1 |
| `v0.16.1.dev` | 0.16.1.dev | Yes | AITER Triton ops, UA-OFF fix |
| `v0.15.2rc1.dev-aiter` | 0.15.2rc1.dev | Yes | Older, first AITER integration |
| `v0.15.2rc1.dev` | 0.15.2rc1.dev | No | Pre-AITER, Triton Flash Attention only |
