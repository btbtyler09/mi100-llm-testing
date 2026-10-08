# MI100 (gfx908) add-ons for AMD's agent skills

These are add-on notes for four skills from **[amd/skills](https://github.com/amd/skills)** (MIT License, Copyright (c) 2026
Advanced Micro Devices, Inc.), written against commit [`6c92b41`](https://github.com/amd/skills/tree/6c92b41304c7). The upstream
skills target MI300-class and newer GPUs. Each `MI100.md` here records what we measured on 4× MI100 (gfx908, CDNA1, ROCm
7.0, 2026-10): what works, what doesn't, and the gfx908 constants the skills need.

**This directory contains only our add-ons.** It does not include the upstream skills. Get those from amd/skills.

| skill (upstream dir) | add-on | the short version |
|---|---|---|
| `tracelens-analysis-orchestrator` | [MI100.md](tracelens-analysis-orchestrator/MI100.md) | MI100 isn't in the platform menu. Use the `MI100.json` below, not MI300X. **Caveat: on our vLLM fork only ~1.1% of kernel time had a TraceLens perf model.** Our decode kernels are JIT HIP extensions inside HIP graphs, so use TraceLens for the timeline, top kernels and idle time, and do rooflines by hand. |
| `magpie-kernel-evaluator` | [MI100.md](magpie-kernel-evaluator/MI100.md) | Magpie itself is unverified on gfx908. What to benchmark (our fork images), the workload rules we use, and the kernel-level works / doesn't-work list. |
| `quark-install` | [MI100.md](quark-install/MI100.md) | amd-quark 0.13 installs and runs on gfx908 with a torch constraint file, the ROCm torchvision, `ninja`, and one visible GPU (Quark touches the GPU at import). |
| `quark-torch-llm-ptq` | [MI100.md](quark-torch-llm-ptq/MI100.md) | **Verdict: Quark is a useful quality reference on MI100, and GPTQModel stays the serving path.** Quark INT4 RTN on Qwen3-8B: PPL 10.29 vs 9.73 bf16, but vLLM can't load Quark-format INT4. AWQ/GPTQ have no qwen3 template in 0.13. Static per-tensor INT8 W8A8 breaks quality (PPL 79). No FP8/FP4 hardware on gfx908. |

## Use them: overlay onto amd/skills

```bash
git clone https://github.com/amd/skills amd-skills && git -C amd-skills checkout 6c92b41
git clone https://github.com/btbtyler09/mi100-llm-testing
for s in tracelens-analysis-orchestrator magpie-kernel-evaluator quark-install quark-torch-llm-ptq; do
  d=amd-skills/skills/$s                                    # upstream layout at 6c92b41
  cp mi100-llm-testing/skills/$s/MI100.md "$d/"
  # tell the agent to read it first on gfx908 (one line at the top of the skill)
  sed -i '0,/^# /s//> **On an MI100 (gfx908) host, read `MI100.md` in this directory first; it overrides conflicting guidance.**\n\n# /' "$d/SKILL.md"
done
```
Then install the skills as amd/skills documents for your agent (e.g. copy the skill folders into your agent's skills
directory). Each add-on is self-contained, so you can also hand the agent the `MI100.md` alone.

## MI100 constants (TraceLens `MI100.json`)

Sustained rates at the stock 290 W cap, all four cards loaded, measured 2026-10-08. The power curve (100/150/200/290 W) and
the analysis of the gap to spec are in the TraceLens add-on.

```json
{"name": "MI100", "mem_bw_gbps": 926, "memory_gb": 32,
 "max_achievable_tflops": {"matrix_fp16": 123.7, "matrix_bf16": 65.8, "matrix_fp32": 34.3, "matrix_int8": 88.0,
                           "vector_fp16": 46.1, "vector_bf16": 23.1, "vector_fp32": 23.1, "vector_fp64": 11.5}}
```
Two findings worth knowing:
- The gap to the 184.6 TFLOPS fp16 spec is the clock, not the kernels: rocBLAS runs at 92–97% of the clock-scaled peak,
  but at 290 W the cards hold ~1.03–1.19 GHz, not 1.5 GHz.
- hipBLASLt runs at ~10% of spec on gfx908. Keep rocBLAS.

Status tags inside the add-ons: **[measured]** on our cards, **[spec]** vendor sheet, **[unverified]** not tested here.
Host-specific details (render-node order, fan behaviour, power policy) are marked as ours; check your own system.

## Credit

The skills these notes extend are [amd/skills](https://github.com/amd/skills), Copyright (c) 2026 Advanced Micro Devices,
Inc., under the [MIT License](https://github.com/amd/skills/blob/main/LICENSE). No upstream files are redistributed here.
