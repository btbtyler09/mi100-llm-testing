#!/usr/bin/env python3
"""
Self-Documenting LLM Benchmark Suite

Runs a comprehensive benchmark suite against a vLLM server, then uses the 
model itself to generate a professional benchmark report.

Usage:
    python benchmark_and_report.py --model <model_name> [--base-url http://localhost:8000]

The script will:
1. Run multiple benchmark scenarios (latency, throughput, long context, etc.)
2. Collect and parse all results
3. Send the data to the model to generate a markdown report
4. Save the report with timestamp

Requirements:
    - vLLM server running with the target model
    - openai python package (pip install openai)
"""

import argparse
import subprocess
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import time

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package required. Install with: pip install openai")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Structured benchmark result data"""
    name: str
    input_len: int
    output_len: int
    concurrency: int
    num_prompts: int
    
    # Results
    successful_requests: int = 0
    failed_requests: int = 0
    duration_s: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    
    # Throughput
    request_throughput: float = 0.0
    output_throughput: float = 0.0
    peak_output_throughput: float = 0.0
    total_throughput: float = 0.0
    
    # Latency
    ttft_mean_ms: float = 0.0
    ttft_median_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_mean_ms: float = 0.0
    tpot_median_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    itl_mean_ms: float = 0.0
    itl_median_ms: float = 0.0
    itl_p99_ms: float = 0.0
    
    # Derived metrics
    pp_speed: float = 0.0  # Prefill tokens/sec
    tg_speed: float = 0.0  # Decode tokens/sec
    
    # Raw output for debugging
    raw_output: str = ""
    error: Optional[str] = None


@dataclass 
class BenchmarkSuite:
    """Collection of benchmark scenarios to run"""
    scenarios: list = field(default_factory=list)
    
    def add_default_scenarios(self):
        """Add the standard benchmark scenarios"""
        self.scenarios = [
            {
                "name": "Single User Latency",
                "description": "Best-case latency with no batching",
                "input_len": 2048,
                "output_len": 512,
                "concurrency": 1,
                "num_prompts": 20,
                "request_rate": "inf",
            },
            {
                "name": "Short Context Throughput",
                "description": "Maximum throughput with short sequences",
                "input_len": 512,
                "output_len": 256,
                "concurrency": 16,
                "num_prompts": 50,
                "request_rate": "inf",
            },
            {
                "name": "Long Context (16K)",
                "description": "Extended context performance",
                "input_len": 16384,
                "output_len": 1024,
                "concurrency": 4,
                "num_prompts": 10,
                "request_rate": "inf",
            },
            {
                "name": "Decode Stress Test",
                "description": "Long generation with minimal prefill",
                "input_len": 128,
                "output_len": 2048,
                "concurrency": 1,
                "num_prompts": 5,
                "request_rate": "inf",
                "extra_args": ["--ignore-eos"],
            },
            {
                "name": "Mixed Traffic",
                "description": "Variable request sizes (±50%)",
                "input_len": 2048,
                "output_len": 512,
                "concurrency": 8,
                "num_prompts": 30,
                "request_rate": "inf",
                "range_ratio": 0.5,
            },
            {
                "name": "Concurrency Scaling (c=2)",
                "description": "Scaling test at concurrency 2",
                "input_len": 1024,
                "output_len": 256,
                "concurrency": 2,
                "num_prompts": 30,
                "request_rate": "inf",
            },
            {
                "name": "Concurrency Scaling (c=4)",
                "description": "Scaling test at concurrency 4",
                "input_len": 1024,
                "output_len": 256,
                "concurrency": 4,
                "num_prompts": 30,
                "request_rate": "inf",
            },
            {
                "name": "Concurrency Scaling (c=8)",
                "description": "Scaling test at concurrency 8",
                "input_len": 1024,
                "output_len": 256,
                "concurrency": 8,
                "num_prompts": 30,
                "request_rate": "inf",
            },
            {
                "name": "Concurrency Scaling (c=16)",
                "description": "Scaling test at concurrency 16",
                "input_len": 1024,
                "output_len": 256,
                "concurrency": 16,
                "num_prompts": 30,
                "request_rate": "inf",
            },
            {
                "name": "Concurrency Scaling (c=32)",
                "description": "Scaling test at concurrency 32",
                "input_len": 1024,
                "output_len": 256,
                "concurrency": 32,
                "num_prompts": 30,
                "request_rate": "inf",
            },
        ]


def _strip_wrapping_markdown_fence(text: str) -> str:
    """Remove a single outer ```markdown ... ``` fence if present."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return t


def _strip_leading_heading(text: str, heading: str) -> str:
    """Strip a redundant leading heading like '### Interpretation' returned by the model."""
    t = text.strip()
    if not t:
        return t
    lines = t.splitlines()
    if not lines:
        return t

    first = lines[0].strip()
    normalized = re.sub(r"[^a-zA-Z]", "", first).lower()
    target = re.sub(r"[^a-zA-Z]", "", heading).lower()

    if normalized == target or normalized.endswith(target):
        return "\n".join(lines[1:]).strip()
    return t


def _format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def _safe_int(value: float | int) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return 0


def _scenario_category(name: str) -> str:
    if "Scaling" in name:
        return "Concurrency Scaling"
    if "Single User" in name:
        return "Latency"
    if "Short Context" in name:
        return "Throughput"
    if "Long Context" in name:
        return "Long Context"
    if "Decode" in name:
        return "Decode"
    if "Mixed" in name:
        return "Mixed"
    return "Other"


def build_performance_summary_table(results: list[BenchmarkResult]) -> str:
    header = (
        "| Scenario | Category | Input (tok) | Output (tok) | Concurrency | Output Throughput (tok/s) | "
        "TTFT mean (ms) | TPOT mean (ms) | TPOT p99 (ms) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in results:
        if r.error:
            rows.append(
                f"| {r.name} | {_scenario_category(r.name)} | {r.input_len} | {r.output_len} | {r.concurrency} | ERROR | - | - | - |"
            )
            continue

        rows.append(
            "| "
            + " | ".join(
                [
                    r.name,
                    _scenario_category(r.name),
                    str(r.input_len),
                    str(r.output_len),
                    str(r.concurrency),
                    _format_float(r.output_throughput, 2),
                    _format_float(r.ttft_mean_ms, 2),
                    _format_float(r.tpot_mean_ms, 2),
                    _format_float(r.tpot_p99_ms, 2),
                ]
            )
            + " |"
        )

    return header + "\n" + "\n".join(rows)


def build_pp_tg_table(results: list[BenchmarkResult]) -> str:
    header = (
        "| Scenario | Prefill (PP) tok/s | Token Generation (TG) tok/s |\n"
        "|---|---:|---:|"
    )
    rows = []
    for r in results:
        if r.error:
            rows.append(f"| {r.name} | ERROR | ERROR |")
        else:
            rows.append(
                f"| {r.name} | {_format_float(r.pp_speed, 1)} | {_format_float(r.tg_speed, 1)} |"
            )
    return header + "\n" + "\n".join(rows)


def _scaling_points(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    pts = [r for r in results if (not r.error and "Scaling" in r.name)]
    pts.sort(key=lambda r: r.concurrency)
    return pts


def build_concurrency_scaling_table(results: list[BenchmarkResult]) -> str:
    pts = _scaling_points(results)
    if not pts:
        return "(No concurrency scaling scenarios found.)"

    header = (
        "| Concurrency | Output Throughput (tok/s) | Per-user Throughput (tok/s) | TTFT p99 (ms) | TPOT p99 (ms) |\n"
        "|---:|---:|---:|---:|---:|"
    )

    rows = []
    for p in pts:
        per_user = p.output_throughput / max(1, p.concurrency)
        rows.append(
            f"| {p.concurrency} | {_format_float(p.output_throughput, 2)} | {_format_float(per_user, 2)} | {_format_float(p.ttft_p99_ms, 2)} | {_format_float(p.tpot_p99_ms, 2)} |"
        )
    return header + "\n" + "\n".join(rows)


def infer_interactive_sweet_spot(results: list[BenchmarkResult]) -> str:
    """Interactive-first heuristic.

    Goal: keep TTFT/TPOT tail latency low while still maintaining decent per-user throughput.
    Strategy:
      1) Find the minimum TTFT p99 and TPOT p99 across scaling points.
      2) Keep candidates within 2× of BOTH minima.
      3) Choose the candidate with the highest per-user throughput (output_throughput / concurrency).
      4) Break ties by choosing lower concurrency.
    """
    pts = _scaling_points(results)
    if not pts:
        return "N/A"

    min_ttft = min((p.ttft_p99_ms for p in pts if p.ttft_p99_ms > 0), default=0.0)
    min_tpot = min((p.tpot_p99_ms for p in pts if p.tpot_p99_ms > 0), default=0.0)
    if min_ttft <= 0 or min_tpot <= 0:
        return "N/A"

    ttft_cap = 2.0 * min_ttft
    tpot_cap = 2.0 * min_tpot
    candidates = [p for p in pts if p.ttft_p99_ms <= ttft_cap and p.tpot_p99_ms <= tpot_cap]
    if not candidates:
        candidates = pts

    def key(p: BenchmarkResult) -> tuple[float, int]:
        per_user = p.output_throughput / max(1, p.concurrency)
        return (per_user, -p.concurrency)

    best = max(candidates, key=key)
    return (
        f"c={best.concurrency} (interactive-optimal by TTFT/TPOT p99; per-user {_format_float(best.output_throughput / best.concurrency, 1)} tok/s)"
    )


def infer_peak_throughput(results: list[BenchmarkResult]) -> str:
    pts = _scaling_points(results)
    if not pts:
        return "N/A"
    best = max(pts, key=lambda p: p.output_throughput)
    return f"c={best.concurrency} ({_format_float(best.output_throughput, 1)} tok/s)"


def build_mermaid_throughput_chart(results: list[BenchmarkResult]) -> str:
    pts = _scaling_points(results)
    if not pts:
        return ""
    x = [str(p.concurrency) for p in pts]
    y = [str(_format_float(p.output_throughput, 2)) for p in pts]
    return (
        "```mermaid\n"
        "xychart-beta\n"
        '    title "Output throughput vs concurrency"\n'
        '    x-axis "Concurrency" ["' + '\", \"'.join(x) + '"]\n'
        '    y-axis "tok/s"\n'
        "    line [" + ", ".join(y) + "]\n"
        "```"
    )


def parse_benchmark_output(output: str) -> dict:
    """Parse vllm bench serve output into structured data"""
    metrics = {}
    
    patterns = {
        "successful_requests": r"Successful requests:\s+([\d.]+)",
        "failed_requests": r"Failed requests:\s+([\d.]+)",
        "duration_s": r"Benchmark duration \(s\):\s+([\d.]+)",
        "total_input_tokens": r"Total input tokens:\s+([\d.]+)",
        "total_output_tokens": r"Total generated tokens:\s+([\d.]+)",
        "request_throughput": r"Request throughput \(req/s\):\s+([\d.]+)",
        "output_throughput": r"Output token throughput \(tok/s\):\s+([\d.]+)",
        "peak_output_throughput": r"Peak output token throughput \(tok/s\):\s+([\d.]+)",
        "total_throughput": r"Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)",
        "ttft_mean_ms": r"Mean TTFT \(ms\):\s+([\d.]+)",
        "ttft_median_ms": r"Median TTFT \(ms\):\s+([\d.]+)",
        "ttft_p99_ms": r"P99 TTFT \(ms\):\s+([\d.]+)",
        "tpot_mean_ms": r"Mean TPOT \(ms\):\s+([\d.]+)",
        "tpot_median_ms": r"Median TPOT \(ms\):\s+([\d.]+)",
        "tpot_p99_ms": r"P99 TPOT \(ms\):\s+([\d.]+)",
        "itl_mean_ms": r"Mean ITL \(ms\):\s+([\d.]+)",
        "itl_median_ms": r"Median ITL \(ms\):\s+([\d.]+)",
        "itl_p99_ms": r"P99 ITL \(ms\):\s+([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics


def run_benchmark(
    model: str,
    base_url: str,
    scenario: dict,
) -> BenchmarkResult:
    """Run a single benchmark scenario"""
    
    result = BenchmarkResult(
        name=scenario["name"],
        input_len=scenario["input_len"],
        output_len=scenario["output_len"],
        concurrency=scenario["concurrency"],
        num_prompts=scenario["num_prompts"],
    )
    
    cmd = [
        "vllm", "bench", "serve",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--random-input-len", str(scenario["input_len"]),
        "--random-output-len", str(scenario["output_len"]),
        "--max-concurrency", str(scenario["concurrency"]),
        "--num-prompts", str(scenario["num_prompts"]),
        "--request-rate", str(scenario.get("request_rate", "inf")),
    ]
    
    if scenario.get("range_ratio"):
        cmd.extend(["--random-range-ratio", str(scenario["range_ratio"])])
    
    if scenario.get("extra_args"):
        cmd.extend(scenario["extra_args"])
    
    print(f"\n{'='*60}")
    print(f"Running: {scenario['name']}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )
        
        output = proc.stdout + proc.stderr
        result.raw_output = output
        
        if proc.returncode != 0:
            result.error = f"Process exited with code {proc.returncode}"
            print(f"Error: {result.error}")
            return result
        
        # Parse the output
        metrics = parse_benchmark_output(output)
        
        # Update result with parsed metrics
        for key, value in metrics.items():
            if hasattr(result, key):
                setattr(result, key, value)
        
        # Calculate derived metrics
        if result.ttft_mean_ms > 0 and result.input_len > 0:
            result.pp_speed = (result.input_len / result.ttft_mean_ms) * 1000
        
        if result.tpot_mean_ms > 0:
            result.tg_speed = 1000 / result.tpot_mean_ms
        
        print(f"✓ Completed: {result.output_throughput:.1f} tok/s, "
              f"TTFT: {result.ttft_mean_ms:.1f}ms, TPOT: {result.tpot_mean_ms:.1f}ms")
        
    except subprocess.TimeoutExpired:
        result.error = "Benchmark timed out after 30 minutes"
        print(f"Error: {result.error}")
    except Exception as e:
        result.error = str(e)
        print(f"Error: {result.error}")
    
    return result


def generate_report_prompt(model: str, hardware_info: str, results: list[BenchmarkResult]) -> str:
    """Generate the prompt for the model to write its own report"""
    
    # Format results as structured data
    results_text = ""
    for r in results:
        if r.error:
            results_text += f"\n### {r.name}\n**ERROR:** {r.error}\n"
            continue
            
        results_text += f"""
### {r.name}
- Input/Output: {r.input_len} / {r.output_len} tokens
- Concurrency: {r.concurrency}
- Prompts: {r.num_prompts}
- Duration: {r.duration_s:.1f}s
- Success/Fail: {int(r.successful_requests)}/{int(r.failed_requests)}
- **Output Throughput: {r.output_throughput:.2f} tok/s**
- Peak Throughput: {r.peak_output_throughput:.2f} tok/s
- Total Throughput: {r.total_throughput:.2f} tok/s
- TTFT: mean={r.ttft_mean_ms:.2f}ms, median={r.ttft_median_ms:.2f}ms, p99={r.ttft_p99_ms:.2f}ms
- TPOT: mean={r.tpot_mean_ms:.2f}ms, median={r.tpot_median_ms:.2f}ms, p99={r.tpot_p99_ms:.2f}ms
- ITL: mean={r.itl_mean_ms:.2f}ms, median={r.itl_median_ms:.2f}ms, p99={r.itl_p99_ms:.2f}ms
- Derived PP (prefill): {r.pp_speed:.1f} tok/s
- Derived TG (decode): {r.tg_speed:.1f} tok/s
"""
    
    prompt = f"""You are writing a professional benchmark report for an LLM inference deployment. 
Write a comprehensive markdown report based on the following benchmark data.

**Model:** {model}
**Hardware:** {hardware_info}
**Date:** {datetime.now().strftime("%B %d, %Y")}

## Raw Benchmark Results
{results_text}

---

Please write a professional benchmark report in markdown format that includes:

1. **Executive Summary** - Brief overview of the model, hardware, and key findings (2-3 sentences)

2. **Hardware Configuration** - Table of hardware specs

3. **Benchmark Results** - For each major test category:
   - What was tested and why
   - Key metrics in a readable format
   - Brief interpretation

4. **Performance Summary Table** - A consolidated table showing all scenarios with Input, Output, Concurrency, Throughput, TTFT, and TPOT

5. **Prefill (PP) and Token Generation (TG) Speeds** - Derived metrics table with explanation

6. **Concurrency Scaling Analysis** - How does performance change with concurrency? Is there a sweet spot?

7. **Key Observations** - Strengths and limitations based on the data

8. **Recommendations** - Practical advice for different use cases (interactive, batch, etc.)

Use clear formatting, tables where appropriate, and highlight the most important metrics in bold.
Be analytical and draw conclusions from the data - don't just repeat numbers.
Note any anomalies or interesting patterns in the results.

Write the report now:"""

    return prompt


def generate_interpretation_prompt(
    model: str,
    hardware_info: str,
    scenario: dict,
    result: BenchmarkResult,
) -> str:
    """Prompt for a single scenario interpretation section (no tables, no guesses)."""
    if result.error:
        metrics_block = f"ERROR: {result.error}"
    else:
        metrics_block = "\n".join(
            [
                f"Input tokens: {result.input_len}",
                f"Output tokens: {result.output_len}",
                f"Concurrency: {result.concurrency}",
                f"Prompts: {result.num_prompts}",
                f"Duration (s): {_format_float(result.duration_s, 2)}",
                f"Output throughput (tok/s): {_format_float(result.output_throughput, 2)}",
                f"Peak output throughput (tok/s): {_format_float(result.peak_output_throughput, 2)}",
                f"Total token throughput (tok/s): {_format_float(result.total_throughput, 2)}",
                f"TTFT mean/median/p99 (ms): {_format_float(result.ttft_mean_ms, 2)} / {_format_float(result.ttft_median_ms, 2)} / {_format_float(result.ttft_p99_ms, 2)}",
                f"TPOT mean/median/p99 (ms): {_format_float(result.tpot_mean_ms, 2)} / {_format_float(result.tpot_median_ms, 2)} / {_format_float(result.tpot_p99_ms, 2)}",
                f"ITL mean/median/p99 (ms): {_format_float(result.itl_mean_ms, 2)} / {_format_float(result.itl_median_ms, 2)} / {_format_float(result.itl_p99_ms, 2)}",
                f"Derived prefill speed PP (tok/s): {_format_float(result.pp_speed, 1)}",
                f"Derived decode speed TG (tok/s): {_format_float(result.tg_speed, 1)}",
            ]
        )

    return f"""You are writing the Interpretation bullet list for ONE benchmark scenario.

Constraints:
- Output ONLY bullet points (no headings like 'Interpretation').
- Do NOT output tables.
- Do NOT output code fences.
- Do NOT restate all numbers; interpret what they imply.
- Do NOT invent hardware details, VRAM, CPU, interconnect, or theoretical maxima/efficiency.
- Do NOT speculate on causes (e.g., "quantization overhead", "scheduler bottleneck", "no hardware saturation") unless directly supported by the measured metrics.
- Keep it to 4-7 bullet points.

Interpretation guidance:
- TTFT mainly reflects queueing + prefill cost; higher TTFT is worse for interactive UX.
- TPOT mainly reflects decode speed; lower TPOT is faster token generation.
- Output throughput reflects batching/utilization; higher throughput can increase TTFT/TPOT.
- Compare mean vs p99: a large gap implies tail-latency/jitter.
- For long-context tests, TTFT can rise sharply; call out whether TPOT stays stable.

Context:
Model: {model}
Hardware (verbatim): {hardware_info}

Scenario name: {scenario['name']}
Scenario purpose: {scenario.get('description','')}

Measured metrics:
{metrics_block}
"""


def generate_exec_summary_prompt(
    model: str,
    hardware_info: str,
    results: list[BenchmarkResult],
    sweet_spot: str,
) -> str:
    ok = [r for r in results if not r.error]
    peak = max(ok, key=lambda r: r.output_throughput) if ok else None
    best_latency = min(ok, key=lambda r: r.tpot_mean_ms) if ok else None
    return f"""Write a concise Executive Summary (2-4 sentences) for a benchmark report.

Constraints:
- Output NO heading.
- No tables.
- No code fences.
- Do not guess hardware specs beyond: {hardware_info}
- Do not claim theoretical maxima/efficiency.
- This report prioritizes INTERACTIVE UX (low TTFT/TPOT, good per-user experience) over max aggregate throughput.
- Mention the interactive recommended concurrency (sweet spot) and mention peak throughput as secondary.

Facts:
- Model: {model}
- Hardware: {hardware_info}
- Interactive sweet spot (heuristic): {sweet_spot}
- Peak throughput: {('scenario '+peak.name+' at '+_format_float(peak.output_throughput,2)+' tok/s') if peak else 'N/A'}
- Best TPOT: {('scenario '+best_latency.name+' at '+_format_float(best_latency.tpot_mean_ms,2)+' ms') if best_latency else 'N/A'}
"""


def generate_recommendations_prompt(
    model: str,
    hardware_info: str,
    results: list[BenchmarkResult],
    sweet_spot: str,
) -> str:
    return f"""Write deployment recommendations based on the benchmark outcomes.

Constraints:
- Output ONLY bullet points (no headings).
- No tables.
- No code fences.
- No guessed hardware specs beyond: {hardware_info}
- Do not mention VRAM size, CPU model, or specific GPU product names unless explicitly provided.
- Base recommendations only on observed TTFT/TPOT/throughput/scaling patterns.
- Use 5-8 bullet points; keep them practical.

Additional constraints:
- Do NOT invent specific token bucket settings, refill times, rate limits, batching schedules, warm-up durations, or capacity numbers unless they are directly computed from the provided results.
- Prefer qualitative guidance (e.g., "cap concurrency", "separate batch vs interactive pools") and reference the reported metrics (TTFT/TPOT/throughput) when justifying.

Required:
- The first bullet MUST be exactly: "- Use {sweet_spot} for interactive UX."
- Do NOT recommend maximizing concurrency for throughput except as a clearly labeled batch/offline trade-off.

Context:
Model: {model}
Hardware: {hardware_info}
Interactive sweet spot (heuristic): {sweet_spot}
"""


def generate_report(
    client: OpenAI,
    model: str,
    prompt: str,
    max_tokens: int = 4096,
) -> str:
    """Use the model to generate its own benchmark report"""
    
    print("\n" + "="*60)
    print("Generating report using the model...")
    print("="*60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return f"# Error Generating Report\n\nFailed to generate report: {e}"


def get_hardware_info() -> str:
    """Attempt to gather hardware information"""
    info_parts: list[str] = []

    # Best effort: use torch device properties (works on ROCm too and avoids rocm-smi/libdrm noise).
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            names: list[str] = []
            for idx in range(torch.cuda.device_count()):
                name = str(torch.cuda.get_device_name(idx)).strip()
                arch = ""
                try:
                    props = torch.cuda.get_device_properties(idx)
                    if not name:
                        name = str(getattr(props, "name", "")).strip()
                    arch = str(getattr(props, "gcnArchName", "")).strip()
                except Exception:
                    arch = ""

                if arch:
                    arch = arch.split(":", 1)[0].strip()

                if not name:
                    name = "AMD GPU"

                # Reduce hallucination risk: gfx908 is commonly AMD Instinct MI100.
                if arch == "gfx908" and name in {"AMD GPU", "AMD Radeon Graphics", ""}:
                    name = "AMD Instinct MI100"

                if arch and arch not in name:
                    names.append(f"{name} ({arch})")
                else:
                    names.append(name)

            gpu_counts: dict[str, int] = {}
            for name in names:
                gpu_counts[name] = gpu_counts.get(name, 0) + 1

            for name, count in sorted(gpu_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                info_parts.append(f"{count}× {name}")
            return ", ".join(info_parts)
    except Exception:
        pass

    # Fallback: try rocm-smi. Some environments emit extra comma-separated diagnostics on the same line;
    # we keep only the first comma-separated token after the colon.
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = re.findall(r"GPU\[\d+\].*?:\s*([^\n]+)", result.stdout)
            cleaned = []
            for gpu in gpus:
                first = gpu.split(",", 1)[0].strip()
                if first:
                    cleaned.append(first)
            if cleaned:
                gpu_counts: dict[str, int] = {}
                for gpu in cleaned:
                    gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                for gpu, count in sorted(gpu_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    info_parts.append(f"{count}× {gpu}")
                return ", ".join(info_parts)
    except Exception:
        pass

    # Fallback: try nvidia-smi.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            gpus = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if gpus:
                gpu_counts: dict[str, int] = {}
                for gpu in gpus:
                    gpu_counts[gpu] = gpu_counts.get(gpu, 0) + 1
                for gpu, count in sorted(gpu_counts.items(), key=lambda kv: (-kv[1], kv[0])):
                    info_parts.append(f"{count}× {gpu}")
                return ", ".join(info_parts)
    except Exception:
        pass

    return "GPU information not available"


def get_system_configuration_markdown(hardware_info: str) -> str:
    """Best-effort system configuration table (deterministic, no LLM)."""
    os_pretty = "Unknown"
    kernel = "Unknown"
    try:
        with open("/etc/os-release", "r") as f:
            osr = f.read()
        m = re.search(r"^PRETTY_NAME=\"(.+)\"$", osr, flags=re.MULTILINE)
        if m:
            os_pretty = m.group(1).strip()
    except Exception:
        pass

    try:
        r = subprocess.run(["uname", "-r"], capture_output=True, text=True, timeout=3)
        if r.returncode == 0 and r.stdout.strip():
            kernel = r.stdout.strip()
    except Exception:
        pass

    cpu_model = "Unknown"
    try:
        r = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            m = re.search(r"^Model name:\s*(.+)$", r.stdout, flags=re.MULTILINE)
            if m:
                cpu_model = m.group(1).strip()
    except Exception:
        pass

    ram_gb = "Unknown"
    try:
        with open("/proc/meminfo", "r") as f:
            meminfo = f.read()
        m = re.search(r"^MemTotal:\s*(\d+)\s*kB$", meminfo, flags=re.MULTILINE)
        if m:
            kb = int(m.group(1))
            ram_gb = f"{kb / 1024 / 1024:.1f} GB"
    except Exception:
        pass

    rocm = "Unknown"
    try:
        r = subprocess.run(
            ["rocm-smi", "--showdriverversion"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0:
            # Prefer explicit version fields if present; avoid footer lines.
            text_out = r.stdout
            candidates: list[str] = []

            for pat in [
                r"ROCm\s*(?:Version|version)\s*:\s*([^\n]+)",
                r"Driver\s*(?:Version|version)\s*:\s*([^\n]+)",
                r"Kernel\s*(?:Version|version)\s*:\s*([^\n]+)",
            ]:
                m = re.search(pat, text_out)
                if m:
                    val = m.group(1).strip()
                    # Drop obvious separators/footers.
                    if val and "End of ROCm" not in val and not set(val) <= set("= "):
                        candidates.append(val)

            if candidates:
                rocm = "; ".join(dict.fromkeys(candidates))
            else:
                # Last-resort: pick the first line that looks like a version.
                for ln in [l.strip() for l in text_out.splitlines() if l.strip()]:
                    if "End of ROCm" in ln:
                        continue
                    if re.search(r"\d+\.\d+", ln):
                        rocm = ln
                        break
    except Exception:
        pass

    xgmi: str | None = None

    # Best effort: detect XGMI via ROCm KFD topology (reliable on multi-GPU ROCm systems).
    try:
        nodes_dir = Path("/sys/class/kfd/kfd/topology/nodes")
        gpu_hive_ids: list[int] = []
        gpu_xgmi_engines: list[int] = []
        if nodes_dir.exists():
            for node in sorted(nodes_dir.iterdir()):
                props = node / "properties"
                if not props.is_file():
                    continue
                try:
                    text_out = props.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                def _read_int(key: str) -> int:
                    m = re.search(rf"^{re.escape(key)}\s+(\d+)\s*$", text_out, flags=re.MULTILINE)
                    return int(m.group(1)) if m else 0

                simd_count = _read_int("simd_count")
                # On this system, CPU node has simd_count=0, GPUs have simd_count>0.
                if simd_count <= 0:
                    continue

                hive_id = _read_int("hive_id")
                num_xgmi = _read_int("num_sdma_xgmi_engines")
                gpu_hive_ids.append(hive_id)
                gpu_xgmi_engines.append(num_xgmi)

        if gpu_hive_ids:
            unique_nonzero_hives = {h for h in gpu_hive_ids if h != 0}
            any_xgmi_engines = any(n > 0 for n in gpu_xgmi_engines)

            if len(gpu_hive_ids) == 1:
                xgmi = "N/A (single GPU)"
            elif any_xgmi_engines and unique_nonzero_hives:
                # If all GPUs share one hive, show it for clarity.
                if len(unique_nonzero_hives) == 1:
                    hive = next(iter(unique_nonzero_hives))
                    xgmi = f"Present (hive_id 0x{hive:x})"
                else:
                    xgmi = "Present (multiple hives)"
            elif not any_xgmi_engines and not unique_nonzero_hives:
                xgmi = "Not detected"
    except Exception:
        pass

    # Fallback: rocm-smi topology text heuristics.
    if xgmi is None:
        try:
            r = subprocess.run(
                ["rocm-smi", "--showtopology"],
                capture_output=True,
                text=True,
                timeout=7,
            )
            if r.returncode == 0:
                topo = r.stdout
                if re.search(r"\bXGMI\b", topo, flags=re.IGNORECASE) or re.search(r"XGMI\d+", topo):
                    xgmi = "Present"
                elif re.search(r"\bNot\s+Supported\b", topo, flags=re.IGNORECASE):
                    xgmi = "Not supported"
                elif topo.strip():
                    # Topology printed but doesn't mention XGMI.
                    xgmi = "Unknown"
        except Exception:
            pass

    rows = [
        "| Component | Value |",
        "|---|---|",
        f"| GPU(s) | {hardware_info} |",
        f"| OS | {os_pretty} |",
        f"| Kernel | {kernel} |",
        f"| CPU | {cpu_model} |",
        f"| System RAM | {ram_gb} |",
        f"| ROCm | {rocm} |",
    ]
    if xgmi is not None and xgmi != "Unknown":
        rows.append(f"| xGMI | {xgmi} |")
    return "\n".join(rows)


def format_key_metrics_block(r: BenchmarkResult) -> str:
    if r.error:
        return f"- ERROR: {r.error}"
    return (
        f"- Input/Output: {r.input_len} / {r.output_len} tokens\n"
        f"- Concurrency: {r.concurrency}\n"
        f"- Prompts: {r.num_prompts}\n"
        f"- Output throughput: {_format_float(r.output_throughput,2)} tok/s (peak {_format_float(r.peak_output_throughput,2)})\n"
        f"- TTFT mean/median/p99: {_format_float(r.ttft_mean_ms,2)} / {_format_float(r.ttft_median_ms,2)} / {_format_float(r.ttft_p99_ms,2)} ms\n"
        f"- TPOT mean/median/p99: {_format_float(r.tpot_mean_ms,2)} / {_format_float(r.tpot_median_ms,2)} / {_format_float(r.tpot_p99_ms,2)} ms\n"
        f"- PP/TG (derived): {_format_float(r.pp_speed,1)} / {_format_float(r.tg_speed,1)} tok/s"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks and generate a self-documented report"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model name/path (as served by vLLM)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--hardware",
        default=None,
        help="Hardware description (auto-detected if not provided)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: benchmark_<model>_<timestamp>.md)"
    )
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Skip benchmarks and use existing results JSON"
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Path to existing results JSON (for --skip-benchmarks)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save raw results to JSON file"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark suite (fewer scenarios, fewer prompts)"
    )

    parser.add_argument(
        "--scaffolded-report",
        action="store_true",
        help="Generate a deterministic report scaffold (tables/charts) and ask the LLM only for short prose sections"
    )

    parser.add_argument(
        "--include-mermaid",
        action="store_true",
        help="Include a Mermaid throughput-vs-concurrency chart (requires a Mermaid-capable markdown renderer)"
    )
    
    args = parser.parse_args()
    
    # Setup
    hardware_info = args.hardware or get_hardware_info()
    print(f"Hardware detected: {hardware_info}")
    
    # Initialize OpenAI client pointing to vLLM
    client = OpenAI(
        base_url=f"{args.base_url}/v1",
        api_key="not-needed",  # vLLM doesn't require API key by default
    )
    
    # Run benchmarks or load existing results
    results = []
    
    if args.skip_benchmarks and args.results_json:
        print(f"Loading results from {args.results_json}")
        with open(args.results_json) as f:
            data = json.load(f)
            for r in data:
                results.append(BenchmarkResult(**r))
    else:
        suite = BenchmarkSuite()
        suite.add_default_scenarios()
        
        # Reduce scenarios for quick mode
        if args.quick:
            suite.scenarios = [s for s in suite.scenarios 
                            if "Scaling" not in s["name"] or s["concurrency"] in [1, 8, 32]]
            for s in suite.scenarios:
                s["num_prompts"] = min(s["num_prompts"], 10)
        
        print(f"\nRunning {len(suite.scenarios)} benchmark scenarios...")
        print(f"Model: {args.model}")
        print(f"Server: {args.base_url}")
        
        for scenario in suite.scenarios:
            result = run_benchmark(args.model, args.base_url, scenario)
            results.append(result)
            
            # Brief pause between benchmarks
            time.sleep(2)
        
        # Optionally save raw results
        if args.save_results:
            results_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump([asdict(r) for r in results], f, indent=2)
            print(f"\nRaw results saved to: {results_file}")
    
    # Generate report using the model (interactive UX is the default priority).
    sweet_spot = infer_interactive_sweet_spot(results)
    peak_scaling = infer_peak_throughput(results)

    if args.scaffolded_report:
        perf_table = build_performance_summary_table(results)
        pp_tg_table = build_pp_tg_table(results)
        scaling_table = build_concurrency_scaling_table(results)
        mermaid = build_mermaid_throughput_chart(results) if args.include_mermaid else ""
        system_table = get_system_configuration_markdown(hardware_info)

        scenario_sections = []
        suite = BenchmarkSuite()
        suite.add_default_scenarios()
        by_name = {r.name: r for r in results}
        for scenario in suite.scenarios:
            r = by_name.get(scenario["name"])
            if r is None:
                continue
            interp = _strip_wrapping_markdown_fence(
                generate_report(
                    client,
                    args.model,
                    generate_interpretation_prompt(args.model, hardware_info, scenario, r),
                    max_tokens=450,
                )
            )
            interp = _strip_leading_heading(interp, "Interpretation")
            scenario_sections.append(
                "\n".join(
                    [
                        f"### {r.name}",
                        f"**Purpose:** {scenario.get('description','')}",
                        "**Key metrics:**",
                        format_key_metrics_block(r),
                        "",
                        "---",
                        "",
                        "**Interpretation:**",
                        "",
                        interp.strip(),
                    ]
                )
            )

        recommendations = _strip_wrapping_markdown_fence(
            generate_report(
                client,
                args.model,
                generate_recommendations_prompt(args.model, hardware_info, results, sweet_spot),
                max_tokens=500,
            )
        )
        recommendations = _strip_leading_heading(recommendations, "Recommendations")

        # Build report body first, then ask for an executive summary last (better context).
        report_body_without_summary = "\n\n".join(
            [
                "# LLM Inference Benchmark Report",
                f"**Model:** {args.model}",
                f"**Hardware:** {hardware_info}",
                f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
                "",
                "## System Configuration",
                system_table,
                "",
                "## Performance Summary",
                perf_table,
                "",
                "## Prefill (PP) and Decode (TG)",
                pp_tg_table,
                "",
                "## Concurrency Scaling",
                f"**Interactive sweet spot:** {sweet_spot}",
                f"**Peak throughput (scaling):** {peak_scaling}",
                "",
                scaling_table,
                ("\n\n" + mermaid) if mermaid else "",
                "",
                "## Scenario Results",
                "\n\n".join(scenario_sections),
                "",
                "## Recommendations",
                recommendations.strip(),
            ]
        )

        exec_summary = _strip_wrapping_markdown_fence(
            generate_report(
                client,
                args.model,
                generate_exec_summary_prompt(args.model, hardware_info, results, sweet_spot)
                + "\n\nAdditional context (do not quote verbatim; do not add tables):\n\n"
                + report_body_without_summary[:6000],
                max_tokens=350,
            )
        )
        exec_summary = _strip_leading_heading(exec_summary, "Executive Summary")

        report = "\n\n".join(
            [
                "# LLM Inference Benchmark Report",
                f"**Model:** {args.model}",
                f"**Hardware:** {hardware_info}",
                f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
                "",
                "## Executive Summary",
                exec_summary.strip(),
                "",
                "## System Configuration",
                system_table,
                "",
                "## Performance Summary",
                perf_table,
                "",
                "## Prefill (PP) and Decode (TG)",
                pp_tg_table,
                "",
                "## Concurrency Scaling",
                f"**Interactive sweet spot:** {sweet_spot}",
                f"**Peak throughput (scaling):** {peak_scaling}",
                "",
                scaling_table,
                ("\n\n" + mermaid) if mermaid else "",
                "",
                "## Scenario Results",
                "\n\n".join(scenario_sections),
                "",
                "## Recommendations",
                recommendations.strip(),
            ]
        )
    else:
        # Legacy single-shot report generation (LLM produces everything).
        prompt = generate_report_prompt(args.model, hardware_info, results)
        report = _strip_wrapping_markdown_fence(generate_report(client, args.model, prompt))
    
    # Add metadata header
    model_short = args.model.split("/")[-1] if "/" in args.model else args.model
    full_report = f"""<!-- 
Benchmark Report Generated: {datetime.now().isoformat()}
Model: {args.model}
Hardware: {hardware_info}
Generated by: benchmark_and_report.py
Note: This report was written by the model being benchmarked.
-->

{report}

---

*Report generated {datetime.now().strftime("%B %d, %Y at %H:%M")} by {model_short}*
"""
    
    # Save report
    if args.output:
        output_path = args.output
    else:
        safe_model_name = re.sub(r'[^\w\-]', '_', model_short)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"benchmark_{safe_model_name}_{timestamp}.md"
    
    with open(output_path, 'w') as f:
        f.write(full_report)
    
    print(f"\n{'='*60}")
    print(f"✓ Report saved to: {output_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())