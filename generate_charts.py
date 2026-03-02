#!/usr/bin/env python3
"""
Generate interactive Plotly benchmark charts from JSON sidecar files.

Reads Model_Reports/json_data/*.json and produces:
  - charts/benchmark_charts.html  (interactive dark-themed dashboard)
  - charts/pp_tg_comparison.png   (prefill & decode bar chart)
  - charts/mixed_traffic.png      (mixed traffic bar chart)
  - charts/concurrency_scaling.png (throughput vs concurrency lines)
  - charts/per_user_scaling.png   (per-user throughput vs concurrency lines)
"""

import json
import os
from glob import glob
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSON_DIR = Path("Model_Reports/json_data")
OUT_DIR = Path("charts")
OUT_DIR.mkdir(exist_ok=True)

# Consistent palette (6 colours)
COLORS = [
    "#636EFA",  # blue
    "#EF553B",  # red
    "#00CC96",  # green
    "#AB63FA",  # purple
    "#FFA15A",  # orange
    "#19D3F3",  # cyan
]


def short_name(model: str) -> str:
    """Derive a concise display name from the model path."""
    name = model.split("/")[-1]
    # Trim common prefixes/suffixes to keep legend readable
    for prefix in ("benchmark_",):
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all():
    files = sorted(glob(str(JSON_DIR / "*.json")))
    models = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        data["_short"] = short_name(data["model"])
        models.append(data)
    return models


def find_result(model_data, name):
    """Find a result entry by scenario name."""
    for r in model_data["results"]:
        if r["name"] == name:
            return r
    return None


def find_scaling_results(model_data):
    """Return scaling results sorted by concurrency."""
    scaling = [r for r in model_data["results"]
               if r["name"].startswith("Concurrency Scaling")]
    return sorted(scaling, key=lambda r: r["concurrency"])


# ---------------------------------------------------------------------------
# Chart 1: Prefill & Decode (Single User, c=1)
# ---------------------------------------------------------------------------
def make_pp_tg_chart(models):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Prefill Speed (tok/s)", "Decode Speed (tok/s)"),
        shared_yaxes=True,
        horizontal_spacing=0.15,
    )

    names = []
    pp_vals = []
    tg_vals = []
    colors_used = []

    for i, m in enumerate(models):
        r = find_result(m, "Single User Latency")
        if r is None:
            continue
        names.append(m["_short"])
        pp_vals.append(r.get("pp_speed", 0))
        tg_vals.append(r.get("tg_speed", 0))
        colors_used.append(COLORS[i % len(COLORS)])

    fig.add_trace(go.Bar(
        y=names, x=pp_vals, orientation="h",
        marker_color=colors_used, name="Prefill (PP)",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=names, x=tg_vals, orientation="h",
        marker_color=colors_used, name="Decode (TG)",
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title_text="Single-User Prefill & Decode Performance (c=1)",
        height=max(350, 80 * len(names)),
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 2: Mixed Traffic
# ---------------------------------------------------------------------------
def make_mixed_traffic_chart(models):
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Output Throughput (tok/s)", "TTFT & TPOT (ms)"),
        horizontal_spacing=0.18,
        shared_yaxes=True,
    )

    names = []
    throughputs = []
    ttfts = []
    tpots = []
    colors_used = []

    for i, m in enumerate(models):
        r = find_result(m, "Mixed Traffic")
        if r is None:
            continue
        names.append(m["_short"])
        throughputs.append(r["output_throughput"])
        ttfts.append(r["ttft_mean_ms"])
        tpots.append(r["tpot_mean_ms"])
        colors_used.append(COLORS[i % len(COLORS)])

    fig.add_trace(go.Bar(
        y=names, x=throughputs, orientation="h",
        marker_color=colors_used, name="Output tok/s",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=names, x=ttfts, orientation="h",
        marker_color=colors_used, name="TTFT (ms)",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Bar(
        y=names, x=tpots, orientation="h",
        marker_color=[c.replace(")", ", 0.5)").replace("rgb", "rgba")
                       if c.startswith("rgb") else c
                       for c in colors_used],
        name="TPOT (ms)", showlegend=False,
        marker_line_width=2, marker_line_color=colors_used,
        opacity=0.5,
    ), row=1, col=2)

    fig.update_layout(
        title_text="Mixed Traffic Performance (c=8, variable input lengths)",
        height=max(350, 80 * len(names)),
        barmode="group",
        margin=dict(l=20, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3: Concurrency Scaling (aggregate throughput)
# ---------------------------------------------------------------------------
def make_concurrency_scaling_chart(models):
    fig = go.Figure()

    for i, m in enumerate(models):
        scaling = find_scaling_results(m)
        if not scaling:
            continue
        xs = [r["concurrency"] for r in scaling]
        ys = [r["output_throughput"] for r in scaling]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=m["_short"],
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title_text="Concurrency Scaling — Aggregate Output Throughput",
        xaxis_title="Concurrency",
        yaxis_title="Output Throughput (tok/s)",
        xaxis=dict(type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=500,
        margin=dict(l=60, r=20, t=60, b=100),
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4: Per-User Throughput vs Concurrency
# ---------------------------------------------------------------------------
def make_per_user_chart(models):
    fig = go.Figure()

    for i, m in enumerate(models):
        scaling = find_scaling_results(m)
        if not scaling:
            continue
        xs = [r["concurrency"] for r in scaling]
        ys = [r["output_throughput"] / r["concurrency"] for r in scaling]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers",
            name=m["_short"],
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title_text="Per-User Throughput vs Concurrency",
        xaxis_title="Concurrency",
        yaxis_title="Per-User Throughput (tok/s)",
        xaxis=dict(type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        height=500,
        margin=dict(l=60, r=20, t=60, b=100),
    )
    return fig


# ---------------------------------------------------------------------------
# Build all charts
# ---------------------------------------------------------------------------
def main():
    models = load_all()
    if not models:
        print("No JSON files found in", JSON_DIR)
        return

    print(f"Loaded {len(models)} model(s):")
    for m in models:
        print(f"  - {m['_short']}")

    fig_pp_tg = make_pp_tg_chart(models)
    fig_mixed = make_mixed_traffic_chart(models)
    fig_scaling = make_concurrency_scaling_chart(models)
    fig_per_user = make_per_user_chart(models)

    # --- Interactive HTML dashboard (dark theme) ---
    dashboard = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            "Single-User Prefill & Decode Performance (c=1)",
            "Mixed Traffic Performance (c=8)",
            "Concurrency Scaling — Aggregate Throughput",
            "Per-User Throughput vs Concurrency",
        ),
        vertical_spacing=0.08,
        specs=[[{"type": "xy"}]] * 4,
    )

    # For the dashboard we rebuild simplified versions inline
    for i, m in enumerate(models):
        r = find_result(m, "Single User Latency")
        if r:
            dashboard.add_trace(go.Bar(
                x=[r.get("pp_speed", 0), r.get("tg_speed", 0)],
                y=["Prefill (PP)", "Decode (TG)"],
                orientation="h", name=m["_short"],
                marker_color=COLORS[i % len(COLORS)],
                legendgroup=m["_short"],
                showlegend=True,
            ), row=1, col=1)

    for i, m in enumerate(models):
        r = find_result(m, "Mixed Traffic")
        if r:
            dashboard.add_trace(go.Bar(
                x=[r["output_throughput"]],
                y=["Throughput"],
                orientation="h", name=m["_short"],
                marker_color=COLORS[i % len(COLORS)],
                legendgroup=m["_short"],
                showlegend=False,
            ), row=2, col=1)

    for i, m in enumerate(models):
        scaling = find_scaling_results(m)
        if scaling:
            dashboard.add_trace(go.Scatter(
                x=[r["concurrency"] for r in scaling],
                y=[r["output_throughput"] for r in scaling],
                mode="lines+markers", name=m["_short"],
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=8),
                legendgroup=m["_short"],
                showlegend=False,
            ), row=3, col=1)

    for i, m in enumerate(models):
        scaling = find_scaling_results(m)
        if scaling:
            dashboard.add_trace(go.Scatter(
                x=[r["concurrency"] for r in scaling],
                y=[r["output_throughput"] / r["concurrency"] for r in scaling],
                mode="lines+markers", name=m["_short"],
                line=dict(color=COLORS[i % len(COLORS)], width=2),
                marker=dict(size=8),
                legendgroup=m["_short"],
                showlegend=False,
            ), row=4, col=1)

    dashboard.update_layout(
        template="plotly_dark",
        title_text="MI100 LLM Benchmark Dashboard (4x AMD Instinct MI100)",
        height=1800,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.03, xanchor="center", x=0.5),
    )

    html_path = OUT_DIR / "benchmark_charts.html"
    dashboard.write_html(str(html_path), include_plotlyjs=True)
    print(f"Wrote {html_path}")

    # --- Static PNGs (light theme for README) ---
    png_opts = dict(width=1200, height=500, scale=2)

    for fig_obj, name in [
        (fig_pp_tg, "pp_tg_comparison"),
        (fig_mixed, "mixed_traffic"),
        (fig_scaling, "concurrency_scaling"),
        (fig_per_user, "per_user_scaling"),
    ]:
        fig_obj.update_layout(template="plotly_white")
        png_path = OUT_DIR / f"{name}.png"
        fig_obj.write_image(str(png_path), **png_opts)
        print(f"Wrote {png_path}")

    print("\nDone! Open charts/benchmark_charts.html for the interactive dashboard.")


if __name__ == "__main__":
    main()
