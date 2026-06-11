#!/usr/bin/env python3
"""
UA vs TRITON_ATTN backend comparison on Qwen3.6-27B-GPTQ-8bit (gfx908, MTP n=3 + P82).
Reads the 12-tier A1 (UA) and A2 (TRITON) results from Model_Reports/json_data/ and
appends the dataset-generation workload A/B (4k in / 6k out), then renders a grouped
bar chart with per-tier UA deltas. Output: charts/ua_vs_triton_27b.png
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_DIR = Path("Model_Reports/json_data")
OUT = Path("charts/ua_vs_triton_27b.png")

def load(fn):
    d = json.load(open(JSON_DIR / fn))
    rows = d if isinstance(d, list) else d.get("results", d)
    return {r["name"]: r["output_throughput"] for r in rows}

ua = load("ua_eval_A1_UA_MTP_27B8.json")
tr = load("ua_eval_A2_TRITON_MTP_27B8.json")

# scenario -> (short label). Pick the decision-relevant tiers + workload.
TIERS = [
    ("Single User Latency", "Interactive\nc=1"),
    ("Decode Stress Test", "Decode\nc=1"),
    ("Long Context (16K)", "Long ctx\n16K c=4"),
    ("Short Context Throughput", "Short ctx\nc=16"),
    ("Concurrency Scaling (c=32)", "c=32\n(1k/256)"),
    ("Concurrency Scaling (c=128)", "c=128\n(1k/256)"),
]
labels, ua_v, tr_v = [], [], []
for key, lab in TIERS:
    labels.append(lab); ua_v.append(ua[key]); tr_v.append(tr[key])

# dataset-gen workload A/B (4k in / 6k out, sonnet, MTP-P82) — measured separately
for lab, u, t in [("Dataset-gen\n4k/6k c=32", 745.7, 646.0),
                  ("Dataset-gen\n4k/6k c=64", 742.7, 648.0)]:
    labels.append(lab); ua_v.append(u); tr_v.append(t)

x = range(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(13, 6))
b1 = ax.bar([i - w/2 for i in x], ua_v, w, label="Unified Attention (UA)", color="#2a9d8f")
b2 = ax.bar([i + w/2 for i in x], tr_v, w, label="TRITON_ATTN", color="#e76f51")

for i in x:
    d = (ua_v[i] - tr_v[i]) / tr_v[i] * 100
    ax.annotate(f"{d:+.0f}%", (i, max(ua_v[i], tr_v[i])),
                textcoords="offset points", xytext=(0, 4), ha="center",
                fontsize=9, fontweight="bold",
                color=("#1d7a6e" if d > 1 else ("#c1442e" if d < -1 else "#666")))

ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Output throughput (tok/s)")
ax.set_title("Qwen3.6-27B-GPTQ-8bit (gfx908, 4×MI100) — UA vs TRITON_ATTN under MTP n=3 + P82\n"
             "% = UA advantage. UA wins interactive / long-output / long-context; "
             "TRITON wins short-context batch.", fontsize=11)
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
OUT.parent.mkdir(exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
