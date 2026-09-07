#!/usr/bin/env python3
"""Render Model_Reports/energy_qwen38_flash_next.json into the energy SVG chart and print the README table."""
import json, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'Model_Reports/energy_qwen38_flash_next.json'
dst = sys.argv[2] if len(sys.argv) > 2 else 'Model_Reports/energy_curve_qwen38_flash_next.svg'
d = json.load(open(src))
caps = sorted(c['cap_w'] for c in d['caps'])
order = ["decode c=1 (128 in / 2048 out)", "c=1 (1024 in / 256 out)", "c=4 (1024 / 256)", "c=8 (1024 / 256)", "c=16 (1024 / 256)", "c=64 (1024 / 256)", "16K prefill c=4 (16384 / 1024)"]
short = {"decode c=1 (128 in / 2048 out)": "c=1 decode", "c=1 (1024 in / 256 out)": "c=1 1K/256", "c=4 (1024 / 256)": "c=4", "c=8 (1024 / 256)": "c=8", "c=16 (1024 / 256)": "c=16", "c=64 (1024 / 256)": "c=64", "16K prefill c=4 (16384 / 1024)": "16K prefill c=4"}
cols = {"c=1 decode": "#c0392b", "c=1 1K/256": "#e67e22", "c=4": "#d4a017", "c=8": "#7f8c8d", "c=16": "#2980b9", "c=64": "#16a085", "16K prefill c=4": "#8e44ad"}
def val(cap, tier, key):
    c = next(c for c in d['caps'] if c['cap_w'] == cap); t = next((t for t in c['tiers'] if t['tier'] == tier), None)
    return None if t is None else t.get(key)
tiers = [t for t in order if all(val(c, t, 'kwh_per_mtok_output') is not None for c in caps)]
W, H = 920, 440; L, R, T, B = 70, 300, 40, 50; pw = W - L - R; ph = H - T - B; ymax = 1.8
X = lambda c: L + pw * (c - 90) / (300 - 90); Y = lambda v: T + ph * (1 - v / ymax)
out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica,Arial,sans-serif" font-size="12">', f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
       f'<text x="{L}" y="22" font-size="15" font-weight="bold">Qwen3.8-Flash-Next on 4x MI100 (rc9): measured energy per output token vs power cap</text>']
for g in range(0, 19, 2):
    v = g / 10; out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{L+pw}" y2="{Y(v):.1f}" stroke="#e5e5e5"/>'); out.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" text-anchor="end" fill="#555">{v:.1f}</text>')
for c in caps:
    out.append(f'<line x1="{X(c):.1f}" y1="{T}" x2="{X(c):.1f}" y2="{T+ph}" stroke="#eeeeee"/>'); out.append(f'<text x="{X(c):.1f}" y="{T+ph+18}" text-anchor="middle" fill="#555">{c} W</text>')
out.append(f'<text x="{L+pw/2:.0f}" y="{H-8}" text-anchor="middle" fill="#333">power cap per MI100; package power sampled at {d["hz"]:g} Hz from sysfs during each tier (idle {d["idle"]["avg_w_total"]:.0f} W total)</text>')
out.append(f'<text transform="translate(16,{T+ph/2:.0f}) rotate(-90)" text-anchor="middle" fill="#333">kWh per million output tokens (4 cards)</text>')
for i, tier in enumerate(tiers):
    vals = [val(c, tier, 'kwh_per_mtok_output') for c in caps]; col = cols[short[tier]]
    out.append(f'<polyline points="{" ".join(f"{X(c):.1f},{Y(v):.1f}" for c, v in zip(caps, vals))}" fill="none" stroke="{col}" stroke-width="2.5"/>')
    for c, v in zip(caps, vals): out.append(f'<circle cx="{X(c):.1f}" cy="{Y(v):.1f}" r="4" fill="{col}"/>'); out.append(f'<text x="{X(c)+6:.1f}" y="{Y(v)-6:.1f}" fill="{col}" font-size="11">{v:.2f}</text>')
    ly = T + 18 + i * 20; out.append(f'<line x1="{L+pw+20}" y1="{ly}" x2="{L+pw+48}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>'); out.append(f'<text x="{L+pw+54}" y="{ly+4}" fill="#222">{short[tier]}</text>')
out.append('</svg>'); open(dst, 'w').write('\n'.join(out))
cols_t = [t for t in ["decode c=1 (128 in / 2048 out)", "c=8 (1024 / 256)", "c=16 (1024 / 256)", "c=64 (1024 / 256)", "16K prefill c=4 (16384 / 1024)"] if t in tiers]
print("| cap | " + " | ".join(("c=1 decode: tok/s / mean W / kWh per Mtok" if i == 0 else short[t]) for i, t in enumerate(cols_t)) + " |")
print("|---|" + "---|" * len(cols_t))
for c in caps:
    print(f"| {c} W | " + " | ".join(f"{val(c,t,'tok_s'):.{1 if i==0 else 0}f} / {val(c,t,'power')['avg_w_total']:.0f} W / {val(c,t,'kwh_per_mtok_output'):.2f}" for i, t in enumerate(cols_t)) + " |")
