#!/usr/bin/env python3
"""Energy per token / per request for a running vLLM server on 4x MI100.

Samples the four cards' package power from sysfs (hwmon power1_average, ~0.3 ms
per read) at --hz while each benchmark tier runs, changes the power cap live
between passes (sudo -n /usr/local/sbin/mi100-power <W>), and reports measured
watts, Wh per request and kWh per million output tokens per tier and cap.

Usage: energy_bench.py [--caps 200,100,150,290] [--hz 2] [--container vllm-q38fn]
                       [--out Model_Reports/energy_qwen38_flash_next]
"""
import argparse, glob, json, os, re, subprocess, threading, time

SENSORS = sorted(glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*/power1_average'))
CAPS = sorted(glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*/power1_cap'))

TIERS = [  # name, input, output, prompts, concurrency
    ("decode c=1 (128 in / 2048 out)", 128, 2048, 3, 1),
    ("c=1 (1024 in / 256 out)", 1024, 256, 8, 1),
    ("c=4 (1024 / 256)", 1024, 256, 16, 4),
    ("c=8 (1024 / 256)", 1024, 256, 32, 8),
    ("c=16 (1024 / 256)", 1024, 256, 48, 16),
    ("c=64 (1024 / 256)", 1024, 256, 128, 64),
    ("16K prefill c=4 (16384 / 1024)", 16384, 1024, 8, 4),
]


class Sampler(threading.Thread):
    def __init__(self, hz):
        super().__init__(daemon=True); self.dt = 1.0 / hz; self.samples = []; self.stop = threading.Event()
    def run(self):
        nxt = time.perf_counter()
        while not self.stop.is_set():
            w = [int(open(f).read()) / 1e6 for f in SENSORS]
            self.samples.append((time.perf_counter(), w))
            nxt += self.dt
            time.sleep(max(0.0, nxt - time.perf_counter()))
    def summary(self):
        if not self.samples: return None
        n = len(self.samples); per = [sum(s[1][i] for s in self.samples) / n for i in range(len(SENSORS))]
        tot = [sum(s[1]) for s in self.samples]
        return {"samples": n, "avg_w_per_gpu": per, "avg_w_total": sum(per), "peak_w_total": max(tot), "min_w_total": min(tot)}


def set_cap(w):
    out = subprocess.run(["sudo", "-n", "/usr/local/sbin/mi100-power", str(w)], capture_output=True, text=True).stdout
    caps = [int(open(f).read()) / 1e6 for f in CAPS]
    if any(abs(c - w) > 0.5 for c in caps): raise SystemExit(f"cap {w} W not applied on all GPUs: {caps}\n{out}")
    time.sleep(3)


def bench(container, model, tokenizer, inp, out, prompts, conc):
    cmd = (f"vllm bench serve --backend openai --base-url http://localhost:8000 --model {model} --tokenizer {tokenizer} "
           f"--dataset-name random --random-input-len {inp} --random-output-len {out} --num-prompts {prompts} "
           f"--max-concurrency {conc} --ignore-eos 2>&1")
    t0 = time.perf_counter()
    txt = subprocess.run(["docker", "exec", container, "bash", "-c", cmd], capture_output=True, text=True).stdout
    wall = time.perf_counter() - t0
    def num(pat):
        m = re.search(pat + r"\s*:\s*([0-9.]+)", txt); return float(m.group(1)) if m else None
    r = {"wall_s": wall, "duration_s": num(r"Benchmark duration \(s\)"), "output_tokens": num(r"Total generated tokens"),
         "input_tokens": num(r"Total input tokens"), "tok_s": num(r"Output token throughput \(tok/s\)"),
         "ttft_mean_ms": num(r"Mean TTFT \(ms\)"), "tpot_mean_ms": num(r"Mean TPOT \(ms\)"), "requests": prompts}
    if r["duration_s"] is None: r["error"] = txt[-800:]
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--caps", default="200,100,150,290"); ap.add_argument("--hz", type=float, default=2.0)
    ap.add_argument("--container", default="vllm-q38fn"); ap.add_argument("--model", default="qwen38-flash-next")
    ap.add_argument("--tokenizer", default="/mnt/slow-storage/quant/Qwen3.8-Flash-Next-GPTQ-4bit"); ap.add_argument("--out", default="Model_Reports/energy_qwen38_flash_next")
    ap.add_argument("--restore", type=int, default=200)
    ap.add_argument("--only", default="", help="regex: run only matching tiers")
    ap.add_argument("--merge", action="store_true", help="merge into an existing --out JSON instead of replacing it")
    a = ap.parse_args()
    tiers_to_run = [t for t in TIERS if not a.only or re.search(a.only, t[0])]
    results = {"hz": a.hz, "sensors": SENSORS, "caps": []}
    if a.merge and os.path.exists(a.out + ".json"):
        results = json.load(open(a.out + ".json"))
    # idle draw
    if not results.get("idle"):
        s = Sampler(a.hz); s.start(); time.sleep(10); s.stop.set(); s.join(); results["idle"] = s.summary()
    print(f"idle: {results['idle']['avg_w_total']:.0f} W total ({[round(x) for x in results['idle']['avg_w_per_gpu']]})")
    try:
        for cap in [int(c) for c in a.caps.split(",")]:
            set_cap(cap); print(f"== cap {cap} W")
            bench(a.container, a.model, a.tokenizer, 1024, 128, 8, 4)  # warm-up at this cap
            tiers = []
            for name, inp, out, prompts, conc in tiers_to_run:
                s = Sampler(a.hz); s.start()
                r = bench(a.container, a.model, a.tokenizer, inp, out, prompts, conc)
                s.stop.set(); s.join(); p = s.summary(); r.update({"tier": name, "power": p})
                if r.get("duration_s") and p:
                    wh = p["avg_w_total"] * r["duration_s"] / 3600.0
                    r["wh_total"] = wh; r["wh_per_request"] = wh / prompts
                    r["kwh_per_mtok_output"] = wh / r["output_tokens"] * 1e6 / 1000.0
                    r["kwh_per_mtok_all"] = wh / (r["output_tokens"] + r["input_tokens"]) * 1e6 / 1000.0
                    print(f"  {name:34s} {r['tok_s']:7.1f} tok/s  {p['avg_w_total']:6.0f} W (peak {p['peak_w_total']:.0f})  "
                          f"{r['wh_per_request']:.3f} Wh/req  {r['kwh_per_mtok_output']:.3f} kWh/Mtok(out)  {r['kwh_per_mtok_all']:.3f} kWh/Mtok(all)")
                else:
                    print(f"  {name}: FAILED {r.get('error','')[:200]}")
                tiers.append(r)
            entry = next((c for c in results["caps"] if c["cap_w"] == cap), None)
            if entry is None:
                results["caps"].append({"cap_w": cap, "tiers": tiers})
            else:
                names = {t["tier"] for t in tiers}
                entry["tiers"] = [t for t in entry["tiers"] if t["tier"] not in names] + tiers
            json.dump(results, open(a.out + ".json", "w"), indent=1)
    finally:
        set_cap(a.restore)
    # markdown
    lines = ["# Energy per token, Qwen3.8-Flash-Next GPTQ-4bit on 4x MI100 (rc9)", "",
             f"Measured package power (sysfs hwmon power1_average, {a.hz:g} Hz, all four cards summed) during each `vllm bench serve` tier; "
             f"idle draw {results['idle']['avg_w_total']:.0f} W total. Energy = mean watts x tier duration. Output-token figures exclude prompt tokens; 'all' includes them.", ""]
    for name, *_ in TIERS:
        if not any(t["tier"] == name for c in results["caps"] for t in c["tiers"]):
            continue
        lines += [f"## {name}", "", "| cap | tok/s | mean W (4 cards) | peak W | Wh / request | kWh / Mtok (output) | kWh / Mtok (in+out) | TTFT | TPOT |", "|---|---|---|---|---|---|---|---|---|"]
        for c in results["caps"]:
            r = next((t for t in c["tiers"] if t["tier"] == name), None)
            if r and r.get("wh_total") is not None:
                lines.append(f"| {c['cap_w']} W | {r['tok_s']:.1f} | {r['power']['avg_w_total']:.0f} | {r['power']['peak_w_total']:.0f} | {r['wh_per_request']:.3f} | {r['kwh_per_mtok_output']:.3f} | {r['kwh_per_mtok_all']:.3f} | {r['ttft_mean_ms']:.0f} ms | {r['tpot_mean_ms']:.2f} ms |")
            else:
                lines.append(f"| {c['cap_w']} W | failed | | | | | | | |")
        lines.append("")
    open(a.out + ".md", "w").write("\n".join(lines)); print("wrote", a.out + ".md")


if __name__ == "__main__":
    main()
