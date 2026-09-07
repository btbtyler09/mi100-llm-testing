# Model reports

One report per model and per *use-case configuration* that is worth running: the
current recommended image, start script and settings for that model. Multiple
configs for one model are fine when they serve different use cases (for example
interactive spec-decode vs batch throughput, or a 290 W power-cap run); release
candidates, ablation arms and superseded runs are not kept here.

- `benchmark_<model>.md` — the current recommended configuration (200 W cap).
- `benchmark_<model>_<variant>.md` — a use-case variant, named for what differs.
- `json_data/` — the raw numbers behind each report in the root.
- `archive/<family>/` — superseded runs, older releases and candidate reports,
  kept for history with their `json_data/`.

Each report names the image tag and the fork commit it was produced from; the
matching start script is in `../scripts/`.

## Measured energy per token (rc9, 2 Hz sysfs power sampling on all four cards)

![energy per token](energy_curve_qwen38_flash_next.svg)

| cap | c=1 decode: tok/s / mean W / kWh per Mtok | c=16 | c=64 | 16K prefill c=4 |
|---|---|---|---|---|
| 100 W | 77.6 / 369 W / 1.32 | 246 / 358 W / 0.41 | 315 / 371 W / 0.33 | 88 / 363 W / 1.15 |
| 150 W | 100.1 / 523 W / 1.45 | 437 / 472 W / 0.30 | 504 / 525 W / 0.29 | 138 / 515 W / 1.04 |
| 200 W | 105.8 / 628 W / 1.65 | 453 / 520 W / 0.32 | 591 / 664 W / 0.31 | 151 / 650 W / 1.20 |
| 290 W | 105.8 / 614 W / 1.61 | 531 / 683 W / 0.36 | 613 / 775 W / 0.35 | 166 / 761 W / 1.27 |

Full per-tier table with Wh per request and prompt-inclusive figures: [`energy_qwen38_flash_next.md`](energy_qwen38_flash_next.md); raw samples in the JSON; script `scripts/energy_bench.py`.
Idle draw is 194 W for the four cards. Note the earlier "45-60 W during decode" reading was an idle-gap snapshot; under load a c=1 decode step draws ~155 W per card at the 200 W cap. Energy per output token is minimised at 100 W for c=1 (-20% vs 200 W, at -27% speed) and at 150 W for everything batched (c=16, c=64, 16K prefill); 290 W costs 8-15% more energy per token than 150 W.

## Qwen3.8-Flash-Next power curve (rc9, same image and settings, cap changed live)

![power curve](power_curve_qwen38_flash_next.svg)

| cap | c=1 decode (tok/s) | single-user TPOT | c=4 | c=8 | c=16 | c=64 | 16K c=4 | c=1 tok/s per kW (4 cards) |
|---|---|---|---|---|---|---|---|---|
| 100 W | 77.5 | 12.9 ms | 149 | 218 | 281 | 304 | 81 | 194 |
| 150 W | 100.9 | 10.0 ms | 209 | 331 | 485 | 512 | 127 | 168 |
| 200 W (recommended) | 107.5 | 9.4 ms | 232 | 361 | 542 | 567 | 138 | 134 |
| 290 W (rc8 halo) | 107.3 | 9.4 ms | 233 | 376 | 571 | 619 | 148 | 93 |

The cap clips clocks even though the sampled average draw during decode is only ~45-60 W per card,
so c=1 is not power-insensitive below 200 W; 150 W is the throughput-per-watt sweet spot for
c>=16, 100 W the c=1 efficiency point, 200 W the recommended balance.
