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

## Qwen3.8-Flash-Next power curve (rc9, same image and settings, cap changed live)

| cap | c=1 decode (tok/s) | single-user TPOT | c=16 | c=64 | 16K c=4 | c=1 tok/s per kW (4 cards) |
|---|---|---|---|---|---|---|
| 100 W | 77.5 | 12.9 ms | 281 | 304 | 81 | 194 |
| 150 W | 100.9 | 10.0 ms | 485 | 512 | 127 | 168 |
| 200 W (recommended) | 107.5 | 9.4 ms | 542 | 567 | 138 | 134 |
| 290 W (rc8 halo) | 107.3 | 9.4 ms | 571 | 619 | 148 | 93 |

The cap clips clocks even though the sampled average draw during decode is only ~45-60 W per card,
so c=1 is not power-insensitive below 200 W; 150 W is the throughput-per-watt sweet spot for
c>=16, 100 W the c=1 efficiency point, 200 W the recommended balance.
