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
