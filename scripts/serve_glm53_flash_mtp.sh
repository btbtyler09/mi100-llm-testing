#!/usr/bin/env bash
# GLM-5.3-Flash W4A16, native MTP K=1, eight MI100 GPUs.
# Supply GLM53_MODEL_DIR and GLM53_CACHE_DIR or a GLM53_ENV_FILE.
# Actions: up (default), build, config, status, logs, stop.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_dir/dockerfiles/glm53-mi100"
action="${1:-up}"
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [up|build|config|status|logs|stop]" >&2
  exit 2
fi
if [[ "$action" == build ]]; then
  exec docker build -t "${GLM53_IMAGE:-glm53-flash-mi100:local}" "$build_dir"
fi
compose=(docker compose --env-file "${GLM53_ENV_FILE:-$build_dir/serving.env.example}" -f "$build_dir/compose.yaml")
case "$action" in
  config) exec "${compose[@]}" config ;;
  status) exec "${compose[@]}" ps ;;
  logs) exec "${compose[@]}" logs --tail 100 glm53 ;;
  stop) exec "${compose[@]}" stop glm53 ;;
  up) ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
resolved="$(mktemp)"
trap 'rm -f "$resolved"' EXIT
"${compose[@]}" config --format json > "$resolved"
python3 "$build_dir/scripts/check_launch.py" "$resolved" "$build_dir/source-lock.json"
image="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"]["glm53"]["image"])' "$resolved")"
docker image inspect "$image" >/dev/null
"${compose[@]}" up -d --no-deps --pull never glm53
base="$(python3 - "$resolved" <<'PY'
import json, sys
argv = json.load(open(sys.argv[1]))["services"]["glm53"]["command"]
host = argv[argv.index("--host") + 1]
if host in ("0.0.0.0", "::"):
    host = "127.0.0.1"
elif ":" in host:
    host = "[" + host + "]"
print("http://" + host + ":" + argv[argv.index("--port") + 1])
PY
)"
container="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["services"]["glm53"]["container_name"])' "$resolved")"
python3 "$build_dir/scripts/wait_ready.py" --base "$base" --container "$container"
python3 - "$base" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1] + "/v1/models", timeout=10) as response:
    print(json.dumps(json.load(response), indent=2))
PY
