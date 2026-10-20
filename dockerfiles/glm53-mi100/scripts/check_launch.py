#!/usr/bin/env python3
"""Validate resolved Compose inputs before allocating the eight GPUs."""
import hashlib
import json
from pathlib import Path
import stat
import sys


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    lock = json.loads(Path(sys.argv[2]).read_text())
    service = config["services"]["glm53"]
    volumes = {v["target"]: v for v in service["volumes"]}
    model = Path(volumes["/model"]["source"])
    if not model.is_dir() or not volumes["/model"].get("read_only"):
        raise SystemExit("The checkpoint must be an existing read-only directory mount.")
    for name, expected in lock["checkpoint_metadata_sha256"].items():
        path = model / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Checkpoint metadata differs from the source lock: {name}")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    missing = [name for name in set(index["weight_map"].values())
               if not (model / name).is_file()]
    if missing:
        raise SystemExit(f"Missing checkpoint shards: {missing}")
    devices = [d["source"] for d in service["devices"] if d["target"] != "/dev/kfd"]
    if len(devices) != 8 or len(set(devices)) != 8:
        raise SystemExit("Exactly eight distinct GPU render devices are required (TP8).")
    for device in ["/dev/kfd", *devices]:
        path = Path(device)
        if not path.exists() or not stat.S_ISCHR(path.stat().st_mode):
            raise SystemExit(f"GPU character device unavailable: {device}")
    for device in devices:
        vendor = Path("/sys/class/drm") / Path(device).name / "device/vendor"
        if not vendor.is_file() or vendor.read_text().strip() != "0x1002":
            raise SystemExit(f"Expected an AMD GPU: {device}")
    cache = Path(volumes["/cache"]["source"])
    if not cache.is_absolute() or cache == model or model in cache.parents:
        raise SystemExit("Use a separate absolute persistent cache directory.")
    cache.mkdir(parents=True, exist_ok=True)
    print("Checkpoint metadata, shard presence and eight AMD GPU devices verified.")


if __name__ == "__main__":
    main()
