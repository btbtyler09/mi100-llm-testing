#!/usr/bin/env python3
"""Fail a build on base drift, including collisions with new overlay modules."""
import hashlib
import json
from pathlib import Path

root = Path('/usr/local/lib/python3.12/dist-packages')
manifest = json.loads(Path('/opt/glm53-build/overlay-manifest.json').read_text())
for rel, hashes in manifest.items():
    path = root / rel
    actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    assert actual == hashes['base'], (rel, actual, hashes['base'])
print('BASE_FILES_VERIFIED', len(manifest))
