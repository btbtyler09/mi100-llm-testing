#!/usr/bin/env python3
"""Wait for readiness while failing promptly if the trial container exits."""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--base', default='http://127.0.0.1:8000')
p.add_argument('--timeout', type=int, default=1800)
p.add_argument('--container', default='vllm-glm53-repo')
args = p.parse_args()
started = time.monotonic()
while time.monotonic() - started < args.timeout:
    state = json.loads(subprocess.check_output(['docker', 'inspect', args.container,
        '--format', '{{json .State}}'], text=True))
    if not state['Running']:
        raise RuntimeError(f'Server exited: {state}')
    try:
        with urllib.request.urlopen(args.base + '/health', timeout=3) as response:
            assert response.status == 200
        print(json.dumps({'ready': True, 'wait_seconds': time.monotonic() - started,
                          'container_started': state['StartedAt']}), flush=True)
        break
    except (urllib.error.URLError, TimeoutError):
        time.sleep(5)
else:
    raise TimeoutError('Server readiness deadline expired')
