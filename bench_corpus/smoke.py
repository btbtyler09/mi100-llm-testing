import sys, json
sys.path.insert(0, '/bench_corpus')
from mixed_dataset import MixedCorpus
c = MixedCorpus('/bench_corpus', '/models/Qwen3.8-27B-GPTQ-8bit')
p = '/tmp/mixed_smoke.jsonl'
c.write_jsonl(p, 8, 1024, 256)
rows = [json.loads(l) for l in open(p)]
print('domains:', [r['domain'] for r in rows])
print('token lens (target 1024):', [len(c.tok.encode(r['prompt'])) for r in rows])
c.write_jsonl(p, 8, 256, 64)
print('short domains:', [json.loads(l)['domain'] for l in open(p)])
