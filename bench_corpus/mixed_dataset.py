#!/usr/bin/env python3
"""Mixed-domain prompt assembly for BenchAndReport (--dataset mixed).

Builds length-controlled prompt JSONLs (vllm bench serve --dataset-name
custom format: {"prompt": ..., "output_tokens": N}) from corpus.json's
domain slices (wiki / code / book / chat), and measures per-domain
speculative-decode acceptance via the server's /metrics counters.

Length control mirrors the sonnet mode: a shared ~40-token preamble
(prefix-cache parity with --sonnet-prefix-len 50), then domain units
concatenated and truncated to the target token count. The chat slice
only naturally reaches short lengths, so tiers with input_len > 512 draw
from wiki/code/book only; chat still appears in the acceptance-by-domain
diagnostic at its natural length.
"""
import json
import os
import random
import urllib.request

PREAMBLES = {
    "wiki": "Read the following encyclopedic material carefully and then "
            "continue it in the same style, staying factual and neutral:\n\n",
    "code": "Review the following Python source code and then continue the "
            "module with consistent style and conventions:\n\n",
    "book": "Read the following passage of prose and then continue the "
            "narrative in the same voice:\n\n",
    "chat": "You are a helpful, knowledgeable assistant. Answer the "
            "following request clearly and completely:\n\n",
}
LONG_DOMAINS = ("wiki", "code", "book")


class MixedCorpus:
    def __init__(self, corpus_dir: str, tokenizer_path: str):
        from transformers import AutoTokenizer
        self.corpus = json.load(open(os.path.join(corpus_dir, "corpus.json")))
        self.tok = AutoTokenizer.from_pretrained(tokenizer_path)

    def _assemble(self, domain: str, target_tokens: int, rng: random.Random) -> str:
        preamble = PREAMBLES[domain]
        units = self.corpus[domain]
        if domain == "chat":
            return preamble + rng.choice(units)
        parts, n_tok = [preamble], len(self.tok.encode(preamble))
        start = rng.randrange(len(units))
        i = 0
        while n_tok < target_tokens + 64 and i < len(units):
            u = units[(start + i) % len(units)]
            parts.append(u)
            n_tok += len(self.tok.encode("\n\n" + u))
            i += 1
        text = "\n\n".join([parts[0].rstrip("\n")] + parts[1:])
        ids = self.tok.encode(text)[:target_tokens]
        return self.tok.decode(ids, skip_special_tokens=True)

    def write_jsonl(
        self,
        path: str,
        num_prompts: int,
        input_len: int,
        output_len: int,
        domains: tuple[str, ...] | None = None,
        seed: int = 1234,
    ) -> None:
        """Round-robin domains; one JSONL row per prompt."""
        rng = random.Random(seed)
        if domains is None:
            domains = LONG_DOMAINS if input_len > 512 else LONG_DOMAINS + ("chat",)
        with open(path, "w") as f:
            for i in range(num_prompts):
                d = domains[i % len(domains)]
                prompt = self._assemble(d, input_len, rng)
                f.write(json.dumps(
                    {"prompt": prompt, "output_tokens": output_len, "domain": d}
                ) + "\n")


def _spec_counters(base_url: str) -> tuple[float, float]:
    """(accepted, drafted) speculative token counters from /metrics; (0,0) if absent."""
    acc = drf = 0.0
    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("#"):
                    continue
                if "spec_decode_num_accepted_tokens" in line:
                    acc += float(line.rsplit(" ", 1)[1])
                elif "spec_decode_num_draft_tokens" in line:
                    drf += float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return acc, drf


def acceptance_by_domain(
    corpus: "MixedCorpus",
    base_url: str,
    model: str,
    workdir: str,
    input_len: int = 512,
    output_len: int = 128,
    prompts_per_domain: int = 12,
    tokenizer: str | None = None,
) -> dict[str, dict[str, float]]:
    """Sequential c=1 mini-runs per domain, acceptance from /metrics deltas.

    Diagnostic only — never a performance tier. Requires an active
    speculative-decode config on the server; returns {} otherwise.
    """
    import subprocess
    results: dict[str, dict[str, float]] = {}
    for domain in ("wiki", "code", "book", "chat"):
        path = os.path.join(workdir, f"accept_{domain}.jsonl")
        corpus.write_jsonl(
            path, prompts_per_domain,
            input_len if domain != "chat" else 128,
            output_len, domains=(domain,),
        )
        before = _spec_counters(base_url)
        cmd = ["vllm", "bench", "serve", "--base-url", base_url, "--model", model,
               "--dataset-name", "custom", "--dataset-path", path,
               "--num-prompts", str(prompts_per_domain), "--max-concurrency", "1",
               "--custom-output-len", str(output_len)]
        if tokenizer:
            cmd += ["--tokenizer", tokenizer]
        subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        after = _spec_counters(base_url)
        acc, drf = after[0] - before[0], after[1] - before[1]
        if drf > 0:
            results[domain] = {
                "accepted": acc, "drafted": drf,
                "draft_acceptance_rate": round(acc / drf, 4),
            }
    return results
