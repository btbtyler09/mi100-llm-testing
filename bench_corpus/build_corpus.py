#!/usr/bin/env python3
"""Build the processed mixed-domain corpus from raw/ into corpus.json.

Domains: wiki (encyclopedic prose), code (Python from Apache/MIT repos),
book (public-domain Gutenberg prose), chat (hand-curated instructions).
Output: corpus.json = {domain: [unit_text, ...]}. Units are plain text
blocks the prompt assembler concatenates to reach a target token length.

Run once after changing raw/; commit corpus.json for reproducibility.
"""
import json
import os
import re

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.json")


def gutenberg_paragraphs() -> list[str]:
    paras = []
    for fn in sorted(os.listdir(RAW)):
        if not fn.startswith("gutenberg_"):
            continue
        txt = open(os.path.join(RAW, fn), encoding="utf-8-sig").read()
        # Strip Gutenberg header/footer boilerplate
        m = re.search(r"\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", txt)
        if m:
            txt = txt[m.end():]
        m = re.search(r"\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG", txt)
        if m:
            txt = txt[: m.start()]
        # Paragraphs = blank-line separated; join hard-wrapped lines
        for p in re.split(r"\n\s*\n", txt):
            p = " ".join(line.strip() for line in p.splitlines()).strip()
            if 200 < len(p) < 3000 and not p.isupper():
                paras.append(p)
    return paras


def wiki_units() -> list[str]:
    units = []
    for fn in ("wiki_full.jsonl", "wiki_extracts.jsonl"):
        path = os.path.join(RAW, fn)
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            d = json.loads(line)
            text = d["text"]
            if fn == "wiki_full.jsonl":
                # Split full articles into section-ish paragraphs
                for p in re.split(r"\n\s*\n|\n(?=== )", text):
                    p = p.strip().strip("= ")
                    if 200 < len(p) < 3000:
                        units.append(p)
            else:
                units.append(text.strip())
    return units


def code_units() -> list[str]:
    data = json.load(open(os.path.join(RAW, "code_units.json"), encoding="utf-8"))
    return [u["text"] for u in data]


def chat_units() -> list[str]:
    data = json.load(open(os.path.join(RAW, "chat_prompts.json"), encoding="utf-8"))
    return [u["text"] for u in data]


def main() -> None:
    corpus = {
        "wiki": wiki_units(),
        "code": code_units(),
        "book": gutenberg_paragraphs(),
        "chat": chat_units(),
    }
    for k, v in corpus.items():
        print(f"{k}: {len(v)} units, {sum(len(x) for x in v)//1000}k chars")
    json.dump(corpus, open(OUT, "w"))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
