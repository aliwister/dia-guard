#!/usr/bin/env python3
"""
Build a dialect-stratified KD training subset from dia_splits/train.jsonl
in the {prompt, response, label} format expected by the KD train scripts.

Usage:
  python subsample_dia.py --n 50000 \
      --out /data/vibe_exp/dia-guard/dataset/dia_splits/train_kd_50k.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DEFAULT_SRC = "/data/vibe_exp/dia-guard/dataset/dia_splits/train.jsonl"

# Same instruction template the Shield CE FT runs use, so the KD'd
# student learns the same I/O contract and the test set / FT pipelines
# downstream are apples-to-apples.
PROMPT_TEMPLATE = (
    "Classify the following text as safe or unsafe.\n\n"
    "Text: {text}\n\n"
    "Answer:"
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", required=True, help="Output JSONL path")
    p.add_argument("--n", type=int, default=50000, help="Total samples to write")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)

    # First pass: bucket by dialect
    print(f"Reading {args.src}")
    by_dialect = defaultdict(list)
    n_in = 0
    with open(args.src) as f:
        for line in f:
            r = json.loads(line)
            n_in += 1
            by_dialect[r.get("dialect", "unknown")].append(r)
    print(f"  {n_in:,} input records across {len(by_dialect)} dialects")

    # Stratified sample: roughly equal counts per dialect
    per_dialect = max(1, args.n // len(by_dialect))
    extra_pool = []
    sampled = []
    for d, recs in by_dialect.items():
        random.shuffle(recs)
        take = recs[:per_dialect]
        sampled.extend(take)
        extra_pool.extend(recs[per_dialect:])

    # Top up to exactly args.n with random extras
    if len(sampled) < args.n:
        random.shuffle(extra_pool)
        sampled.extend(extra_pool[: args.n - len(sampled)])
    sampled = sampled[: args.n]
    random.shuffle(sampled)

    # Convert and write
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    label_counts = {"safe": 0, "unsafe": 0}
    with open(args.out, "w") as g:
        for r in sampled:
            label_str = r["label_str"]
            label_counts[label_str] += 1
            out = {
                "prompt": PROMPT_TEMPLATE.format(text=r["text"]),
                "response": label_str,
                "label": r["label"],
                # Carry-through for downstream eval scripts that may want them
                "dialect": r.get("dialect"),
                "text_type": r.get("text_type"),
                "sample_id": r.get("sample_id"),
            }
            g.write(json.dumps(out) + "\n")

    print(f"Wrote {len(sampled):,} records to {args.out}")
    print(f"  label dist: {label_counts}")
    n_dialects_in_sample = len({r.get('dialect') for r in sampled})
    print(f"  dialects represented: {n_dialects_in_sample}")


if __name__ == "__main__":
    main()
