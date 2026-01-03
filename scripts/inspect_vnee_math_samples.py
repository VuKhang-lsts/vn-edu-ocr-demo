#!/usr/bin/env python3
"""Inspect a few Math-domain samples in the VNEE dataset and print their fields/types.

Run:
  python scripts/inspect_vnee_math_samples.py --count 5

Paste the printed output here so I can diagnose the image format.
"""
import argparse
from datasets import load_dataset
import json

VNEE = "Intelligent-Internet/Vietnamese-Entrance-Exam"


def canonical_domain(raw):
    s = str(raw or "").strip()
    if not s:
        return "Unknown"
    sl = s.lower()
    if "math" in sl or "toán" in sl:
        return "Math"
    if "phys" in sl or "vật" in sl:
        return "Physics"
    if "chem" in sl or "hóa" in sl:
        return "Chemistry"
    return s


def inspect(count: int = 5):
    ds = load_dataset(VNEE, split="train")
    print(f"Loaded {len(ds)} samples")

    math_idxs = [i for i in range(len(ds)) if canonical_domain(ds[i].get("domain")) == "Math"]
    print(f"Found {len(math_idxs)} Math indices. Showing first {min(count, len(math_idxs))} examples:\n")

    for j, idx in enumerate(math_idxs[:count]):
        sample = ds[idx]
        print(f"=== Sample idx={idx} (#{j+1}) ===")
        # print keys and types
        for k in sorted(list(sample.keys())):
            v = sample.get(k)
            t = type(v).__name__
            info = ""
            try:
                if isinstance(v, (bytes, bytearray)):
                    info = f"len={len(v)} bytes"
                elif isinstance(v, str):
                    info = f"len={len(v)} chars"
                elif hasattr(v, "shape"):
                    info = f"shape={getattr(v, 'shape')}"
                else:
                    s = repr(v)
                    if len(s) > 200:
                        info = s[:200] + "..."
                    else:
                        info = s
            except Exception as e:
                info = f"(repr error: {e})"
            print(f" - {k}: type={t} • {info}")
        # print a small JSON of text fields
        txt_fields = {k: sample.get(k) for k in ["vi_problem", "vi_choices", "domain"] if k in sample}
        print("Summary text fields:")
        print(json.dumps(txt_fields, ensure_ascii=False, indent=2))
        print("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()
    inspect(args.count)
