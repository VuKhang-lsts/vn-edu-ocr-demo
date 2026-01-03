#!/usr/bin/env python3
"""Check VNEE dataset image decoding and domain counts.

Run from the project root (activate venv first):

  python scripts/check_vnee_images.py

This will print domain counts and list sample indices where image decoding fails.
"""
from datasets import load_dataset
import io
import base64
from PIL import Image
import numpy as np
import cv2

VNEE_DATASET = "Intelligent-Internet/Vietnamese-Entrance-Exam"


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


def try_decode(x):
    if x is None:
        return None
    try:
        if isinstance(x, Image.Image):
            return x.convert("RGB")
        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(x))).convert("RGB")
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("data:") and "," in s:
                s = s.split(",", 1)[1]
            try:
                b = base64.b64decode(s)
                return Image.open(io.BytesIO(b)).convert("RGB")
            except Exception:
                pass
        if isinstance(x, dict):
            if x.get("bytes"):
                try:
                    return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
                except Exception:
                    pass
            if x.get("path"):
                try:
                    return Image.open(x["path"]).convert("RGB")
                except Exception:
                    pass
            if x.get("array") is not None:
                arr = np.asarray(x["array"])
                if arr.ndim == 2:
                    return Image.fromarray(arr).convert("RGB")
                if arr.ndim == 3:
                    try:
                        rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        return Image.fromarray(rgb).convert("RGB")
                    except Exception:
                        return Image.fromarray(arr).convert("RGB")
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr).convert("RGB")
            if arr.ndim == 3:
                try:
                    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(rgb).convert("RGB")
                except Exception:
                    return Image.fromarray(arr).convert("RGB")
    except Exception:
        return None
    return None


def main():
    ds = load_dataset(VNEE_DATASET, split="train")
    n = len(ds)
    print(f"Loaded {VNEE_DATASET} split=train → {n} samples")

    domain_counts = {}
    failures = []
    for i in range(n):
        sample = ds[i]
        dom = canonical_domain(sample.get("domain"))
        domain_counts[dom] = domain_counts.get(dom, 0) + 1

        img_field = sample.get("image") if "image" in sample else None
        img = try_decode(img_field)
        if img is None:
            # try other common keys
            tried = False
            for k in ["img", "image_bytes", "image_url"]:
                if k in sample:
                    tried = True
                    if try_decode(sample.get(k)) is not None:
                        img = True
                        break
            if img is None:
                failures.append((i, dom, list(sample.keys())))

    print("\nDomain counts:")
    for k, v in sorted(domain_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {k}: {v}")

    print(f"\nTotal failures: {len(failures)}")
    if failures:
        print("First failing samples (index, domain, keys):")
        for t in failures[:50]:
            print(f"  {t}")


if __name__ == "__main__":
    main()
