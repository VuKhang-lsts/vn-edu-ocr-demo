# -*- coding: utf-8 -*-
"""
Streamlit demo: Vietnamese educational document OCR + NLI + Full-dataset experiment dashboard
Dataset (public): Intelligent-Internet/Vietnamese-Entrance-Exam
- OCR tab: upload image/PDF or pick a sample from dataset
- Experiments tab: run OCR over the whole dataset (or a subset), compare configs, export CSV, show charts & error analysis
- NLI tab: premise (OCR text) vs hypothesis with evidence highlight

Note:
- This app expects your existing project modules:
  src.ocr_engine.OCREngine
  src.pdf_utils.pdf_bytes_to_images
  src.text_utils.split_sentences_vi
  src.viz.draw_quads
  src.export.lines_to_text, lines_to_json, lines_to_docx_bytes
  (optional) src.table, src.nontext, src.postprocess_vi, src.nli_engine, src.evidence
"""

import io
import re
import time
import random
import unicodedata
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import base64
try:
    import cv2
except Exception:
    cv2 = None

# Project modules
from src.text_utils import split_sentences_vi
from src.pdf_utils import pdf_bytes_to_images
from src.ocr_engine import OCREngine
from src.viz import draw_quads
from src.export import lines_to_text, lines_to_json, lines_to_docx_bytes

# HF datasets
from datasets import load_dataset

# =========================================================
# Page config + CSS
# =========================================================
st.set_page_config(page_title="OCR Tài liệu giáo dục tiếng Việt", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 1.6rem; max-width: 1500px; }
      h1, h2, h3 { letter-spacing: -0.2px; }
      .small-note { font-size: 0.92rem; opacity: 0.85; }
      textarea { font-size: 16px !important; line-height: 1.38 !important; }
      .stDownloadButton button { width: 100%; }
      section[data-testid="stSidebar"] .block-container { padding-top: 0.8rem; }
      .pill { display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; border:1px solid rgba(49,51,63,0.2); margin-right:6px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# Metrics helpers
# =========================================================
def _edit_distance(a: str, b: str) -> int:
    a = a or ""
    b = b or ""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]

def cer(ref: str, hyp: str) -> float:
    ref = ref or ""
    hyp = hyp or ""
    return _edit_distance(ref, hyp) / max(1, len(ref))

def wer(ref: str, hyp: str) -> float:
    r = (ref or "").split()
    h = (hyp or "").split()
    n, m = len(r), len(h)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m] / max(1, n)

def normalize_text(s: str) -> str:
    """Light normalization for EM metric."""
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    # normalize quotes and dashes
    s = s.replace("–", "-").replace("−", "-").replace("“", '"').replace("”", '"').replace("’", "'")
    return s

def strip_accents(s: str) -> str:
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)

MATH_SYMBOLS = set(list("√π∞∑∫≈≠≤≥±×÷∂∆∇∈∉∅∝∠°µΩαβγδθλσφω"))
def symbol_recall(ref: str, hyp: str) -> float:
    """Heuristic: fraction of math/science symbols in ref that appear in hyp."""
    ref_syms = [c for c in (ref or "") if c in MATH_SYMBOLS]
    if not ref_syms:
        return 1.0
    hyp_set = set([c for c in (hyp or "") if c in MATH_SYMBOLS])
    hit = sum(1 for c in ref_syms if c in hyp_set)
    return hit / max(1, len(ref_syms))

# =========================================================
# OCR line geometry helpers (safe)
# =========================================================
def _ensure_list(lines) -> List:
    if lines is None:
        return []
    if isinstance(lines, list):
        return [ln for ln in lines if ln is not None]
    try:
        return [ln for ln in list(lines) if ln is not None]
    except Exception:
        return []

def _get_box(ln) -> Optional[Tuple[float, float, float, float]]:
    if ln is None:
        return None
    b = getattr(ln, "bbox", None)
    if b and len(b) == 4:
        x0, y0, x1, y1 = b
        return float(x0), float(y0), float(x1), float(y1)

    q = getattr(ln, "quad", None)
    if q and len(q) >= 4:
        try:
            xs = [p[0] for p in q]
            ys = [p[1] for p in q]
            return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        except Exception:
            return None
    return None

def _box_center(b: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0

def _box_area(b: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def _intersect_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0 = max(ax0, bx0)
    y0 = max(ay0, by0)
    x1 = min(ax1, bx1)
    y1 = min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)

def _draw_rects(pil_img: Image.Image, boxes, width: int = 3, color="red") -> Image.Image:
    if pil_img is None:
        return pil_img
    im = pil_img.copy()
    draw = ImageDraw.Draw(im)
    for b in (boxes or []):
        if not b or len(b) != 4:
            continue
        x0, y0, x1, y1 = map(int, b)
        try:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
        except TypeError:
            for k in range(width):
                draw.rectangle([x0 - k, y0 - k, x1 + k, y1 + k], outline=color)
    return im

def _estimate_line_height(lines) -> float:
    hs = []
    for ln in _ensure_list(lines):
        b = _get_box(ln)
        if not b:
            continue
        hs.append(max(1.0, b[3] - b[1]))
    if not hs:
        return 14.0
    hs.sort()
    return float(hs[len(hs) // 2])

# =========================================================
# Reading order (1c / 2c)
# =========================================================
def _auto_detect_two_col(lines, page_w: int) -> bool:
    xs = []
    for ln in _ensure_list(lines):
        b = _get_box(ln)
        if not b:
            continue
        cx, _ = _box_center(b)
        xs.append(cx)
    if len(xs) < 20:
        return False
    left = sum(1 for x in xs if x < 0.45 * page_w)
    right = sum(1 for x in xs if x > 0.55 * page_w)
    return (left / len(xs) >= 0.30) and (right / len(xs) >= 0.30)

def _reorder_lines_internal(
    lines,
    page_w: int,
    mode: str = "auto",
    two_col_order: str = "column",
    row_bucket_factor: float = 0.85,
    col_split: Optional[float] = None,
):
    lines = _ensure_list(lines)
    if not lines:
        return []

    lh = _estimate_line_height(lines)
    bucket_h = max(8.0, lh * row_bucket_factor)

    def key_single(ln):
        b = _get_box(ln)
        if not b:
            return (10**9, 10**9)
        x0, y0, x1, y1 = b
        y_bucket = int(y0 / bucket_h)
        return (y_bucket, x0)

    if mode == "auto":
        mode = "two" if _auto_detect_two_col(lines, page_w) else "single"

    if mode in ("single", "one", "1"):
        return sorted(lines, key=key_single)

    split = col_split if col_split is not None else 0.5 * page_w
    left_col, right_col = [], []
    for ln in lines:
        b = _get_box(ln)
        if not b:
            left_col.append(ln)
            continue
        cx, _ = _box_center(b)
        (left_col if cx < split else right_col).append(ln)

    left_col = sorted(left_col, key=key_single)
    right_col = sorted(right_col, key=key_single)

    if two_col_order == "column":
        return left_col + right_col

    def bucket_of(ln):
        b = _get_box(ln)
        if not b:
            return 10**9
        return int(b[1] / bucket_h)

    buckets = sorted(set([bucket_of(ln) for ln in left_col + right_col]))
    out = []
    for bk in buckets:
        ls = [ln for ln in left_col if bucket_of(ln) == bk]
        rs = [ln for ln in right_col if bucket_of(ln) == bk]
        out.extend(ls + rs)
    return out

# =========================================================
# Non-text filter fallback (conservative)
# =========================================================
def _nontext_filter_fallback(lines, page_w: int, page_h: int, keep_boxes=None, strength: float = 0.25):
    lines = _ensure_list(lines)
    keep_boxes = keep_boxes or []
    out = []
    page_area = float(page_w * page_h)
    big_area_thr = (0.035 + 0.03 * strength) * page_area
    min_chars_drop = 1 if strength < 0.5 else 2

    def _in_keep(b, boxes, thr=0.20):
        if not b:
            return False
        la = _box_area(b)
        if la <= 1e-6:
            return False
        for kb in boxes:
            if not kb or len(kb) != 4:
                continue
            inter = _intersect_area(b, kb)
            if inter / la >= thr:
                return True
        return False

    for ln in lines:
        txt = (getattr(ln, "text", "") or "").strip()
        b = _get_box(ln)

        if b and _in_keep(b, keep_boxes):
            out.append(ln)
            continue
        if not txt:
            continue
        if not b:
            out.append(ln)
            continue

        area = _box_area(b)
        alnum = sum(1 for c in txt if c.isalnum())
        ratio = alnum / max(1, len(txt))

        if len(txt) <= min_chars_drop and area >= big_area_thr:
            continue
        if ratio < (0.10 + 0.10 * (1.0 - strength)) and len(txt) < 8 and area >= big_area_thr:
            continue
        out.append(ln)

    return out

# =========================================================
# Vietnamese-Entrance-Exam dataset helpers
# =========================================================
VNEE_DATASET = "Intelligent-Internet/Vietnamese-Entrance-Exam"

def _pick_first_key(d: dict, keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None

def _decode_hf_image(x) -> Optional[Image.Image]:
    # Robust decoder for HF datasets.Image variants and common encodings.
    if x is None:
        return None

    try:
        # Already a PIL Image
        if isinstance(x, Image.Image):
            return x.convert("RGB")

        # Raw bytes
        if isinstance(x, (bytes, bytearray)):
            return Image.open(io.BytesIO(bytes(x))).convert("RGB")

        # Base64-encoded string (possibly data URL)
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("data:") and "," in s:
                s = s.split(",", 1)[1]
            b = base64.b64decode(s)
            return Image.open(io.BytesIO(b)).convert("RGB")

        # dict-style from datasets (various keys)
        if isinstance(x, dict):
            if x.get("bytes"):
                return Image.open(io.BytesIO(x["bytes"])).convert("RGB")
            if x.get("path"):
                return Image.open(x["path"]).convert("RGB")
            if x.get("array") is not None:
                arr = np.asarray(x["array"])
                if arr.ndim == 2:
                    return Image.fromarray(arr).convert("RGB")
                if arr.ndim == 3:
                    return Image.fromarray(arr).convert("RGB")

        # numpy array directly  ✅ phải nằm TRONG try:
        if isinstance(x, np.ndarray):
            arr = x
            if arr.dtype != np.uint8:
                arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            if arr.ndim == 2:
                return Image.fromarray(arr).convert("RGB")
            if arr.ndim == 3:
                return Image.fromarray(arr).convert("RGB")

    except Exception:
        return None

    return None



def canonical_domain(raw: Any) -> str:
    s = str(raw or "").strip()
    if not s:
        return "Unknown"
    # Normalize and also strip accents to match common variants (e.g. 'toan', 'toán', 'mathematics')
    sl = s.lower()
    sl_na = strip_accents(sl)
    # match common tokens in un-accented form
    if any(tok in sl_na for tok in ("math", "toan", "mathematics", "maths")):
        return "Math"
    if any(tok in sl_na for tok in ("phys", "vat", "vatly", "vat ly", "physics")):
        return "Physics"
    if any(tok in sl_na for tok in ("chem", "hoa", "hoá", "chemistry")):
        return "Chemistry"
    return s  # keep original if no canonical match

def build_gt_from_sample(sample: Dict[str, Any]) -> str:
    """
    Build OCR ground-truth text for evaluation:
    vi_problem + newline + vi_choices (formatted)
    """
    problem = str(sample.get("vi_problem", "") or "")
    choices = sample.get("vi_choices", None)

    choice_txt = ""
    if choices is None:
        choice_txt = ""
    elif isinstance(choices, str):
        choice_txt = choices
    elif isinstance(choices, (list, tuple)):
        # list of strings: add A/B/C/D...
        letters = "ABCD"
        parts = []
        for i, c in enumerate(list(choices)[:10]):
            label = letters[i] if i < len(letters) else f"Opt{i+1}"
            parts.append(f"{label}. {str(c)}")
        choice_txt = "\n".join(parts)
    elif isinstance(choices, dict):
        # dict like {"A": "...", "B": "..."}
        parts = []
        for k in sorted(choices.keys()):
            parts.append(f"{k}. {str(choices[k])}")
        choice_txt = "\n".join(parts)
    else:
        choice_txt = str(choices)

    gt = (problem.strip() + ("\n" + choice_txt.strip() if choice_txt.strip() else "")).strip()
    return gt

@st.cache_data(show_spinner=False)
def load_vnee_split(split: str = "train"):
    # 432 samples, safe to load fully.
    return load_dataset(VNEE_DATASET, split=split)


def _get_sample_domain(sample: Any) -> str:
    """Extract a domain-like value from a dataset sample robustly.

    Checks common keys and handles list/dict values, then canonicalizes.
    """
    if sample is None:
        return "Unknown"
    # sample may be a dict-like
    if isinstance(sample, dict):
        # candidate keys to look for
        keys = ["domain", "domains", "subject", "subjects", "tags", "category"]
        for k in keys:
            if k in sample and sample[k] is not None:
                v = sample[k]
                # if list/tuple: examine elements and pick the first reasonable candidate
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    for elem in v:
                        if elem is None:
                            continue
                        if isinstance(elem, dict):
                            for nk in ["name", "label", "type", "value", "title", "text"]:
                                if nk in elem and elem[nk] is not None:
                                    return canonical_domain(elem[nk])
                            # fallback to string representation of the element
                            return canonical_domain(str(elem))
                        # primitive (string/number/etc.) -> use directly
                        return canonical_domain(elem)

                # if dict: try more nested keys before falling back
                if isinstance(v, dict):
                    for nk in ["name", "label", "type", "value", "title", "text"]:
                        if nk in v and v[nk] is not None:
                            return canonical_domain(v[nk])
                    return canonical_domain(str(v))

                # otherwise, direct value
                return canonical_domain(v)
    # fallback: treat as raw value
    return canonical_domain(sample)

@st.cache_data(show_spinner=False)
def get_vnee_domain_index(split: str = "train") -> Dict[str, List[int]]:
    ds = load_vnee_split(split)
    dom_map: Dict[str, List[int]] = {}
    for i in range(len(ds)):
        d = _get_sample_domain(ds[i])
        dom_map.setdefault(d, []).append(i)
    return dom_map

# =========================================================
# Image preprocessing
# =========================================================
def preprocess_image(pil_img: Image.Image, mode: str) -> Image.Image:
    if pil_img is None:
        return pil_img
    mode = (mode or "none").lower()
    im = pil_img.convert("RGB")

    if mode == "none":
        return im
    if mode == "gray":
        return im.convert("L").convert("RGB")
    if mode == "contrast_sharp":
        # conservative: increase contrast + sharpen slightly
        im2 = ImageEnhance.Contrast(im).enhance(1.35)
        im2 = ImageEnhance.Sharpness(im2).enhance(1.25)
        return im2
    if mode == "binarize":
        g = im.convert("L")
        # Otsu-like simple threshold
        arr = np.array(g)
        thr = int(np.mean(arr))
        bw = (arr > thr).astype(np.uint8) * 255
        return Image.fromarray(bw).convert("RGB")
    return im

# =========================================================
# Pipeline config + engine
# =========================================================
@st.cache_resource
def load_engine(device: str):
    return OCREngine(device=device)

@st.cache_resource
def load_nli(model_name: str, device: str):
    from src.nli_engine import NLIEdge
    return NLIEdge(model_name=model_name, device=device)

@st.cache_data(show_spinner=False)
def _cached_pdf_to_images(data: bytes, dpi: int, max_pages: int):
    return pdf_bytes_to_images(data, dpi=dpi, max_pages=max_pages)

@dataclass
class PipelineCfg:
    name: str
    preprocess: str = "none"
    enable_nontext: bool = True
    nontext_strength: float = 0.25
    enable_table: bool = True
    enable_post_vi: bool = True
    reading_mode: str = "auto"
    two_col_order: str = "column"
    col_split_ratio: float = 0.50
    row_bucket_factor: float = 0.85

PIPELINE_REGISTRY: Dict[str, PipelineCfg] = {
    "C0_Baseline": PipelineCfg(
        name="C0_Baseline",
        preprocess="none",
        enable_nontext=False,
        enable_table=False,
        enable_post_vi=False,
        reading_mode="auto",
        row_bucket_factor=0.85
    ),
    "C1_PostVI+NonText": PipelineCfg(
        name="C1_PostVI+NonText",
        preprocess="none",
        enable_nontext=True,
        nontext_strength=0.25,
        enable_table=True,
        enable_post_vi=True,
        reading_mode="auto",
        row_bucket_factor=0.85
    ),
    "C2_Contrast+PostVI+NonText": PipelineCfg(
        name="C2_Contrast+PostVI+NonText",
        preprocess="contrast_sharp",
        enable_nontext=True,
        nontext_strength=0.25,
        enable_table=True,
        enable_post_vi=True,
        reading_mode="auto",
        row_bucket_factor=0.85
    ),
}

def run_ocr_pipeline(img_pil: Image.Image, engine: OCREngine, cfg: PipelineCfg):
    """Run OCR pipeline on a PIL image with a given PipelineCfg."""
    if img_pil is None:
        return [], "", [], [], {}, 0.0, img_pil

    t0 = time.perf_counter()

    # 0) preprocess
    img_pp = preprocess_image(img_pil, cfg.preprocess)
    img_np = np.array(img_pp)
    h, w = img_np.shape[:2]

    # 1) OCR
    try:
        lines_raw = engine.run(img_np)
    except Exception as e:
        st.error(f"OCR engine lỗi: {e}")
        lines_raw = []
    lines_raw = _ensure_list(lines_raw)

    # 2) table region detection (optional)
    table_boxes: List[Tuple[int, int, int, int]] = []
    tables: List[Dict[str, Any]] = []
    if cfg.enable_table:
        try:
            from src.table import detect_table_regions
            tb = detect_table_regions(img_np, lines=lines_raw) or []
            table_boxes = [tuple(map(int, b)) for b in tb if b and len(b) == 4]
        except Exception:
            table_boxes = []

    table_boxes = [tuple(map(int, b)) for b in (table_boxes or []) if b and len(b) == 4]
    table_boxes.sort(key=lambda b: (b[1], b[0], (b[3] - b[1]), (b[2] - b[0])))

    # 3) filter non-text (optional)
    lines = lines_raw
    if cfg.enable_nontext:
        try:
            from src.nontext import filter_nontext_lines
            filtered = filter_nontext_lines(lines_raw, page_w=w, page_h=h, keep_boxes=table_boxes)
            lines = _ensure_list(filtered) if filtered is not None else lines_raw
        except Exception:
            lines = _nontext_filter_fallback(
                lines_raw, page_w=w, page_h=h,
                keep_boxes=table_boxes, strength=float(cfg.nontext_strength)
            )

    # 4) reorder
    col_split = float(cfg.col_split_ratio) * float(w)
    lines = _reorder_lines_internal(
        lines,
        page_w=w,
        mode=str(cfg.reading_mode),
        two_col_order=str(cfg.two_col_order),
        row_bucket_factor=float(cfg.row_bucket_factor),
        col_split=col_split,
    )

    # 5) extract tables (optional)
    if cfg.enable_table and table_boxes:
        try:
            from src.table import extract_tables_from_lines
            tt = extract_tables_from_lines(lines, table_boxes)
            tables = tt if tt is not None else []
        except Exception:
            tables = []

    # 6) postprocess_vi (optional)
    pp_stats: Dict[str, Any] = {}
    lines = _ensure_list(lines)
    if cfg.enable_post_vi and lines:
        try:
            from src.postprocess_vi import postprocess_lines
            lines, pp_stats = postprocess_lines(lines)
            lines = _ensure_list(lines)
        except Exception as e:
            pp_stats = {"warning": f"postprocess_vi lỗi: {e}"}

    # 7) output text
    text_out = lines_to_text(lines) if lines else ""

    dt = time.perf_counter() - t0
    return lines, text_out, table_boxes, tables, (pp_stats or {}), dt, img_pp

# =========================================================
# NLI helpers
# =========================================================
NLI_MODELS = [
    "lizNguyen235/phobert-nli-vi",
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    "joeddav/xlm-roberta-large-xnli",
]

def _vi_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    vi_chars = set(
        "ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩị"
        "óòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
        "ĂÂĐÊÔƠƯÁÀẢÃẠẤẦẨẪẬẮẰẲẴẶÉÈẺẼẸẾỀỂỄỆÍÌỈĨỊ"
        "ÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤỨỪỬỮỰÝỲỶỸỴ"
    )
    hit = sum(1 for c in text if c in vi_chars)
    return hit / max(1, len(text))

def auto_pick_model_heuristic(premise_text: str) -> str:
    return "lizNguyen235/phobert-nli-vi" if _vi_char_ratio(premise_text) >= 0.01 else "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

def _cut_premise(text: str, max_sent: int, max_chars: int) -> str:
    sents = split_sentences_vi(text or "")
    out = " ".join(sents[:max_sent]) if sents else (text or "")
    return out[:max_chars]

# =========================================================
# Session state
# =========================================================
def _init_state():
    defaults = {
        "ocr_lines": [],
        "ocr_text": "",
        "ocr_text_area": "",
        "ocr_image": None,
        "ocr_image_pp": None,
        "ocr_meta": {"source": None, "id": None, "page_idx": None, "dpi": None},
        "pp_stats": {},
        "tables": [],
        "table_boxes": [],
        "last_runtime": None,
        "last_pipeline": "C2_Contrast+PostVI+NonText",

        "nli_res": None,
        "nli_chosen_model": None,
        "nli_evidence": ([], "", 0.0, "neutral"),
        "nli_evidence_debug": {},

        "exp_rows": None,  # full results (pandas df)
        "exp_summary": None,
        "exp_by_domain": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# =========================================================
# Header
# =========================================================
st.title("📄 OCR Tài liệu giáo dục tiếng Việt")
st.caption("Demo: **Upload/Chọn mẫu → OCR → BBox/Text/Export** • + **Experiments** (full dataset) • + **NLI evidence**")

# =========================================================
# Sidebar controls
# =========================================================
with st.sidebar:
    st.header("⚙️ Thiết lập nhanh")
    demo_mode = st.checkbox("Chế độ Demo (tối giản)", value=True)

    device = st.selectbox("Device", ["cpu"], index=0)  # keep cpu for portability
    dpi = st.slider("DPI (PDF)", 120, 300, 200, 10)
    max_pages = st.slider("Tối đa trang PDF", 1, 25, 8)

    st.markdown("---")
    st.subheader("Pipeline")
    pipeline_name = st.selectbox("Chọn cấu hình OCR", list(PIPELINE_REGISTRY.keys()),
                                 index=list(PIPELINE_REGISTRY.keys()).index(st.session_state.get("last_pipeline", "C2_Contrast+PostVI+NonText")))
    st.session_state["last_pipeline"] = pipeline_name

    if not demo_mode:
        cfg = PIPELINE_REGISTRY[pipeline_name]
        st.caption("Nâng cao (debug)")
        st.write(cfg)

    st.markdown('<div class="small-note">Gợi ý: dùng C0/C1/C2 để so sánh trong Experiments.</div>', unsafe_allow_html=True)

engine = load_engine(device=device)

# =========================================================
# Tabs
# =========================================================
tab_ocr, tab_exp, tab_nli, tab_help = st.tabs(["🧾 OCR", "📊 Experiments", "🧠 NLI", "ℹ️ Hướng dẫn"])

# =========================================================
# TAB: OCR (single sample demo)
# =========================================================
with tab_ocr:
    st.subheader("1) Chọn nguồn dữ liệu")
    src = st.radio("Nguồn", ["Upload ảnh/PDF", "Vietnamese-Entrance-Exam (dataset)"], horizontal=True)

    img: Optional[Image.Image] = None
    pages: List[Image.Image] = []
    source_id = None
    page_idx = 0
    gt_text = ""
    domain = "Unknown"

    if src == "Upload ảnh/PDF":
        up = st.file_uploader("Chọn ảnh hoặc PDF", type=["png", "jpg", "jpeg", "pdf"], label_visibility="visible")
        if up is None:
            st.info("⬆️ Tải một ảnh/PDF để bắt đầu.")
            st.stop()

        data = up.getvalue()
        is_pdf = (up.type == "application/pdf") or up.name.lower().endswith(".pdf")

        if is_pdf:
            pages = _cached_pdf_to_images(data, dpi=int(dpi), max_pages=int(max_pages))
            if not pages:
                st.error("Không đọc được PDF.")
                st.stop()
            if len(pages) > 1:
                page_idx = st.slider("Chọn trang", 1, len(pages), 1) - 1
            img = pages[page_idx]
        else:
            img = Image.open(io.BytesIO(data)).convert("RGB")
            pages = [img]

        source_id = up.name
        domain = "Upload"
        st.caption(f"📌 Nguồn: **{up.name}** • Trang: {page_idx+1}/{len(pages)} • DPI: {dpi}")

    else:
        split = st.selectbox("Split", ["train"], index=0, help="Dataset hiện public có split train (432 mẫu).")
        dom_map = get_vnee_domain_index(split=split)
        # show counts
        dom_items = sorted([(k, len(v)) for k, v in dom_map.items()], key=lambda x: (-x[1], x[0]))
        # Only expose Physics and Chemistry domains (remove Math and others)
        dom_items = [item for item in dom_items if item[0] in ("Physics", "Chemistry")]
        dom_labels = ["All"] + [f"{k} ({n})" for k, n in dom_items]
        dom_choice = st.selectbox("Lọc domain", dom_labels, index=0, help="Đã fix lỗi 'không tìm thấy domain math' bằng cách chuẩn hoá domain.")
        if dom_choice == "All":
            indices = list(range(len(load_vnee_split(split))))
            domain = "All"
        else:
            dom_name = dom_choice.split(" (")[0]
            indices = dom_map.get(dom_name, [])
            domain = dom_name

        if not indices:
            st.error("Không có mẫu nào cho domain đã chọn.")
            st.stop()

        idx = st.slider("Chọn mẫu", 1, len(indices), 1) - 1
        real_idx = indices[idx]
        ds = load_vnee_split(split)
        sample = ds[real_idx]

        img = _decode_hf_image(sample.get("image"))
        if img is None:
            st.error("Không decode được ảnh từ dataset.")
            st.stop()

        gt_text = build_gt_from_sample(sample)
        source_id = f"{VNEE_DATASET}:{split}:{real_idx}"
        pages = [img]

        st.caption(f"📌 Dataset: **{VNEE_DATASET}** • split={split} • idx={real_idx} • domain={canonical_domain(sample.get('domain'))}")

    # display input image
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    st.caption(f"Kích thước ảnh: {w}×{h}")

    c1, c2, c3, c4 = st.columns([1.1, 1.2, 1.2, 2.5])
    with c1:
        run = st.button("🚀 Chạy OCR", type="primary", use_container_width=True)
    with c2:
        show_overlay = st.checkbox("Hiện bbox chữ", value=True)
    with c3:
        show_table_overlay = st.checkbox("Hiện vùng bảng", value=False)
    with c4:
        st.markdown(f"<span class='pill'>Pipeline: <b>{pipeline_name}</b></span>", unsafe_allow_html=True)

    if run:
        cfg = PIPELINE_REGISTRY[pipeline_name]
        prog = st.progress(0, text="Đang chạy OCR…")
        prog.progress(10, text="Preprocess + OCR…")
        lines, text_out, table_boxes, tables, pp_stats, dt, img_pp = run_ocr_pipeline(img, engine, cfg)
        prog.progress(90, text="Đang tổng hợp…")

        st.session_state["ocr_lines"] = _ensure_list(lines)
        st.session_state["ocr_text"] = text_out or ""
        st.session_state["ocr_text_area"] = text_out or ""
        st.session_state["ocr_image"] = img
        st.session_state["ocr_image_pp"] = img_pp
        st.session_state["ocr_meta"] = {"source": src, "id": source_id, "page_idx": page_idx, "dpi": dpi, "domain": domain, "pipeline": pipeline_name}
        st.session_state["pp_stats"] = pp_stats or {}
        st.session_state["tables"] = tables or []
        st.session_state["table_boxes"] = table_boxes or []
        st.session_state["last_runtime"] = dt

        # reset NLI state (new OCR)
        st.session_state["nli_res"] = None
        st.session_state["nli_chosen_model"] = None
        st.session_state["nli_evidence"] = ([], "", 0.0, "neutral")
        st.session_state["nli_evidence_debug"] = {}

        prog.progress(100, text="Hoàn tất ✅")
        st.toast(f"OCR xong ({dt:.2f}s).", icon="✅")
        time.sleep(0.05)

    # render results
    lines = _ensure_list(st.session_state.get("ocr_lines", []))
    ocr_img = st.session_state.get("ocr_image", None)
    ocr_img_pp = st.session_state.get("ocr_image_pp", None)
    tables = st.session_state.get("tables", []) or []
    table_boxes = st.session_state.get("table_boxes", []) or []
    dt = st.session_state.get("last_runtime", None)

    st.subheader("2) Kết quả OCR")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số dòng OCR", f"{len(lines)}")
    m2.metric("Số bảng", f"{len(tables)}")
    m3.metric("Trang", f"{page_idx + 1}/{len(pages)}")
    m4.metric("Thời gian", f"{dt:.2f}s" if isinstance(dt, (int, float)) else "—")

    colL, colR = st.columns([1.05, 1.15], gap="large")
    with colL:
        st.markdown("#### Ảnh xem nhanh")
        if ocr_img is None:
            st.image(img, use_container_width=True)
        else:
            im_vis = ocr_img_pp if ocr_img_pp is not None else ocr_img
            if show_overlay and lines:
                im_vis = draw_quads(im_vis, lines)
            if show_table_overlay and table_boxes:
                im_vis = _draw_rects(im_vis, table_boxes, width=3, color="red")
            st.image(im_vis, use_container_width=True)

    with colR:
        tab_text, tab_table, tab_export = st.tabs(["📝 Văn bản", "📊 Bảng", "⬇️ Tải về"])

        with tab_text:
            st.text_area("Văn bản OCR", key="ocr_text_area", height=520)
            pp_stats = st.session_state.get("pp_stats", {}) or {}
            if pp_stats:
                with st.expander("🔧 Nhật ký hậu xử lý", expanded=False):
                    st.json(pp_stats)

            if src == "Vietnamese-Entrance-Exam (dataset)" and gt_text.strip() and st.session_state.get("ocr_text", "").strip():
                hyp = st.session_state.get("ocr_text", "")
                c_cer = cer(gt_text, hyp)
                c_wer = wer(gt_text, hyp)
                em = 1.0 if normalize_text(gt_text) == normalize_text(hyp) else 0.0
                a1, a2, a3, a4 = st.columns(4)
                a1.metric("CER", f"{c_cer:.4f}")
                a2.metric("WER", f"{c_wer:.4f}")
                a3.metric("EM", f"{em:.0f}")
                a4.metric("Char-Acc", f"{(1.0 - c_cer):.4f}")

                with st.expander("Xem Ground-truth (GT)", expanded=False):
                    st.text_area("GT text", value=gt_text, height=220)

        with tab_table:
            if tables:
                st.caption("Bảng đã được sắp theo thứ tự **từ trên xuống**.")
                for k, tb in enumerate(tables, start=1):
                    df = tb.get("df", None) if isinstance(tb, dict) else None
                    b = tb.get("bbox", None) if isinstance(tb, dict) else None
                    bbox_txt = f" • bbox={b}" if b else ""
                    if df is not None:
                        st.markdown(f"**Bảng {k}** — rows={tb.get('n_rows')} • cols={tb.get('n_cols')}{bbox_txt}")
                        st.dataframe(df, use_container_width=True, height=260)
                        try:
                            csv_bytes = df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                f"Tải CSV — Bảng {k}",
                                data=csv_bytes,
                                file_name=f"table_{k}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.warning(f"Không xuất CSV được: {e}")
                    else:
                        st.markdown(f"**Bảng {k}**: (không có dataframe){bbox_txt}")
            else:
                st.info("Chưa phát hiện được bảng (hoặc trang này không có bảng).")

        with tab_export:
            text_out = st.session_state.get("ocr_text_area", "") or ""
            st.download_button("Tải TXT", data=text_out.encode("utf-8"), file_name="ocr_output.txt", mime="text/plain", use_container_width=True)
            st.download_button("Tải JSON (bbox + text)", data=lines_to_json(lines).encode("utf-8"), file_name="ocr_output.json", mime="application/json", use_container_width=True)
            st.download_button("Tải DOCX", data=lines_to_docx_bytes(lines), file_name="ocr_output.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               use_container_width=True)

# =========================================================
# TAB: Experiments (full dataset evaluation + charts)
# =========================================================
with tab_exp:
    st.subheader("📊 Thực nghiệm toàn bộ dataset (so sánh cấu hình OCR)")

    split = "train"
    try:
        ds = load_vnee_split(split)
        dom_map = get_vnee_domain_index(split)
        dom_items = sorted([(k, len(v)) for k, v in dom_map.items()], key=lambda x: (-x[1], x[0]))
        # Only expose Physics and Chemistry domains (remove Math and others)
        dom_items = [item for item in dom_items if item[0] in ("Physics", "Chemistry")]
    except Exception as e:
        st.error(f"Không load được dataset {VNEE_DATASET}: {e}")
        st.stop()

    st.markdown("#### 1) Chọn phạm vi chạy")
    cA, cB, cC, cD = st.columns([1.2, 1.2, 1.2, 1.2])
    with cA:
        dom_labels = ["All"] + [f"{k} ({n})" for k, n in dom_items]
        dom_choice = st.multiselect("Domain (chọn 1 hoặc nhiều)", dom_labels, default=["All"], help="Chọn một hoặc nhiều domain; chọn 'All' để chạy toàn bộ")
    with cB:
        limit = st.number_input("Giới hạn số mẫu (0 = toàn bộ)", min_value=0, max_value=len(ds), value=0, step=50)
    with cC:
        shuffle = st.checkbox("Shuffle", value=False)
    with cD:
        seed = st.number_input("Seed", min_value=0, max_value=999999, value=42, step=1)

    st.markdown("#### 2) Chọn cấu hình để so sánh")
    default_configs = ["C0_Baseline", "C1_PostVI+NonText", "C2_Contrast+PostVI+NonText"]
    picked = st.multiselect("Configs", options=list(PIPELINE_REGISTRY.keys()), default=default_configs)

    if not picked:
        st.warning("Chọn ít nhất 1 cấu hình.")
        st.stop()

    run_all = st.button("📊 Chạy thực nghiệm (so sánh cấu hình)", type="primary", use_container_width=True)

    if run_all:
        # Build indices (support multiple selected domains)
        if not dom_choice or "All" in dom_choice:
            indices = list(range(len(ds)))
            dom_name = "All"
        else:
            selected_names = [c.split(" (")[0] for c in dom_choice]
            indices = []
            for name in selected_names:
                indices.extend(dom_map.get(name, []))
            indices = sorted(set(indices))
            dom_name = ", ".join(selected_names)

        if shuffle:
            random.seed(int(seed))
            random.shuffle(indices)

        if int(limit) > 0:
            indices = indices[: int(limit)]

        if not indices:
            st.error("Không có mẫu nào để chạy.")
            st.stop()

        rows: List[Dict[str, Any]] = []
        prog = st.progress(0, text="Đang chạy…")
        t_start = time.perf_counter()

        total_steps = len(indices) * len(picked)
        step = 0

        skipped_indices = []
        for ii, idx in enumerate(indices, start=1):
            sample = ds[idx]
            img0 = _decode_hf_image(sample.get("image"))
            if img0 is None:
                # record skipped index for debugging and skip
                skipped_indices.append(idx)
                continue
            gt = build_gt_from_sample(sample)
            dom = _get_sample_domain(sample)

            for cfg_name in picked:
                cfg = PIPELINE_REGISTRY[cfg_name]
                lines, hyp, _, _, _, dt, _ = run_ocr_pipeline(img0, engine, cfg)

                gt_n = normalize_text(gt)
                hyp_n = normalize_text(hyp)
                c1 = cer(gt_n, hyp_n)
                w1 = wer(gt_n, hyp_n)
                em = 1.0 if gt_n == hyp_n else 0.0

                gt_na = normalize_text(strip_accents(gt_n))
                hyp_na = normalize_text(strip_accents(hyp_n))
                c_no_acc = cer(gt_na, hyp_na)

                rows.append({
                    "sample_idx": idx,
                    "domain": dom,
                    "config": cfg_name,
                    "cer": float(c1),
                    "wer": float(w1),
                    "em": float(em),
                    "char_acc": float(1.0 - c1),
                    "cer_no_acc": float(c_no_acc),
                    "diac_penalty": float(max(0.0, c1 - c_no_acc)),
                    "sym_recall": float(symbol_recall(gt_n, hyp_n)),
                    "runtime_s": float(dt),
                    "gt_chars": int(len(gt_n)),
                    "hyp_chars": int(len(hyp_n)),
                })

                step += 1
                prog.progress(int(step / max(1, total_steps) * 100), text=f"{ii}/{len(indices)} mẫu • {cfg_name}")

        t_total = time.perf_counter() - t_start
        prog.progress(100, text=f"Hoàn tất ✅ Tổng thời gian: {t_total:.1f}s")

        try:
            import pandas as pd
        except Exception:
            st.error("Thiếu pandas. Cài: pip install pandas")
            st.stop()

        df = pd.DataFrame(rows)
        if df.empty:
            msg = "Không có kết quả (df rỗng)."
            if skipped_indices:
                msg += f"\nSố mẫu bị skip (không decode được ảnh): {len(skipped_indices)}. Ví dụ các idx đầu: {skipped_indices[:20]}"
            st.error(msg)
            st.stop()

        # Summaries
        summary = (
            df.groupby("config", as_index=False)
              .agg(
                  n=("cer", "count"),
                  mean_cer=("cer", "mean"),
                  mean_wer=("wer", "mean"),
                  mean_em=("em", "mean"),
                  mean_char_acc=("char_acc", "mean"),
                  mean_sym_recall=("sym_recall", "mean"),
                  mean_runtime=("runtime_s", "mean"),
              )
              .sort_values("mean_cer")
        )

        by_domain = (
            df.groupby(["domain", "config"], as_index=False)
              .agg(
                  n=("cer", "count"),
                  mean_cer=("cer", "mean"),
                  mean_wer=("wer", "mean"),
                  mean_em=("em", "mean"),
                  mean_runtime=("runtime_s", "mean"),
              )
              .sort_values(["domain", "mean_cer"])
        )

        st.session_state["exp_rows"] = df
        st.session_state["exp_summary"] = summary
        st.session_state["exp_by_domain"] = by_domain

        st.toast("Đã lưu kết quả thực nghiệm vào session.", icon="✅")

    # Show results if available
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception:
        pd, plt = None, None

    df = st.session_state.get("exp_rows", None)
    summary = st.session_state.get("exp_summary", None)
    by_domain = st.session_state.get("exp_by_domain", None)

    if df is None or summary is None:
        st.info("Chưa có kết quả. Chọn cấu hình rồi bấm **📊 Chạy thực nghiệm**.")
    else:
        st.markdown("### 3) Kết quả tổng hợp")
        st.dataframe(summary, use_container_width=True, height=240)

        st.markdown("### 4) Kết quả theo domain")
        st.dataframe(by_domain, use_container_width=True, height=280)

        st.markdown("### 5) Biểu đồ phân tích")
        if plt is None:
            st.warning("Thiếu matplotlib. Cài: pip install matplotlib")
        else:
            # Mean CER by config
            fig1 = plt.figure()
            plt.bar(summary["config"], summary["mean_cer"])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Mean CER")
            plt.title("Mean CER by OCR config")
            st.pyplot(fig1, clear_figure=True)

            # Mean WER by config
            fig2 = plt.figure()
            plt.bar(summary["config"], summary["mean_wer"])
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("Mean WER")
            plt.title("Mean WER by OCR config")
            st.pyplot(fig2, clear_figure=True)

            # CER distribution (boxplot)
            fig3 = plt.figure()
            # build list per config
            configs = list(summary["config"])
            data_box = [df[df["config"] == c]["cer"].values for c in configs]
            plt.boxplot(data_box, labels=configs, showfliers=False)
            plt.xticks(rotation=20, ha="right")
            plt.ylabel("CER")
            plt.title("CER distribution (boxplot) by config")
            st.pyplot(fig3, clear_figure=True)

            # Domain breakdown: mean CER lines
            pivot = by_domain.pivot(index="domain", columns="config", values="mean_cer")
            fig4 = plt.figure()
            for c in pivot.columns:
                plt.plot(pivot.index, pivot[c].values, marker="o", label=str(c))
            plt.ylabel("Mean CER")
            plt.title("Mean CER by domain and config")
            plt.xticks(rotation=15, ha="right")
            plt.legend()
            st.pyplot(fig4, clear_figure=True)

        st.markdown("### 6) Phân tích lỗi (top CER cao)")
        topk = st.slider("Top-K mẫu lỗi lớn nhất (mỗi config)", 3, 30, 10)
        show_images = st.checkbox("Hiển thị ảnh + GT/HYP", value=False)

        try:
            ds = load_vnee_split("train")
        except Exception:
            ds = None

        for cfg_name in list(summary["config"]):
            st.markdown(f"#### {cfg_name}")
            sub = df[df["config"] == cfg_name].sort_values("cer", ascending=False).head(int(topk))
            st.dataframe(sub[["sample_idx", "domain", "cer", "wer", "em", "diac_penalty", "sym_recall", "runtime_s"]], use_container_width=True, height=240)

            if show_images and ds is not None:
                with st.expander(f"Xem minh hoạ lỗi — {cfg_name}", expanded=False):
                    for _, r in sub.head(5).iterrows():
                        idx = int(r["sample_idx"])
                        sample = ds[idx]
                        img0 = _decode_hf_image(sample.get("image"))
                        gt = build_gt_from_sample(sample)
                        # rerun OCR to display hypothesis (avoid storing huge text in df)
                        cfg = PIPELINE_REGISTRY[cfg_name]
                        _, hyp, _, _, _, _, img_pp = run_ocr_pipeline(img0, engine, cfg)
                        st.markdown(f"**idx={idx} • domain={_get_sample_domain(sample)} • CER={float(r['cer']):.4f}**")
                        st.image(img_pp, use_container_width=True)
                        cA, cB = st.columns(2)
                        with cA:
                                        st.text_area("GT", value=gt, height=160, key=f"gt_{cfg_name}_{idx}")
                        with cB:
                                        st.text_area("OCR", value=hyp, height=160, key=f"ocr_{cfg_name}_{idx}")

        st.markdown("### 7) Xuất kết quả (CSV)")
        # Downloads
        df_csv = df.to_csv(index=False).encode("utf-8")
        sum_csv = summary.to_csv(index=False).encode("utf-8")
        dom_csv = by_domain.to_csv(index=False).encode("utf-8")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button("⬇️ CSV chi tiết (từng mẫu × config)", data=df_csv, file_name="vnee_ocr_eval_all_configs.csv", mime="text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇️ CSV tổng hợp theo config", data=sum_csv, file_name="vnee_ocr_eval_summary_by_config.csv", mime="text/csv", use_container_width=True)
        with c3:
            st.download_button("⬇️ CSV theo domain × config", data=dom_csv, file_name="vnee_ocr_eval_by_domain_and_config.csv", mime="text/csv", use_container_width=True)

        st.markdown("### 8) Gợi ý nội dung đưa vào Chương 2–4")
        with st.expander("Copy-paste nội dung (mẫu) cho báo cáo", expanded=False):
            n_total = int(df["sample_idx"].nunique())
            dom_counts = df.drop_duplicates(["sample_idx", "domain"])["domain"].value_counts().to_dict()
            st.markdown(
                f"""
**Chương 2 – Dữ liệu & Pipeline**  
- Dataset thí điểm: **{VNEE_DATASET}** (split train), tổng **{n_total}** mẫu.  
- Phân bố domain: {", ".join([f"{k}={v}" for k, v in dom_counts.items()])}.  
- GT cho OCR được dựng từ **vi_problem + vi_choices** (đề bài + lựa chọn).  
- Pipeline: tiền xử lý (tuỳ config) → OCR engine → sắp xếp thứ tự đọc → lọc non-text (tuỳ config) → sửa tiếng Việt (tuỳ config) → xuất TXT/JSON/DOCX.

**Chương 3 – Thực nghiệm & Kết quả**  
- So sánh {len(summary)} cấu hình: {", ".join(list(summary["config"]))}.  
- Kết quả tổng hợp (điền số từ bảng Summary):  
  - CER thấp nhất: **{summary.iloc[0]['config']}** với mean CER ≈ **{summary.iloc[0]['mean_cer']:.4f}**  
  - WER thấp nhất: **{summary.sort_values('mean_wer').iloc[0]['config']}** với mean WER ≈ **{summary.sort_values('mean_wer').iloc[0]['mean_wer']:.4f}**  
- Phân tích lỗi: xem Top-K mẫu CER cao (mất dấu, sai ký hiệu, sai xuống dòng/thứ tự đọc).

**Chương 4 – Kết luận & Hướng phát triển**  
- Kết luận: cấu hình có tiền xử lý + hậu xử lý giúp cải thiện CER/WER so với baseline.  
- Hạn chế: OCR ký hiệu toán – khoa học và dấu tiếng Việt vẫn là điểm khó.  
- Hướng mở rộng: mở rộng dữ liệu scan/ảnh chụp thực tế; finetune OCR theo miền giáo dục; tích hợp sâu DocVQA/NLI để hỗ trợ suy luận.
                """.strip()
            )

# =========================================================
# TAB: NLI
# =========================================================
with tab_nli:
    st.subheader("🧠 NLI Reasoning (Premise–Hypothesis)")

    ocr_text_current = st.session_state.get("ocr_text", "") or ""
    meta = st.session_state.get("ocr_meta", {"id": None})

    if not meta.get("id") or not ocr_text_current.strip():
        st.info("Chưa có OCR text. Hãy chạy OCR ở tab **🧾 OCR** trước.")
        st.stop()

    st.caption(f"OCR source: {meta.get('id')} • pipeline={meta.get('pipeline')}")

    with st.expander("⚙️ Cấu hình NLI (nâng cao)", expanded=False):
        nli_mode = st.radio("Chế độ chọn model", ["Manual", "Auto"], index=1, horizontal=True)
        if nli_mode == "Manual":
            nli_model = st.selectbox("Model", NLI_MODELS, index=0)
        else:
            st.selectbox("Auto strategy", ["Heuristic (nhanh)"], index=0, disabled=True)
            nli_model = None
        abstain_thr = st.slider("Ngưỡng abstain", 0.0, 1.0, 0.65, 0.01)
        max_sent = st.slider("Giới hạn số câu", 1, 30, 8)
        max_chars = st.slider("Giới hạn ký tự", 500, 5000, 2000, 100)
        show_debug = st.checkbox("Hiển thị debug evidence", value=False)

    with st.form("nli_form"):
        st.text_area("Premise (OCR text)", value=ocr_text_current, height=180, disabled=True)
        hypothesis = st.text_input("Hypothesis (nhận định/câu trả lời)", value="")
        submit = st.form_submit_button("🧩 Chạy NLI")

    if submit:
        if not hypothesis.strip():
            st.warning("Bạn cần nhập Hypothesis.")
        else:
            premise_for_nli = _cut_premise(ocr_text_current, max_sent=max_sent, max_chars=max_chars)
            chosen_model = nli_model if nli_mode == "Manual" else auto_pick_model_heuristic(premise_for_nli)

            nli = load_nli(chosen_model, device="cpu")
            res = nli.predict(premise_for_nli, hypothesis)

            st.session_state["nli_res"] = res
            st.session_state["nli_chosen_model"] = chosen_model

            # evidence selection (optional module)
            try:
                from src.evidence import pick_evidence_window
                ocr_lines = _ensure_list(st.session_state.get("ocr_lines", []))
                idxs, ev_text, ev_score, ev_label, ev_debug = pick_evidence_window(
                    nli=nli,
                    lines=ocr_lines,
                    hypothesis=hypothesis,
                    target_label=res.label,
                    max_window=4,
                    topk_lines=80,
                    min_final=0.35,
                    alpha_nli=0.60,
                )
                st.session_state["nli_evidence"] = (idxs, ev_text, float(ev_score), ev_label)
                st.session_state["nli_evidence_debug"] = ev_debug
            except Exception as e:
                st.session_state["nli_evidence"] = ([], "", 0.0, "neutral")
                st.session_state["nli_evidence_debug"] = {"warning": f"evidence module lỗi: {e}"}

    res = st.session_state.get("nli_res", None)
    chosen_model = st.session_state.get("nli_chosen_model", None)

    if res is None:
        st.info("Nhập Hypothesis rồi bấm **🧩 Chạy NLI**.")
    else:
        st.success(f"Model: {chosen_model}")
        st.write("**Label:**", res.label)
        st.write("**Confidence:**", round(float(res.confidence), 4))
        st.json(res.probs)

        if float(res.confidence) < abstain_thr:
            st.warning("Model không chắc chắn → nên xác nhận thủ công (abstain).")

        idxs, ev_text, ev_score, _ = st.session_state.get("nli_evidence", ([], "", 0.0, "neutral"))
        st.markdown("### 🔎 Evidence")
        st.write(ev_text if ev_text else "(Chưa đủ chắc để highlight)")
        st.caption(f"score={ev_score:.4f}")

        ocr_img_pp = st.session_state.get("ocr_image_pp", None)
        ocr_lines = _ensure_list(st.session_state.get("ocr_lines", []))
        if ocr_img_pp is not None and idxs:
            hi_lines = [ocr_lines[i] for i in idxs if 0 <= i < len(ocr_lines)]
            st.image(draw_quads(ocr_img_pp, hi_lines), use_container_width=True)

        if show_debug:
            with st.expander("Debug evidence", expanded=False):
                st.json(st.session_state.get("nli_evidence_debug", {}) or {})

# =========================================================
# TAB: Help
# =========================================================
with tab_help:
    st.markdown(
        """
**Luồng demo nhanh (30–60s):**
1) Tab **🧾 OCR** → chọn nguồn (Upload hoặc dataset) → chọn **Pipeline** → bấm **🚀 Chạy OCR**  
2) Trình bày: Ảnh bbox → Văn bản → (Bảng nếu có) → Tải TXT/JSON/DOCX  
3) Tab **🧠 NLI**: nhập hypothesis → xem label + highlight evidence  
4) Tab **📊 Experiments**: chạy toàn bộ dataset để lấy bảng + biểu đồ + CSV phục vụ Chương 2–4

**Fix lỗi domain 'math':**
- App đã **chuẩn hoá domain** về {Math, Physics, Chemistry, Unknown} bằng `canonical_domain()` nên không còn phụ thuộc đúng chính tả/hoa-thường.

**Nếu chạy Experiments lâu:**
- Giảm "Giới hạn số mẫu" (ví dụ 50/100) để test nhanh,
- Sau đó chạy 0 (toàn bộ) để lấy kết quả cuối.
        """
    )




