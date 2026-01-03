# src/evidence.py
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import re
from difflib import SequenceMatcher
from statistics import median

from .nli_engine import NLIEdge
from .ocr_engine import OcrLine
from .text_utils import normalize_vi, split_sentences_vi


# -------------------------
# Normalization + features
# -------------------------
_KEEP_CHARS = r"\/\;\[\]\(\)\.\-\%\:\+"

def _norm(s: str) -> str:
    s = normalize_vi(s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _norm_for_match(s: str) -> str:
    s = _norm(s)
    s = re.sub(rf"[^\w\s{_KEEP_CHARS}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _char_ngrams(s: str, n: int = 3) -> set:
    s = _norm_for_match(s)
    s = re.sub(r"\s+", " ", s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}

def _tokens(s: str) -> List[str]:
    s = _norm_for_match(s)
    return [t for t in s.split() if t]

def _anchors(s: str) -> List[str]:
    """
    Anchors định vị mạnh: số, ngày, % , khoảng [a;b)...
    """
    s = _norm_for_match(s)
    out = []
    out += re.findall(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", s)        # 21/05/2025
    out += re.findall(r"\[\s*\d+\s*;\s*\d+\s*\)", s)          # [159;162)
    out += re.findall(r"\[\s*\d+\s*;\s*\d+\s*\]", s)          # [..;..]
    out += re.findall(r"\b\d+(?:\.\d+)?\s*%?\b", s)           # 120, 8%, 1.5
    # unique preserve order
    seen = set()
    uniq = []
    for a in out:
        a = a.strip()
        if a and a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

def _seq_sim(a: str, b: str) -> float:
    a2, b2 = _norm_for_match(a), _norm_for_match(b)
    if not a2 or not b2:
        return 0.0
    return SequenceMatcher(None, a2, b2).ratio()

def _f1_overlap(query_tokens: List[str], line_tokens: List[str]) -> float:
    if not query_tokens or not line_tokens:
        return 0.0
    q = set(query_tokens)
    l = set(line_tokens)
    inter = len(q & l)
    if inter == 0:
        return 0.0
    prec = inter / max(1, len(l))
    rec = inter / max(1, len(q))
    return 2 * prec * rec / max(1e-9, (prec + rec))


# -------------------------
# Geometry helpers (IMPORTANT)
# -------------------------
def _get_bbox(ln: OcrLine) -> Optional[Tuple[float, float, float, float]]:
    """
    Support OcrLine.bbox = (x0,y0,x1,y1) OR OcrLine.quad = [(x,y)...]
    Return (x0,y0,x1,y1) or None.
    """
    if hasattr(ln, "bbox") and ln.bbox is not None:
        b = ln.bbox
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x0, y0, x1, y1 = b
            return float(x0), float(y0), float(x1), float(y1)

    if hasattr(ln, "quad") and ln.quad is not None:
        q = ln.quad
        try:
            xs = [p[0] for p in q]
            ys = [p[1] for p in q]
            return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
        except Exception:
            return None

    return None

def _reading_order_indices(lines: List[OcrLine]) -> Tuple[List[int], Dict[int, Dict]]:
    """
    Trả về:
      - order: list original indices đã sắp xếp reading-order (có detect 2 cột)
      - geom: map original idx -> dict geom (x0,y0,x1,y1,col,height)
    Nếu không có bbox/quad, giữ nguyên thứ tự.
    """
    geom: Dict[int, Dict] = {}
    items = []
    for i, ln in enumerate(lines):
        b = _get_bbox(ln)
        txt = (ln.text or "").strip() if hasattr(ln, "text") else ""
        if b is None:
            continue
        x0, y0, x1, y1 = b
        h = max(1.0, (y1 - y0))
        geom[i] = {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "h": h, "txt": txt}
        items.append(i)

    # fallback: không có geom -> giữ nguyên
    if not items:
        return list(range(len(lines))), geom

    # ước lượng page width
    max_x1 = max(geom[i]["x1"] for i in items)
    min_x0 = min(geom[i]["x0"] for i in items)
    page_w = max(1.0, (max_x1 - min_x0))

    # detect 2 columns bằng khoảng cách lớn trong phân bố x0
    x0s = sorted(geom[i]["x0"] for i in items)
    gaps = [(x0s[j+1] - x0s[j], j) for j in range(len(x0s) - 1)]
    gap_thr = 0.22 * page_w  # heuristic: gap lớn ~ 2 cột
    split_x = None
    if gaps:
        best_gap, best_j = max(gaps, key=lambda t: t[0])
        if best_gap >= gap_thr:
            split_x = (x0s[best_j] + x0s[best_j + 1]) / 2.0

    # assign column
    for i in items:
        if split_x is None:
            col = 0
        else:
            col = 0 if geom[i]["x0"] < split_x else 1
        geom[i]["col"] = col

    # sort reading order: col -> y -> x
    order = sorted(items, key=lambda i: (geom[i]["col"], geom[i]["y0"], geom[i]["x0"]))

    # add any lines without bbox at end (keep stable)
    missing = [i for i in range(len(lines)) if i not in geom]
    order.extend(missing)

    return order, geom

def _build_next_ok(order: List[int], geom: Dict[int, Dict], gap_factor: float = 1.8) -> Dict[int, bool]:
    """
    next_ok[pos] = True nếu order[pos] và order[pos+1] thuộc cùng block/col và không cách xa dọc.
    """
    heights = [geom[i]["h"] for i in order if i in geom]
    if not heights:
        return {p: True for p in range(len(order) - 1)}

    h_med = median(heights)
    gap_thr = gap_factor * h_med

    next_ok = {}
    for p in range(len(order) - 1):
        i = order[p]
        j = order[p + 1]
        if i in geom and j in geom:
            same_col = geom[i].get("col", 0) == geom[j].get("col", 0)
            gap = geom[j]["y0"] - geom[i]["y1"]  # >0 nếu xuống dòng
            ok = same_col and (gap <= gap_thr)
            next_ok[p] = ok
        else:
            # thiếu bbox -> cho qua (khó kiểm soát)
            next_ok[p] = True
    return next_ok


# -------------------------
# Evidence sentence (for reporting)
# -------------------------
def pick_evidence_sentence(nli: NLIEdge, premise_text: str, hypothesis: str) -> Tuple[str, float, str]:
    sents = split_sentences_vi(premise_text)
    if not sents:
        return "", 0.0, "neutral"
    best = ("", 0.0, "neutral")
    for s in sents:
        r = nli.predict(s, hypothesis)
        if r.confidence > best[1]:
            best = (s, r.confidence, r.label)
    return best


# -------------------------
# Robust evidence selection for ANY document
# Retrieve -> Re-rank -> Verify (with geometry-aware ordering)
# -------------------------
def pick_evidence_window(
    nli: NLIEdge,
    lines: List[OcrLine],
    hypothesis: str,
    target_label: Optional[str] = None,
    max_window: int = 4,
    topk_lines: int = 80,
    min_final: float = 0.35,
    alpha_nli: float = 0.60,
) -> Tuple[List[int], str, float, str, Dict]:
    """
    Trả: (idxs(original), evidence_text, final_score, evidence_label, debug)

    Nâng cấp quan trọng:
    - Tự reorder lines theo bbox (reading-order + detect 2 cột)
    - Khi build window, KHÔNG nối qua khoảng cách dọc quá lớn (tránh nhảy đoạn)
    """
    debug: Dict = {
        "stage": "init",
        "target_label": target_label,
        "top_lines": [],
        "top_windows": [],
        "has_geom": False
    }

    if not lines or not (hypothesis or "").strip():
        return [], "", 0.0, "neutral", debug

    # Geometry-aware reading order
    order, geom = _reading_order_indices(lines)
    debug["has_geom"] = bool(geom)

    # adjacency constraint (same block-ish)
    next_ok = _build_next_ok(order, geom, gap_factor=1.8)

    # Precompute query features
    q_text = hypothesis.strip()
    q_ngr = _char_ngrams(q_text, 3)
    q_toks = _tokens(q_text)
    q_anc = _anchors(q_text)

    # Map ordered position -> original idx
    # Build features in ordered space
    feats = []
    for pos, orig_i in enumerate(order):
        txt = (lines[orig_i].text or "").strip()
        if not txt:
            feats.append(None)
            continue
        feats.append({
            "pos": pos,
            "i": orig_i,
            "txt": txt,
            "ngr": _char_ngrams(txt, 3),
            "toks": _tokens(txt),
            "anc": _anchors(txt),
        })

    # 1) Retrieve: score each line against hypothesis (ordered positions)
    line_scores: List[Tuple[float, int, int]] = []  # (score, pos, orig_i)
    for f in feats:
        if f is None:
            continue

        s_ng = _jaccard(q_ngr, f["ngr"])
        s_seq = _seq_sim(q_text, f["txt"])
        s_f1 = _f1_overlap(q_toks, f["toks"])

        bonus = 0.0
        if q_anc:
            hit = sum(1 for a in q_anc if a in _norm_for_match(f["txt"]))
            bonus = min(0.25, 0.08 * hit)

        score = 0.45 * s_ng + 0.35 * s_seq + 0.20 * s_f1 + bonus
        line_scores.append((score, f["pos"], f["i"]))

    line_scores.sort(reverse=True)
    cand_pos = [pos for s, pos, _ in line_scores[:topk_lines] if s > 0.10]
    if not cand_pos:
        cand_pos = [f["pos"] for f in feats if f is not None]

    debug["stage"] = "retrieved"
    debug["top_lines"] = [
        {"pos": pos, "i": orig_i, "score": float(s), "txt": (lines[orig_i].text or "")[:120]}
        for (s, pos, orig_i) in line_scores[:10]
    ]

    # 2) Candidate window starts around candidate positions
    cand_starts = set()
    for pos in cand_pos:
        for start in range(max(0, pos - (max_window - 1)), pos + 1):
            cand_starts.add(start)

    # helper: retrieval score for window = avg(line_score) in window
    score_map = {pos: s for s, pos, _ in line_scores}

    def window_retrieval_score(pos_list: List[int]) -> float:
        ss = [score_map.get(p, 0.0) for p in pos_list]
        return float(sum(ss) / max(1, len(ss)))

    # check window validity by next_ok constraints
    def is_valid_window(start: int, end: int) -> bool:
        # window positions [start, end)
        if end - start <= 1:
            return True
        # require all consecutive links ok
        for p in range(start, end - 1):
            if p in next_ok and not next_ok[p]:
                return False
        return True

    # 3) Re-rank windows with NLI (ordered positions)
    best = {"final": -1.0, "pos": [], "idxs": [], "text": "", "label": "neutral", "nli": 0.0, "ret": 0.0}
    top_windows_dbg = []

    for start in sorted(cand_starts):
        for w in range(1, max_window + 1):
            end = start + w
            if end > len(order):
                break
            if not is_valid_window(start, end):
                break  # càng dài càng sai -> dừng sớm

            pos_list = list(range(start, end))
            orig_idxs = [order[p] for p in pos_list]

            txts = [(lines[i].text or "").strip() for i in orig_idxs]
            if not any(txts):
                continue
            cand_text = " ".join([t for t in txts if t])

            r = nli.predict(cand_text, q_text)
            if target_label and isinstance(r.probs, dict):
                nli_score = float(r.probs.get(target_label.lower(), 0.0))
            else:
                nli_score = float(r.confidence)

            ret_score = window_retrieval_score(pos_list)

            # geometry penalty: prefer tighter vertical span (if geom available)
            geo_pen = 0.0
            if geom and all(i in geom for i in orig_idxs):
                y0 = min(geom[i]["y0"] for i in orig_idxs)
                y1 = max(geom[i]["y1"] for i in orig_idxs)
                span = max(1.0, (y1 - y0))
                h_med = median([geom[i]["h"] for i in orig_idxs])
                # span lớn bất thường -> trừ điểm nhẹ
                if span > 4.0 * max(1.0, h_med):
                    geo_pen = 0.03

            final = alpha_nli * nli_score + (1.0 - alpha_nli) * ret_score - geo_pen

            # debug top windows
            if len(top_windows_dbg) < 8:
                top_windows_dbg.append((final, orig_idxs, r.label, nli_score, ret_score, cand_text[:140]))
                top_windows_dbg.sort(reverse=True)
            else:
                if final > top_windows_dbg[-1][0]:
                    top_windows_dbg[-1] = (final, orig_idxs, r.label, nli_score, ret_score, cand_text[:140])
                    top_windows_dbg.sort(reverse=True)

            if final > best["final"]:
                best.update(final=final, pos=pos_list, idxs=orig_idxs, text=cand_text, label=r.label, nli=nli_score, ret=ret_score)

    debug["stage"] = "reranked"
    debug["top_windows"] = [
        {"final": float(f), "idxs": idxs, "label": lab, "nli": float(ns), "ret": float(rs), "text": tx}
        for (f, idxs, lab, ns, rs, tx) in top_windows_dbg[:8]
    ]

    # 4) Verify
    if best["final"] < min_final:
        debug["stage"] = "low_confidence_no_highlight"
        return [], best["text"], float(best["final"]), best["label"], debug

    debug["stage"] = "selected"
    return best["idxs"], best["text"], float(best["final"]), best["label"], debug


# -------------------------
# Manual search helper (guarantee correctness)
# -------------------------
def search_lines(lines: List[OcrLine], query: str, topk: int = 20) -> List[int]:
    q = _norm_for_match(query)
    if not q:
        return []
    scored = []
    for i, ln in enumerate(lines):
        t = _norm_for_match(getattr(ln, "text", "") or "")
        if not t:
            continue
        sim = _seq_sim(q, t)
        if q in t:
            sim += 0.35
        scored.append((sim, i))
    scored.sort(reverse=True)
    return [i for s, i in scored[:topk] if s >= 0.35]
