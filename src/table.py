# src/table.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from statistics import median
import numpy as np
import re

from .ocr_engine import OcrLine
from .layout import get_bbox


def _merge_boxes(
    boxes: List[Tuple[int, int, int, int]],
    iou_thr: float = 0.12,
    pad: int = 6
) -> List[Tuple[int, int, int, int]]:
    def area(b):
        x0, y0, x1, y1 = b
        return max(1, (x1 - x0)) * max(1, (y1 - y0))

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        x0 = max(ax0, bx0)
        y0 = max(ay0, by0)
        x1 = min(ax1, bx1)
        y1 = min(ay1, by1)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        inter = (x1 - x0) * (y1 - y0)
        return inter / (area(a) + area(b) - inter)

    merged = []
    for b in boxes:
        x0, y0, x1, y1 = b
        b2 = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        merged.append(b2)

    changed = True
    while changed:
        changed = False
        out = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            a = merged[i]
            ax0, ay0, ax1, ay1 = a
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                if iou(a, merged[j]) >= iou_thr:
                    bx0, by0, bx1, by1 = merged[j]
                    ax0 = min(ax0, bx0)
                    ay0 = min(ay0, by0)
                    ax1 = max(ax1, bx1)
                    ay1 = max(ay1, by1)
                    used[j] = True
                    changed = True
            used[i] = True
            out.append((ax0, ay0, ax1, ay1))
        merged = out

    merged = [(max(0, x0), max(0, y0), max(0, x1), max(0, y1)) for x0, y0, x1, y1 in merged]
    return merged


def _intersect_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _overlap_ratio(inner_box, outer_box) -> float:
    ax0, ay0, ax1, ay1 = inner_box
    a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    inter = _intersect_area(inner_box, outer_box)
    return float(inter / a_area)


def _cluster_1d(vals: List[float], tol: float) -> List[float]:
    """Greedy 1D clustering -> trả centers sorted"""
    if not vals:
        return []
    xs = sorted(vals)
    clusters = [[xs[0]]]
    for x in xs[1:]:
        if abs(x - clusters[-1][-1]) <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    centers = [sum(c) / len(c) for c in clusters]
    return centers


def _center_in_box(xc: float, yc: float, box: Tuple[int, int, int, int]) -> bool:
    x0, y0, x1, y1 = box
    return (x0 <= xc <= x1) and (y0 <= yc <= y1)


def _make_inner_box(tb: Tuple[int, int, int, int], pad_in: int) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = tb
    x0i = min(x1 - 1, x0 + pad_in)
    y0i = min(y1 - 1, y0 + pad_in)
    x1i = max(x0i + 1, x1 - pad_in)
    y1i = max(y0i + 1, y1 - pad_in)
    return (x0i, y0i, x1i, y1i)


def _centers_to_spans(centers: List[float], lo: float, hi: float) -> List[Tuple[float, float]]:
    """centers sorted -> spans theo midpoint"""
    if not centers:
        return []
    cs = sorted(centers)
    bounds = []
    for i in range(len(cs) - 1):
        bounds.append((cs[i] + cs[i + 1]) / 2.0)
    spans = []
    for i in range(len(cs)):
        left = lo if i == 0 else bounds[i - 1]
        right = hi if i == len(cs) - 1 else bounds[i]
        spans.append((left, right))
    return spans


def _assign_by_overlap_1d(seg: Tuple[float, float], spans: List[Tuple[float, float]]) -> int:
    """Gán seg vào span có overlap ratio lớn nhất."""
    a0, a1 = seg
    w = max(1e-6, a1 - a0)
    best_i = 0
    best = -1.0
    for i, (s0, s1) in enumerate(spans):
        inter = max(0.0, min(a1, s1) - max(a0, s0))
        ratio = inter / w
        if ratio > best:
            best = ratio
            best_i = i
    return best_i


def _count_overlapped_spans(x0: float, x1: float, spans: List[Tuple[float, float]], thr: float = 0.22) -> int:
    """đếm số span mà đoạn [x0,x1] overlap đủ lớn"""
    w = max(1e-6, x1 - x0)
    cnt = 0
    for s0, s1 in spans:
        inter = max(0.0, min(x1, s1) - max(x0, s0))
        if (inter / w) >= thr:
            cnt += 1
    return cnt


def _split_table_like_text(text: str) -> Optional[List[str]]:
    """
    Tách 1 dòng OCR thành nhiều cell nếu có pattern bảng:
    - khoảng [a;b)
    - dãy số ngắn (1 2 3 4 5)
    - hoặc nhiều cụm tách bởi 2+ spaces
    """
    t = (text or "").strip()
    if not t:
        return None

    # 1) interval tokens: [153;156)
    interval_pat = re.compile(r"\[\s*\d+\s*;\s*\d+\s*\)")
    ms = list(interval_pat.finditer(t))
    if len(ms) >= 2:
        parts: List[str] = []
        # prefix trước interval đầu tiên (vd: "Chiều cao")
        pre = t[:ms[0].start()].strip()
        if pre:
            parts.append(pre)
        for m in ms:
            parts.append(m.group(0).strip())
        # suffix sau interval cuối (hiếm)
        suf = t[ms[-1].end():].strip()
        if suf:
            parts.append(suf)
        # nếu tách được >=2 cell -> ok
        if len(parts) >= 2:
            return parts

    # 2) split by 2+ spaces (OCR đôi khi giữ khoảng cách cột)
    parts2 = [p.strip() for p in re.split(r"\s{2,}", t) if p.strip()]
    if len(parts2) >= 2:
        return parts2

    # 3) dãy số ngắn, thường là header bảng (1 2 3 4 5 6)
    nums = re.findall(r"\b\d+\b", t)
    if len(nums) >= 3 and len(t) <= 50:
        # nếu gần như toàn số -> tách theo số
        non_num = re.sub(r"[\d\s]", "", t)
        if len(non_num) <= 2:
            return nums

    return None


def _approx_segments_xspans(
    x0: float, x1: float, segments: List[str]
) -> List[Tuple[float, float]]:
    """
    Ước lượng (sx0,sx1) cho từng segment trong bbox [x0,x1]
    theo tỉ lệ độ dài ký tự (heuristic, nhưng đủ tốt để gán cột).
    """
    W = max(1.0, x1 - x0)
    lens = [max(1, len(s)) for s in segments]
    total = float(sum(lens))
    spans = []
    cur = x0
    for L in lens:
        w = W * (float(L) / total)
        spans.append((cur, cur + w))
        cur += w
    # đảm bảo segment cuối chạm x1
    if spans:
        a0, _ = spans[-1]
        spans[-1] = (a0, x1)
    return spans


# =========================
# detect_table_regions: giữ nguyên như bạn đang dùng
# =========================
def detect_table_regions(
    img_rgb: np.ndarray,
    lines: Optional[List[OcrLine]] = None,
) -> List[Tuple[int, int, int, int]]:
    H, W = img_rgb.shape[:2]
    boxes: List[Tuple[int, int, int, int]] = []

    # ---------- A) Image-based (grid lines) ----------
    try:
        import cv2

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        bw = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 15)
        bw = cv2.medianBlur(bw, 3)

        hk = max(12, W // 28)
        vk = max(12, H // 28)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))

        h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel, iterations=1)

        grid = cv2.addWeighted(h_lines, 0.5, v_lines, 0.5, 0.0)
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

        contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 0.012 * W * H
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_area:
                continue
            if w < 0.18 * W or h < 0.06 * H:
                continue
            boxes.append((x, y, x + w, y + h))
    except Exception:
        pass

    # ---------- B) OCR-based (no borders) ----------
    if lines:
        items = []
        hs = []
        for ln in lines:
            b = get_bbox(ln)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            t = (ln.text or "").strip()
            if not t:
                continue
            h = max(1.0, y1 - y0)
            hs.append(h)
            items.append((x0, y0, x1, y1, (x0 + x1) / 2.0, (y0 + y1) / 2.0, t))

        if len(items) >= 20:
            h_med = median(hs) if hs else 14.0
            row_tol = max(10.0, 0.75 * h_med)

            ycs = [it[5] for it in items]
            row_centers = _cluster_1d(ycs, tol=row_tol)

            rows: Dict[int, List[Tuple[float, float, float, float, float, str]]] = {i: [] for i in range(len(row_centers))}
            for x0, y0, x1, y1, xc, yc, t in items:
                rid = min(range(len(row_centers)), key=lambda k: abs(yc - row_centers[k]))
                rows[rid].append((x0, y0, x1, y1, xc, t))

            table_row_flags = []
            for rid, cells in rows.items():
                cells_sorted = sorted(cells, key=lambda z: z[4])
                cnt = len(cells_sorted)
                if cnt < 3:
                    table_row_flags.append(False)
                    continue
                short_ratio = sum(1 for c in cells_sorted if len(c[5]) <= 14) / max(1, cnt)
                table_row_flags.append(short_ratio >= 0.55)

            segs = []
            start = None
            for i, ok in enumerate(table_row_flags):
                if ok and start is None:
                    start = i
                if (not ok) and start is not None:
                    if i - start >= 3:
                        segs.append((start, i - 1))
                    start = None
            if start is not None and (len(table_row_flags) - start) >= 3:
                segs.append((start, len(table_row_flags) - 1))

            pad = int(max(6, 0.6 * h_med))
            for a, b in segs:
                xs0, ys0, xs1, ys1 = [], [], [], []
                for rid in range(a, b + 1):
                    for x0, y0, x1, y1, xc, t in rows[rid]:
                        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
                if xs0:
                    x0 = max(0, int(min(xs0)) - pad)
                    y0 = max(0, int(min(ys0)) - pad)
                    x1 = min(W, int(max(xs1)) + pad)
                    y1 = min(H, int(max(ys1)) + pad)
                    if (x1 - x0) >= 0.35 * W and (y1 - y0) >= 0.06 * H:
                        boxes.append((x0, y0, x1, y1))

    boxes = _merge_boxes(boxes, iou_thr=0.10, pad=6)
    return boxes


def extract_tables_from_lines(
    lines: List[OcrLine],
    table_boxes: List[Tuple[int, int, int, int]],
) -> List[Dict]:
    """
    Fix spill cột khi OCR gộp nhiều cell thành 1 line:
    - Nếu 1 line overlap >=2 cột + text có pattern nhiều cell -> split line thành segments
    - Ước lượng x-span cho từng segment -> gán cột theo overlap span
    """
    if not lines or not table_boxes:
        return []

    import pandas as pd

    line_items = []
    hs = []
    for ln in lines:
        b = get_bbox(ln)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        t = (ln.text or "").strip()
        if not t:
            continue
        h = max(1.0, y1 - y0)
        hs.append(h)
        line_items.append((ln, (x0, y0, x1, y1), (x0 + x1) / 2.0, (y0 + y1) / 2.0, t))

    h_med = median(hs) if hs else 14.0
    row_tol = max(10.0, 0.80 * h_med)
    col_tol = max(18.0, 2.2 * h_med)

    tables = []
    for tb in table_boxes:
        tbx0, tby0, tbx1, tby1 = tb
        tb_h = max(1, tby1 - tby0)

        # shrink để tránh dính text ngoài bảng
        pad_in = int(max(4, 0.55 * h_med))
        tb_inner = _make_inner_box(tb, pad_in=pad_in)

        cand = []
        for ln, b, xc, yc, t in line_items:
            ov = _overlap_ratio(b, tb)
            if ov < 0.18:
                continue
            if not _center_in_box(xc, yc, tb_inner):
                continue
            cand.append((ln, b, xc, yc, t))

        if len(cand) < 6:
            continue

        # cluster rows/cols
        ycs = [c[3] for c in cand]
        row_centers = _cluster_1d(ycs, tol=row_tol)
        if len(row_centers) < 2:
            continue

        xcs = [c[2] for c in cand]
        col_centers = _cluster_1d(xcs, tol=col_tol)
        if len(col_centers) < 2:
            continue

        n_rows = len(row_centers)
        n_cols = len(col_centers)

        row_spans = _centers_to_spans(row_centers, lo=float(tby0), hi=float(tby1))
        col_spans = _centers_to_spans(col_centers, lo=float(tbx0), hi=float(tbx1))

        grid = [[[] for _ in range(n_cols)] for _ in range(n_rows)]
        row_meta = [{"y0": 10**9, "y1": -1} for _ in range(n_rows)]

        for ln, b, xc, yc, t in cand:
            x0, y0, x1, y1 = b

            r = _assign_by_overlap_1d((float(y0), float(y1)), row_spans)

            # --- NEW: split line nếu nó span nhiều cột ---
            overlapped_cols = _count_overlapped_spans(float(x0), float(x1), col_spans, thr=0.18)
            parts = None
            if overlapped_cols >= 2:
                parts = _split_table_like_text(t)

            if parts and len(parts) >= 2:
                # giới hạn số part để không vỡ bảng nếu OCR noise
                if len(parts) > (n_cols + 2):
                    parts = parts[: (n_cols + 2)]

                seg_spans = _approx_segments_xspans(float(x0), float(x1), parts)
                for part, (sx0, sx1) in zip(parts, seg_spans):
                    if not part.strip():
                        continue
                    c = _assign_by_overlap_1d((sx0, sx1), col_spans)
                    grid[r][c].append((sx0, part.strip()))
            else:
                # normal assign (một cell)
                c = _assign_by_overlap_1d((float(x0), float(x1)), col_spans)
                grid[r][c].append((float(x0), t))

            row_meta[r]["y0"] = min(row_meta[r]["y0"], y0)
            row_meta[r]["y1"] = max(row_meta[r]["y1"], y1)

        # compose cell text
        mat = []
        for r in range(n_rows):
            row = []
            for c in range(n_cols):
                cell = grid[r][c]
                cell_sorted = sorted(cell, key=lambda z: z[0])
                text = " ".join([z[1] for z in cell_sorted]).strip()
                row.append(text)
            mat.append(row)

        # drop caption rows (1 cell dài sát mép)
        cleaned = []
        for r in range(n_rows):
            row = mat[r]
            non_empty = [i for i, v in enumerate(row) if (v or "").strip()]
            if len(non_empty) <= 1:
                long_txt = (row[non_empty[0]] if non_empty else "").strip()
                if len(long_txt) >= 22:
                    y0r, y1r = row_meta[r]["y0"], row_meta[r]["y1"]
                    near_top = (y0r - tby0) <= 0.10 * tb_h
                    near_bot = (tby1 - y1r) <= 0.10 * tb_h
                    if near_top or near_bot:
                        continue
            cleaned.append(row)

        if len(cleaned) < 2:
            continue

        df = pd.DataFrame(cleaned)

        # drop empty columns
        keep_cols = []
        for c in range(df.shape[1]):
            s = df.iloc[:, c].astype(str).str.strip()
            s = s.replace("nan", "")
            if s.str.len().sum() > 0:
                keep_cols.append(c)
        if len(keep_cols) >= 2:
            df = df.iloc[:, keep_cols]

        tables.append({
            "bbox": tb,
            "df": df,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
        })

    return tables
