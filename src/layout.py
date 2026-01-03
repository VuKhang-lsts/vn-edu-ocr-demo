# src/layout.py
from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from statistics import median

from .ocr_engine import OcrLine


def get_bbox(ln: OcrLine) -> Optional[Tuple[float, float, float, float]]:
    """
    Trả bbox (x0,y0,x1,y1) từ ln.bbox hoặc ln.quad.
    """
    if hasattr(ln, "bbox") and ln.bbox is not None:
        b = ln.bbox
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x0, y0, x1, y1 = b
            return float(x0), float(y0), float(x1), float(y1)

    if hasattr(ln, "quad") and ln.quad is not None:
        q = ln.quad
        # quad: [(x,y),...]
        xs = [p[0] for p in q]
        ys = [p[1] for p in q]
        return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))
    return None


def _kmeans_1d_2clusters(xs: List[float], iters: int = 12) -> Optional[Tuple[float, float]]:
    """
    1D kmeans (k=2). Trả (c0,c1) sorted nếu hợp lệ.
    """
    if len(xs) < 8:
        return None
    xs2 = sorted(xs)
    c0 = xs2[len(xs2) // 4]
    c1 = xs2[(3 * len(xs2)) // 4]
    if c0 == c1:
        return None

    for _ in range(iters):
        g0, g1 = [], []
        for x in xs2:
            if abs(x - c0) <= abs(x - c1):
                g0.append(x)
            else:
                g1.append(x)
        if not g0 or not g1:
            return None
        nc0 = sum(g0) / len(g0)
        nc1 = sum(g1) / len(g1)
        if abs(nc0 - c0) < 1e-3 and abs(nc1 - c1) < 1e-3:
            break
        c0, c1 = nc0, nc1

    if c0 > c1:
        c0, c1 = c1, c0
    return c0, c1


def reorder_lines_reading_order(
    lines: List[OcrLine],
    page_w: int,
    mode: str = "auto",               # auto | single | two
    two_col_order: str = "column",    # column | row
    fullwidth_thr: float = 0.78,      # >= 78% page => full-width/spanning
    min_sep_ratio: float = 0.22,      # khoảng cách 2 cụm x-center đủ lớn
    min_cluster_ratio: float = 0.18,  # mỗi cột phải >= 18% số dòng
) -> List[OcrLine]:
    """
    Reading order cho tài liệu giáo dục:
    - Single: y rồi x
    - Two columns:
        * column: đọc cột trái từ trên xuống hết -> cột phải (chuẩn SGK)
          nhưng vẫn giữ dòng full-width đúng theo y bằng segment theo spanning lines.
        * row: y rồi (col,left->right) (hợp cho bố cục “song song theo hàng”)
    """
    if not lines:
        return lines

    # Collect geometry
    items = []
    hs = []
    xcenters = []
    for i, ln in enumerate(lines):
        b = get_bbox(ln)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        items.append({
            "i": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "w": w, "h": h, "xc": xc, "yc": yc,
            "is_full": (w >= fullwidth_thr * page_w),
        })
        hs.append(h)
        if not (w >= fullwidth_thr * page_w):
            xcenters.append(xc)

    # Nếu thiếu bbox quá nhiều -> giữ nguyên
    if len(items) < max(3, len(lines) // 5):
        return lines

    h_med = median(hs) if hs else 14.0
    y_bucket = max(8.0, 0.65 * h_med)

    # Decide split for columns
    split_x = None
    if mode == "single":
        split_x = None
    else:
        # auto/two: thử detect 2 cột
        if mode == "two":
            # ép two-columns: dùng kmeans nếu có, fallback median
            km = _kmeans_1d_2clusters(xcenters)
            if km:
                c0, c1 = km
                split_x = (c0 + c1) / 2.0
            elif xcenters:
                xs2 = sorted(xcenters)
                split_x = xs2[len(xs2)//2]
        else:
            # auto
            km = _kmeans_1d_2clusters(xcenters)
            if km:
                c0, c1 = km
                sep = abs(c1 - c0)
                # kiểm tra separation đủ lớn
                if sep >= min_sep_ratio * page_w:
                    split_x = (c0 + c1) / 2.0

    # Assign col_id
    for it in items:
        if it["is_full"] or split_x is None:
            it["col"] = -1 if it["is_full"] else 0
        else:
            it["col"] = 0 if it["xc"] < split_x else 1

    # Nếu split_x tồn tại nhưng cột mất cân bằng quá -> về single
    if split_x is not None:
        n0 = sum(1 for it in items if it["col"] == 0)
        n1 = sum(1 for it in items if it["col"] == 1)
        tot = max(1, n0 + n1)
        if (n0 / tot) < min_cluster_ratio or (n1 / tot) < min_cluster_ratio:
            for it in items:
                it["col"] = -1 if it["is_full"] else 0
            split_x = None

    def yk(it) -> int:
        return int(round(it["yc"] / y_bucket))

    # -------- Case: SINGLE (or no split) --------
    if split_x is None:
        items_sorted = sorted(items, key=lambda it: (yk(it), it["x0"]))
        return [lines[it["i"]] for it in items_sorted]

    # -------- Case: TWO COLUMNS --------
    # Group spanning lines (full-width) by y_bucket
    spans = [it for it in items if it["col"] == -1]
    nonsp = [it for it in items if it["col"] in (0, 1)]

    # Nếu user chọn row-wise (y then left->right)
    if two_col_order == "row":
        col_rank = {-1: 0, 0: 1, 1: 2}
        items_sorted = sorted(items, key=lambda it: (yk(it), col_rank[it["col"]], it["x0"]))
        return [lines[it["i"]] for it in items_sorted]

    # Column-wise nhưng giữ spanning đúng vị trí theo y bằng segment
    spans_sorted = sorted(spans, key=lambda it: (yk(it), it["x0"]))

    # Gom spans có cùng yk thành group
    span_groups = []
    for sp in spans_sorted:
        if not span_groups or yk(span_groups[-1][0]) != yk(sp):
            span_groups.append([sp])
        else:
            span_groups[-1].append(sp)

    # Segment boundaries theo yk của span group
    # Mỗi segment: (prev_yk, curr_yk) => lấy nonsp trong khoảng đó
    out: List[OcrLine] = []
    prev = -10**9
    used = set()

    def emit_segment(y_from: int, y_to: int):
        seg = [it for it in nonsp if (y_from <= yk(it) < y_to)]
        if not seg:
            return
        left = sorted([it for it in seg if it["col"] == 0], key=lambda it: (yk(it), it["x0"]))
        right = sorted([it for it in seg if it["col"] == 1], key=lambda it: (yk(it), it["x0"]))
        for it in left + right:
            if it["i"] not in used:
                used.add(it["i"])
                out.append(lines[it["i"]])

    for grp in span_groups:
        curr = yk(grp[0])
        emit_segment(prev, curr)
        # emit spans in this y band (order by x)
        grp_sorted = sorted(grp, key=lambda it: it["x0"])
        for it in grp_sorted:
            if it["i"] not in used:
                used.add(it["i"])
                out.append(lines[it["i"]])
        prev = curr + 1  # tránh dính cùng y band

    # tail segment
    emit_segment(prev, 10**9)

    # nếu còn nonsp nào chưa emit (do rounding), emit theo left+right toàn cục
    remaining = [it for it in nonsp if it["i"] not in used]
    if remaining:
        left = sorted([it for it in remaining if it["col"] == 0], key=lambda it: (yk(it), it["x0"]))
        right = sorted([it for it in remaining if it["col"] == 1], key=lambda it: (yk(it), it["x0"]))
        for it in left + right:
            out.append(lines[it["i"]])

    return out
