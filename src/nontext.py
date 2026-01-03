# src/nontext.py
from __future__ import annotations
from typing import List, Optional, Tuple
import re
from statistics import median

from .ocr_engine import OcrLine
from .layout import get_bbox  # dùng lại bbox extractor


def _has_alnum_vi(s: str) -> bool:
    # Có chữ/số (unicode) => coi là text hợp lệ
    return bool(re.search(r"[0-9A-Za-zÀ-ỹà-ỹ]", s or ""))


def _is_obvious_noise(s: str) -> bool:
    """
    Chỉ coi là rác khi cực kỳ chắc chắn:
    - toàn dấu/punct kéo dài: "-----", "____", "......", "|||||"
    - quá ngắn và không có chữ/số
    """
    s0 = (s or "").strip()
    if not s0:
        return True
    if _has_alnum_vi(s0):
        return False

    # chuỗi lặp 1 ký tự/punct dài
    if len(s0) >= 6 and len(set(s0)) <= 2:
        return True

    # toàn punctuation / ký tự trang trí
    if re.fullmatch(r"[\W_]+", s0) and len(s0) >= 4:
        return True

    # rất ngắn và không alnum
    if len(s0) <= 2 and not _has_alnum_vi(s0):
        return True

    return False


def _intersect_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def _overlap_ratio(inner_box, outer_box) -> float:
    # inner_box overlap with outer_box by inner area
    ax0, ay0, ax1, ay1 = inner_box
    a_area = max(1.0, (ax1 - ax0) * (ay1 - ay0))
    inter = _intersect_area(inner_box, outer_box)
    return float(inter / a_area)


def filter_nontext_lines(
    lines: List[OcrLine],
    page_w: int,
    page_h: int,
    keep_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
    keep_overlap_thr: float = 0.20,
) -> List[OcrLine]:
    """
    Lọc non-text theo hướng "đừng làm rơi ký tự".
    - Giữ gần như tất cả dòng có chữ/số.
    - Chỉ loại những dòng cực chắc là rác.
    - Nếu dòng nằm trong keep_boxes (vùng bảng) => luôn giữ.
    """
    if not lines:
        return lines

    # median height để nhận biết bbox bất thường
    hs = []
    bxs = []
    for ln in lines:
        b = get_bbox(ln)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        hs.append(max(1.0, y1 - y0))
        bxs.append(b)

    h_med = median(hs) if hs else 14.0

    out: List[OcrLine] = []
    for ln in lines:
        txt = (ln.text or "").strip()

        b = get_bbox(ln)
        if b is None:
            # không có bbox => đừng lọc, giữ
            out.append(ln)
            continue

        x0, y0, x1, y1 = b
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        area = bw * bh
        page_area = max(1.0, page_w * page_h)

        # 1) nếu thuộc vùng bảng => luôn giữ (để không mất số/điểm)
        if keep_boxes:
            if any(_overlap_ratio(b, kb) >= keep_overlap_thr for kb in keep_boxes):
                out.append(ln)
                continue

        # 2) có chữ/số => luôn giữ (kể cả ngắn: "8%", "120", "(1)")
        if _has_alnum_vi(txt):
            out.append(ln)
            continue

        # 3) chỉ loại khi cực chắc rác
        if _is_obvious_noise(txt):
            continue

        # 4) loại bbox quá lớn mà text không alnum (thường là vùng hình bị OCR nhầm)
        #    (conservative: chỉ khi ch
