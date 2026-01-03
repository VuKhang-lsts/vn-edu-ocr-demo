# src/ocr_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from paddleocr import TextDetection
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

Point = Tuple[float, float]
Quad = List[Point]


@dataclass
class OcrLine:
    quad: Quad
    text: str


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def crop_quad(image_bgr: np.ndarray, quad: Quad, pad: int = 2) -> np.ndarray:
    pts = np.array(quad, dtype="float32")
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)

    maxW = max(10, int(max(wA, wB)) + 2 * pad)
    maxH = max(10, int(max(hA, hB)) + 2 * pad)

    dst = np.array(
        [[pad, pad], [maxW - pad, pad], [maxW - pad, maxH - pad], [pad, maxH - pad]],
        dtype="float32",
    )

    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image_bgr, M, (maxW, maxH))


def sort_lines_reading_order(lines: List[OcrLine], y_thresh: int = 12) -> List[OcrLine]:
    def key_xy(line: OcrLine):
        xs = [p[0] for p in line.quad]
        ys = [p[1] for p in line.quad]
        return (min(ys), min(xs))

    lines_sorted = sorted(lines, key=key_xy)

    grouped: List[List[OcrLine]] = []
    for ln in lines_sorted:
        y = min(p[1] for p in ln.quad)
        if not grouped:
            grouped.append([ln])
            continue
        last_y = min(p[1] for p in grouped[-1][0].quad)
        if abs(y - last_y) <= y_thresh:
            grouped[-1].append(ln)
        else:
            grouped.append([ln])

    final: List[OcrLine] = []
    for g in grouped:
        g2 = sorted(g, key=lambda l: min(p[0] for p in l.quad))
        final.extend(g2)
    return final


class OCREngine:
    def __init__(self, device: str = "cpu"):
        # ✅ Indent đúng: mọi dòng bên dưới thuộc __init__ đều thụt vào 4 spaces
        self.detector = TextDetection(model_name="PP-OCRv5_mobile_det")

        cfg = Cfg.load_config_from_name("vgg_transformer")
        cfg["device"] = device
        self.recognizer = Predictor(cfg)

    def detect_quads(self, image_rgb: np.ndarray) -> List[Quad]:
        """
        PaddleOCR 3.x: TextDetection.predict trả về generator.
        Mỗi res có dt_polys là danh sách bbox 4 điểm. :contentReference[oaicite:0]{index=0}
        """
        quads: List[Quad] = []
        output = self.detector.predict(input=image_rgb, batch_size=1)

        for res in output:  # res là dict cho 1 ảnh
            dt_polys = res.get("dt_polys", [])
            for poly in dt_polys:
                poly_np = np.array(poly, dtype="float32")
                if poly_np.shape[0] < 4:
                    continue
                quad = [(float(p[0]), float(p[1])) for p in poly_np[:4]]
                quads.append(quad)

        return quads

    def recognize(self, image_bgr: np.ndarray, quads: List[Quad]) -> List[OcrLine]:
        lines: List[OcrLine] = []
        for quad in quads:
            crop = crop_quad(image_bgr, quad)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)
            text = (self.recognizer.predict(pil) or "").strip()
            if text:
                lines.append(OcrLine(quad=quad, text=text))
        return sort_lines_reading_order(lines)

    def run(self, image_rgb: np.ndarray) -> List[OcrLine]:
        quads = self.detect_quads(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        return self.recognize(image_bgr, quads)
