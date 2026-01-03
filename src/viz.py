# src/viz.py
from __future__ import annotations
from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .ocr_engine import OcrLine


def draw_quads(image: Image.Image, lines: List[OcrLine]) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    for i, ln in enumerate(lines, start=1):
        draw.polygon(ln.quad, outline=(31, 119, 180), width=2)
        x0, y0 = ln.quad[0]
        draw.text((x0, y0 - 14), f"{i}", fill=(31, 119, 180))
    return out
