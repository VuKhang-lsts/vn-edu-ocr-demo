# src/viz_debug.py
from __future__ import annotations
from typing import List, Any, Tuple, Optional
from PIL import Image, ImageDraw

from src.layout import get_bbox

def draw_numbered_boxes(img: Image.Image, lines: List[Any], max_n: int = 200) -> Image.Image:
    im = img.copy()
    draw = ImageDraw.Draw(im)
    for k, ln in enumerate(lines[:max_n]):
        bb = get_bbox(ln)
        if not bb:
            continue
        x0,y0,x1,y1 = bb
        x0,y0,x1,y1 = map(int, [x0,y0,x1,y1])
        draw.rectangle([x0,y0,x1,y1], width=2)
        draw.text((x0+2, max(0,y0-12)), str(k), fill=(255,0,0))
    return im
