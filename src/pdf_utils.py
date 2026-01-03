# src/pdf_utils.py
from __future__ import annotations
import io
from typing import List
from PIL import Image

import pymupdf  # PyMuPDF (newer name). "fitz" vẫn thường dùng, nhưng pymupdf là tên mới.  :contentReference[oaicite:7]{index=7}


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200, max_pages: int = 10) -> List[Image.Image]:
    """
    Convert PDF bytes -> list PIL Images.
    dpi càng cao thì OCR càng tốt nhưng sẽ nặng hơn.
    """
    images: List[Image.Image] = []
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    page_count = min(len(doc), max_pages)
    zoom = dpi / 72.0  # PyMuPDF uses 72 dpi base
    mat = pymupdf.Matrix(zoom, zoom)

    for i in range(page_count):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        images.append(img)

    return images
