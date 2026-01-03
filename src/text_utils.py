# src/text_utils.py
from __future__ import annotations
import re
import unicodedata
from typing import List

def normalize_vi(text: str) -> str:
    # Chuẩn hóa Unicode giúp giảm lỗi dấu (NFC)
    return unicodedata.normalize("NFC", text).strip()

_SENT_SPLIT = re.compile(r"(?<=[\.\?\!…])\s+")

def split_sentences_vi(text: str) -> List[str]:
    text = normalize_vi(text)
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    return sents
