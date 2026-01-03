# src/edu_lexicon.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import unicodedata
import re


def remove_diacritics(s: str) -> str:
    """
    Bỏ dấu tiếng Việt: "Thời gian làm bài" -> "Thoi gian lam bai"
    """
    s = s or ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s)


def norm_spaces(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


# =========================
# Phrase-level lexicon (header / common patterns)
# =========================
EDU_PHRASES: List[str] = [
    "UBND",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO",
    "PHÒNG GIÁO DỤC VÀ ĐÀO TẠO",
    "TRƯỜNG THCS",
    "TRƯỜNG THPT",
    "ĐỀ KHẢO SÁT",
    "ĐỀ THI",
    "ĐỀ CHÍNH THỨC",
    "(Đề thi gồm 02 trang)",
    "Môn:",
    "Ngày kiểm tra:",
    "Thời gian làm bài:",
    "phút",
    "Bài 1",
    "Bài 2",
    "Bài 3",
    "Câu 1",
    "Câu 2",
    "Câu 3",
    "điểm",
    "Xét biến cố",
    "Tính xác suất",
    "Cho hai biểu thức",
    "Chứng minh",
    "Tìm tất cả giá trị của x",
    "Bảng tần số",
    "Tần số",
    "Tần số tương đối",
    "Số học sinh",
    "Chiều cao",
    "Số chấm xuất hiện",
]


# =========================
# Word-level lexicon (for token corrections)
# - dùng để sửa dấu nhẹ và sửa nhầm d/đ, l/1/I, O/0...
# =========================
EDU_WORDS: Set[str] = {
    # hành chính / trường
    "UBND", "QUẬN", "HUYỆN", "PHƯỜNG", "XÃ",
    "TRƯỜNG", "THCS", "THPT",

    # cấu trúc đề
    "ĐỀ", "KHẢO", "SÁT", "CHÍNH", "THỨC", "Môn", "MÔN",
    "Ngày", "kiểm", "tra", "Thời", "gian", "làm", "bài",
    "phút", "trang",

    # toán / thống kê
    "Bài", "Câu", "điểm", "tần", "số", "tương", "đối",
    "bảng", "nhóm", "chiều", "cao", "học", "sinh",
    "xúc", "xắc", "đồng", "chất", "biến", "cố", "xác", "suất",
    "cho", "hai", "biểu", "thức", "chứng", "minh", "tìm", "giá", "trị",
}


# =========================
# Diacriticless -> canonical (very conservative)
# - chỉ map những từ "ít mơ hồ" trong ngữ cảnh đề thi
# =========================
MANUAL_DIACRITIC_MAP: Dict[str, str] = {
    "mon": "Môn",
    "ngay": "Ngày",
    "kiem": "kiểm",
    "tra": "tra",
    "thoi": "Thời",
    "gian": "gian",
    "lam": "làm",
    "bai": "bài",
    "bai1": "Bài 1",
    "bai2": "Bài 2",
    "bai3": "Bài 3",
    "cau": "Câu",
    "diem": "điểm",
    "tan": "tần",
    "so": "số",
    "tuong": "tương",
    "doi": "đối",
    "xac": "xác",
    "suat": "suất",
    "xuc": "xúc",
    "xac": "xắc",
    "dong": "đồng",
    "chat": "chất",
    "chung": "chứng",
    "minh": "minh",
    "gia": "giá",
    "tri": "trị",
}


def build_phrase_keys(phrases: List[str]) -> List[Tuple[str, str]]:
    """
    Trả về list (key_khong_dau_lower, phrase_goc)
    """
    out: List[Tuple[str, str]] = []
    for p in phrases:
        k = remove_diacritics(norm_spaces(p)).lower()
        out.append((k, p))
    return out


PHRASE_KEYS: List[Tuple[str, str]] = build_phrase_keys(EDU_PHRASES)
