# src/postprocess.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re
import unicodedata
from dataclasses import is_dataclass, replace

from .edu_lexicon import (
    EDU_WORDS,
    MANUAL_DIACRITIC_MAP,
    PHRASE_KEYS,
    remove_diacritics,
    norm_spaces,
)


_ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]


def _normalize_unicode(s: str) -> str:
    s = s or ""
    for z in _ZERO_WIDTH:
        s = s.replace(z, "")
    s = unicodedata.normalize("NFC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _is_mostly_numberlike(tok: str) -> bool:
    """
    token kiểu số: '21/05/2025', '120', '8%', '[159;162)', 'x=20'...
    """
    t = tok.strip()
    if not t:
        return False
    if re.fullmatch(r"[\d\.\,\%\:/\-\[\]\(\);]+", t):
        return True
    # chứa >= 2 chữ số => ưu tiên coi là số
    return sum(ch.isdigit() for ch in t) >= 2


def _fix_confusions(tok: str) -> str:
    """
    Rules sửa nhầm phổ biến:
    - O/0 trong token số
    - l/1/I trong token số
    - d/đ trong 1 số từ rất đặc thù (không sửa bừa)
    """
    t = tok

    if _is_mostly_numberlike(t):
        # O -> 0, o -> 0
        t = re.sub(r"[Oo]", "0", t)
        # I, l -> 1
        t = re.sub(r"[Il]", "1", t)

    return t


def _tokenize_keep_punct(s: str) -> List[str]:
    """
    Tách token nhưng giữ dấu câu đơn giản.
    """
    # ví dụ: "[159;162)." "21/05/2025" "8%" vẫn là 1 token
    return re.findall(r"\[\s*\d+\s*;\s*\d+\s*\)|\d{1,2}/\d{1,2}/\d{2,4}|\d+(?:[.,]\d+)?%?|[A-Za-zÀ-ỹĐđ]+|[^\s]", s)


def _maybe_restore_diacritics(tok: str, strict: float) -> str:
    """
    Khôi phục dấu cực kỳ bảo thủ:
    - chỉ áp dụng cho token chữ cái (không lẫn số)
    - dùng map thủ công (ít mơ hồ) + ưu tiên ngữ cảnh giáo dục
    """
    t = tok.strip()
    if not t:
        return tok

    # Không động vào token có số hoặc token toàn caps dài (UBND/THCS...)
    if any(ch.isdigit() for ch in t):
        return tok
    if t.isupper() and len(t) >= 3:
        return tok

    key = remove_diacritics(t).lower()

    # map thủ công (bảo thủ)
    if key in MANUAL_DIACRITIC_MAP:
        cand = MANUAL_DIACRITIC_MAP[key]
        # strict cao -> chỉ sửa nếu token đang "mất dấu hoàn toàn"
        if strict >= 0.70:
            if remove_diacritics(t).lower() == key and t.lower() == key:
                return cand if t[0].isupper() else cand.lower()
            return tok
        else:
            return cand if t[0].isupper() else cand.lower()

    return tok


def _apply_phrase_correction(line: str, strict: float) -> str:
    """
    Sửa phrase-level cho các dòng ngắn (header) bằng so khớp không dấu.
    Nếu line ngắn và rất giống phrase trong EDU_PHRASES -> thay toàn dòng.
    """
    s = norm_spaces(line)
    if not s:
        return s

    # chỉ áp dụng cho dòng ngắn (header) để tránh "đổi bừa"
    if len(s) > 55:
        return s

    s_key = remove_diacritics(s).lower()

    # match exact key
    for k, phrase in PHRASE_KEYS:
        if s_key == k:
            return phrase

    # fuzzy nhẹ: chỉ khi strict thấp hơn và độ giống rất cao
    if strict <= 0.70:
        # so khớp kiểu "contains" với các phrase rất đặc thù
        for k, phrase in PHRASE_KEYS:
            if k and k in s_key and len(k) >= 10:
                return phrase

    return s


def postprocess_text(text: str, strict: float = 0.75, enable_phrase: bool = True, enable_token: bool = True) -> Tuple[str, Dict]:
    """
    Hậu xử lý 1 đoạn text:
    - Unicode NFC + remove zero-width
    - Phrase correction (header)
    - Token correction: O/0, l/1/I, và restore dấu cực bảo thủ bằng edu lexicon
    """
    debug: Dict = {"changed": False, "steps": []}
    before = text or ""
    s = _normalize_unicode(before)

    if enable_phrase:
        s2 = _apply_phrase_correction(s, strict=strict)
        if s2 != s:
            debug["steps"].append({"phrase": {"before": s, "after": s2}})
        s = s2

    if enable_token:
        toks = _tokenize_keep_punct(s)
        new_toks: List[str] = []
        for tok in toks:
            t0 = tok
            t1 = _fix_confusions(t0)
            t2 = _maybe_restore_diacritics(t1, strict=strict)

            # edu word guard: nếu token chữ thường mà gần giống edu word (không dấu) thì cho sửa
            # (đã bảo thủ ở _maybe_restore_diacritics)
            new_toks.append(t2)

        s2 = " ".join(new_toks)
        # clean spaces around punct
        s2 = re.sub(r"\s+([,\.\;\:\)\]\}])", r"\1", s2)
        s2 = re.sub(r"([\(\[\{])\s+", r"\1", s2)
        s2 = re.sub(r"\s+", " ", s2).strip()

        if s2 != s:
            debug["steps"].append({"token": {"before": s, "after": s2}})
        s = s2

    debug["changed"] = (s != before)
    return s, debug


def _set_line_text(line_obj: Any, new_text: str) -> Any:
    """
    Hỗ trợ nhiều kiểu OcrLine:
    - dataclass: replace
    - object: setattr
    """
    if line_obj is None:
        return line_obj

    # dataclass
    try:
        if is_dataclass(line_obj):
            return replace(line_obj, text=new_text)
    except Exception:
        pass

    # normal object
    try:
        setattr(line_obj, "text", new_text)
        return line_obj
    except Exception:
        # fallback: trả về nguyên (không sửa)
        return line_obj


def postprocess_lines(
    lines: List[Any],
    strict: float = 0.75,
    enable_phrase: bool = True,
    enable_token: bool = True,
    max_debug_samples: int = 12,
) -> Tuple[List[Any], Dict]:
    """
    Hậu xử lý list OCR lines:
    - sửa line.text
    - trả debug: số dòng sửa + samples before/after
    """
    if not lines:
        return lines, {"changed": 0, "samples": []}

    changed = 0
    samples = []

    out_lines: List[Any] = []
    for ln in lines:
        txt = getattr(ln, "text", "") or ""
        new_txt, dbg = postprocess_text(txt, strict=strict, enable_phrase=enable_phrase, enable_token=enable_token)

        if new_txt != txt:
            changed += 1
            if len(samples) < max_debug_samples:
                token_changes = []
                # chỉ log ngắn: line-level before/after
                samples.append(
                    {
                        "before": txt,
                        "after": new_txt,
                        "token_changes": token_changes,
                    }
                )

        out_lines.append(_set_line_text(ln, new_txt))

    debug = {
        "changed": changed,
        "samples": samples,
        "strict": strict,
        "enable_phrase": enable_phrase,
        "enable_token": enable_token,
    }
    return out_lines, debug
