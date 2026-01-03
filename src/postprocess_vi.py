# src/postprocess_vi.py
from __future__ import annotations

import os
import re
import json
import unicodedata
from dataclasses import is_dataclass, replace
from typing import Dict, List, Tuple, Optional, Iterable

# =========================
# Unicode + token helpers
# =========================

_WORD_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-/][0-9A-Za-zÀ-ỹĐđ]+)*", re.UNICODE)
_TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+(?:[-/][0-9A-Za-zÀ-ỹĐđ]+)*|\s+|.", re.UNICODE)

_VI_COMBINING_MARKS = {  # used by strip_accents
    "\u0300", "\u0301", "\u0303", "\u0309", "\u0323",  # tone: huyền sắc ngã hỏi nặng
    "\u0302", "\u0306", "\u031B",                      # â ă ơ/ư
}

def normalize_unicode(text: str) -> str:
    """NFC normalize + clean invisible/odd spaces."""
    if not text:
        return ""
    t = text
    # normalize weird spaces
    t = t.replace("\u00A0", " ")  # nbsp
    t = t.replace("\u200B", "")   # zero width space
    t = t.replace("\uFEFF", "")   # bom
    # normalize unicode composition
    t = unicodedata.normalize("NFC", t)
    # collapse spaces but keep newlines elsewhere handled by caller
    t = re.sub(r"[ \t]+", " ", t)
    return t

def strip_accents(s: str) -> str:
    """Remove Vietnamese diacritics; đ/Đ -> d/D."""
    if not s:
        return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    # NFD then remove combining marks
    nfd = unicodedata.normalize("NFD", s)
    out = []
    for ch in nfd:
        if unicodedata.combining(ch):
            continue
        # also remove specific marks if any left
        if ch in _VI_COMBINING_MARKS:
            continue
        out.append(ch)
    return unicodedata.normalize("NFC", "".join(out))

def _is_word(tok: str) -> bool:
    return bool(tok) and bool(_WORD_RE.fullmatch(tok))

def _case_style(words: List[str]) -> str:
    """Detect span casing: UPPER / TITLE / LOWER / MIXED."""
    ws = [w for w in words if w]
    if not ws:
        return "MIXED"
    if all(w.isupper() for w in ws):
        return "UPPER"
    if all(w.islower() for w in ws):
        return "LOWER"
    # simple title heuristic
    if all((w[:1].isupper() and w[1:].islower()) or w.isupper() for w in ws):
        return "TITLE"
    return "MIXED"

def _apply_case(word: str, style: str) -> str:
    if not word:
        return word
    if style == "UPPER":
        return word.upper()
    if style == "LOWER":
        return word.lower()
    if style == "TITLE":
        # keep acronyms as is
        if len(word) <= 4 and word.isupper():
            return word
        return word[:1].upper() + word[1:].lower()
    return word  # MIXED: keep canonical

# =========================
# Default gazetteer (small)
# You SHOULD extend using resources files
# =========================

_DEFAULT_PHRASES = [
    # admin/edu common
    "UBND QUẬN HOÀN KIẾM",
    "QUẬN HOÀN KIẾM",
    "HOÀN KIẾM",
    "SỞ GIÁO DỤC VÀ ĐÀO TẠO",
    "PHÒNG GIÁO DỤC VÀ ĐÀO TẠO",
    "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
    # subjects
    "NGỮ VĂN", "TIẾNG ANH", "TIN HỌC", "TOÁN", "VẬT LÍ", "HÓA HỌC", "SINH HỌC",
    "LỊCH SỬ", "ĐỊA LÍ", "GIÁO DỤC CÔNG DÂN", "CÔNG NGHỆ", "GIÁO DỤC THỂ CHẤT",
    "GIÁO DỤC QUỐC PHÒNG VÀ AN NINH",
    # question types
    "TRẮC NGHIỆM", "TỰ LUẬN", "ĐÚNG SAI", "GHÉP ĐÔI", "ĐIỀN KHUYẾT",
]

# Small word mapping fallback (you SHOULD provide a bigger one)
_DEFAULT_UNACCENT2BEST = {
    "quan": "quận",
    "uyban": "ủy ban",
    "ubnd": "ubnd",
    "hoan": "hoàn",
    "kiem": "kiếm",  # NOTE: ambiguous; phrase layer should decide; this is only fallback
    "giaoduc": "giáo dục",
    "daotao": "đào tạo",
    "tinhoc": "tin học",
    "tienganh": "tiếng anh",
    "nguvan": "ngữ văn",
}

# =========================
# OCR confusion rules
# =========================

def fix_ambiguous_chars(token: str) -> str:
    """
    Fix common OCR confusions:
    - O/0, I/1/l (contextual)
    - d/đ handled later via lexicon mapping
    """
    if not token:
        return token

    t = token

    # If token is numeric-like (mostly digits), map letter-lookalikes to digits
    digits = sum(ch.isdigit() for ch in t)
    letters = sum(ch.isalpha() for ch in t)
    if digits > 0:
        if digits >= max(1, letters):  # numeric-like
            t = (t.replace("O", "0").replace("o", "0")
                   .replace("I", "1").replace("l", "1"))
        else:
            # code-like or mixed: keep but fix obvious 0->O if surrounded by letters
            # ex: "H0A" -> "HOA"
            if letters >= 2:
                t = re.sub(r"(?<=\D)0(?=\D)", "O", t)

    # Normalize common punctuation forms inside tokens
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return t

# =========================
# Loading resources
# =========================

def _load_phrases_from_file(path: str) -> List[str]:
    out = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                out.append(s)
    except Exception:
        return []
    return out

def _load_unaccent2best_from_json(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}

def _load_unaccent2best_from_wordfreq(path: str) -> Dict[str, str]:
    """
    Optional: build mapping from a word frequency list file.
    Format supported:
      - "word<TAB>freq"
      - "word freq"
      - "word"
    Keeps the highest freq candidate for each unaccented form.
    """
    best: Dict[str, Tuple[float, str]] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = re.split(r"[\t ]+", s)
                w = parts[0].strip()
                if not w:
                    continue
                freq = 1.0
                if len(parts) >= 2:
                    try:
                        freq = float(parts[1])
                    except Exception:
                        freq = 1.0
                key = strip_accents(w.lower())
                prev = best.get(key)
                if (prev is None) or (freq > prev[0]):
                    best[key] = (freq, w)
    except Exception:
        return {}
    return {k: v for k, (sc, v) in best.items()}

def _resources_dir() -> str:
    # resources next to this file: src/resources/...
    return os.path.join(os.path.dirname(__file__), "resources")

def load_resources() -> Tuple[List[str], Dict[str, str]]:
    """
    Loads:
    - phrases: resources/vi_phrases.txt (one phrase per line)
    - phrases_edu: resources/vi_phrases_edu.txt
    - phrases_admin: resources/vi_phrases_admin.txt
    - unaccent2best: resources/unaccent2best.json OR resources/vi_words_freq.txt
    """
    rdir = _resources_dir()

    phrases = list(_DEFAULT_PHRASES)
    for fn in ["vi_phrases.txt", "vi_phrases_edu.txt", "vi_phrases_admin.txt"]:
        phrases.extend(_load_phrases_from_file(os.path.join(rdir, fn)))

    # dedup while preserving order
    seen = set()
    phrases2 = []
    for p in phrases:
        p2 = normalize_unicode(p).strip()
        if not p2:
            continue
        k = strip_accents(p2.lower())
        if k in seen:
            continue
        seen.add(k)
        phrases2.append(p2)

    # unaccent2best
    unaccent2best = dict(_DEFAULT_UNACCENT2BEST)
    j = _load_unaccent2best_from_json(os.path.join(rdir, "unaccent2best.json"))
    if j:
        unaccent2best.update(j)
    else:
        wf = _load_unaccent2best_from_wordfreq(os.path.join(rdir, "vi_words_freq.txt"))
        if wf:
            unaccent2best.update(wf)

    # normalize keys/values
    unaccent2best2 = {}
    for k, v in unaccent2best.items():
        kk = strip_accents(normalize_unicode(str(k)).lower())
        vv = normalize_unicode(str(v)).strip()
        if kk and vv:
            unaccent2best2[kk] = vv

    return phrases2, unaccent2best2

# Build phrase index by first unaccented token
def build_phrase_index(phrases: List[str]) -> Dict[str, List[List[str]]]:
    idx: Dict[str, List[List[str]]] = {}
    for ph in phrases:
        toks = [t for t in re.findall(_WORD_RE, normalize_unicode(ph)) if t]
        if not toks:
            continue
        base0 = strip_accents(toks[0].lower())
        if not base0:
            continue
        idx.setdefault(base0, []).append(toks)
    # prefer longer phrases first
    for k in idx:
        idx[k].sort(key=lambda xs: len(xs), reverse=True)
    return idx

# Cache resources in module
_PHRASES, _UNACCENT2BEST = load_resources()
_PHRASE_IDX = build_phrase_index(_PHRASES)

# =========================
# Phrase + word correction
# =========================

def apply_phrase_gazetteer(text: str, stats: Dict[str, int]) -> str:
    """
    Replace multi-word phrases using accent-insensitive matching on WORD tokens.
    Keeps original casing style across the matched span.
    """
    if not text:
        return text

    toks = re.findall(_TOKEN_RE, text)
    if not toks:
        return text

    # word positions
    word_pos = [i for i, t in enumerate(toks) if _is_word(t)]
    words = [toks[i] for i in word_pos]
    bases = [strip_accents(w.lower()) for w in words]

    i = 0
    while i < len(words):
        b0 = bases[i]
        cands = _PHRASE_IDX.get(b0)
        if not cands:
            i += 1
            continue

        matched = None
        matched_len = 0

        for ph_tokens in cands:
            L = len(ph_tokens)
            if i + L > len(words):
                continue
            ok = True
            for j in range(L):
                if strip_accents(ph_tokens[j].lower()) != bases[i + j]:
                    ok = False
                    break
            if ok:
                matched = ph_tokens
                matched_len = L
                break  # because we sorted by len desc

        if matched is None:
            i += 1
            continue

        # detect casing style from original span
        span_words = words[i : i + matched_len]
        style = _case_style(span_words)

        # apply replacement word-by-word
        for j in range(matched_len):
            w_new = _apply_case(matched[j], style)
            tok_idx = word_pos[i + j]
            toks[tok_idx] = w_new

        stats["phrase_replaced"] = stats.get("phrase_replaced", 0) + 1
        i += matched_len

    return "".join(toks)

def correct_word_token(tok: str, stats: Dict[str, int]) -> str:
    """
    Correct a single word token (no spaces), safe & conservative.
    """
    if not tok:
        return tok

    original = tok

    # 1) unicode normalize inside token
    tok = normalize_unicode(tok)

    # 2) fix ambiguous chars (O/0, I/1/l)
    tok2 = fix_ambiguous_chars(tok)
    if tok2 != tok:
        stats["char_fixed"] = stats.get("char_fixed", 0) + 1
        tok = tok2

    # 3) try lexicon-based diacritic correction
    # only attempt if alphabetic-ish (allow Đđ)
    if any(ch.isalpha() for ch in tok):
        base = strip_accents(tok.lower())
        cand = _UNACCENT2BEST.get(base)
        if cand:
            # preserve casing of the original token
            if tok.isupper():
                tok = cand.upper()
            elif tok.islower():
                tok = cand.lower()
            elif tok[:1].isupper() and tok[1:].islower():
                tok = _apply_case(cand, "TITLE")
            else:
                # mixed -> keep candidate as canonical (often best)
                tok = cand

            if tok != original:
                stats["word_corrected"] = stats.get("word_corrected", 0) + 1

    return tok

def postprocess_text(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Full postprocess for a text block (one OCR line).
    Order:
    - normalize unicode
    - phrase gazetteer (fix proper nouns / terms)
    - token-level corrections
    """
    stats: Dict[str, int] = {}

    if not text:
        return "", stats

    t = normalize_unicode(text)

    # Phrase-level first (fix HOÀN KIỂM -> HOÀN KIẾM if gazetteer contains HOÀN KIẾM)
    t2 = apply_phrase_gazetteer(t, stats)

    # Token-level scan (preserve whitespaces & punctuation)
    toks = re.findall(_TOKEN_RE, t2)
    out = []
    for tk in toks:
        if _is_word(tk):
            out.append(correct_word_token(tk, stats))
        else:
            # keep spaces/punct as-is
            out.append(tk)
    t3 = "".join(out)

    # normalize again to ensure NFC after replacements
    t3 = normalize_unicode(t3)
    return t3, stats

def _set_line_text(ln, new_text: str):
    """
    Robustly set ln.text even if dataclass/frozen.
    """
    if ln is None:
        return ln
    try:
        setattr(ln, "text", new_text)
        return ln
    except Exception:
        # dataclass replace
        try:
            if is_dataclass(ln):
                return replace(ln, text=new_text)
        except Exception:
            pass
    return ln  # fallback (cannot set)

def postprocess_lines(lines: List, ) -> Tuple[List, Dict[str, int]]:
    """
    Main entry used by app.py:
      lines, stats = postprocess_lines(lines)
    """
    lines = [ln for ln in (lines or []) if ln is not None]
    agg: Dict[str, int] = {"n_lines": len(lines)}

    out_lines = []
    for ln in lines:
        txt = getattr(ln, "text", "") or ""
        fixed, stt = postprocess_text(txt)
        for k, v in stt.items():
            agg[k] = agg.get(k, 0) + int(v)
        out_lines.append(_set_line_text(ln, fixed))

    return out_lines, agg
