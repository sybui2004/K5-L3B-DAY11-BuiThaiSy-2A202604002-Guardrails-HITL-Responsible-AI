"""
Secret / obfuscation scanner dùng chung cho input + output guardrails (không cần NeMo).

Ý tưởng: kẻ tấn công không cần secret xuất hiện NGUYÊN VĂN - chỉ cần người đọc
khôi phục được. Vì vậy ta sinh nhiều "biến thể giải mã" của văn bản rồi so với
dấu vân tay (fingerprint) của từng secret:

  - NFKC + bỏ ký tự vô hình + map homoglyph (Cyrillic/Greek) + bỏ dấu tiếng Việt
  - chỉ giữ chữ-số  (a-d-m-i-n 1 2 3, tách dòng, chèn ký tự)
  - leetspeak skeleton (4dm1n123, @dmin123)
  - số đọc bằng chữ (admin one two three / một hai ba), bảng chữ NATO
  - viết ngược, ROT13
  - giải base64 / hex / percent-encoding các token dài
  - acrostic: chữ cái đầu mỗi dòng / mỗi từ
  - nhiều mảnh của cùng một secret (admin ... 123, secret ... 2024, internal ... 5432)
"""
from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from urllib.parse import unquote

try:  # giá trị thật lấy từ data/protected/vinbank_secrets.json
    from core.config import DEMO_SECRETS as _CONFIG_SECRETS
except Exception:  # noqa: BLE001
    _CONFIG_SECRETS = []

_FALLBACK_SECRETS = ["admin123", "sk-vinbank-secret-2024", "db.vinbank.internal:5432"]

INVISIBLE = "​‌‍⁠﻿­᠎‎‏‪‫‬‭‮"

# Homoglyph thường gặp (Cyrillic / Greek) -> Latin
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "к": "k", "м": "m", "т": "t", "в": "b", "н": "h", "ԁ": "d",
    "А": "a", "Е": "e", "О": "o", "Р": "p", "С": "c", "Х": "x", "І": "i", "К": "k",
    "М": "m", "Т": "t", "В": "b", "Н": "h",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ρ": "p", "τ": "t",
    "υ": "u", "χ": "x",
})

# Leetspeak skeleton - áp dụng cho CẢ secret lẫn văn bản nên vẫn khớp
_LEET = str.maketrans({"0": "o", "1": "i", "l": "i", "!": "i", "|": "i", "3": "e",
                       "4": "a", "@": "a", "5": "s", "$": "s", "7": "t", "+": "t"})

_NUMBER_WORDS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "khong": "0", "mot": "1", "hai": "2", "ba": "3", "bon": "4", "tu": "4",
    "nam": "5", "lam": "5", "sau": "6", "bay": "7", "tam": "8", "chin": "9",
}
_NATO = {
    "alpha": "a", "alfa": "a", "bravo": "b", "charlie": "c", "delta": "d", "echo": "e",
    "foxtrot": "f", "golf": "g", "hotel": "h", "india": "i", "juliet": "j", "juliett": "j",
    "kilo": "k", "lima": "l", "mike": "m", "november": "n", "oscar": "o", "papa": "p",
    "quebec": "q", "romeo": "r", "sierra": "s", "tango": "t", "uniform": "u",
    "victor": "v", "whiskey": "w", "whisky": "w", "xray": "x", "yankee": "y", "zulu": "z",
    "dash": "", "hyphen": "", "dot": "", "colon": "",
}

# Từ không đặc trưng (tên ngân hàng, tiền tố) - không dùng làm "mảnh" secret
_COMMON_TOKENS = {"vinbank", "db", "sk", "com", "www"}


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(c for c in text if unicodedata.category(c) != "Mn")


def canonicalize(text: str) -> str:
    """NFKC + bỏ ký tự vô hình + homoglyph -> Latin + gộp khoảng trắng (giữ dấu)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(str.maketrans("", "", INVISIBLE)).translate(_CONFUSABLES)
    return re.sub(r"\s+", " ", text).strip()


def fold(text: str) -> str:
    """canonicalize + lowercase + bỏ dấu tiếng Việt."""
    return strip_accents(canonicalize(text)).casefold()


def alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text)


def _secret_values() -> list[str]:
    values = [str(s) for s in (_CONFIG_SECRETS or []) if s]
    return values or list(_FALLBACK_SECRETS)


def _build_fingerprints():
    full, skeleton, fragments = set(), set(), []
    for value in _secret_values():
        squashed = alnum(fold(value))
        if len(squashed) < 5:
            continue
        full.add(squashed)
        skeleton.add(squashed.translate(_LEET))
        # Biến thể bỏ port / hậu tố số: db.vinbank.internal
        no_port = alnum(fold(value.split(":")[0]))
        if len(no_port) >= 8:
            full.add(no_port)
            skeleton.add(no_port.translate(_LEET))
        tokens = [t for t in re.findall(r"[a-z]+|\d+", fold(value)) if t not in _COMMON_TOKENS]
        if len(tokens) >= 2:
            fragments.append(tokens)
    # Mảnh đặc trưng không phụ thuộc phần còn lại
    full |= {"vinbanksecret", "vinbankinternal", "skvinbank"}
    return full, skeleton, fragments


SECRET_FULL, SECRET_SKELETON, SECRET_FRAGMENTS = _build_fingerprints()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


def _decode_tokens(text: str) -> list[str]:
    """Giải base64 / hex / percent-encoding cho các token dài."""
    out = []
    if "%" in text:
        out.append(unquote(text))
    for tok in re.findall(r"[A-Za-z0-9+/_\-=]{8,}", text):
        padded = tok + "=" * (-len(tok) % 4)
        for decoder in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                out.append(decoder(padded).decode("utf-8", "ignore"))
            except (binascii.Error, ValueError):
                pass
    for tok in re.findall(r"(?:[0-9a-fA-F]{2}[\s:]?){6,}", text):
        try:
            out.append(bytes.fromhex(re.sub(r"[\s:]", "", tok)).decode("utf-8", "ignore"))
        except ValueError:
            pass
    return out


def text_variants(text: str) -> list[str]:
    """Các chuỗi chữ-số (lowercase) mà người đọc có thể khôi phục từ ``text``."""
    base = fold(text)
    words = _words(base)
    squashed = alnum(base)
    spoken = "".join(_NUMBER_WORDS.get(w, _NATO.get(w, w)) for w in words)
    lines = [ln.strip() for ln in fold(text).splitlines() if ln.strip()]
    acrostic_lines = "".join(alnum(ln)[:1] for ln in lines)
    acrostic_words = "".join(w[:1] for w in words)
    variants = [
        squashed,
        spoken,
        squashed[::-1],
        codecs.encode(squashed, "rot13"),
        acrostic_lines,
        acrostic_words,
    ]
    for decoded in _decode_tokens(canonicalize(text)):
        variants.append(alnum(fold(decoded)))
    return [v for v in variants if v]


def find_secret_leaks(text: str) -> list[str]:
    """Trả danh sách lý do nếu ``text`` chứa (hoặc khôi phục được) một secret."""
    if not text:
        return []
    reasons = []
    for variant in text_variants(text):
        skel = variant.translate(_LEET)
        if any(fp in variant for fp in SECRET_FULL) or any(fp in skel for fp in SECRET_SKELETON):
            reasons.append("secret_obfuscated")
            break
    # Nhiều mảnh của cùng một secret xuất hiện rải rác (vd "admin" ... "123")
    words = set(_words(fold(text)))
    for tokens in SECRET_FRAGMENTS:
        if all(t in words for t in tokens):
            reasons.append("secret_fragments")
            break
    if re.search(r"\bsk\s*[-_]\s*[a-z0-9]{3,}", fold(text)):
        reasons.append("api_key_shape")
    if re.search(r"\b[a-z0-9-]+\.(internal|local|corp|lan)\b", fold(text)):
        reasons.append("internal_host_shape")
    return reasons
