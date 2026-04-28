from __future__ import annotations

import re
import unicodedata
from collections import OrderedDict
from typing import Callable


_KANJI_DIGITS = {
    "零": 0, "〇": 0, "○": 0,
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}
_KANJI_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}


def _kanji_to_arabic(text: str) -> str:
    def parse_kanji_number(s: str) -> int:
        result = 0
        current = 0
        for ch in s:
            if ch in _KANJI_DIGITS:
                current = _KANJI_DIGITS[ch]
            elif ch in _KANJI_UNITS:
                unit = _KANJI_UNITS[ch]
                if current == 0:
                    current = 1
                result += current * unit
                current = 0
        result += current
        return result

    pattern = re.compile(r"[零〇○一二三四五六七八九十百千万]+")
    return pattern.sub(lambda m: str(parse_kanji_number(m.group())), text)


def _normalize_japan(citation: str) -> str:
    s = unicodedata.normalize("NFKC", citation)
    s = re.sub(r"\s+", "", s)
    s = _kanji_to_arabic(s)
    s = re.sub(r"(?<![第之\d])(\d+)条", r"第\1条", s)
    return s


def _normalize_us(citation: str) -> str:
    s = citation.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\b(Section|Sec\.|Sect\.|§)\s*", "§", s, flags=re.IGNORECASE)
    s = re.sub(r"\busc\b", "U.S.C.", s, flags=re.IGNORECASE)
    return s


def _normalize_eu(citation: str) -> str:
    s = unicodedata.normalize("NFKC", citation).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\bArt\.\s*", "Article ", s, flags=re.IGNORECASE)
    return s


def _normalize_china(citation: str) -> str:
    s = unicodedata.normalize("NFKC", citation)
    s = re.sub(r"\s+", "", s)
    s = _kanji_to_arabic(s)
    s = re.sub(r"(?<![第\d])(\d+)条", r"第\1条", s)
    return s


def _normalize_default(citation: str) -> str:
    s = unicodedata.normalize("NFKC", citation).strip()
    s = re.sub(r"\s+", " ", s)
    return s


NORMALIZATION_RULES: dict[str, Callable[[str], str]] = {
    "Japan": _normalize_japan,
    "United States": _normalize_us,
    "European Union": _normalize_eu,
    "China": _normalize_china,
    "default": _normalize_default,
}


def normalize_citation(citation: str, jurisdiction: str) -> str:
    rule = NORMALIZATION_RULES.get(jurisdiction, NORMALIZATION_RULES["default"])
    return rule(citation)


def dedupe_citations(
    citations: list[str],
    jurisdiction: str,
) -> tuple[list[str], dict[str, list[str]]]:
    seen: OrderedDict[str, None] = OrderedDict()
    back_map: dict[str, list[str]] = {}
    for original in citations:
        canonical = normalize_citation(original, jurisdiction)
        if canonical not in seen:
            seen[canonical] = None
        back_map.setdefault(canonical, []).append(original)
    return list(seen.keys()), back_map
