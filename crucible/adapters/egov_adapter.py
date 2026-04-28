"""
e-Gov e-LAWS API adapter for the CRUCIBLE RAG expander.

Fetches Japanese statutory articles from the public e-Gov e-LAWS API v1
(https://elaws.e-gov.go.jp/api/1/) and returns well-formed Document objects.

Public API: fetch_egov_article(ref: ArticleRef) -> Document | None
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from crucible.schemas import ArticleRef, Document

_log = logging.getLogger("crucible.egov")

_BASE = "https://elaws.e-gov.go.jp/api/1"
_TIMEOUT = 10  # seconds per HTTP request

# Statute short-name → 法令番号 accepted by /lawdata/{lawId}
_LAW_NUM_MAP: dict[str, str] = {
    "民法": "明治二十九年法律第八十九号",
    "商法": "明治三十二年法律第四十八号",
    "会社法": "平成十七年法律第八十六号",
    "労働基準法": "昭和二十二年法律第四十九号",
    "労働契約法": "平成十九年法律第百二十八号",
    "個人情報の保護に関する法律": "平成十五年法律第五十七号",
    "個人情報保護法": "平成十五年法律第五十七号",
    "不正競争防止法": "平成五年法律第四十七号",
    "著作権法": "昭和四十五年法律第四十八号",
    "私的独占の禁止及び公正取引の確保に関する法律": "昭和二十二年法律第五十四号",
    "独占禁止法": "昭和二十二年法律第五十四号",
    "独禁法": "昭和二十二年法律第五十四号",
    "下請代金支払遅延等防止法": "昭和三十一年法律第百二十号",
    "下請法": "昭和三十一年法律第百二十号",
    "借地借家法": "平成三年法律第九十号",
    "消費者契約法": "平成十二年法律第六十一号",
    "特定商取引法": "昭和五十一年法律第五十七号",
    "製造物責任法": "平成六年法律第八十五号",
    "PL法": "平成六年法律第八十五号",
    "特許法": "昭和三十四年法律第百二十一号",
    "商標法": "昭和三十四年法律第百二十七号",
    "景品表示法": "昭和三十七年法律第百三十四号",
    "労働者派遣法": "昭和六十年法律第八十八号",
    "育児・介護休業法": "平成三年法律第七十六号",
    "民事訴訟法": "平成八年法律第百九号",
    "民事執行法": "昭和五十四年法律第四号",
    "民事保全法": "平成元年法律第九十一号",
    "破産法": "平成十六年法律第七十五号",
    "刑法": "明治四十年法律第四十五号",
    "憲法": "昭和二十一年憲法",
    "日本国憲法": "昭和二十一年憲法",
    "国家賠償法": "昭和二十二年法律第百二十五号",
    "国税通則法": "昭和三十七年法律第六十六号",
    "所得税法": "昭和四十年法律第三十三号",
    "法人税法": "昭和四十年法律第三十四号",
    "消費税法": "昭和六十三年法律第百八号",
    "行政手続法": "平成五年法律第八十八号",
    "行政事件訴訟法": "昭和三十七年法律第百三十九号",
    "金融商品取引法": "昭和二十三年法律第二十五号",
    "銀行法": "昭和五十六年法律第五十九号",
    "保険業法": "平成七年法律第百五号",
    "電子記録債権法": "平成十九年法律第百二号",
    "割賦販売法": "昭和三十六年法律第百五十九号",
    "宅地建物取引業法": "昭和二十七年法律第百七十六号",
    "建設業法": "昭和二十四年法律第百号",
    "農地法": "昭和二十七年法律第二百二十九号",
    # Additional common abbreviations / short names
    "個情法": "平成十五年法律第五十七号",
    "不競法": "平成五年法律第四十七号",
    "景表法": "昭和三十七年法律第百三十四号",
    "労基法": "昭和二十二年法律第四十九号",
    "労契法": "平成十九年法律第百二十八号",
    "派遣法": "昭和六十年法律第八十八号",
    "育介法": "平成三年法律第七十六号",
    "下請代金法": "昭和三十一年法律第百二十号",
}

# Short name / alias → canonical (full) statute name used in _LAW_NUM_MAP
_ALIAS_TO_CANONICAL: dict[str, str] = {
    "個人情報保護法": "個人情報の保護に関する法律",
    "独禁法": "独占禁止法",
    "PL法": "製造物責任法",
    "派遣法": "労働者派遣法",
    "労契法": "労働契約法",
    "労基法": "労働基準法",
    "不競法": "不正競争防止法",
    "景表法": "景品表示法",
    "下請代金法": "下請代金支払遅延等防止法",
    "個情法": "個人情報の保護に関する法律",
    "育介法": "育児・介護休業法",
}


def _normalize_statute_name(name: str) -> str:
    """Expand common abbreviations/aliases to canonical statute names."""
    return _ALIAS_TO_CANONICAL.get(name, name)


# Session-level caches
_law_id_cache: dict[str, str | None] = {}
_xml_cache: dict[str, bytes] = {}
_law_list_index: dict[str, str] = {}  # law_name → law_id, populated lazily
_law_list_loaded: bool = False


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_get(url: str) -> bytes:
    """GET with timeout; returns b'' on any network or HTTP error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CRUCIBLE/3.0 (legal-research)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return b""


# ---------------------------------------------------------------------------
# Law-ID resolution
# ---------------------------------------------------------------------------

def _resolve_law_id(statute: str) -> str | None:
    """Map a statute name to the law identifier used by /lawdata/."""
    # Normalize alias first
    statute = _normalize_statute_name(statute)

    if statute in _law_id_cache:
        return _law_id_cache[statute]

    # 1. Hardcoded table — covers common commercial law statutes
    law_num = _LAW_NUM_MAP.get(statute)
    if law_num:
        _law_id_cache[statute] = law_num
        return law_num

    # 2. Dynamic search via law list API
    _log.debug("_LAW_NUM_MAP miss for %r — trying law list search", statute)
    result = _search_law_list(statute)
    if result is None:
        _log.warning("FAIL:law_id_resolution statute=%r — not found in map or law list", statute)
    _law_id_cache[statute] = result
    return result


def _search_law_list(statute: str) -> str | None:
    """Search the Acts law list for a statute name; returns law_id or None."""
    global _law_list_loaded
    if not _law_list_loaded:
        _populate_law_list_index()
        _law_list_loaded = True

    # Exact match
    if statute in _law_list_index:
        return _law_list_index[statute]

    # Containment match: statute is a suffix/abbreviation of the full name
    for name, law_id in _law_list_index.items():
        if statute in name or name in statute:
            return law_id

    return None


def _populate_law_list_index() -> None:
    """Fetch the Acts law list (type 2) and index name → law_id."""
    url = f"{_BASE}/lawlists/2"
    data = _http_get(url)
    if not data:
        return
    try:
        root = ET.fromstring(data)
    except ET.ParseError:
        return
    for info in root.iter("LawNameListInfo"):
        name = (info.findtext("LawName") or "").strip()
        law_id = (info.findtext("LawId") or "").strip()
        if name and law_id:
            _law_list_index[name] = law_id


# ---------------------------------------------------------------------------
# Law XML fetching
# ---------------------------------------------------------------------------

def _fetch_law_xml(law_id: str) -> bytes:
    """Return raw law XML bytes, using a session cache."""
    if law_id in _xml_cache:
        return _xml_cache[law_id]
    encoded = urllib.parse.quote(law_id, safe="")
    url = f"{_BASE}/lawdata/{encoded}"
    data = _http_get(url)
    if data:
        _xml_cache[law_id] = data
    return data


# ---------------------------------------------------------------------------
# XML element finders
# ---------------------------------------------------------------------------

def _find_law_elem(root: ET.Element) -> ET.Element | None:
    """Navigate the DataRoot wrapper to the <Law> element."""
    # The API wraps: <DataRoot><ApplData><LawFullText><Law>...</Law>
    law = root.find(".//Law")
    if law is not None:
        return law
    return root if root.tag == "Law" else None


def _find_article_elem(law: ET.Element, article_num: int) -> ET.Element | None:
    for article in law.iter("Article"):
        num_str = article.get("Num", "")
        # Strip sub-article suffix (e.g. "415_2" → "415"); e-Gov uses underscore or hyphen
        base = num_str.split("_")[0].split("-")[0].strip()
        try:
            if int(base) == article_num:
                return article
        except ValueError:
            continue
    return None


def _find_paragraph_elem(article: ET.Element, paragraph_num: int) -> ET.Element | None:
    for para in article.findall("Paragraph"):
        try:
            if int(para.get("Num", "")) == paragraph_num:
                return para
        except ValueError:
            continue
    return None


def _find_item_elem(para: ET.Element, item_num: int) -> ET.Element | None:
    for item in para.findall(".//Item"):
        try:
            if int(item.get("Num", "")) == item_num:
                return item
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Text extraction from law XML elements
# ---------------------------------------------------------------------------

def _item_text(item: ET.Element) -> str:
    title_elem = item.find("ItemTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    sentence_elem = item.find("ItemSentence")
    body = "".join(sentence_elem.itertext()).strip() if sentence_elem is not None else ""

    if title and body:
        return f"　{title}　{body}"
    return f"　{title or body}"


def _paragraph_text(para: ET.Element) -> str:
    num_elem = para.find("ParagraphNum")
    num_str = "".join(num_elem.itertext()).strip() if num_elem is not None else ""

    sentence_elem = para.find("ParagraphSentence")
    body = "".join(sentence_elem.itertext()).strip() if sentence_elem is not None else ""

    header = f"{num_str}　" if num_str else ""
    lines = [f"{header}{body}"] if body else []

    for item in para.findall(".//Item"):
        lines.append(_item_text(item))

    return "\n".join(l for l in lines if l.strip())


def _article_text(article: ET.Element) -> str:
    lines: list[str] = []

    caption_elem = article.find("ArticleCaption")
    if caption_elem is not None:
        caption = "".join(caption_elem.itertext()).strip()
        if caption:
            lines.append(caption)

    for para in article.findall("Paragraph"):
        pt = _paragraph_text(para)
        if pt:
            lines.append(pt)

    return "\n".join(lines)


def _elem_text(elem: ET.Element) -> str:
    """Dispatch to the appropriate text extractor based on element tag."""
    tag = elem.tag
    if tag == "Article":
        return _article_text(elem)
    if tag == "Paragraph":
        return _paragraph_text(elem)
    if tag == "Item":
        return _item_text(elem)
    return "".join(elem.itertext()).strip()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_egov_article(ref: ArticleRef) -> Document | None:
    """Fetch a specific article from the e-Gov e-LAWS API.

    Maps ref.statute to a law ID, retrieves the full law XML (cached per
    session), and extracts the text for the requested article/paragraph/item.
    Returns None if the statute cannot be resolved or the article is absent.
    """
    _log.info("fetch_egov_article called: statute=%r article=%d", ref.statute, ref.article_num)

    law_id = _resolve_law_id(ref.statute)
    if not law_id:
        _log.warning("FAIL:law_id_resolution statute=%r — skipping fetch", ref.statute)
        return None

    raw = _fetch_law_xml(law_id)
    if not raw:
        _log.warning("FAIL:xml_fetch law_id=%r statute=%r — network/HTTP error", law_id, ref.statute)
        return None

    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        _log.warning("FAIL:xml_parse law_id=%r statute=%r", law_id, ref.statute)
        return None

    law_elem = _find_law_elem(root)
    if law_elem is None:
        _log.warning("FAIL:law_elem_not_found law_id=%r statute=%r", law_id, ref.statute)
        return None

    article_elem = _find_article_elem(law_elem, ref.article_num)
    if article_elem is None:
        _log.warning("FAIL:article_not_found statute=%r article=%d law_id=%r",
                     ref.statute, ref.article_num, law_id)
        return None

    # Narrow to paragraph / item if requested
    target = article_elem
    if ref.paragraph_num is not None:
        para_elem = _find_paragraph_elem(article_elem, ref.paragraph_num)
        if para_elem is not None:
            target = para_elem
            if ref.item_num is not None:
                item_elem = _find_item_elem(para_elem, ref.item_num)
                if item_elem is not None:
                    target = item_elem

    text = _elem_text(target)
    if not text:
        return None

    _log.info("OK:fetch statute=%r article=%d (law_id=%r)", ref.statute, ref.article_num, law_id)
    return Document(
        canonical_ref=ref.canonical,
        text=text,
        statute=ref.statute,
        article_num=ref.article_num,
        expansion_metadata={
            "source": "egov_elaws_api_v1",
            "law_id": law_id,
            "paragraph_num": ref.paragraph_num,
            "item_num": ref.item_num,
        },
    )


def resolve_unknown_statutes_batch(
    statute_names: list[str],
    llm_adapter: Any,
) -> None:
    """Batch-resolve unknown statute names to official names via LLM, then cache.

    For statute names that cannot be resolved through the normal lookup path
    (_LAW_NUM_MAP + law list API), sends a single LLM call to get the official
    name and law number.  Resolved entries are written into _LAW_NUM_MAP so
    subsequent fetch_egov_article calls can find them.  Always safe to call;
    never raises — resolution is best-effort.
    """
    if not statute_names:
        return

    # Only attempt LLM resolution for names that have no known mapping yet
    unknown = [n for n in statute_names if not _LAW_NUM_MAP.get(_normalize_statute_name(n))]
    if not unknown:
        return

    _log.info("resolve_unknown_statutes_batch: %d unknown names: %r", len(unknown), unknown)

    prompt = (
        "Given these Japanese statute name variants, return the official name (正式名称) "
        "as used in e-Gov and the law number (法令番号) for each. "
        "If truly unknown, set official and law_num to null.\n\n"
        f"Input: {json.dumps(unknown, ensure_ascii=False)}\n\n"
        "Output: JSON array only, exactly this format: "
        '[{"input": "...", "official": "...", "law_num": "..."}]'
    )
    try:
        from crucible.mpsr import strip_markdown_fences_for_json
        text = llm_adapter.call_text("", prompt, max_tokens=1024)
        cleaned = strip_markdown_fences_for_json(text)
        results = json.loads(cleaned)
        for item in results:
            if not isinstance(item, dict):
                continue
            original = item.get("input", "")
            official = item.get("official")
            law_num = item.get("law_num")
            if original and official and law_num:
                _LAW_NUM_MAP[official] = law_num
                _LAW_NUM_MAP[original] = law_num
                if original != official:
                    _ALIAS_TO_CANONICAL[original] = official
                _log.info("batch_resolved: %r → official=%r law_num=%r", original, official, law_num)
    except Exception as exc:
        _log.warning("resolve_unknown_statutes_batch failed: %r", exc)
