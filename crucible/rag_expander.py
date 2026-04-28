from __future__ import annotations

import re
from collections import deque
from typing import Callable

from crucible.enums import _TokenType
from crucible.schemas import ArticleRef, Document, ExpansionResult


# ===========================================================================
# Canonical ref parser
# ===========================================================================

_CANONICAL_REF_PATTERN = re.compile(
    r"^(?P<statute>[^\d第]*?)第(?P<article>\d+)条"
    r"(?:第(?P<paragraph>\d+)項)?"
    r"(?:第(?P<item>\d+)号)?$"
)


def parse_canonical_ref(canonical: str) -> ArticleRef | None:
    """Inverse of ArticleRef.canonical. Returns None on parse failure."""
    m = _CANONICAL_REF_PATTERN.match(canonical.strip())
    if not m:
        return None
    return ArticleRef(
        statute=m.group("statute") or "",
        article_num=int(m.group("article")),
        paragraph_num=int(m.group("paragraph")) if m.group("paragraph") else None,
        item_num=int(m.group("item")) if m.group("item") else None,
    )


# ===========================================================================
# Japan tokenizer
# ===========================================================================

from dataclasses import dataclass, field as dc_field


@dataclass
class _Token:
    type: _TokenType
    text: str
    start: int
    end: int
    article_num: int | None = None
    paragraph_num: int | None = None
    item_num: int | None = None
    statute: str | None = None
    relative_kind: str | None = None
    negated: bool = False


_DEFAULT_JP_STATUTE_NAMES: tuple[str, ...] = (
    "民法", "商法", "会社法", "民事訴訟法", "民事執行法",
    "刑法", "刑事訴訟法", "労働基準法", "労働契約法",
    "独占禁止法", "私的独占の禁止及び公正取引の確保に関する法律",
    "個人情報保護法", "個人情報の保護に関する法律",
    "金融商品取引法", "下請法", "下請代金支払遅延等防止法",
    "著作権法", "特許法", "商標法",
)


def _build_japan_lexer_rules(
    statute_names: tuple[str, ...],
) -> list[tuple[re.Pattern, _TokenType, Callable[[re.Match], dict] | None]]:
    sorted_statutes = sorted(statute_names, key=len, reverse=True)
    statute_alt = "|".join(re.escape(s) for s in sorted_statutes)

    rules: list[tuple[re.Pattern, _TokenType, Callable[[re.Match], dict] | None]] = []

    rules.append((
        re.compile(statute_alt),
        _TokenType.STATUTE_NAME,
        lambda m: {"statute": m.group(0)},
    ))

    rules.append((
        re.compile(
            r"第(\d+)条"
            r"(?:第(\d+)項)?"
            r"(?:第(\d+)号)?"
        ),
        _TokenType.ARTICLE_REF,
        lambda m: {
            "article_num": int(m.group(1)),
            "paragraph_num": int(m.group(2)) if m.group(2) else None,
            "item_num": int(m.group(3)) if m.group(3) else None,
        },
    ))

    rules.append((re.compile(r"前条"), _TokenType.RELATIVE_REF,
                  lambda m: {"relative_kind": "prev_article"}))
    rules.append((re.compile(r"次条"), _TokenType.RELATIVE_REF,
                  lambda m: {"relative_kind": "next_article"}))
    rules.append((re.compile(r"同条"), _TokenType.RELATIVE_REF,
                  lambda m: {"relative_kind": "same_article"}))
    rules.append((re.compile(r"前項"), _TokenType.RELATIVE_REF,
                  lambda m: {"relative_kind": "prev_para"}))
    rules.append((re.compile(r"次項"), _TokenType.RELATIVE_REF,
                  lambda m: {"relative_kind": "next_para"}))
    rules.append((re.compile(r"同法"), _TokenType.SAME_STATUTE, None))

    rules.append((re.compile(r"から"), _TokenType.RANGE_FROM, None))
    rules.append((re.compile(r"まで"), _TokenType.RANGE_TO, None))

    # Negated verbs BEFORE positive verbs — critical ordering
    for verb in ("準用されない", "準用しない", "適用されない", "適用しない"):
        rules.append((
            re.compile(re.escape(verb)),
            _TokenType.APPLY_VERB,
            lambda m: {"negated": True},
        ))
    for verb in ("準用する", "準用し", "適用する", "適用し"):
        rules.append((re.compile(re.escape(verb)), _TokenType.APPLY_VERB, None))

    for prep in ("にかかわらず", "を除き", "を除く"):
        rules.append((re.compile(re.escape(prep)), _TokenType.EXCLUDE_PREP, None))

    for joiner in ("及び", "並びに", "若しくは", "又は"):
        rules.append((re.compile(re.escape(joiner)), _TokenType.LIST_JOINER, None))

    rules.append((re.compile(r"。"), _TokenType.SENTENCE_END, None))
    rules.append((re.compile(r"、"), _TokenType.COMMA, None))

    return rules


def _tokenize_japan(
    text: str,
    statute_names: tuple[str, ...] = _DEFAULT_JP_STATUTE_NAMES,
) -> list[_Token]:
    """Produce a token stream for Japanese statute text."""
    rules = _build_japan_lexer_rules(statute_names)
    tokens: list[_Token] = []
    pos = 0
    other_start: int | None = None

    def flush_other(end: int) -> None:
        nonlocal other_start
        if other_start is not None and end > other_start:
            tokens.append(_Token(
                type=_TokenType.OTHER,
                text=text[other_start:end],
                start=other_start,
                end=end,
            ))
        other_start = None

    while pos < len(text):
        best_match: tuple[int, _TokenType, Callable | None, re.Match] | None = None
        for pattern, ttype, extractor in rules:
            m = pattern.match(text, pos)
            if m is None:
                continue
            match_len = m.end() - m.start()
            if best_match is None or match_len > best_match[0]:
                best_match = (match_len, ttype, extractor, m)

        if best_match is None:
            if other_start is None:
                other_start = pos
            pos += 1
            continue

        flush_other(pos)
        match_len, ttype, extractor, m = best_match
        payload = extractor(m) if extractor else {}
        tokens.append(_Token(
            type=ttype,
            text=m.group(0),
            start=m.start(),
            end=m.end(),
            **payload,
        ))
        pos = m.end()

    flush_other(len(text))
    return tokens


# ===========================================================================
# Japan scope resolver
# ===========================================================================

_GROUP_CONTINUATIONS = frozenset({
    _TokenType.ARTICLE_REF,
    _TokenType.RELATIVE_REF,
    _TokenType.LIST_JOINER,
    _TokenType.COMMA,
    _TokenType.RANGE_FROM,
    _TokenType.RANGE_TO,
    _TokenType.STATUTE_NAME,
    _TokenType.SAME_STATUTE,
})

_GROUP_STARTERS = frozenset({
    _TokenType.ARTICLE_REF,
    _TokenType.RELATIVE_REF,
    _TokenType.STATUTE_NAME,
    _TokenType.SAME_STATUTE,
})


def _split_sentences(tokens: list[_Token]) -> list[list[_Token]]:
    """Split a token stream into sentence-level sublists on SENTENCE_END."""
    sentences: list[list[_Token]] = []
    current: list[_Token] = []
    for t in tokens:
        current.append(t)
        if t.type == _TokenType.SENTENCE_END:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def _find_reference_groups(sentence: list[_Token]) -> list[tuple[int, int]]:
    """Find contiguous reference groups within a sentence."""
    groups: list[tuple[int, int]] = []
    i = 0
    n = len(sentence)
    while i < n:
        if sentence[i].type not in _GROUP_STARTERS:
            i += 1
            continue
        start = i
        last_ref_idx = i if sentence[i].type in (
            _TokenType.ARTICLE_REF, _TokenType.RELATIVE_REF
        ) else -1
        j = i + 1
        while j < n and sentence[j].type in _GROUP_CONTINUATIONS:
            if sentence[j].type in (_TokenType.ARTICLE_REF, _TokenType.RELATIVE_REF):
                last_ref_idx = j
            j += 1
        if last_ref_idx >= 0:
            groups.append((start, last_ref_idx))
        i = j if j > i else i + 1
    return groups


def _find_governing_verb(
    sentence: list[_Token],
    group_end: int,
) -> tuple[str | None, int] | None:
    """Walk forward from group_end+1 to find the governing verb."""
    n = len(sentence)
    i = group_end + 1
    while i < n:
        t = sentence[i]
        if t.type == _TokenType.APPLY_VERB:
            return ("exclude" if t.negated else "apply"), i
        if t.type == _TokenType.EXCLUDE_PREP:
            return "exclude", i
        if t.type == _TokenType.SENTENCE_END:
            return None
        i += 1
    return None


def _expand_group_to_refs(
    sentence: list[_Token],
    group_start: int,
    group_end: int,
    default_statute: str,
) -> list[tuple[ArticleRef, int]]:
    """Convert a reference group (token span) into a list of (ArticleRef, token_idx)."""
    refs: list[tuple[ArticleRef, int]] = []
    current_statute = default_statute

    i = group_start
    while i <= group_end:
        t = sentence[i]

        if t.type == _TokenType.STATUTE_NAME:
            current_statute = t.statute or current_statute
            i += 1
            continue

        if t.type == _TokenType.SAME_STATUTE:
            current_statute = "<SAME_STATUTE>"
            i += 1
            continue

        if t.type == _TokenType.ARTICLE_REF:
            if (i + 2 <= group_end
                    and sentence[i + 1].type == _TokenType.RANGE_FROM
                    and sentence[i + 2].type == _TokenType.ARTICLE_REF):
                start_num = t.article_num
                end_num = sentence[i + 2].article_num
                assert start_num is not None and end_num is not None
                if start_num <= end_num:
                    for n in range(start_num, end_num + 1):
                        refs.append((
                            ArticleRef(statute=current_statute, article_num=n),
                            i,
                        ))
                i += 3
                if i <= group_end and sentence[i].type == _TokenType.RANGE_TO:
                    i += 1
                continue

            assert t.article_num is not None
            refs.append((
                ArticleRef(
                    statute=current_statute,
                    article_num=t.article_num,
                    paragraph_num=t.paragraph_num,
                    item_num=t.item_num,
                ),
                i,
            ))
            i += 1
            continue

        if t.type == _TokenType.RELATIVE_REF:
            refs.append((
                ArticleRef(
                    statute=f"<RELATIVE:{t.relative_kind}>",
                    article_num=0,
                ),
                i,
            ))
            i += 1
            continue

        i += 1

    return refs


def _resolve_sentence(
    sentence: list[_Token],
    default_statute: str,
) -> list[ArticleRef]:
    """Resolve a single sentence to its list of apply-scope ArticleRefs."""
    resolved: list[ArticleRef] = []
    groups = _find_reference_groups(sentence)

    for start, end in groups:
        gv = _find_governing_verb(sentence, end)
        if gv is None:
            continue
        scope, _ = gv
        if scope != "apply":
            continue
        refs = _expand_group_to_refs(sentence, start, end, default_statute)
        for ref, _token_idx in refs:
            resolved.append(ref)

    return resolved


def _resolve_relative_and_same_statute(
    refs: list[ArticleRef],
    referrer_statute: str,
    referrer_article_num: int,
    document_statute_context: str | None,
) -> list[ArticleRef]:
    """Second pass: convert RELATIVE sentinels into real article numbers."""
    out: list[ArticleRef] = []
    fallback_statute = document_statute_context or referrer_statute

    for ref in refs:
        if ref.statute.startswith("<RELATIVE:"):
            kind = ref.statute[len("<RELATIVE:"):-1]
            if kind == "prev_article" and referrer_article_num > 1:
                out.append(ArticleRef(
                    statute=referrer_statute,
                    article_num=referrer_article_num - 1,
                ))
            elif kind == "next_article":
                out.append(ArticleRef(
                    statute=referrer_statute,
                    article_num=referrer_article_num + 1,
                ))
            elif kind == "same_article":
                out.append(ArticleRef(
                    statute=referrer_statute,
                    article_num=referrer_article_num,
                ))
            continue

        if ref.statute == "<SAME_STATUTE>":
            out.append(ref.with_statute(fallback_statute))
            continue

        if ref.statute == "":
            out.append(ref.with_statute(referrer_statute))
            continue

        out.append(ref)

    return out


# ===========================================================================
# Public dispatcher
# ===========================================================================

def extract_referenced_articles(
    text: str,
    jurisdiction: str,
    referrer_canonical: str,
    statute_names: tuple[str, ...] | None = None,
) -> list[ArticleRef]:
    """Extract apply-scope ArticleRefs from article text. Japan only; other jurisdictions return []."""
    if jurisdiction != "Japan":
        return []

    referrer = parse_canonical_ref(referrer_canonical)
    if referrer is None:
        return []

    statutes = statute_names or _DEFAULT_JP_STATUTE_NAMES
    tokens = _tokenize_japan(text, statute_names=statutes)
    sentences = _split_sentences(tokens)

    # v3.0 (P2-A): 同法 resolved per-sentence, falls back to document-level only when
    # current sentence has no STATUTE_NAME tokens.
    document_statute_context: str | None = None
    raw_refs: list[ArticleRef] = []

    for sentence in sentences:
        sentence_statutes: list[str] = [
            t.statute for t in sentence
            if t.type == _TokenType.STATUTE_NAME and t.statute
        ]

        if sentence_statutes:
            document_statute_context = sentence_statutes[-1]

        sentence_refs = _resolve_sentence(sentence, default_statute="")

        sentence_context = sentence_statutes[-1] if sentence_statutes else None
        effective_context = (
            sentence_context
            or document_statute_context
            or referrer.statute
        )

        resolved_sentence_refs: list[ArticleRef] = []
        for ref in sentence_refs:
            if ref.statute == "<SAME_STATUTE>":
                resolved_sentence_refs.append(ref.with_statute(effective_context))
            else:
                resolved_sentence_refs.append(ref)

        raw_refs.extend(resolved_sentence_refs)

    return _resolve_relative_and_same_statute(
        raw_refs,
        referrer_statute=referrer.statute,
        referrer_article_num=referrer.article_num,
        document_statute_context=document_statute_context,
    )


# ===========================================================================
# BFS expander
# ===========================================================================

def expand_references_bfs(
    initial_doc: Document,
    jurisdiction: str,
    article_fetcher: Callable[[ArticleRef], Document | None],
    max_depth: int = 3,
    max_total_articles: int = 8,
    max_total_chars: int = 4000,
    statute_names: tuple[str, ...] | None = None,
) -> ExpansionResult:
    """Walk the reference graph from initial_doc using BFS with three independent caps."""
    visited: set[str] = {initial_doc.canonical_ref}
    queue: deque[tuple[Document, int]] = deque([(initial_doc, 0)])
    expansions: list[Document] = []
    total_chars = 0
    truncated = False
    max_depth_reached = 0
    skipped_cycle = 0
    skipped_depth = 0
    skipped_article_cap = 0
    skipped_char_cap = 0

    while queue:
        doc, depth = queue.popleft()
        max_depth_reached = max(max_depth_reached, depth)

        if depth >= max_depth:
            skipped_depth += 1
            continue

        refs = extract_referenced_articles(
            doc.text,
            jurisdiction,
            referrer_canonical=doc.canonical_ref,
            statute_names=statute_names,
        )

        for ref in refs:
            if ref.canonical in visited:
                skipped_cycle += 1
                continue

            if len(expansions) >= max_total_articles:
                skipped_article_cap += 1
                truncated = True
                continue

            expanded = article_fetcher(ref)
            if expanded is None:
                visited.add(ref.canonical)
                continue

            if total_chars + len(expanded.text) > max_total_chars:
                skipped_char_cap += 1
                truncated = True
                visited.add(ref.canonical)
                continue

            visited.add(ref.canonical)
            expansions.append(expanded)
            total_chars += len(expanded.text)
            queue.append((expanded, depth + 1))

    if jurisdiction == "Japan":
        inlined_text = _format_expansion_japan(initial_doc, expansions, truncated)
    else:
        inlined_text = initial_doc.text

    final_doc = Document(
        canonical_ref=initial_doc.canonical_ref,
        text=inlined_text,
        statute=initial_doc.statute,
        article_num=initial_doc.article_num,
        expansion_metadata={
            "expanded_count": len(expansions),
            "truncated": truncated,
            "max_depth_reached": max_depth_reached,
            "expanded_canonicals": [e.canonical_ref for e in expansions],
            "skipped_for_cycle": skipped_cycle,
            "skipped_for_depth": skipped_depth,
            "skipped_for_article_cap": skipped_article_cap,
            "skipped_for_char_cap": skipped_char_cap,
        },
    )

    return ExpansionResult(
        document=final_doc,
        expanded_count=len(expansions),
        truncated=truncated,
        max_depth_reached=max_depth_reached,
        expanded_canonicals=[e.canonical_ref for e in expansions],
        skipped_for_cycle=skipped_cycle,
        skipped_for_depth=skipped_depth,
        skipped_for_article_cap=skipped_article_cap,
        skipped_for_char_cap=skipped_char_cap,
    )


def _format_expansion_japan(
    initial_doc: Document,
    expansions: list[Document],
    truncated: bool,
) -> str:
    """Inline the expansions into the initial document as an appendix."""
    if not expansions:
        return initial_doc.text

    lines: list[str] = [initial_doc.text.rstrip(), "", "── 参照条文 ──"]
    for ex in expansions:
        lines.append("")
        lines.append(f"【{ex.canonical_ref}】")
        lines.append(ex.text.rstrip())
    if truncated:
        lines.append("")
        lines.append("（参照条文の一部は上限により省略されました。）")
    return "\n".join(lines)


def expand_document_dict(
    doc_dict: dict,
    jurisdiction: str,
    article_fetcher: Callable[[ArticleRef], Document | None],
    max_depth: int = 3,
    max_total_articles: int = 8,
    max_total_chars: int = 4000,
    statute_names: tuple[str, ...] | None = None,
) -> dict:
    """Convenience wrapper for host rag_query handler."""
    initial = Document(
        canonical_ref=doc_dict["canonical_ref"],
        text=doc_dict["text"],
        statute=doc_dict.get("statute", ""),
        article_num=doc_dict.get("article_num", 0),
    )
    result = expand_references_bfs(
        initial_doc=initial,
        jurisdiction=jurisdiction,
        article_fetcher=article_fetcher,
        max_depth=max_depth,
        max_total_articles=max_total_articles,
        max_total_chars=max_total_chars,
        statute_names=statute_names,
    )
    return {
        **doc_dict,
        "text": result.document.text,
        "expansion_metadata": result.document.expansion_metadata,
    }
