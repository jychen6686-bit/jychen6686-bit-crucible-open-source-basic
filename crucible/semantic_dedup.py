from __future__ import annotations

import math
from typing import Any, Callable

from crucible.enums import ConsensusLevel
from crucible.schemas import CharterEntry


_SEMANTIC_DEDUP_CATEGORIES = frozenset({
    "doctrine_theme",
    "mandatory_clause",
    "risk_domain",
    "jurisdictional_trap",
})


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _cluster_by_similarity(
    labels: list[str],
    embeddings: list[list[float]],
    threshold: float,
) -> list[list[int]]:
    """Single-linkage clustering via Union-Find. O(n²), fine for n < 50."""
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for i in range(n):
        for j in range(i + 1, n):
            if _cosine_similarity(embeddings[i], embeddings[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    return list(clusters.values())


def compute_consensus_level(raised_by_count: int, panel_size: int) -> ConsensusLevel:
    if raised_by_count == panel_size:
        return ConsensusLevel.UNANIMOUS
    if raised_by_count > panel_size / 2:
        return ConsensusLevel.MAJORITY
    return ConsensusLevel.MINORITY


def semantic_dedup_charter_entries(
    entries: list[CharterEntry],
    panel_size: int,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    similarity_threshold: float = 0.85,
) -> tuple[list[CharterEntry], list[dict[str, Any]]]:
    """Post-merge semantic deduplication. If embed_fn is None, returns entries unchanged."""
    if embed_fn is None:
        return entries, []

    merge_log: list[dict[str, Any]] = []

    eligible: list[CharterEntry] = []
    pass_through: list[CharterEntry] = []
    for entry in entries:
        if entry.type in _SEMANTIC_DEDUP_CATEGORIES:
            eligible.append(entry)
        else:
            pass_through.append(entry)

    if len(eligible) <= 1:
        return entries, []

    by_category: dict[str, list[CharterEntry]] = {}
    for entry in eligible:
        by_category.setdefault(entry.type, []).append(entry)

    deduped_eligible: list[CharterEntry] = []

    for category, cat_entries in by_category.items():
        if len(cat_entries) <= 1:
            deduped_eligible.extend(cat_entries)
            continue

        refs = [e.canonical_reference for e in cat_entries]

        try:
            embeddings = embed_fn(refs)
        except Exception as exc:
            merge_log.append({
                "category": category,
                "action": "embed_failed",
                "error": str(exc)[:200],
                "fallback": "kept_all_entries",
            })
            deduped_eligible.extend(cat_entries)
            continue

        if len(embeddings) != len(refs):
            deduped_eligible.extend(cat_entries)
            continue

        clusters = _cluster_by_similarity(refs, embeddings, similarity_threshold)

        for cluster_indices in clusters:
            if len(cluster_indices) == 1:
                deduped_eligible.append(cat_entries[cluster_indices[0]])
                continue

            cluster_entries = [cat_entries[i] for i in cluster_indices]
            representative = max(
                cluster_entries, key=lambda e: len(e.canonical_reference)
            )
            total_raised = min(
                sum(e.raised_by_count for e in cluster_entries),
                panel_size,
            )
            new_consensus = compute_consensus_level(total_raised, panel_size)

            merged_entry = CharterEntry(
                type=category,
                canonical_reference=representative.canonical_reference,
                payload=representative.payload,
                consensus_level=new_consensus,
                raised_by_count=total_raised,
                panel_size=panel_size,
            )
            deduped_eligible.append(merged_entry)

            merge_log.append({
                "category": category,
                "action": "semantic_merge",
                "merged_refs": [e.canonical_reference for e in cluster_entries],
                "representative": representative.canonical_reference,
                "old_consensus_levels": [e.consensus_level.value for e in cluster_entries],
                "new_consensus": new_consensus.value,
                "total_raised_by": total_raised,
            })

    result = pass_through + deduped_eligible
    return result, merge_log
