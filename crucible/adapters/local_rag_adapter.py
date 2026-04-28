"""
Local treatise RAG adapter interface for the CRUCIBLE pipeline.

Defines the Protocol that any local-corpus RAG backend must satisfy to be used
as an article_fetcher / treatise query source in the pipeline.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from crucible.schemas import Document


@runtime_checkable
class LocalTreatiseRagAdapter(Protocol):
    """Protocol for local authoritative legal treatise retrieval.

    Implementations back this with a vector store, BM25 index, or hybrid
    retrieval over a corpus of 基本書/体系書 (foundational civil law treatises).
    The pipeline calls query_treatises to seed the RAG context passed to
    framework, red_team, and verdict subagents.
    """

    def query_treatises(
        self,
        keywords: list[str],
        max_results: int,
    ) -> list[Document]:
        """Retrieve treatise passages relevant to the given keywords.

        Args:
            keywords: Search terms derived from MpsrRagSeed entries whose
                layer == "treatise". Typically statute names, legal concepts,
                or academic doctrine labels in Japanese or English.
            max_results: Upper bound on returned documents. Implementations
                should return at most this many; fewer is acceptable.

        Returns:
            List of Document objects, each representing a passage from an
            authoritative legal treatise. Documents must have non-empty
            canonical_ref and text fields. Ordering should be by relevance
            (most relevant first). Returns [] if no results found or if the
            backend is unavailable.
        """
        ...
