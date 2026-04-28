"""
ChromaDB + Gemini RAG adapter for CRUCIBLE v3.0.

Implements the LocalTreatiseRagAdapter protocol using:
  - ChromaDB for vector storage and retrieval
  - Gemini text-embedding-004 for query-time embedding

Usage:
    from crucible.adapters.chroma_rag_adapter import ChromaRagAdapter

    adapter = ChromaRagAdapter(
        chroma_db_path="/path/to/rag_index/chroma_db",
        gemini_api_key="<google-api-key>",
    )
    docs = adapter.query_treatises(["民法第90条", "公序良俗"], max_results=5)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from crucible.schemas import Document

# Default collection names (must match rag_build_pipeline.py constants)
_DEFAULT_CHILD_COLLECTION = "jp_legal_treatises"
_DEFAULT_PARENT_COLLECTION = "jp_legal_treatises_parents"

# Gemini embedding model
_EMBED_MODEL = "gemini-embedding-001"

# Parent text cache: limit to avoid unbounded memory growth on long runs
_PARENT_CACHE_MAX_SIZE = 2000


class ChromaRagAdapter:
    """Retrieves authoritative Japanese legal treatise passages from a ChromaDB index.

    Implements the LocalTreatiseRagAdapter protocol.

    Query flow:
      1. Embed the joined keywords string via Gemini text-embedding-004
         (task_type=RETRIEVAL_QUERY)
      2. ANN search in the child chunk collection
      3. Optionally expand each hit to its parent chunk for richer context
      4. Return list[Document] ordered by relevance
    """

    def __init__(
        self,
        chroma_db_path: str | Path,
        gemini_api_key: str,
        child_collection_name: str = _DEFAULT_CHILD_COLLECTION,
        parent_collection_name: str = _DEFAULT_PARENT_COLLECTION,
        return_parent_context: bool = True,
        max_cache_size: int = _PARENT_CACHE_MAX_SIZE,
    ) -> None:
        chroma_db_path = Path(chroma_db_path)
        if not chroma_db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB path not found: {chroma_db_path}. "
                "Run rag_build_pipeline.py to build the index first."
            )

        try:
            import chromadb as _chromadb
        except ImportError as exc:
            raise ImportError(
                "chromadb required: pip install chromadb"
            ) from exc

        self._chroma_client = _chromadb.PersistentClient(path=str(chroma_db_path))
        self._child_collection = self._chroma_client.get_collection(name=child_collection_name)
        self._return_parent_context = return_parent_context
        self._max_cache_size = max_cache_size
        self._api_key = gemini_api_key

        # Lazy-load parent collection (graceful if not built)
        self._parent_collection: Any | None = None
        if return_parent_context:
            try:
                self._parent_collection = self._chroma_client.get_collection(
                    name=parent_collection_name
                )
            except Exception:
                self._parent_collection = None

        # LRU-style parent text cache (dict preserves insertion order in Python 3.7+)
        self._parent_cache: dict[str, str] = {}

        # Lazy Gemini client — created on first query call
        self._gemini_client: Any | None = None

    # ------------------------------------------------------------------
    # Public protocol method
    # ------------------------------------------------------------------

    def query_treatises(
        self,
        keywords: list[str],
        max_results: int,
    ) -> list[Document]:
        """Retrieve treatise passages relevant to the given keywords.

        Args:
            keywords: Search terms from MpsrRagSeed entries (Japanese legal concepts,
                statute names, doctrine labels).
            max_results: Upper bound on returned Document objects.

        Returns:
            list[Document] ordered by cosine similarity (most relevant first).
            Empty list if ChromaDB is unavailable or no results found.
        """
        if not keywords:
            return []

        query_text = " ".join(k.strip() for k in keywords if k.strip())
        if not query_text:
            return []

        try:
            embedding = self._embed_query(query_text)
        except Exception:
            # Embedding failure → graceful degrade
            return []

        try:
            results = self._child_collection.query(
                query_embeddings=[embedding],
                n_results=min(max_results * 2, 50),
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        raw_docs = results.get("documents", [[]])[0]
        raw_metas = results.get("metadatas", [[]])[0]
        raw_distances = results.get("distances", [[]])[0]

        documents: list[Document] = []
        seen_parent_ids: set[str] = set()

        for child_text, meta, distance in zip(raw_docs, raw_metas, raw_distances):
            if len(documents) >= max_results:
                break

            if not child_text or not meta:
                continue

            chunk_id = meta.get("chunk_id", "")
            book = meta.get("book", "")
            legal_authority = meta.get("legal_authority", "commentary")
            parent_chunk_id = meta.get("parent_chunk_id", "")

            # Prefer parent text for richer context (one parent per query result)
            text = child_text
            if (
                self._return_parent_context
                and parent_chunk_id
                and parent_chunk_id not in seen_parent_ids
            ):
                parent_text = self._fetch_parent_text(parent_chunk_id)
                if parent_text:
                    text = parent_text
                    seen_parent_ids.add(parent_chunk_id)

            documents.append(Document(
                canonical_ref=chunk_id,
                text=text,
                statute=book,
                article_num=0,
                expansion_metadata={
                    "legal_authority": legal_authority,
                    "heading_path": meta.get("heading_path_str", ""),
                    "doc_id": meta.get("doc_id", ""),
                    "page_start": meta.get("page_start", 0),
                    "page_end": meta.get("page_end", 0),
                    "distance": float(distance),
                },
            ))

        return documents

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ImportError(
                    "google-genai required: pip install google-genai"
                ) from exc
            self._gemini_client = genai.Client(api_key=self._api_key)
        return self._gemini_client

    def _embed_query(self, text: str) -> list[float]:
        """Embed a query string for retrieval."""
        from google.genai import types

        client = self._get_gemini_client()
        result = client.models.embed_content(
            model=_EMBED_MODEL,
            contents=[text],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return list(result.embeddings[0].values)

    # ------------------------------------------------------------------
    # Parent chunk fetch with simple LRU eviction
    # ------------------------------------------------------------------

    def _fetch_parent_text(self, parent_chunk_id: str) -> str | None:
        """Fetch parent chunk text from ChromaDB. Uses a size-bounded cache."""
        if not parent_chunk_id:
            return None

        # Cache hit
        if parent_chunk_id in self._parent_cache:
            return self._parent_cache[parent_chunk_id]

        if self._parent_collection is None:
            return None

        try:
            result = self._parent_collection.get(
                ids=[parent_chunk_id],
                include=["documents"],
            )
            docs = result.get("documents") or []
            if not docs or not docs[0]:
                return None

            text = docs[0]

            # Simple cache eviction: drop oldest entry when at capacity
            if len(self._parent_cache) >= self._max_cache_size:
                oldest_key = next(iter(self._parent_cache))
                del self._parent_cache[oldest_key]

            self._parent_cache[parent_chunk_id] = text
            return text
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def collection_size(self) -> int:
        """Return the number of child chunks indexed."""
        try:
            return self._child_collection.count()
        except Exception:
            return 0

    def __repr__(self) -> str:
        try:
            n = self._child_collection.count()
        except Exception:
            n = "?"
        return (
            f"ChromaRagAdapter("
            f"child_chunks={n}, "
            f"parent_context={self._return_parent_context})"
        )
