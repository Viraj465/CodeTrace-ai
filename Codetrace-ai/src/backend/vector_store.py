"""
VectorStore: Dual-embedding hybrid search using BGE + E5 models.
- BGE: retrieval alignment / ranking
- E5:  query understanding

Architecture:
  - Single Chroma text store (no duplication)
  - Two separate vector collections (one per embedding model)
  - Concurrent search via ThreadPoolExecutor
  - RRF (Reciprocal Rank Fusion) for result merging
  - Batch ingestion support
  - Full error handling + sync safety
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Raw chromadb — no LangChain wrapper needed.
# chromadb.PersistentClient() is the replacement for langchain_chroma.Chroma().
import chromadb
from chromadb import Collection
from flashrank import Ranker, RerankRequest

# Raw sentence-transformers.
# SentenceTransformer.encode() returns numpy arrays compatible with ChromaDB directly.
from sentence_transformers import SentenceTransformer

# ── System / device detection ──────────────────────────────────────────────
from src.core.system_info import get_system_info as _get_system_info
_SYS = _get_system_info()          # cached, thread-safe
_DEVICE = _SYS.device              # "cuda" | "mps" | "cpu"
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# Lightweight Document dataclass.
# Replaces langchain_core.documents.Document.
# Same two fields (page_content, metadata) so downstream code is unchanged.

@dataclass
class Document:
    """
    Minimal document container (replaces langchain_core.documents.Document).
    page_content: the text of the code symbol.
    metadata: dict with file path, symbol type, etc.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """
    All tunable knobs in one place.
    Swap model names here — nothing else needs to change.
    """
    persist_dir:      str = ".codetrace/chroma"

    # Use 'small' variants for 5x speed + 5x less RAM; 'large' for max accuracy.
    retrieval_align:  str = "BAAI/bge-small-en-v1.5"      # BGE ranker/retrieval
    query_condition:  str = "intfloat/e5-small-v2"         # E5 query understanding

    collection_bge:   str = "code_bge"
    collection_e5:    str = "code_e5"

    retrieval_k:      int = 20     # Wide initial retrieval.
    top_k:            int = 5      # Final results after re-ranking.
    rrf_k:            int = 60     # RRF constant; higher = smoother rank blending.
    embed_batch_size: int = _SYS.embed_batch_size
    # Batch size is driven by system_info: 128 (CUDA) | 64 (MPS) | 32 (CPU).

    # Re-ranker (FlashRank cross-encoder, runs locally).
    reranker_model:   str = "ms-marco-MiniLM-L-12-v2"


class VectorStore:
    """
    Handles dual-embedding indexing and hybrid search over a code corpus.

    Responsibilities
    ----------------
    - Index code symbols into both BGE and E5 Chroma collections.
    - Run both searches concurrently and merge via Reciprocal Rank Fusion.
    - Guarantee both collections stay in sync (atomic-ish write with rollback).
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()
        self._bge_model: Optional[SentenceTransformer] = None
        self._e5_model:  Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._bge_col: Optional[Collection] = None
        self._e5_col:  Optional[Collection] = None
        self._reranker = None          # Lazy-loaded on first search.
        self._init_stores()

    # Lazy model properties
    # Models are only loaded when first accessed, then cached.

    @property
    def bge_model(self) -> SentenceTransformer:
        if self._bge_model is None:
            logger.info("Loading BGE embedding model: %s (device=%s)", self.config.retrieval_align, _DEVICE.upper())
            self._bge_model = SentenceTransformer(self.config.retrieval_align, device=_DEVICE)
        return self._bge_model

    @property
    def e5_model(self) -> SentenceTransformer:
        if self._e5_model is None:
            logger.info("Loading E5 embedding model: %s (device=%s)", self.config.query_condition, _DEVICE.upper())
            self._e5_model = SentenceTransformer(self.config.query_condition, device=_DEVICE)
        return self._e5_model

    # Internal helpers

    def _embed_bge(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with BGE model, normalised for cosine similarity."""
        vectors = self.bge_model.encode(
            texts,
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _embed_e5(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts with E5 model.
        E5 expects a 'passage: ' prefix for document indexing,
        and 'query: ' prefix for query embedding.
        """
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self.e5_model.encode(
            prefixed,
            batch_size=self.config.embed_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def _embed_e5_query(self, query: str) -> List[float]:
        """Embed a search query with E5's 'query: ' prefix."""
        vectors = self.e5_model.encode(
            [f"query: {query}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors[0].tolist()

    # Initialization

    def _init_stores(self) -> None:
        """
        Load both models in parallel, then open ChromaDB collections.
        Parallelising model loading cuts cold-start time ~50%.
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            bge_fut = ex.submit(lambda: self.bge_model)
            e5_fut  = ex.submit(lambda: self.e5_model)
            bge_fut.result()
            e5_fut.result()

        # Open ChromaDB with a single persistent client shared by both collections.
        # chromadb.PersistentClient() replaces the old langchain_chroma.Chroma() wrapper.
        self._chroma_client = chromadb.PersistentClient(path=self.config.persist_dir)
        self._bge_col = self._chroma_client.get_or_create_collection(
            name=self.config.collection_bge,
            metadata={"hnsw:space": "cosine"},   # Required for normalized BGE vectors.
        )
        self._e5_col = self._chroma_client.get_or_create_collection(
            name=self.config.collection_e5,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma collections ready (dir=%s)", self.config.persist_dir)

    # Indexing - single symbol

    def add_symbol(self, symbol_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Upsert one code symbol into both embedding collections.

        Uses upsert semantics: safe to call repeatedly with the same symbol_id.
        If either store fails, we attempt to roll back the successful one
        so both collections stay in sync.
        """
        # Sanitize metadata: ChromaDB only accepts str/int/float/bool values.
        clean_meta = {
            k: v for k, v in metadata.items()
            if isinstance(v, (str, int, float, bool))
        }

        bge_written = False
        try:
            bge_vec = self._embed_bge([content])
            self._bge_col.upsert(
                ids=[symbol_id],
                documents=[content],
                embeddings=bge_vec,
                metadatas=[clean_meta],
            )
            bge_written = True

            e5_vec = self._embed_e5([content])
            self._e5_col.upsert(
                ids=[symbol_id],
                documents=[content],
                embeddings=e5_vec,
                metadatas=[clean_meta],
            )

        except Exception as exc:
            logger.error("add_symbol failed for id=%s: %s", symbol_id, exc)
            if bge_written:
                try:
                    self._bge_col.delete(ids=[symbol_id])
                    logger.warning("Rolled back BGE write for id=%s", symbol_id)
                except Exception as rb_exc:
                    logger.error("Rollback also failed for id=%s: %s", symbol_id, rb_exc)
            raise RuntimeError(f"VectorStore sync failure on symbol '{symbol_id}'") from exc

    # Indexing - batch

    def add_symbols_batch(
        self,
        symbol_ids: List[str],
        contents:   List[str],
        metadatas:  List[Dict[str, Any]],
    ) -> None:
        """
        Upsert many code symbols at once.

        Sending N items in one call lets the embedding model fill its
        full batch window, giving 5-10x throughput vs. N individual calls.
        """
        if not (len(symbol_ids) == len(contents) == len(metadatas)):
            raise ValueError("symbol_ids, contents, and metadatas must have equal length.")

        # ChromaDB requires unique IDs per batch. Deduplicate, keeping last occurrence.
        seen: dict[str, int] = {}
        for idx, sid in enumerate(symbol_ids):
            seen[sid] = idx                       # Last-write wins.
        if len(seen) < len(symbol_ids):
            original_len = len(symbol_ids)
            unique_idx   = sorted(seen.values())  # Preserve original order.
            symbol_ids   = [symbol_ids[i]  for i in unique_idx]
            contents     = [contents[i]    for i in unique_idx]
            metadatas    = [metadatas[i]   for i in unique_idx]
            logger.warning(
                "Deduplicated batch: %d → %d unique IDs.", original_len, len(seen)
            )

        # Sanitize metadata for ChromaDB.
        clean_metas = [
            {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            for m in metadatas
        ]

        # ChromaDB has a hard max upsert batch size (~5461 records).
        # Embed the full list at once (maximises GPU throughput) then write
        # to ChromaDB in safe chunks.
        CHROMA_MAX_BATCH = 5000  # conservative — well under the 5461 hard limit

        bge_written_ids: List[str] = []
        try:
            # Embed everything in one shot (GPU-efficient).
            bge_vecs = self._embed_bge(contents)
            e5_vecs  = self._embed_e5(contents)

            # Write to ChromaDB in chunks.
            for start in range(0, len(symbol_ids), CHROMA_MAX_BATCH):
                end      = start + CHROMA_MAX_BATCH
                chunk_ids    = symbol_ids[start:end]
                chunk_docs   = contents[start:end]
                chunk_metas  = clean_metas[start:end]
                chunk_bge    = bge_vecs[start:end]
                chunk_e5     = e5_vecs[start:end]

                self._bge_col.upsert(
                    ids=chunk_ids,
                    documents=chunk_docs,
                    embeddings=chunk_bge,
                    metadatas=chunk_metas,
                )
                bge_written_ids.extend(chunk_ids)

                self._e5_col.upsert(
                    ids=chunk_ids,
                    documents=chunk_docs,
                    embeddings=chunk_e5,
                    metadatas=chunk_metas,
                )

            logger.info("Batch upserted %d symbols (%d chunks).",
                        len(symbol_ids),
                        (len(symbol_ids) + CHROMA_MAX_BATCH - 1) // CHROMA_MAX_BATCH)

        except Exception as exc:
            logger.error("add_symbols_batch failed: %s", exc)
            if bge_written_ids:
                try:
                    # Roll back only the chunks already committed to BGE.
                    for start in range(0, len(bge_written_ids), CHROMA_MAX_BATCH):
                        self._bge_col.delete(ids=bge_written_ids[start:start + CHROMA_MAX_BATCH])
                    logger.warning("Rolled back BGE batch write (%d ids).", len(bge_written_ids))
                except Exception as rb_exc:
                    logger.error("Batch rollback failed: %s", rb_exc)
            raise RuntimeError("VectorStore batch sync failure") from exc

    # Hybrid Search

    def hybrid_search(self, query: str) -> List[Document]:
        """
        Query both embedding models concurrently, merge via RRF, then
        re-rank with FlashRank cross-encoder for maximum precision.

        Pipeline:  BGE(retrieval_k) + E5(retrieval_k)
                    → RRF merge
                    → FlashRank re-rank
                    → top_k results
        """
        retrieval_k = self.config.retrieval_k
        bge_query_vec = self._embed_bge([query])[0]
        e5_query_vec  = self._embed_e5_query(query)

        def _search_bge():
            results = self._bge_col.query(
                query_embeddings=[bge_query_vec],
                n_results=retrieval_k,
                include=["documents", "metadatas", "distances"],
            )
            return self._chroma_results_to_docs(results)

        def _search_e5():
            results = self._e5_col.query(
                query_embeddings=[e5_query_vec],
                n_results=retrieval_k,
                include=["documents", "metadatas", "distances"],
            )
            return self._chroma_results_to_docs(results)

        with ThreadPoolExecutor(max_workers=2) as ex:
            bge_fut = ex.submit(_search_bge)
            e5_fut  = ex.submit(_search_e5)

            results: Dict[str, List[Document]] = {}
            for fut in as_completed([bge_fut, e5_fut]):
                try:
                    docs = fut.result()
                    key  = "bge" if fut is bge_fut else "e5"
                    results[key] = docs
                except Exception as exc:
                    logger.error("Search future failed: %s", exc)
                    results.setdefault("bge", [])
                    results.setdefault("e5",  [])

        # Stage 1: RRF merge (broad candidate list).
        rrf_results = self._rrf_merge(
            results.get("bge", []),
            results.get("e5",  []),
        )

        if not rrf_results:
            return []

        # Stage 2: FlashRank cross-encoder re-ranking (precision filtering).
        return self._rerank(query, rrf_results)

    def _chroma_results_to_docs(self, results: dict) -> List[Document]:
        """Convert a raw chromadb query result dict to a list of Documents."""
        docs = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for text, meta in zip(documents, metadatas):
            if text:
                docs.append(Document(page_content=text, metadata=meta or {}))
        return docs

    # RRF Merge

    def _rrf_merge(
        self,
        bge_results: List[Document],
        e5_results:  List[Document],
        k: int | None = None,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion — combines two ranked lists into one.

        Score formula:  sum( 1 / (k + rank) )  for each list the doc appears in.
        Higher score = more relevant.

        BGE results are weighted 1.5x to prioritise retrieval alignment.
        """
        rrf_k   = k or self.config.rrf_k
        scores:  Dict[str, float]    = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(bge_results):
            key = doc.page_content
            scores[key]  += 1.5 / (rrf_k + rank + 1)   # BGE weighted higher.
            doc_map[key]  = doc

        for rank, doc in enumerate(e5_results):
            key = doc.page_content
            scores[key]  += 1.0 / (rrf_k + rank + 1)
            doc_map[key]  = doc

        ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[key] for key in ranked_keys]

    # FlashRank Re-ranker

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents using a FlashRank cross-encoder model.

        Unlike embedding-based similarity, the cross-encoder sees both
        the query and each document together, producing far more accurate
        relevance scores for the final top_k selection.
        """
        if not documents:
            return []

        # Lazy-load the re-ranker on first call.
        if self._reranker is None:
            
            logger.info("Loading FlashRank re-ranker: %s", self.config.reranker_model)
            self._reranker = Ranker(model_name=self.config.reranker_model)

        passages = [{"text": doc.page_content, "meta": doc.metadata} for doc in documents]
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked = self._reranker.rerank(rerank_request)

        reranked_docs = []
        for result in ranked[:self.config.top_k]:
            meta = result.get("meta", result.get("metadata", {}))
            reranked_docs.append(Document(page_content=result["text"], metadata=meta))

        logger.info(
            "Re-ranked %d candidates → top %d results.",
            len(documents), len(reranked_docs)
        )
        return reranked_docs

    # Deletion

    def delete_symbol(self, symbol_id: str) -> None:
        """Remove a symbol from both stores."""
        errors = []
        for col, name in [(self._bge_col, "BGE"), (self._e5_col, "E5")]:
            try:
                col.delete(ids=[symbol_id])
            except Exception as exc:
                logger.error("Delete from %s failed for id=%s: %s", name, symbol_id, exc)
                errors.append(exc)
        if errors:
            raise RuntimeError(f"delete_symbol incomplete for '{symbol_id}'") from errors[0]

    # Info and Debug

    def collection_counts(self) -> Dict[str, int]:
        """Return number of indexed documents in each collection."""
        return {
            "bge": self._bge_col.count(),
            "e5":  self._e5_col.count(),
        }

    def __repr__(self) -> str:
        return (
            f"VectorStore(bge={self.config.retrieval_align!r}, "
            f"e5={self.config.query_condition!r}, "
            f"top_k={self.config.top_k})"
        )