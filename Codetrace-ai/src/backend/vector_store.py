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
from typing import List, Dict, Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """
    All tunable knobs in one place.
    Swap model names here — nothing else needs to change.
    """
    persist_dir:      str = ".codetrace/chroma"

    # Use 'small' variants for 5x speed + 5x less RAM; 'large' for max accuracy.
    retrieval_align:  str = "BAAI/bge-small-en-v1.5"      # BGE — ranker/retrieval
    query_condition:  str = "intfloat/e5-small-v2"         # E5  — query understanding

    collection_bge:   str = "code_bge"
    collection_e5:    str = "code_e5"

    retrieval_k:      int = 20     # wide initial retrieval (candidates for re-ranker)
    top_k:            int = 5      # final results after re-ranking
    rrf_k:            int = 60     # RRF constant; higher = smoother rank blending
    embed_batch_size: int = 32     # sent to HuggingFace encode

    # Re-ranker (FlashRank cross-encoder, runs locally)
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
        self._bge_embeddings: HuggingFaceEmbeddings | None = None
        self._e5_embeddings:  HuggingFaceEmbeddings | None = None
        self._bge_store: Chroma | None = None
        self._e5_store:  Chroma | None = None
        self._reranker = None          # lazy-loaded on first search
        self._init_stores()

    
    # Lazy properties — models are only loaded when first accessed
    

    @property
    def bge_embeddings(self) -> HuggingFaceEmbeddings:
        if self._bge_embeddings is None:
            logger.info("Loading BGE embedding model: %s", self.config.retrieval_align)
            self._bge_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.retrieval_align,
                encode_kwargs={
                    "batch_size": self.config.embed_batch_size,
                    "normalize_embeddings": True,   # required for cosine similarity
                },
            )
        return self._bge_embeddings

    @property
    def e5_embeddings(self) -> HuggingFaceEmbeddings:
        if self._e5_embeddings is None:
            logger.info("Loading E5 embedding model: %s", self.config.query_condition)
            self._e5_embeddings = HuggingFaceEmbeddings(
                model_name=self.config.query_condition,
                encode_kwargs={
                    "batch_size": self.config.embed_batch_size,
                    "normalize_embeddings": True,
                },
            )
        return self._e5_embeddings

    
    # Initialization
    

    def _init_stores(self) -> None:
        """
        Load both models in parallel, then open Chroma collections.
        Parallelising model loading cuts cold-start time ~50%.
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            bge_fut = ex.submit(lambda: self.bge_embeddings)
            e5_fut  = ex.submit(lambda: self.e5_embeddings)
            # force resolution so errors surface here, not later
            bge_fut.result()
            e5_fut.result()

        self._bge_store = Chroma(
            collection_name=self.config.collection_bge,
            persist_directory=self.config.persist_dir,
            embedding_function=self.bge_embeddings,
        )
        self._e5_store = Chroma(
            collection_name=self.config.collection_e5,
            persist_directory=self.config.persist_dir,
            embedding_function=self.e5_embeddings,
        )
        logger.info("Chroma collections ready (dir=%s)", self.config.persist_dir)

    
    # Indexing — single symbol
    

    def add_symbol(self, symbol_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Upsert one code symbol into both embedding collections.

        Uses upsert semantics: safe to call repeatedly with the same symbol_id.
        If either store fails, we attempt to roll back the successful one
        so both collections stay in sync.
        """
        bge_written = False
        try:
            self._bge_store.add_texts(
                ids=[symbol_id],
                texts=[content],
                metadatas=[metadata],
            )
            bge_written = True

            self._e5_store.add_texts(
                ids=[symbol_id],
                texts=[content],
                metadatas=[metadata],
            )

        except Exception as exc:
            logger.error("add_symbol failed for id=%s: %s", symbol_id, exc)
            if bge_written:
                # rollback BGE so stores don't drift apart
                try:
                    self._bge_store.delete(ids=[symbol_id])
                    logger.warning("Rolled back BGE write for id=%s", symbol_id)
                except Exception as rb_exc:
                    logger.error("Rollback also failed for id=%s: %s", symbol_id, rb_exc)
            raise RuntimeError(f"VectorStore sync failure on symbol '{symbol_id}'") from exc

    
    # Indexing — batch (much faster than looping add_symbol)
    

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
            seen[sid] = idx                       # last-write wins
        if len(seen) < len(symbol_ids):
            original_len = len(symbol_ids)
            unique_idx   = sorted(seen.values())  # preserve original order
            symbol_ids   = [symbol_ids[i]  for i in unique_idx]
            contents     = [contents[i]    for i in unique_idx]
            metadatas    = [metadatas[i]   for i in unique_idx]
            logger.warning(
                "Deduplicated batch: %d → %d unique IDs.", original_len, len(seen)
            )

        bge_written = False
        try:
            self._bge_store.add_texts(
                ids=symbol_ids,
                texts=contents,
                metadatas=metadatas,
            )
            bge_written = True

            self._e5_store.add_texts(
                ids=symbol_ids,
                texts=contents,
                metadatas=metadatas,
            )
            logger.info("Batch upserted %d symbols.", len(symbol_ids))

        except Exception as exc:
            logger.error("add_symbols_batch failed: %s", exc)
            if bge_written:
                try:
                    self._bge_store.delete(ids=symbol_ids)
                    logger.warning("Rolled back BGE batch write (%d ids).", len(symbol_ids))
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

        with ThreadPoolExecutor(max_workers=2) as ex:
            bge_fut = ex.submit(
                self._bge_store.similarity_search, query, k=retrieval_k
            )
            e5_fut = ex.submit(
                self._e5_store.similarity_search, query, k=retrieval_k
            )

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

        # Stage 1: RRF merge (broad candidate list)
        rrf_results = self._rrf_merge(
            results.get("bge", []),
            results.get("e5",  []),
        )

        if not rrf_results:
            return []

        # Stage 2: FlashRank cross-encoder re-ranking (precision filtering)
        return self._rerank(query, rrf_results)

    
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
            scores[key]  += 1.5 / (rrf_k + rank + 1)   # BGE weighted higher
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

        # Lazy-load the re-ranker on first call
        if self._reranker is None:
            from flashrank import Ranker
            logger.info("Loading FlashRank re-ranker: %s", self.config.reranker_model)
            self._reranker = Ranker(model_name=self.config.reranker_model)

        from flashrank import RerankRequest

        # Build passages for FlashRank (expects list of dicts with "text" key)
        passages = []
        for doc in documents:
            passages.append({
                "text": doc.page_content,
                "meta": doc.metadata,
            })

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked = self._reranker.rerank(rerank_request)

        # Map back to LangChain Documents, sorted by FlashRank score (descending)
        reranked_docs = []
        for result in ranked[:self.config.top_k]:
            meta = result.get("meta", result.get("metadata", {}))
            reranked_docs.append(
                Document(
                    page_content=result["text"],
                    metadata=meta,
                )
            )

        logger.info(
            "Re-ranked %d candidates → top %d results.",
            len(documents), len(reranked_docs)
        )
        return reranked_docs

    
    # Deletion
    

    def delete_symbol(self, symbol_id: str) -> None:
        """Remove a symbol from both stores."""
        errors = []
        for store, name in [(self._bge_store, "BGE"), (self._e5_store, "E5")]:
            try:
                store.delete(ids=[symbol_id])
            except Exception as exc:
                logger.error("Delete from %s failed for id=%s: %s", name, symbol_id, exc)
                errors.append(exc)
        if errors:
            raise RuntimeError(f"delete_symbol incomplete for '{symbol_id}'") from errors[0]

    
    # Info / Debug
    

    def collection_counts(self) -> Dict[str, int]:
        """Return number of indexed documents in each collection."""
        return {
            "bge": self._bge_store._collection.count(),
            "e5":  self._e5_store._collection.count(),
        }

    def __repr__(self) -> str:
        return (
            f"VectorStore(bge={self.config.retrieval_align!r}, "
            f"e5={self.config.query_condition!r}, "
            f"top_k={self.config.top_k})"
        )