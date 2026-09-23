"""
VectorStore: hybrid code search backed by two embedding models, BGE and E5.
BGE handles retrieval alignment / ranking; E5 handles query understanding.

How it fits together: one Chroma store holds the text, with a separate vector
collection per model. Searches run against both models at once (ThreadPoolExecutor),
and the two ranked lists get fused with Reciprocal Rank Fusion. Ingestion supports
batching, and writes across the two collections are kept in sync (with rollback).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# We talk to chromadb directly instead of going through LangChain —
# chromadb.PersistentClient() covers what langchain_chroma.Chroma() used to.
import chromadb
from chromadb import Collection
from flashrank import Ranker, RerankRequest

# Same story for sentence-transformers: encode() hands back numpy arrays that
# ChromaDB accepts as-is.
from sentence_transformers import SentenceTransformer

# ── System / device detection ──────────────────────────────────────────────
from src.core.system_info import get_system_info as _get_system_info
_SYS = _get_system_info()          # cached, thread-safe
_DEVICE = _SYS.device              # "cuda" | "mps" | "cpu"
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# Our stand-in for langchain_core.documents.Document — same two fields, so the
# rest of the code didn't have to change when we dropped LangChain.

@dataclass
class Document:
    """
    Minimal document container.
    page_content: the text of the code symbol.
    metadata: dict with file path, symbol type, etc.
    """
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """
    Every tunable in one place. To switch models, change the names here and
    nothing else needs touching.
    """
    persist_dir:      str = ".codetrace/chroma"

    # The 'small' variants are ~5x faster and lighter on RAM; go 'large' if you
    # want the last bit of accuracy.
    retrieval_align:  str = "BAAI/bge-small-en-v1.5"      # BGE ranker/retrieval
    query_condition:  str = "intfloat/e5-small-v2"         # E5 query understanding

    collection_bge:   str = "code_bge"
    collection_e5:    str = "code_e5"

    retrieval_k:      int = 20     # How many candidates each model pulls back.
    top_k:            int = 5      # How many survive the final re-rank.
    rrf_k:            int = 60     # RRF constant — bigger smooths out the rank blend.
    embed_batch_size: int = _SYS.embed_batch_size
    # embed_batch_size comes from system_info: 128 on CUDA, 64 on MPS, 32 on CPU.

    # Local FlashRank cross-encoder used for the final re-rank.
    reranker_model:   str = "ms-marco-MiniLM-L-12-v2"
    # Candidates scored per onnxruntime call. The cross-encoder's attention
    # buffer grows with (batch x seq^2), so the whole candidate set in one call
    # is a several-hundred-MB allocation — enough to fail on a CPU-only box
    # that's also hosting the LLM. 8 keeps each call small; the ranking is
    # identical either way, since scores are per (query, passage) pair.
    rerank_batch_size: int = 8


class VectorStore:
    """
    Indexes code symbols and runs hybrid search over them using two embedding
    models.

    It writes each symbol into both the BGE and E5 Chroma collections, searches
    them concurrently and fuses the results with RRF, and tries hard to keep the
    two collections consistent — writes roll back if one side fails.
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self.config = config or VectorStoreConfig()
        self._bge_model: Optional[SentenceTransformer] = None
        self._e5_model:  Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.PersistentClient] = None
        self._bge_col: Optional[Collection] = None
        self._e5_col:  Optional[Collection] = None
        self._reranker = None          # loaded on the first search, not before
        self._init_stores()

    # The embedding models load on first access and stay cached after that.

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
        Embed texts with the E5 model. E5 wants a 'passage: ' prefix when you're
        indexing documents (and 'query: ' when embedding a query — see below).
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
        Load both models side by side, then open the ChromaDB collections.
        Loading them in parallel roughly halves the cold-start wait.
        """
        with ThreadPoolExecutor(max_workers=2) as ex:
            bge_fut = ex.submit(lambda: self.bge_model)
            e5_fut  = ex.submit(lambda: self.e5_model)
            bge_fut.result()
            e5_fut.result()

        # One persistent client, shared by both collections.
        self._chroma_client = chromadb.PersistentClient(path=self.config.persist_dir)
        self._bge_col = self._chroma_client.get_or_create_collection(
            name=self.config.collection_bge,
            metadata={"hnsw:space": "cosine"},   # cosine, to match the normalized BGE vectors
        )
        self._e5_col = self._chroma_client.get_or_create_collection(
            name=self.config.collection_e5,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma collections ready (dir=%s)", self.config.persist_dir)

    # Indexing - single symbol

    def add_symbol(self, symbol_id: str, content: str, metadata: Dict[str, Any]) -> None:
        """
        Upsert one code symbol into both collections.

        It's an upsert, so calling it again with the same symbol_id is fine. If
        one collection succeeds and the other fails, we undo the successful write
        so the two don't drift apart.
        """
        # ChromaDB metadata values can only be str/int/float/bool — drop the rest.
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
        Upsert a whole pile of symbols at once.

        Handing the model one big batch lets it fill its batch window instead of
        idling between calls — in practice 5-10x the throughput of upserting one
        at a time.
        """
        if not (len(symbol_ids) == len(contents) == len(metadatas)):
            raise ValueError("symbol_ids, contents, and metadatas must have equal length.")

        # ChromaDB wants unique IDs within a batch, so dedupe first — on a repeat
        # ID, the later entry wins.
        seen: dict[str, int] = {}
        for idx, sid in enumerate(symbol_ids):
            seen[sid] = idx
        if len(seen) < len(symbol_ids):
            original_len = len(symbol_ids)
            unique_idx   = sorted(seen.values())  # keep them in the original order
            symbol_ids   = [symbol_ids[i]  for i in unique_idx]
            contents     = [contents[i]    for i in unique_idx]
            metadatas    = [metadatas[i]   for i in unique_idx]
            logger.warning(
                "Deduplicated batch: %d → %d unique IDs.", original_len, len(seen)
            )

        # Same metadata scrub as add_symbol, applied to every item.
        clean_metas = [
            {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            for m in metadatas
        ]

        # ChromaDB caps a single upsert at ~5461 records, so we embed the whole
        # list in one go (keeps the GPU busy) and then write it out in chunks.
        CHROMA_MAX_BATCH = 5000  # sits comfortably under the 5461 hard limit

        bge_written_ids: List[str] = []
        try:
            # Embed everything up front — one big pass is far cheaper than many.
            bge_vecs = self._embed_bge(contents)
            e5_vecs  = self._embed_e5(contents)

            # Now write it to ChromaDB a chunk at a time.
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
                    # Undo just the BGE chunks that already landed.
                    for start in range(0, len(bge_written_ids), CHROMA_MAX_BATCH):
                        self._bge_col.delete(ids=bge_written_ids[start:start + CHROMA_MAX_BATCH])
                    logger.warning("Rolled back BGE batch write (%d ids).", len(bge_written_ids))
                except Exception as rb_exc:
                    logger.error("Batch rollback failed: %s", rb_exc)
            raise RuntimeError("VectorStore batch sync failure") from exc

    # Hybrid Search

    def hybrid_search(self, query: str) -> List[Document]:
        """
        Search both models at once, fuse the two ranked lists with RRF, then
        run a FlashRank cross-encoder over the survivors to sharpen the ordering.

        So the flow is: pull retrieval_k hits from each of BGE and E5, RRF-merge
        them, re-rank with FlashRank, and return the top_k.
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

        # First, fuse the two lists into one broad candidate set.
        rrf_results = self._rrf_merge(
            results.get("bge", []),
            results.get("e5",  []),
        )

        if not rrf_results:
            return []

        # Then let the cross-encoder trim it down to the most relevant few.
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

    @staticmethod
    def _doc_key(doc: Document) -> str:
        """
        A stable identity for a doc during RRF fusion: file path + qualified name.

        We used to key on page_content, but that merged genuinely different symbols
        whenever their code happened to be identical — overloads, boilerplate
        getters, copy-pasted snippets — which silently dropped results and lost
        their metadata. We only fall back to page_content when there's no metadata
        to key on.
        """
        meta = doc.metadata or {}
        file_path = meta.get("file_path")
        symbol = meta.get("qualified_name") or meta.get("symbol_name")
        if file_path and symbol:
            return f"{file_path}:{symbol}"
        return doc.page_content

    def _rrf_merge(
        self,
        bge_results: List[Document],
        e5_results:  List[Document],
        k: int | None = None,
    ) -> List[Document]:
        """
        Reciprocal Rank Fusion — folds two ranked lists into one.

        Each doc scores sum(1 / (k + rank)) across whichever lists it shows up in,
        and higher wins. BGE gets a 1.5x weight so retrieval alignment counts for
        a bit more than query understanding.
        """
        rrf_k   = k or self.config.rrf_k
        scores:  Dict[str, float]    = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for rank, doc in enumerate(bge_results):
            key = self._doc_key(doc)
            scores[key]  += 1.5 / (rrf_k + rank + 1)   # the 1.5x BGE weight
            doc_map[key]  = doc

        for rank, doc in enumerate(e5_results):
            key = self._doc_key(doc)
            scores[key]  += 1.0 / (rrf_k + rank + 1)
            doc_map[key]  = doc

        ranked_keys = sorted(scores, key=scores.__getitem__, reverse=True)
        return [doc_map[key] for key in ranked_keys]

    # FlashRank Re-ranker

    def _rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank documents with a FlashRank cross-encoder.

        A cross-encoder looks at the query and each document together, rather than
        comparing two separate embeddings, so its relevance scores are much sharper
        — which is exactly what we want for picking the final top_k.
        """
        if not documents:
            return []

        # Load the re-ranker the first time we actually need it.
        if self._reranker is None:
            logger.info("Loading FlashRank re-ranker: %s", self.config.reranker_model)
            self._reranker = Ranker(model_name=self.config.reranker_model)

        # Score in small chunks. FlashRank pads a whole request into one padded
        # batch, so ~40 candidates at 512 tokens asks onnxruntime for a single
        # ~380 MB attention buffer — which fails outright on a machine already
        # holding an LLM in RAM ("BFCArena::AllocateRawInternal Failed to
        # allocate memory for requested buffer"). Chunking keeps each allocation
        # ~an order of magnitude smaller for the same final ranking.
        scored: List[tuple[float, Document]] = []
        for start in range(0, len(documents), self.config.rerank_batch_size):
            chunk = documents[start:start + self.config.rerank_batch_size]
            passages = [
                # Index into `chunk` so we can map results back to the original
                # Document (and its metadata) without relying on text equality.
                {"id": i, "text": doc.page_content, "meta": doc.metadata}
                for i, doc in enumerate(chunk)
            ]
            try:
                ranked = self._reranker.rerank(RerankRequest(query=query, passages=passages))
            except Exception as exc:
                # Out of memory, a corrupt model file, an onnxruntime provider
                # blowing up — none of it is worth failing the search over. The
                # RRF order is already a decent ranking; degrade to it.
                logger.warning(
                    "Re-ranker failed (%s: %s) — falling back to RRF order.",
                    type(exc).__name__, exc or "no detail",
                )
                return documents[:self.config.top_k]

            for result in ranked:
                idx = result.get("id")
                doc = chunk[idx] if isinstance(idx, int) and idx < len(chunk) else None
                if doc is None:
                    meta = result.get("meta", result.get("metadata", {}))
                    doc = Document(page_content=result.get("text", ""), metadata=meta)
                scored.append((float(result.get("score", 0.0)), doc))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        reranked_docs = [doc for _score, doc in scored[:self.config.top_k]]

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

    def delete_by_file(self, file_paths: List[str]) -> None:
        """
        Drop every indexed symbol that belongs to the given files from both
        collections, matching on the ``file_path`` metadata.

        We call this right before re-indexing a changed file, so symbols that got
        renamed or removed inside it don't stick around as stale search hits.
        """
        if not file_paths:
            return
        paths = list(file_paths)
        where = {"file_path": {"$in": paths}} if len(paths) > 1 else {"file_path": paths[0]}
        errors = []
        for col, name in [(self._bge_col, "BGE"), (self._e5_col, "E5")]:
            try:
                col.delete(where=where)
            except Exception as exc:
                logger.error("delete_by_file from %s failed for %d file(s): %s",
                             name, len(paths), exc)
                errors.append(exc)
        if errors:
            raise RuntimeError(
                f"delete_by_file incomplete for {len(paths)} file(s)"
            ) from errors[0]

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