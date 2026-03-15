"""
RAG (Retrieval-Augmented Generation) Module

Architecture:
  - Knowledge base: plain text files in /knowledge_base/
  - Chunking: split documents into ~800 char overlapping chunks
  - Embedding: sentence-transformers (all-MiniLM-L6-v2) — free, local, fast
  - Vector store: in-memory numpy cosine similarity (no external DB needed)
  - At startup, all KB docs are embedded and cached in memory
  - At query time, top-k chunks retrieved and injected into LLM system prompt

Install:
  pip install sentence-transformers numpy

For production: swap in-memory store for ChromaDB, Pinecone, or pgvector.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger("rag")

KB_DIR = Path(__file__).parent / "knowledge_base"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.3

# Lazy-loaded sentence transformer model
_embedder = None
_embedder_lock = asyncio.Lock()


def _get_embedder():
    """Lazy-load the sentence transformer model."""
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("RAG: Loading sentence-transformers model (all-MiniLM-L6-v2)...")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("RAG: Embedding model loaded")
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. Run: pip install sentence-transformers"
            )
    return _embedder


class VectorStore:
    """Simple in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.sources: List[str] = []
        self._initialized = False

    async def build(self, documents: List[Tuple[str, str]]):
        """Build the vector index from (source, text) pairs."""
        logger.info(f"RAG: Building index from {len(documents)} documents")

        all_chunks = []
        all_sources = []

        for source, text in documents:
            chunks = _chunk_text(text)
            all_chunks.extend(chunks)
            all_sources.extend([source] * len(chunks))

        if not all_chunks:
            logger.warning("RAG: No chunks to index")
            return

        logger.info(f"RAG: Embedding {len(all_chunks)} chunks locally...")

        # Run embedding in thread pool (CPU/GPU intensive, sync operation)
        loop = asyncio.get_event_loop()
        embedder = _get_embedder()
        embeddings = await loop.run_in_executor(
            None,
            lambda: embedder.encode(all_chunks, show_progress_bar=False, normalize_embeddings=True)
        )

        self.chunks = all_chunks
        self.sources = all_sources
        self.embeddings = np.array(embeddings)
        self._initialized = True
        logger.info(f"RAG: Index built. {len(self.chunks)} chunks, dim={self.embeddings.shape[1]}")

    async def search(self, query: str, top_k: int = TOP_K) -> List[Tuple[str, float, str]]:
        """Search for most relevant chunks. Returns (chunk, score, source) list."""
        if not self._initialized or self.embeddings is None:
            return []

        loop = asyncio.get_event_loop()
        embedder = _get_embedder()

        query_emb = await loop.run_in_executor(
            None,
            lambda: embedder.encode([query], normalize_embeddings=True)[0]
        )

        # Cosine similarity (embeddings already normalized → just dot product)
        similarities = self.embeddings @ query_emb
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= SIMILARITY_THRESHOLD:
                results.append((self.chunks[idx], score, self.sources[idx]))

        return results

    @property
    def is_ready(self) -> bool:
        return self._initialized


# Global vector store singleton
_store = VectorStore()
_init_lock = asyncio.Lock()


async def init_rag():
    """Load knowledge base and build vector index. Call once at startup."""
    async with _init_lock:
        if _store.is_ready:
            return

        KB_DIR.mkdir(exist_ok=True)
        documents = []

        for filepath in KB_DIR.glob("*.txt"):
            try:
                text = filepath.read_text(encoding="utf-8").strip()
                if text:
                    documents.append((filepath.name, text))
                    logger.info(f"RAG: Loaded '{filepath.name}' ({len(text)} chars)")
            except Exception as e:
                logger.error(f"RAG: Failed to load {filepath}: {e}")

        if not documents:
            logger.info("RAG: No knowledge base documents found. RAG disabled.")
            return

        await _store.build(documents)


async def retrieve_context(query: str) -> Optional[str]:
    """Retrieve relevant context for a query. Returns formatted string or None."""
    if not _store.is_ready:
        try:
            await init_rag()
        except Exception as e:
            logger.warning(f"RAG init failed: {e}")
            return None

    if not _store.is_ready:
        return None

    results = await _store.search(query)
    if not results:
        return None

    context_parts = [f"[Source: {src}]\n{chunk}" for chunk, score, src in results]
    return "\n\n".join(context_parts)


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 160) -> List[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    chunks = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            for sep in [". ", "! ", "? ", "\n\n", "\n"]:
                pos = text.rfind(sep, start, end)
                if pos > start + overlap:
                    end = pos + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if start >= len(text):
            break

    return chunks