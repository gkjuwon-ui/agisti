"""
Document Retriever — vector search over external knowledge corpora.

Provides semantic search over ArXiv, Wikipedia, textbooks, and
custom document corpora using FAISS indices and sentence embeddings.

Used by Ceiling Breaker Level 2 (RAG Surgery) to find relevant
documents for problems the model fails.

Design: §11.1.2 — DocumentRetriever.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from agisti.types import Document

logger = logging.getLogger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Source Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class SourceConfig:
    """Configuration for a document source."""
    name: str
    index_path: str
    docs_path: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_docs: int = 100_000
    chunk_size: int = 512  # tokens per chunk


DEFAULT_SOURCES = {
    "arxiv": SourceConfig(
        name="arxiv",
        index_path="indices/arxiv-abstracts-2024.faiss",
        docs_path="indices/arxiv-abstracts-2024.jsonl",
        max_docs=100_000,
    ),
    "wikipedia": SourceConfig(
        name="wikipedia",
        index_path="indices/wikipedia-en-chunks.faiss",
        docs_path="indices/wikipedia-en-chunks.jsonl",
        max_docs=5_000_000,
    ),
    "textbooks": SourceConfig(
        name="textbooks",
        index_path="indices/open-textbooks-stem.faiss",
        docs_path="indices/open-textbooks-stem.jsonl",
        max_docs=500_000,
    ),
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embedding Engine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class EmbeddingEngine:
    """
    Computes embeddings using sentence-transformers.

    Lazily loads the model on first use. Supports batching
    for efficient encoding of multiple queries.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get embedding dimension
                test_emb = self._model.encode(["test"], normalize_embeddings=True)
                self._dimension = test_emb.shape[1]
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for document retrieval. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model  # trigger load
        assert self._dimension is not None
        return self._dimension

    def encode(
        self,
        texts: list[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> Any:
        """
        Encode texts to embeddings.

        Returns numpy array of shape (len(texts), dimension).
        """
        import numpy as np
        return self.model.encode(
            texts,
            normalize_embeddings=normalize,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)

    def encode_single(self, text: str, normalize: bool = True) -> Any:
        """Encode a single text to embedding vector."""
        return self.encode([text], normalize=normalize)[0]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FAISS Index Wrapper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class FAISSIndex:
    """
    Wrapper around FAISS index for vector similarity search.

    Supports loading pre-built indices and building new ones.
    """

    def __init__(self, dimension: int | None = None):
        self._index: Any = None
        self._dimension = dimension
        self._doc_map: dict[int, dict[str, Any]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    def load(self, path: str) -> None:
        """Load a pre-built FAISS index from disk."""
        try:
            import faiss
            self._index = faiss.read_index(path)
            self._dimension = self._index.d
            logger.info(
                "Loaded FAISS index from %s: %d vectors, dim=%d",
                path, self._index.ntotal, self._dimension,
            )
        except ImportError:
            raise ImportError(
                "faiss-cpu or faiss-gpu required for document retrieval. "
                "Install with: pip install faiss-cpu"
            )
        except Exception as e:
            logger.error("Failed to load FAISS index from %s: %s", path, e)
            raise

    def build(
        self,
        embeddings: Any,  # numpy array (N, D)
        use_ivf: bool = True,
        nlist: int = 100,
    ) -> None:
        """Build a new FAISS index from embeddings."""
        import faiss
        import numpy as np

        n, d = embeddings.shape
        self._dimension = d

        if use_ivf and n > 10000:
            # IVF index for large datasets
            nlist = min(nlist, int(n ** 0.5))
            quantizer = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantizer, d, nlist)
            self._index.train(embeddings.astype(np.float32))
            self._index.add(embeddings.astype(np.float32))
            self._index.nprobe = min(10, nlist)
        else:
            # Flat index for smaller datasets
            self._index = faiss.IndexFlatIP(d)
            self._index.add(embeddings.astype(np.float32))

        logger.info("Built FAISS index: %d vectors, dim=%d", n, d)

    def search(
        self,
        query_vec: Any,  # numpy array (1, D) or (D,)
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Search for nearest neighbors.

        Returns list of (doc_id, score) tuples.
        """
        import numpy as np

        if self._index is None:
            return []

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        scores, ids = self._index.search(
            query_vec.astype(np.float32), top_k,
        )

        results: list[tuple[int, float]] = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id >= 0:
                results.append((int(doc_id), float(score)))

        return results

    def save(self, path: str) -> None:
        """Save index to disk."""
        if self._index is None:
            raise ValueError("No index to save")
        import faiss
        faiss.write_index(self._index, path)

    def load_doc_map(self, path: str) -> None:
        """Load document metadata from JSONL file."""
        self._doc_map.clear()
        p = Path(path)
        if not p.exists():
            logger.warning("Doc map file not found: %s", path)
            return

        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            data = json.loads(line)
            self._doc_map[i] = data

        logger.info("Loaded %d document entries from %s", len(self._doc_map), path)

    def get_doc(self, doc_id: int) -> dict[str, Any] | None:
        """Get document metadata by ID."""
        return self._doc_map.get(doc_id)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Document Retriever
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DocumentRetriever:
    """
    External knowledge retriever using FAISS vector search.

    Searches across multiple document sources (ArXiv, Wikipedia,
    textbooks) to find relevant context for problems the model fails.

    Used by Level 2 ceiling breaker to provide context that helps
    the model solve problems, then bake that context knowledge into
    the model weights via activation contrast.
    """

    def __init__(
        self,
        sources: list[str] | None = None,
        source_configs: dict[str, SourceConfig] | None = None,
    ):
        self.source_configs = source_configs or {
            k: v for k, v in DEFAULT_SOURCES.items()
            if sources is None or k in sources
        }
        self._embedder: EmbeddingEngine | None = None
        self._indices: dict[str, FAISSIndex] = {}
        self._initialized: set[str] = set()

    def _ensure_initialized(self, source: str) -> None:
        """Lazily initialize a source index."""
        if source in self._initialized:
            return

        if source not in self.source_configs:
            raise ValueError(f"Unknown source: {source}")

        config = self.source_configs[source]

        # Initialize embedder if needed
        if self._embedder is None:
            self._embedder = EmbeddingEngine(config.embedding_model)

        # Load FAISS index
        index = FAISSIndex()
        index_path = Path(config.index_path)

        if index_path.exists():
            index.load(str(index_path))
            # Load doc map
            docs_path = Path(config.docs_path)
            if docs_path.exists():
                index.load_doc_map(str(docs_path))
        else:
            logger.warning(
                "FAISS index not found for %s at %s. "
                "Build with DocumentIndexBuilder first.",
                source, config.index_path,
            )

        self._indices[source] = index
        self._initialized.add(source)

    def search(
        self,
        query: str,
        top_k: int = 3,
        max_tokens_per_doc: int = 512,
        sources: list[str] | None = None,
    ) -> list[Document]:
        """
        Semantic search across document sources.

        Args:
            query: Natural language search query.
            top_k: Number of documents to return (total across sources).
            max_tokens_per_doc: Maximum tokens per returned document.
            sources: Which sources to search (None = all).

        Returns:
            Top-k documents sorted by relevance score.
        """
        if self._embedder is None:
            first_source = next(iter(self.source_configs))
            self._ensure_initialized(first_source)

        assert self._embedder is not None

        query_vec = self._embedder.encode_single(query, normalize=True)

        all_results: list[Document] = []
        search_sources = sources or list(self.source_configs.keys())

        for source in search_sources:
            try:
                self._ensure_initialized(source)
            except Exception as e:
                logger.warning("Failed to initialize source %s: %s", source, e)
                continue

            index = self._indices.get(source)
            if index is None or not index.is_loaded:
                continue

            hits = index.search(query_vec, top_k=top_k)

            for doc_id, score in hits:
                doc_data = index.get_doc(doc_id)
                if doc_data is None:
                    continue

                text = doc_data.get("text", "")
                # Truncate to max tokens (approximate: 1 token ≈ 4 chars)
                max_chars = max_tokens_per_doc * 4
                if len(text) > max_chars:
                    text = text[:max_chars] + "..."

                all_results.append(Document(
                    source=source,
                    doc_id=doc_id,
                    score=score,
                    text=text,
                ))

        # Sort by score and take top_k
        all_results.sort(key=lambda d: -d.score)
        return all_results[:top_k]

    def search_for_problem(
        self,
        question: str,
        domain: str = "",
        top_k: int = 3,
    ) -> list[Document]:
        """
        Search for documents relevant to a specific problem.

        Constructs a specialized search query from the problem details.
        """
        query = question
        if domain:
            query = f"{domain}: {query}"

        return self.search(query, top_k=top_k)

    def get_context_string(
        self,
        documents: list[Document],
        max_total_tokens: int = 1536,
    ) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents: Documents to format.
            max_total_tokens: Maximum total tokens in output.

        Returns:
            Formatted string suitable for prompt inclusion.
        """
        if not documents:
            return ""

        parts: list[str] = []
        total_chars = 0
        max_chars = max_total_tokens * 4

        for i, doc in enumerate(documents):
            header = f"[Source {i+1}: {doc.source}]"
            remaining = max_chars - total_chars - len(header) - 10
            if remaining <= 0:
                break

            text = doc.text[:remaining] if len(doc.text) > remaining else doc.text
            parts.append(f"{header}\n{text}")
            total_chars += len(header) + len(text) + 2

        return "\n\n".join(parts)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Index Builder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DocumentIndexBuilder:
    """
    Builds FAISS indices from document corpora.

    Used offline to prepare indices for the DocumentRetriever.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ):
        self.embedder = EmbeddingEngine(embedding_model)
        self.batch_size = batch_size

    def build_from_jsonl(
        self,
        input_path: str,
        output_index_path: str,
        text_field: str = "text",
        max_docs: int | None = None,
    ) -> int:
        """
        Build a FAISS index from a JSONL file.

        Each line should have at least a text field.

        Args:
            input_path: Path to JSONL file.
            output_index_path: Where to save the FAISS index.
            text_field: Field name containing the text to embed.
            max_docs: Maximum number of documents to index.

        Returns:
            Number of documents indexed.
        """
        import numpy as np

        # Read documents
        texts: list[str] = []
        p = Path(input_path)

        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get(text_field, "")
            if text:
                texts.append(text)
            if max_docs and len(texts) >= max_docs:
                break

        if not texts:
            logger.warning("No documents found in %s", input_path)
            return 0

        logger.info("Encoding %d documents...", len(texts))

        # Encode in batches
        all_embeddings: list[Any] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.embedder.encode(batch, normalize=True)
            all_embeddings.append(embeddings)

        embeddings_matrix = np.concatenate(all_embeddings, axis=0)

        # Build index
        index = FAISSIndex()
        index.build(embeddings_matrix, use_ivf=len(texts) > 10000)
        index.save(output_index_path)

        logger.info(
            "Built index: %d documents → %s",
            len(texts), output_index_path,
        )

        return len(texts)

    def build_chunked(
        self,
        input_path: str,
        output_index_path: str,
        output_docs_path: str,
        text_field: str = "text",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        max_docs: int | None = None,
    ) -> int:
        """
        Build index with overlapping text chunks.

        Splits long documents into overlapping chunks before indexing.

        Args:
            input_path: Path to JSONL file.
            output_index_path: Where to save FAISS index.
            output_docs_path: Where to save chunk metadata.
            text_field: Field name for text content.
            chunk_size: Words per chunk.
            chunk_overlap: Overlap in words.
            max_docs: Maximum source documents.

        Returns:
            Number of chunks indexed.
        """
        import numpy as np

        chunks: list[dict[str, Any]] = []
        p = Path(input_path)
        doc_count = 0

        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get(text_field, "")
            if not text:
                continue

            doc_count += 1
            if max_docs and doc_count > max_docs:
                break

            # Split into chunks
            words = text.split()
            for start in range(0, len(words), chunk_size - chunk_overlap):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                if len(chunk_text) < 50:
                    continue

                chunks.append({
                    "text": chunk_text,
                    "source_doc": doc_count - 1,
                    "chunk_start": start,
                    "metadata": {
                        k: v for k, v in data.items() if k != text_field
                    },
                })

        if not chunks:
            return 0

        # Save chunk metadata
        out_docs = Path(output_docs_path)
        out_docs.parent.mkdir(parents=True, exist_ok=True)
        with out_docs.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        # Encode and build index
        texts = [c["text"] for c in chunks]
        all_embeddings: list[Any] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self.embedder.encode(batch, normalize=True)
            all_embeddings.append(embeddings)

        embeddings_matrix = np.concatenate(all_embeddings, axis=0)

        index = FAISSIndex()
        index.build(embeddings_matrix, use_ivf=len(chunks) > 10000)
        index.save(output_index_path)

        logger.info(
            "Built chunked index: %d docs → %d chunks → %s",
            doc_count, len(chunks), output_index_path,
        )

        return len(chunks)
