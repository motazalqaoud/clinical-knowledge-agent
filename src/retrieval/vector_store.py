"""Local FAISS-backed vector store for retrieval.

Clinical note: the index and its metadata are persisted entirely on the
local filesystem — no vectors or document text are ever sent to a
third-party service, supporting HIPAA-friendly deployments.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_INDEX_DIR = Path("data/faiss_index")
_INDEX_FILENAME = "index.faiss"
_METADATA_FILENAME = "metadata.json"


@dataclass
class SearchResult:
    """A single retrieved passage.

    Args:
        text: The retrieved chunk's text.
        source: The originating document's filename.
        page: The originating document's page number.
        score: Similarity score (inner product on normalized vectors,
            i.e. cosine similarity), higher is more similar.
    """

    text: str
    source: str
    page: int
    score: float


class FAISSVectorStore:
    """Local vector store wrapping a FAISS inner-product index.

    Clinical note: embeddings must be L2-normalized (as `ClinicalEmbedder`
    produces) so that inner-product search is equivalent to cosine
    similarity.
    """

    def __init__(self, dim: int) -> None:
        """Initializes an empty store.

        Args:
            dim: Dimensionality of the embedding vectors this store holds.
        """
        import faiss

        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._metadata: list[dict] = []

    def add(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        sources: list[str],
        pages: list[int],
    ) -> None:
        """Adds embeddings and their associated metadata to the store.

        Args:
            embeddings: Array of shape (n, dim) of L2-normalized vectors.
            texts: Chunk text for each embedding.
            sources: Source filename for each embedding.
            pages: Page number for each embedding.

        Raises:
            ValueError: If embeddings.shape[0] does not match the length
                of texts, sources, or pages.
        """
        n = embeddings.shape[0]
        if not (n == len(texts) == len(sources) == len(pages)):
            raise ValueError(
                "embeddings, texts, sources, and pages must all have the same "
                f"length, got {n}, {len(texts)}, {len(sources)}, {len(pages)}"
            )
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self._index.add(embeddings)
        for text, source, page in zip(texts, sources, pages):
            self._metadata.append({"text": text, "source": source, "page": page})

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[SearchResult]:
        """Searches for the k most similar stored embeddings.

        Args:
            query_embedding: A single L2-normalized query vector, shape (dim,).
            k: Maximum number of results to return.

        Returns:
            Up to k SearchResult, sorted by descending score. Returns an
            empty list (with a logged warning) if the store is empty.
        """
        if len(self._metadata) == 0:
            logger.warning("search() called on an empty vector store.")
            return []

        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(k, len(self._metadata))
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self._metadata[idx]
            results.append(
                SearchResult(
                    text=entry["text"],
                    source=entry["source"],
                    page=entry["page"],
                    score=float(score),
                )
            )
        return results

    def save(self, path: str | Path = DEFAULT_INDEX_DIR) -> None:
        """Persists the index and metadata to disk.

        Args:
            path: Directory to save into (created if it doesn't exist).
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / _INDEX_FILENAME))
        with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump({"dim": self._dim, "metadata": self._metadata}, f)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_INDEX_DIR) -> "FAISSVectorStore":
        """Loads a previously saved index and metadata from disk.

        Args:
            path: Directory previously written by `save`.

        Returns:
            A restored FAISSVectorStore.

        Raises:
            FileNotFoundError: If the index file does not exist at path.
        """
        import faiss

        path = Path(path)
        index_path = path / _INDEX_FILENAME
        metadata_path = path / _METADATA_FILENAME
        if not index_path.exists():
            raise FileNotFoundError(f"No FAISS index found at: {index_path}")

        with open(metadata_path, encoding="utf-8") as f:
            payload = json.load(f)

        store = cls(dim=payload["dim"])
        store._index = faiss.read_index(str(index_path))
        store._metadata = payload["metadata"]
        return store

    def __len__(self) -> int:
        return len(self._metadata)
