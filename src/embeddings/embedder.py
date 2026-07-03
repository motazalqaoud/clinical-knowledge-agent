"""Local, offline text embedding via sentence-transformers.

Clinical note: embeddings are computed entirely on the local machine —
no network calls at inference time — supporting HIPAA-friendly, fully
local deployments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class ClinicalEmbedder:
    """Wraps a sentence-transformers model for local, offline embedding."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model: Any | None = None,
    ) -> None:
        """Initializes the embedder.

        Args:
            model_name: Name of the sentence-transformers model to load
                lazily on first use. Ignored if ``model`` is provided.
            model: An already-constructed model object (real
                SentenceTransformer or a test double) exposing an
                ``encode(texts, batch_size=...) -> array-like`` method. If
                given, it is used directly and the real model is never
                downloaded or imported.
        """
        self._model_name = model_name
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encodes a list of texts into L2-normalized embedding vectors.

        Args:
            texts: Texts to embed.
            batch_size: Batch size passed through to the underlying model.

        Returns:
            A float32 array of shape (len(texts), embedding_dim), with
            each row L2-normalized so that cosine similarity between two
            embeddings equals their dot product (matching FAISS's
            IndexFlatIP).

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must not be empty")

        model = self._ensure_model()
        vectors = np.asarray(
            model.encode(texts, batch_size=batch_size), dtype=np.float32
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode_query(self, query: str) -> np.ndarray:
        """Encodes a single query string.

        Args:
            query: The query text to embed.

        Returns:
            A 1-D L2-normalized float32 embedding vector.
        """
        return self.encode([query])[0]
