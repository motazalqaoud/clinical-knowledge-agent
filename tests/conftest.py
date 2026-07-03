import hashlib

import numpy as np
import pytest

from src.embeddings.embedder import ClinicalEmbedder
from src.ingestion.document_loader import LoadedDocument


class FakeSentenceTransformer:
    """Deterministic fake standing in for a real SentenceTransformer.

    Encodes each text as a one-hot vector selected by a hash of the text,
    so identical texts always produce identical (similarity 1.0) vectors
    and distinct texts land in different buckets (similarity ~0.0),
    without any network access or model download.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, texts, batch_size: int = 32, **kwargs) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            bucket = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % self.dim
            vectors[i, bucket] = 1.0
        return vectors


class FakeGenerator:
    """Callable fake matching the transformers text2text-generation pipeline
    output contract: __call__(prompt, **kw) -> [{"generated_text": str}]."""

    def __init__(self, response: str):
        self.response = response
        self.call_count = 0

    def __call__(self, prompt, **kwargs):
        self.call_count += 1
        return [{"generated_text": self.response}]


@pytest.fixture
def fake_embedder() -> ClinicalEmbedder:
    return ClinicalEmbedder(model=FakeSentenceTransformer())


@pytest.fixture
def sample_loaded_document() -> LoadedDocument:
    text = (
        "Metformin is dosed at 500mg BID for most adults with type 2 diabetes. "
        "Target HbA1c < 7.0% for most adults. Give insulin 10 units QHS as needed."
    )
    return LoadedDocument(text=text, source="sample.txt", page=1)
