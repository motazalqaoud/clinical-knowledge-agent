import numpy as np
import pytest

from src.embeddings.embedder import ClinicalEmbedder
from tests.conftest import FakeSentenceTransformer


def test_encode_returns_normalized_vectors_of_expected_shape():
    embedder = ClinicalEmbedder(model=FakeSentenceTransformer(dim=32))

    vectors = embedder.encode(["hello world", "second text"])

    assert vectors.shape == (2, 32)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_encode_query_matches_first_row_of_encode():
    embedder = ClinicalEmbedder(model=FakeSentenceTransformer())

    query_vec = embedder.encode_query("some query")
    batch_vec = embedder.encode(["some query"])[0]

    assert query_vec.ndim == 1
    assert np.array_equal(query_vec, batch_vec)


def test_encode_empty_list_raises():
    embedder = ClinicalEmbedder(model=FakeSentenceTransformer())

    with pytest.raises(ValueError):
        embedder.encode([])


def test_real_model_never_constructed_when_fake_is_injected(monkeypatch):
    import sentence_transformers

    def _raise(*args, **kwargs):
        raise AssertionError("real SentenceTransformer should never be constructed")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _raise)

    embedder = ClinicalEmbedder(model=FakeSentenceTransformer())
    vectors = embedder.encode(["no network needed"])

    assert vectors.shape[0] == 1
