import numpy as np
import pytest

from src.retrieval.vector_store import FAISSVectorStore


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_add_and_search_round_trip():
    store = FAISSVectorStore(dim=4)
    embeddings = np.array(
        [_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])]
    )
    store.add(
        embeddings,
        texts=["alpha", "beta", "gamma"],
        sources=["a.txt", "b.txt", "c.txt"],
        pages=[1, 1, 1],
    )

    results = store.search(_unit([1, 0, 0, 0]), k=2)

    assert len(results) == 2
    assert results[0].text == "alpha"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)
    assert results[0].score >= results[1].score


def test_add_length_mismatch_raises():
    store = FAISSVectorStore(dim=4)
    embeddings = np.array([_unit([1, 0, 0, 0])])

    with pytest.raises(ValueError):
        store.add(embeddings, texts=["a", "b"], sources=["x"], pages=[1])


def test_save_and_load_round_trip(tmp_path):
    store = FAISSVectorStore(dim=4)
    embeddings = np.array([_unit([1, 0, 0, 0]), _unit([0, 1, 0, 0])])
    store.add(
        embeddings,
        texts=["alpha", "beta"],
        sources=["a.txt", "b.txt"],
        pages=[1, 2],
    )

    save_path = tmp_path / "index"
    store.save(save_path)
    reloaded = FAISSVectorStore.load(save_path)

    results = reloaded.search(_unit([1, 0, 0, 0]), k=1)
    assert len(reloaded) == 2
    assert results[0].text == "alpha"
    assert results[0].page == 1


def test_load_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        FAISSVectorStore.load(tmp_path / "does_not_exist")


def test_search_on_empty_store_returns_empty_and_warns(caplog):
    import logging

    store = FAISSVectorStore(dim=4)

    with caplog.at_level(logging.WARNING):
        results = store.search(_unit([1, 0, 0, 0]), k=3)

    assert results == []
    assert any("empty" in record.message for record in caplog.records)


def test_len_reflects_stored_count():
    store = FAISSVectorStore(dim=4)
    assert len(store) == 0

    store.add(
        np.array([_unit([1, 0, 0, 0])]), texts=["a"], sources=["a.txt"], pages=[1]
    )
    assert len(store) == 1
