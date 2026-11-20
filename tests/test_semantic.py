"""Semantic and hybrid search with a deterministic stand-in for the embedding model."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from conftest import doc_id
from fastapi.testclient import TestClient

from findex.corpus import iter_documents
from findex.hybrid import ModeSearcher, SemanticUnavailable, bag_of_words
from findex.index import Index
from findex.ranking import Searcher
from findex.semantic import SemanticIndex, build_embeddings, chunk_text, normalize_rows, rrf
from findex.web.app import Engine, create_app, get_engine, get_modes
from findex.web.settings import Settings

SYNONYMS = {"assigns": "walrus", "operator": "walrus", "coroutine": "async", "concurrency": "async"}


class HashingEncoder:
    """Bag of words hashed into 64 dimensions; a few synonyms share a dimension, like a real model would."""

    name = "test/hashing-64"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = SYNONYMS.get(word.strip(".,;:=()"), word.strip(".,;:=()"))
                out[row, zlib.crc32(word.encode()) % 64] += 1.0
        return out


@pytest.fixture(scope="module")
def semantic(corpus_dir: Path) -> SemanticIndex:
    return build_embeddings(iter_documents(corpus_dir), HashingEncoder())


@pytest.fixture
def modes(index: Index, semantic: SemanticIndex) -> ModeSearcher:
    return ModeSearcher(Searcher(index), semantic, HashingEncoder())


def test_chunk_text_overlaps_and_keeps_offsets() -> None:
    text = " ".join(f"w{i}" for i in range(450))
    chunks = chunk_text(text, words=200, stride=150)
    assert [len(c.split()) for _, c in chunks] == [200, 200, 150]
    for start, chunk in chunks:
        assert text[start:start + len(chunk)] == chunk
    assert chunks[1][1].split()[0] == "w150"
    assert chunk_text("") == []


def test_vectors_are_unit_length(semantic: SemanticIndex) -> None:
    assert np.allclose(np.linalg.norm(semantic.vectors, axis=1), 1.0, atol=1e-5)
    assert len(semantic.chunk_doc) == len(semantic.vectors) == len(semantic.chunk_start)


def test_save_and_load(tmp_path: Path, semantic: SemanticIndex) -> None:
    semantic.save(tmp_path / "emb.npy")
    loaded = SemanticIndex.load(tmp_path / "emb.npy")
    assert np.array_equal(loaded.vectors, semantic.vectors)
    assert loaded.chunk_doc.tolist() == semantic.chunk_doc.tolist()
    assert loaded.model == "test/hashing-64" and loaded.doc_paths == semantic.doc_paths


def test_rank_gives_each_document_its_best_fragment(semantic: SemanticIndex) -> None:
    q = normalize_rows(HashingEncoder().encode(["walrus"]))[0]
    ranked = semantic.rank(q, 100)
    ids = [d for d, _, _ in ranked]
    assert len(ids) == len(set(ids)) == semantic.num_docs
    sims = [s for _, s, _ in ranked]
    assert sims == sorted(sims, reverse=True)
    best = semantic.vectors @ q
    for d, s, _ in ranked:
        assert s == pytest.approx(best[semantic.chunk_doc == d].max())


def test_semantic_finds_document_without_shared_words(modes: ModeSearcher, index: Index) -> None:
    # the stand-in model maps "concurrency" and "coroutine" to the dimension of "async"
    assert modes.search("concurrency coroutine", "lexical").total == 0
    top = modes.search("concurrency coroutine", "semantic", k=2).results
    assert {r.doc_id for r in top} & {doc_id(index, "a.rst"), doc_id(index, "g.rst")}


def test_hybrid_fuses_both_rankings(modes: ModeSearcher, index: Index) -> None:
    page = modes.search("walrus", "hybrid", k=3)
    assert page.results[0].doc_id in {doc_id(index, "c.rst"), doc_id(index, "d.rst"), doc_id(index, "pep-0572.rst")}
    assert all(r.score <= 2 / 61 for r in page.results)  # an RRF score, at most first in both lists


def test_rrf() -> None:
    fused = rrf([[1, 2, 3], [3, 1]], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert rrf([[]]) == []


def test_bag_of_words() -> None:
    assert bag_of_words("Event loop") == "event OR loop"
    assert bag_of_words('"event loop"') == '"event loop"'
    assert bag_of_words("a AND b") == "a AND b"


def test_without_embeddings_semantic_is_unavailable(index: Index) -> None:
    with pytest.raises(SemanticUnavailable):
        ModeSearcher(Searcher(index)).search("walrus", "semantic")


def test_embeddings_for_another_corpus_are_rejected(index: Index, semantic: SemanticIndex) -> None:
    other = SemanticIndex(semantic.vectors, semantic.chunk_doc, semantic.chunk_start, "m", ["x"])
    with pytest.raises(ValueError, match="another corpus"):
        ModeSearcher(Searcher(index), other, HashingEncoder())


def test_api_modes(modes: ModeSearcher) -> None:
    app = create_app(Settings(), load_index=False)
    engine = Engine(status="ok", searcher=modes.searcher, modes=modes)
    app.dependency_overrides[get_modes] = lambda: modes
    app.dependency_overrides[get_engine] = lambda: engine
    with TestClient(app) as client:
        assert client.get("/health").json()["modes"] == ["lexical", "semantic", "hybrid"]
        for mode in ("semantic", "hybrid"):
            body = client.get("/search", params={"q": "walrus operator", "mode": mode}).json()
            assert body["mode"] == mode and body["hits"]
        assert client.get("/search", params={"q": "x", "mode": "vector"}).status_code == 422
        page = client.get("/", params={"q": "walrus", "mode": "hybrid"}).text
        assert 'value="hybrid" selected' in page and "<mark>walrus</mark>" in page
        assert client.get("/stats").json()["semantic_model"] == "test/hashing-64"

