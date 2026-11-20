"""Lab 8: the NumPy scorers must rank exactly like the per-posting scorers of lab 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex import storage
from findex.index import Index
from findex.ranking import SCORERS, Searcher

QUERIES = ["walrus", "the", "event loop", "python", "walrus OR python", "the NOT walrus", "async OR the OR year",
           "word1 OR word2 OR walrus", "\"event loop\"", "nothing-here"]


@pytest.mark.parametrize("scorer", ["bm25", "tfidf"])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize(("k", "offset"), [(10, 0), (1, 0), (2, 1), (3, 5)])
def test_numpy_order_matches_lab3(index: Index, query: str, scorer: str, k: int, offset: int) -> None:
    s = Searcher(index)
    a = s.search(query, SCORERS[scorer], k=k, offset=offset, snippets=False, engine="numpy")
    b = s.search(query, SCORERS[scorer], k=k, offset=offset, snippets=False, engine="python")
    assert [(r.doc_id, r.score) for r in a] == [(r.doc_id, r.score) for r in b]


def test_ties_at_the_cut_are_broken_by_doc_id(index: Index) -> None:
    # "the" in TF-IDF: idf of a term in most documents is small and many scores tie
    s = Searcher(index)
    for k in range(1, 6):
        a = s.search("the", SCORERS["tfidf"], k=k, snippets=False, engine="numpy")
        b = s.search("the", SCORERS["tfidf"], k=k, snippets=False, engine="python")
        assert [r.doc_id for r in a] == [r.doc_id for r in b]


@pytest.mark.parametrize("name", ["bm25", "tfidf"])
def test_score_array_equals_score(index: Index, name: str) -> None:
    scorer = SCORERS[name]
    for term in ("the", "walrus", "event", "python"):
        ids, tfs = index.term_arrays(term)
        arr = scorer.score_array(index, len(ids), ids, tfs)
        one = [scorer.score(index, len(ids), p) for p in index[term]]
        assert arr.tolist() == one  # bit for bit, not approximately


def test_term_arrays(index: Index) -> None:
    ids, tfs = index.term_arrays("the")
    assert ids.dtype == np.int32 and tfs.dtype == np.int32
    assert ids.tolist() == [p.doc_id for p in index["the"]]
    assert tfs.tolist() == [p.tf for p in index["the"]]
    assert index.term_arrays("no-such-term")[0].size == 0
    assert index.doc_lengths.tolist() == [d.length for d in index.docs]


def test_loaded_index_is_numpy_backed_and_equal(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.bin"
    storage.save(index, path)
    loaded = storage.load(path)
    assert loaded == index
    storage.save(loaded, tmp_path / "again.bin")  # saving what was loaded writes the same bytes
    assert (tmp_path / "again.bin").read_bytes() == path.read_bytes()


words = st.sampled_from(["the", "walrus", "event", "loop", "python", "async", "year", "match", "word7", "zzz"])


@settings(max_examples=150, deadline=None)
@given(st.lists(words, min_size=1, max_size=4), st.sampled_from(["", " OR "]), st.integers(1, 12))
def test_random_queries_rank_the_same(index: Index, terms: list[str], op: str, k: int) -> None:
    query = (op or " ").join(terms)
    s = Searcher(index)
    for name in ("bm25", "tfidf"):
        a = s.search(query, SCORERS[name], k=k, snippets=False, engine="numpy")
        b = s.search(query, SCORERS[name], k=k, snippets=False, engine="python")
        assert [(r.doc_id, r.score) for r in a] == [(r.doc_id, r.score) for r in b]
