from pathlib import Path

import pytest
from conftest import doc_id

from findex import storage
from findex.index import Index, Posting
from findex.ranking import BM25, Scorer, Searcher, TfIdf, make_snippet
from findex.search import open_index

SCORERS: list[Scorer] = [TfIdf(), BM25()]


@pytest.mark.parametrize("scorer", SCORERS, ids=lambda s: s.name)
def test_rare_term_beats_frequent(index: Index, scorer: Scorer) -> None:
    p = Posting(0, 1)
    assert scorer.score(index, index.df("walrus"), p) > scorer.score(index, index.df("the"), p)


def test_bm25_saturates(index: Index) -> None:
    bm = BM25()
    s19 = bm.score(index, 2, Posting(0, 19))
    s20 = bm.score(index, 2, Posting(0, 20))
    assert s20 - s19 < 0.01 * s20


def test_short_document_wins(index: Index) -> None:
    results = Searcher(index).search("walrus", BM25(), snippets=False)
    assert results[0].doc_id == doc_id(index, "c.rst")
    assert results[-1].doc_id == doc_id(index, "d.rst")


def test_results_sorted_and_k(index: Index) -> None:
    results = Searcher(index).search("the OR python OR event", BM25(), k=3, snippets=False)
    assert len(results) == 3
    assert results == sorted(results, reverse=True)


def test_cache_hit(index: Index) -> None:
    s = Searcher(index)
    s.search("walrus", snippets=False)
    s.search("walrus", snippets=False)
    assert s.matching.cache_info().hits == 1


def test_snippet(corpus_dir: Path) -> None:
    snip = make_snippet(str(corpus_dir / "a.rst"), ("event", "loop"))
    assert "**event** **loop**" in snip


def test_open_index_releases_on_error(index: Index, tmp_path: Path) -> None:
    path = tmp_path / "i.bin"
    storage.save(index, path)
    kept: Index | None = None
    with pytest.raises(RuntimeError), open_index(path) as idx:
        kept = idx
        assert len(idx) == len(index)
        raise RuntimeError("boom")
    assert kept is not None and len(kept) == 0
