"""End-to-end check on the real corpus; skipped when data/ has not been downloaded."""

from pathlib import Path

import pytest

from findex.index import build_index
from findex.ranking import Searcher

DATA = Path(__file__).resolve().parent.parent / "data"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (DATA / "peps").is_dir(), reason="corpus not downloaded"),
]


def test_assignment_expressions_finds_pep_572() -> None:
    index = build_index(DATA / "peps")
    top = Searcher(index).search("assignment expressions", k=5, snippets=False)
    assert any(r.title.startswith("PEP 572") for r in top)


def test_numpy_ranks_like_lab3_on_the_real_corpus() -> None:
    from findex.ranking import SCORERS

    searcher = Searcher(build_index(DATA))
    for query in ("python", "zipapp", "asyncio event loop", "asyncio OR event OR loop", "the", "type hints"):
        for name in ("bm25", "tfidf"):
            a = searcher.search(query, SCORERS[name], k=20, snippets=False, engine="numpy")
            b = searcher.search(query, SCORERS[name], k=20, snippets=False, engine="python")
            assert [(r.doc_id, r.score) for r in a] == [(r.doc_id, r.score) for r in b], (query, name)
