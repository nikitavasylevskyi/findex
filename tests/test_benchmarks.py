"""pytest-benchmark on search and build_index (lab 8).

A synthetic corpus, the same on every machine, so that CI can compare a run with
the previous one and fail on a slowdown. Run with ``--benchmark-enable``; in
the ordinary test run each benchmark executes once as a plain test.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.index import Index, build_index
from findex.ranking import SCORERS, Searcher


@pytest.fixture(scope="module")
def synthetic_corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    rng = random.Random(8)
    vocab = [f"w{i}" for i in range(3000)]
    weights = [1 / (i + 1) for i in range(len(vocab))]  # Zipf-like: a few very frequent words
    root = tmp_path_factory.mktemp("synthetic")
    for d in range(400):
        words = rng.choices(vocab, weights, k=rng.randint(200, 1200))
        (root / f"doc{d:04}.txt").write_text(" ".join(words), encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def synthetic_index(synthetic_corpus: Path) -> Index:
    return build_index(synthetic_corpus, positions=True)


def test_bench_build_index(benchmark: BenchmarkFixture, synthetic_corpus: Path) -> None:
    index = benchmark.pedantic(build_index, args=(synthetic_corpus,), kwargs={"positions": True},
                               rounds=3, iterations=1)
    assert index.num_docs == 400


@pytest.mark.parametrize("query", ["w0", "w2500", "w1 w2 w3", "w1 OR w2 OR w3"])
def test_bench_search(benchmark: BenchmarkFixture, synthetic_index: Index, query: str) -> None:
    searcher = Searcher(synthetic_index)

    def run() -> int:
        searcher.matching.cache_clear()
        return len(searcher.search(query, SCORERS["bm25"], k=10, snippets=False))

    assert benchmark(run) >= 0
