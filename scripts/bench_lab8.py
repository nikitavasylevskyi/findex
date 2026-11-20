"""Lab 8: the scorer of lab 3 (a Python call per posting) against NumPy array expressions.

Only scoring and top-k are timed: the query is parsed and matched once beforehand,
snippets are not made. Median of many runs, arrays already cached in the index.

    uv run python scripts/bench_lab8.py --index index.bin
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_lab5 import cpu_model  # noqa: E402

from findex import storage  # noqa: E402
from findex.ranking import SCORERS, Searcher  # noqa: E402

CASES = [
    ("frequent term", "python"),
    ("rare term", "zipapp"),
    ("three terms, AND", "asyncio event loop"),
    ("three terms, OR", "asyncio OR event OR loop"),
]


def timeit(fn, repeat: int) -> float:  # type: ignore[no-untyped-def]
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index.bin")
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    searcher = Searcher(storage.load(args.index))
    index = searcher.index
    print(f"{cpu_model()}; {index.num_docs} documents")
    print("| query | | candidates | postings | Python, ms | NumPy, ms | speedup |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for label, query in CASES:
        doc_ids, terms = searcher.matching(query)
        postings = sum(index.df(t) for t in terms)
        for t in terms:
            index.term_arrays(t)  # warm the cache: the service does this once per term
        for name in ("bm25", "tfidf"):
            scorer = SCORERS[name]
            py = timeit(lambda: searcher.top_python(doc_ids, terms, scorer, 10), args.repeat)
            np_ = timeit(lambda: searcher.top_numpy(doc_ids, terms, scorer, 10), args.repeat)
            print(f"| {label}: `{query}` | {name} | {len(doc_ids)} | {postings} | {py:.3f} | {np_:.3f} "
                  f"| {py / np_:.1f}× |")


if __name__ == "__main__":
    main()
