"""Lab 8: what to profile, in a form cProfile, py-spy and Scalene can all run.

    uv run python scripts/profile_lab8.py index            # build_index on the full corpus
    uv run python scripts/profile_lab8.py search           # a three-term query, repeated
    uv run python -m cProfile -o out.prof scripts/profile_lab8.py search
    uv run py-spy record -o docs/profile/search.svg -- python scripts/profile_lab8.py search
    uv run scalene --cli --cpu --reduced-profile scripts/profile_lab8.py search
"""

from __future__ import annotations

import sys
import time

from findex import storage
from findex.index import build_index
from findex.ranking import SCORERS, Searcher

QUERY = "asyncio event loop"
REPEAT = int(sys.argv[2]) if len(sys.argv) > 2 else 200


def profile_index() -> None:
    t0 = time.perf_counter()
    build_index("data", positions=True)
    print(f"build_index: {time.perf_counter() - t0:.2f} s", file=sys.stderr)


def profile_search(snippets: bool) -> None:
    searcher = Searcher(storage.load("index.bin"))
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        searcher.matching.cache_clear()  # every run parses and evaluates the query again
        searcher.search(QUERY, SCORERS["bm25"], k=10, snippets=snippets)
    dt = (time.perf_counter() - t0) / REPEAT
    print(f"search {QUERY!r}: {dt * 1000:.1f} ms per query (snippets={snippets})", file=sys.stderr)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "search"
    if what == "index":
        profile_index()
    elif what == "search":
        profile_search(snippets=True)
    elif what == "score":
        profile_search(snippets=False)
    else:
        raise SystemExit(f"unknown target {what!r}: index, search or score")
