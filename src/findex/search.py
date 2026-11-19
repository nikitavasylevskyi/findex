"""Search a saved index.

    python -m findex.search index.bin 'python AND (async OR await) NOT java "event loop"'
    python -m findex.search index.bin "walrus operator" --scorer tfidf -k 5
    python -m findex.search index.bin "async await" --boolean --engine set
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from findex import boolean, storage
from findex.index import Index
from findex.ranking import SCORERS, Searcher


@contextmanager
def open_index(path: str | Path) -> Iterator[Index]:
    """``with open_index("index.bin") as index:`` — data is released on exit, also after an exception."""
    index = storage.load(path)
    try:
        yield index
    finally:
        index.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m findex.search")
    parser.add_argument("index", type=Path)
    parser.add_argument("query", nargs="+", help="one or more queries; repeats show the cache")
    parser.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--boolean", action="store_true", help="unranked Boolean search of lab 2")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    args = parser.parse_args(argv)

    with open_index(args.index) as index:
        print(repr(index))
        if args.boolean:
            for q in args.query:
                t0 = time.perf_counter()
                hits = boolean.search(index, q, args.engine)
                ms = (time.perf_counter() - t0) * 1000
                for doc_id in hits[: args.k]:
                    print(f"{doc_id:5}  {index.docs[doc_id].title}")
                print(f"{len(hits)} documents, {ms:.2f} ms ({args.engine})")
            return 0
        searcher = Searcher(index)
        for q in args.query:
            print(f"\n=== {q}  [{args.scorer}]")
            for rank, r in enumerate(searcher.search(q, SCORERS[args.scorer], args.k), 1):
                print(f"{rank:2}. {r.score:6.2f}  {r.title}")
                print(f"    {r.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
