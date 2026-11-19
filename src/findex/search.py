"""Load a saved index and run a Boolean query.

    python -m findex.search index.bin "async await NOT asyncio" [--engine merge|set]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex import boolean, storage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m findex.search")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--engine", choices=["merge", "set"], default="merge")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = storage.load(args.index)
    t1 = time.perf_counter()
    hits = boolean.search(index, args.query, args.engine)
    t2 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    for doc_id in hits[: args.limit]:
        meta = index.docs[doc_id]
        print(f"{doc_id:5}  {meta.title}")
    print(f"{len(hits)} documents; load: {(t1 - t0) * 1000:.0f} ms, "
          f"search ({args.engine}): {(t2 - t1) * 1000:.2f} ms, peak memory: {peak / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
