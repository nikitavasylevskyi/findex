"""Measurements for lab 2: postings storage, save/load, merge vs set.

    uv run python scripts/bench_lab2.py data/
"""

from __future__ import annotations

import statistics
import sys
import tempfile
import time
import tracemalloc
from array import array
from dataclasses import dataclass
from pathlib import Path

from findex import boolean, storage
from findex.index import build_index


@dataclass(frozen=True)
class PlainPosting:
    doc_id: int
    tf: int


@dataclass(frozen=True, slots=True)
class SlotPosting:
    doc_id: int
    tf: int


def median_time(fn, repeat: int = 5) -> float:
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def traced(fn) -> tuple[object, int]:
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    obj = fn()
    after = tracemalloc.get_traced_memory()[0]
    tracemalloc.stop()
    return obj, after - before


def main(root: str) -> None:
    index = build_index(root)
    pairs = {t: [(p.doc_id, p.tf) for p in pl] for t, pl in index.postings.items()}
    n = sum(len(v) for v in pairs.values())
    print(f"terms: {len(pairs)}, postings: {n}")

    variants = {
        "dataclass": lambda: {t: [PlainPosting(d, f) for d, f in v] for t, v in pairs.items()},
        "dataclass(slots=True)": lambda: {t: [SlotPosting(d, f) for d, f in v] for t, v in pairs.items()},
        "array('I') pairs": lambda: {t: array("I", [x for d, f in v for x in (d, f)]) for t, v in pairs.items()},
    }
    print("\n| storage | MiB | bytes per posting |")
    for name, make in variants.items():
        obj, size = traced(make)
        print(f"| {name} | {size / 2**20:.1f} | {size / n:.1f} |")
        del obj

    print("\n| format | positions | file MiB | save s | load s |")
    with tempfile.TemporaryDirectory() as tmp:
        for positions in (False, True):
            idx = index if not positions else build_index(root, positions=True)
            for fmt in ("pickle", "binary"):
                path = Path(tmp) / f"index.{fmt}"
                save_t = median_time(lambda: storage.save(idx, path, fmt), 3)
                load_t = median_time(lambda: storage.load(path), 3)
                print(f"| {fmt} | {'yes' if positions else 'no'} | {path.stat().st_size / 2**20:.1f} "
                      f"| {save_t:.2f} | {load_t:.2f} |")

    by_df = sorted(index.postings, key=lambda t: len(index.postings[t]))
    frequent = by_df[-2:]
    rare = [t for t in by_df if len(index.postings[t]) >= 2][:2]
    mixed = [frequent[-1], by_df[len(by_df) // 2]]
    print("\n| query | df | result | merge µs | set µs |")
    for a, b in (frequent, rare, mixed):
        q = f"{a} {b}"
        df = f"{len(index.postings[a])} / {len(index.postings[b])}"
        res = boolean.search(index, q, "merge")
        assert res == boolean.search(index, q, "set")
        tm = median_time(lambda: boolean.search(index, q, "merge"), 200) * 1e6
        ts = median_time(lambda: boolean.search(index, q, "set"), 200) * 1e6
        print(f"| `{q}` | {df} | {len(res)} | {tm:.0f} | {ts:.0f} |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data")
