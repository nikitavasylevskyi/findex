"""Inverted index: term -> postings sorted by doc_id.

    python -m findex.index data/ --out index.bin [--format pickle|binary] [--positions]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


@dataclass
class InvertedIndex:
    postings: dict[str, list[Posting]] = field(default_factory=dict)
    docs: list[DocMeta] = field(default_factory=list)
    with_positions: bool = False

    @property
    def num_docs(self) -> int:
        return len(self.docs)


def build_index(root: str | Path, positions: bool = False) -> InvertedIndex:
    """One pass over the lazy pipeline of lab 1.

    Documents arrive in increasing doc_id order, so appending keeps every
    postings list sorted without an extra sort.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    docs: list[DocMeta] = []
    for doc in iter_documents(root):
        term_pos: defaultdict[str, list[int]] = defaultdict(list)
        length = 0
        for pos, tok in enumerate(tokenize(doc.text)):
            term_pos[tok].append(pos)
            length += 1
        for term, plist in term_pos.items():
            postings[term].append(Posting(doc.doc_id, len(plist), tuple(plist) if positions else ()))
        docs.append(DocMeta(doc.doc_id, str(doc.path), doc.title, length))
    return InvertedIndex(dict(postings), docs, positions)


def main(argv: list[str] | None = None) -> int:
    from findex import storage

    parser = argparse.ArgumentParser(prog="python -m findex.index")
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=["pickle", "binary"], default="binary")
    parser.add_argument("--positions", action="store_true", help="store token positions")
    args = parser.parse_args(argv)

    tracemalloc.start()
    t0 = time.perf_counter()
    index = build_index(args.root, positions=args.positions)
    t1 = time.perf_counter()
    storage.save(index, args.out, args.format)
    t2 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"documents: {index.num_docs}, terms: {len(index.postings)}")
    print(f"build: {t1 - t0:.2f} s, save ({args.format}): {t2 - t1:.2f} s, "
          f"file: {args.out.stat().st_size / 2**20:.1f} MiB, peak memory: {peak / 2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
