"""Corpus statistics in one pass, and a deliberately greedy variant for comparison.

    python -m findex.stats data/            # lazy pipeline
    python -m findex.stats data/ --greedy   # everything in lists
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents, iter_paths, guess_title, Document
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    bytes: int
    top: list[tuple[str, int]]


def stats_lazy(root: Path, top_n: int = 50) -> CorpusStats:
    """One pass over a generator pipeline: documents -> tokens -> Counter."""
    counts: Counter[str] = Counter()
    docs = size = 0
    for doc in iter_documents(root):
        docs += 1
        size += len(doc.text.encode("utf-8"))
        # Counter.update consumes the generator in C, token by token,
        # without building a list of the document's tokens.
        counts.update(tokenize(doc.text))
    return CorpusStats(docs, counts.total(), len(counts), size, counts.most_common(top_n))


def stats_greedy(root: Path, top_n: int = 50) -> CorpusStats:
    """The same numbers, but every stage materialises a full list first."""
    paths = list(iter_paths(root))
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in paths]
    documents = [Document(i, p, guess_title(p, t), t) for i, (p, t) in enumerate(zip(paths, texts))]
    token_lists = [list(tokenize(d.text)) for d in documents]
    all_tokens = [tok for toks in token_lists for tok in toks]
    counts = Counter(all_tokens)
    size = sum(len(t.encode("utf-8")) for t in texts)
    return CorpusStats(len(documents), len(all_tokens), len(counts), size, counts.most_common(top_n))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m findex.stats")
    parser.add_argument("root", type=Path)
    parser.add_argument("--greedy", action="store_true", help="use the list-based version")
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args(argv)

    fn = stats_greedy if args.greedy else stats_lazy
    # Two runs: tracemalloc slows every allocation down, so time is measured
    # without it and the peak memory in a separate traced run.
    t0 = time.perf_counter()
    s = fn(args.root, args.top)
    elapsed = time.perf_counter() - t0
    tracemalloc.start()
    fn(args.root, args.top)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"mode:        {'greedy' if args.greedy else 'lazy'}")
    print(f"documents:   {s.documents}")
    print(f"size:        {s.bytes / 2**20:.1f} MiB")
    print(f"tokens:      {s.tokens}")
    print(f"vocabulary:  {s.vocabulary}")
    print(f"time:        {elapsed:.2f} s")
    print(f"peak memory: {peak / 2**20:.1f} MiB")
    print(f"top {args.top}:")
    for i, (term, n) in enumerate(s.top, 1):
        print(f"  {i:2}. {term:<15} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
