"""Corpus statistics in one pass, and a deliberately greedy variant for comparison."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import Document, guess_title, iter_documents, iter_paths
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    bytes: int
    top: list[tuple[str, int]]


def stats_lazy(root: str | Path, top_n: int = 50) -> CorpusStats:
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


def stats_greedy(root: str | Path, top_n: int = 50) -> CorpusStats:
    """The same numbers, but every stage materialises a full list first."""
    paths = list(iter_paths(Path(root)))
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in paths]
    documents = [Document(i, p, guess_title(p, t), t) for i, (p, t) in enumerate(zip(paths, texts, strict=True))]
    token_lists = [list(tokenize(d.text)) for d in documents]
    all_tokens = [tok for toks in token_lists for tok in toks]
    counts = Counter(all_tokens)
    size = sum(len(t.encode("utf-8")) for t in texts)
    return CorpusStats(len(documents), len(all_tokens), len(counts), size, counts.most_common(top_n))
