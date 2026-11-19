"""Ranked retrieval: scorers, search and snippets."""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol

from findex.corpus import display_path, load_text
from findex.index import Index, Posting
from findex.query import Node, parse
from findex.timing import timed
from findex.tokenize import normalize

log = logging.getLogger("findex.search")
ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    """Anything with ``name`` and ``score`` can rank documents — no base class needed."""

    name: str

    def score(self, index: Index, df: int, posting: Posting) -> float: ...


class TfIdf:
    name = "tfidf"

    def score(self, index: Index, df: int, posting: Posting) -> float:
        idf = math.log(index.num_docs / df)
        return (1 + math.log(posting.tf)) * idf

    def __repr__(self) -> str:
        return "TfIdf()"


class BM25:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def score(self, index: Index, df: int, posting: Posting) -> float:
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = index.docs[posting.doc_id].length
        norm = self.k1 * (1 - self.b + self.b * dl / index.avg_doc_length)
        return idf * posting.tf * (self.k1 + 1) / (posting.tf + norm)

    def __repr__(self) -> str:
        return f"BM25(k1={self.k1}, b={self.b})"


SCORERS: dict[ScorerName, Scorer] = {"tfidf": TfIdf(), "bm25": BM25()}


@dataclass(frozen=True, order=True)
class SearchResult:
    score: float
    doc_id: int = field(compare=False)
    title: str = field(compare=False)
    path: str = field(compare=False)
    snippet: str = field(default="", compare=False)


class Searcher:
    """Ranks documents of one index. Caches which documents match a query."""

    def __init__(self, index: Index) -> None:
        self.index = index
        self.matching = lru_cache(maxsize=512)(self._matching)

    def _matching(self, query: str) -> tuple[tuple[int, ...], tuple[str, ...]]:
        node: Node = parse(query)
        return tuple(node.evaluate(self.index)), tuple(dict.fromkeys(node.terms()))

    @timed
    def search(self, query: str, scorer: Scorer | None = None, k: int = 10,
               snippets: bool = True) -> list[SearchResult]:
        scorer = scorer or SCORERS["bm25"]
        hits_before = self.matching.cache_info().hits
        doc_ids, terms = self.matching(query)
        if self.matching.cache_info().hits > hits_before:
            log.debug("cache hit for %r", query)
        candidates = set(doc_ids)
        scores: defaultdict[int, float] = defaultdict(float)
        for term in terms:
            plist = self.index.get(term, [])
            df = len(plist)
            for p in plist:
                if p.doc_id in candidates:
                    scores[p.doc_id] += scorer.score(self.index, df, p)
        for doc_id in candidates:  # e.g. "NOT x" alone: matching, but no positive terms
            scores.setdefault(doc_id, 0.0)
        top = heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))
        results: list[SearchResult] = []
        for doc_id, score in top:
            meta = self.index.docs[doc_id]
            snip = make_snippet(meta.path, terms) if snippets else ""
            results.append(SearchResult(score, doc_id, meta.title, display_path(meta.path), snip))
        return results


_WORD = re.compile(r"[^\W_]+(?:['’_]*[^\W_]+)*")


def make_snippet(path: str, terms: tuple[str, ...], radius: int = 80,
                 mark: tuple[str, str] = ("**", "**")) -> str:
    """Window of ±radius characters around the best occurrence, terms highlighted.

    "Best" = the occurrence whose window contains the most distinct query terms.
    """
    try:
        text = load_text(path)
    except (OSError, ValueError, KeyError):
        return ""
    text = re.sub(r"\s+", " ", text)
    wanted = set(terms)
    hits = [(m.start(), m.end(), t) for m in _WORD.finditer(text)
            if (t := normalize(m.group()).replace("'", "").replace("’", "")) in wanted]
    if not hits:
        return text[: 2 * radius].strip() + "…"
    best_start, best_count = hits[0][0], 0
    for start, _, _ in hits:
        count = len({t for s, _, t in hits if abs(s - start) <= radius})
        if count > best_count:
            best_start, best_count = start, count
    lo = max(0, best_start - radius)
    hi = min(len(text), best_start + radius)
    while lo > 0 and text[lo - 1].isalnum():  # do not cut words in half
        lo -= 1
    while hi < len(text) and text[hi].isalnum():
        hi += 1
    out: list[str] = []
    last = lo
    for s, e, _ in hits:
        if lo <= s and e <= hi:
            out.append(text[last:s])
            out.append(mark[0] + text[s:e] + mark[1])
            last = e
    out.append(text[last:hi])
    return ("…" if lo > 0 else "") + "".join(out).strip() + ("…" if hi < len(text) else "")
