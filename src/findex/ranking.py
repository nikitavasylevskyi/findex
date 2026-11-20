"""Ranked retrieval: scorers, search and snippets."""

from __future__ import annotations

import heapq
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt

from findex.corpus import display_path, load_text
from findex.index import Index, Posting
from findex.query import Node, parse
from findex.timing import timed
from findex.tokenize import normalize

log = logging.getLogger("findex.search")
ScorerName = Literal["bm25", "tfidf"]
Engine = Literal["numpy", "python"]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]


class Scorer(Protocol):
    """Anything with ``name``, ``score`` and ``score_array`` can rank documents — no base class needed.

    ``score`` takes one posting (lab 3); ``score_array`` takes all postings of a term at once
    as arrays and must give the same numbers (lab 8).
    """

    name: str

    def score(self, index: Index, df: int, posting: Posting) -> float: ...

    def score_array(self, index: Index, df: int, doc_ids: IntArray, tfs: IntArray) -> FloatArray: ...


class TfIdf:
    name = "tfidf"

    def score(self, index: Index, df: int, posting: Posting) -> float:
        idf = math.log(index.num_docs / df)
        return (1 + math.log(posting.tf)) * idf

    def score_array(self, index: Index, df: int, doc_ids: IntArray, tfs: IntArray) -> FloatArray:
        idf = math.log(index.num_docs / df)  # one scalar per term, same as in score()
        return (1 + np.log(tfs.astype(np.float64))) * idf

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

    def score_array(self, index: Index, df: int, doc_ids: IntArray, tfs: IntArray) -> FloatArray:
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = index.doc_lengths[doc_ids].astype(np.float64)  # fancy indexing: lengths of these documents
        tf = tfs.astype(np.float64)
        norm = self.k1 * (1 - self.b + self.b * dl / index.avg_doc_length)  # same order of operations as score()
        return idf * tf * (self.k1 + 1) / (tf + norm)

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
               snippets: bool = True, offset: int = 0,
               mark: tuple[str, str] = ("**", "**"), engine: Engine = "numpy") -> list[SearchResult]:
        """Top ``k`` results after skipping ``offset`` (pages of a result list)."""
        scorer = scorer or SCORERS["bm25"]
        hits_before = self.matching.cache_info().hits
        doc_ids, terms = self.matching(query)
        if self.matching.cache_info().hits > hits_before:
            log.debug("cache hit for %r", query)
        if engine == "numpy":
            top = self.top_numpy(doc_ids, terms, scorer, offset + k)[offset:]
        else:
            top = self.top_python(doc_ids, terms, scorer, offset + k)[offset:]
        results: list[SearchResult] = []
        for doc_id, score in top:
            meta = self.index.docs[doc_id]
            snip = make_snippet(meta.path, terms, mark=mark) if snippets else ""
            results.append(SearchResult(score, doc_id, meta.title, display_path(meta.path), snip))
        return results

    def top_numpy(self, doc_ids: tuple[int, ...], terms: tuple[str, ...], scorer: Scorer,
                  k: int) -> list[tuple[int, float]]:
        """Scores as array expressions, top-k with argpartition (lab 8)."""
        if not doc_ids or k <= 0:
            return []
        candidates = np.unique(np.fromiter(doc_ids, dtype=np.int32, count=len(doc_ids)))
        scores = np.zeros(self.index.num_docs, dtype=np.float64)
        for term in terms:  # the same order of additions as in top_python, so the sums are identical
            ids, tfs = self.index.term_arrays(term)
            if len(ids):
                scores[ids] += scorer.score_array(self.index, len(ids), ids, tfs)  # doc_ids are unique per term
        cand_scores = scores[candidates]
        if k < len(candidates):
            # the k-th best score in O(n); keep everything tied with it too, so that ties are broken
            # by doc_id below and not by where argpartition happened to put them
            part = np.argpartition(-cand_scores, k - 1)[:k]
            kth = cand_scores[part].min()
            keep = cand_scores >= kth
            candidates, cand_scores = candidates[keep], cand_scores[keep]
        order = np.lexsort((candidates, -cand_scores))[:k]  # by score, then by doc_id, like lab 3
        return [(int(d), float(sc)) for d, sc in zip(candidates[order], cand_scores[order], strict=True)]

    def top_python(self, doc_ids: tuple[int, ...], terms: tuple[str, ...], scorer: Scorer,
                   k: int) -> list[tuple[int, float]]:
        """The scorer of lab 3: one Python call per posting, heapq.nlargest."""
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
        return heapq.nlargest(k, scores.items(), key=lambda item: (item[1], -item[0]))

    def count(self, query: str) -> int:
        """How many documents match (cached together with the match itself)."""
        return len(self.matching(query)[0])


_WORD = re.compile(r"[^\W_]+(?:['’_]*[^\W_]+)*")


def _fold(word: str) -> str:
    """normalize() as the tokenizer does it, with a fast path: for ASCII, NFKC changes nothing."""
    if word.isascii():
        return word.lower().replace("'", "")
    return normalize(word).replace("'", "").replace("’", "")


def make_snippet(path: str, terms: tuple[str, ...], radius: int = 80,
                 mark: tuple[str, str] = ("**", "**")) -> str:
    """Window of ±radius characters around the best occurrence, terms highlighted.

    "Best" = the occurrence whose window contains the most distinct query terms.
    """
    try:
        text = load_text(path)
    except (OSError, ValueError, KeyError):
        return ""
    return snippet_from_text(text, terms, radius, mark)


def snippet_from_text(text: str, terms: tuple[str, ...], radius: int = 80,
                      mark: tuple[str, str] = ("**", "**")) -> str:
    text = re.sub(r"\s+", " ", text)
    wanted = set(terms)
    hits = [(m.start(), m.end(), t) for m in _WORD.finditer(text) if (t := _fold(m.group())) in wanted]
    if not hits:
        return text[: 2 * radius].strip() + "…"
    # Sliding window over the occurrences (sorted by position): for each occurrence, how many distinct
    # terms lie within ±radius of it. Two pointers make this O(h); the first version compared every
    # occurrence with every other one, O(h²), and was 92 % of search time (docs/profile/HOTSPOTS.md).
    best_start, best_count = hits[0][0], 0
    window: Counter[str] = Counter()
    lo_i = hi_i = 0
    for start, _, _ in hits:
        while hi_i < len(hits) and hits[hi_i][0] <= start + radius:
            window[hits[hi_i][2]] += 1
            hi_i += 1
        while hits[lo_i][0] < start - radius:
            window[hits[lo_i][2]] -= 1
            if not window[hits[lo_i][2]]:
                del window[hits[lo_i][2]]
            lo_i += 1
        if len(window) > best_count:
            best_start, best_count = start, len(window)
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
