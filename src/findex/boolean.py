"""Boolean retrieval over sorted postings.

Query language: words separated by spaces are joined with AND; the keywords
OR and NOT (upper case) are operators. NOT binds tighter than AND, AND tighter
than OR: ``a b OR c NOT d`` means ``(a AND b) OR (c AND NOT d)``.

Two engines give the same result:
    merge — two-pointer walk over the sorted doc_id lists;
    set   — Python set operations on the same lists.
"""

from __future__ import annotations

from bisect import bisect_left
from typing import Literal

from findex.index import Index
from findex.tokenize import tokenize

Engine = Literal["merge", "set"]


def intersect(a: list[int], b: list[int]) -> list[int]:
    if len(a) > len(b):
        a, b = b, a
    if len(a) * 8 < len(b):
        # Very different lengths: instead of stepping through the long list one by
        # one, jump to each doc_id of the short list with binary search.
        out: list[int] = []
        lo = 0
        for x in a:
            lo = bisect_left(b, x, lo)
            if lo == len(b):
                break
            if b[lo] == x:
                out.append(x)
        return out
    i = j = 0
    out = []  # same name as in the branch above, already typed list[int]
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def difference(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def doc_ids(index: Index, word: str) -> list[int]:
    toks = list(tokenize(word))
    if not toks:
        return []
    result = [p.doc_id for p in index.get(toks[0], [])]
    for t in toks[1:]:  # "read-only" as one query word means read AND only
        result = intersect(result, [p.doc_id for p in index.get(t, [])])
    return result


def parse(query: str) -> list[list[tuple[bool, str]]]:
    """Disjunctive form: OR-groups of (negated, word)."""
    groups: list[list[tuple[bool, str]]] = [[]]
    negate = False
    for word in query.split():
        if word == "OR":
            groups.append([])
        elif word == "NOT":
            negate = True
        else:
            groups[-1].append((negate, word))
            negate = False
    return [g for g in groups if g]


def search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    if engine == "set":
        return _search_set(index, query)
    result: list[int] = []
    for group in parse(query):
        positive = sorted((doc_ids(index, w) for neg, w in group if not neg), key=len)
        if positive:
            acc = positive[0]
            for other in positive[1:]:  # shortest lists first: fewer comparisons
                acc = intersect(acc, other)
        else:
            acc = list(range(index.num_docs))
        for neg, w in group:
            if neg:
                acc = difference(acc, doc_ids(index, w))
        result = union(result, acc)
    return result


def _search_set(index: Index, query: str) -> list[int]:
    result: set[int] = set()
    for group in parse(query):
        positive = [set(doc_ids(index, w)) for neg, w in group if not neg]
        acc: set[int] = positive[0].intersection(*positive[1:]) if positive else set(range(index.num_docs))
        for neg, w in group:
            if neg:
                acc -= set(doc_ids(index, w))
        result |= acc
    return sorted(result)
