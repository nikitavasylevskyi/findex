"""Inverted index: term -> postings sorted by doc_id."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import Document, iter_documents
from findex.timing import timed
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


class Index(Mapping[str, list[Posting]]):
    """Read-only mapping term -> postings, plus document metadata.

    Behaves like a dict: ``len(index)`` is the vocabulary size, ``"async" in index``,
    ``index["async"]``, ``for term in index``.
    """

    def __init__(self, postings: dict[str, list[Posting]], docs: list[DocMeta],
                 with_positions: bool = False) -> None:
        self._postings = postings
        self.docs = docs
        self.with_positions = with_positions

    def __getitem__(self, term: str) -> list[Posting]:
        return self._postings[term]

    def __len__(self) -> int:
        return len(self._postings)

    def __iter__(self) -> Iterator[str]:
        return iter(self._postings)

    def __contains__(self, term: object) -> bool:  # faster than Mapping's try/except
        return term in self._postings

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, docs={self.num_docs}, positions={self.with_positions})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return NotImplemented
        return (self._postings == other._postings and self.docs == other.docs
                and self.with_positions == other.with_positions)

    __hash__ = None  # type: ignore[assignment]  # a mutable container is unhashable, like dict

    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @cached_property
    def avg_doc_length(self) -> float:
        return sum(d.length for d in self.docs) / max(self.num_docs, 1)

    def df(self, term: str) -> int:
        return len(self._postings.get(term, ()))

    @property
    def postings(self) -> dict[str, list[Posting]]:
        return self._postings

    def close(self) -> None:
        """Drop the data so that memory is freed even if references to the object remain."""
        self._postings = {}
        self.docs = []
        self.__dict__.pop("avg_doc_length", None)


@timed
def build_index(root: str | Path, positions: bool = False,
                on_document: Callable[[Document], None] | None = None) -> Index:
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
        if on_document is not None:
            on_document(doc)
    return Index(dict(postings), docs, positions)
