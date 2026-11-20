"""Inverted index: term -> postings sorted by doc_id."""

from __future__ import annotations

from array import array
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from findex.corpus import Document, iter_documents
from findex.timing import timed
from findex.tokenize import tokenize


class Posting(NamedTuple):
    """A NamedTuple rather than a frozen dataclass: the index holds over a million
    of them, and a tuple is about 2x cheaper to create and to pickle (lab 5)."""

    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    path: str
    title: str
    length: int


class PackedPostings:
    """Postings of one term as three flat arrays of u32 instead of Posting objects.

    This is the form in which build_partial sends its result back from a worker
    process: pickling three arrays is a memcpy, pickling a million small tuples
    is not. Rebuilding all Posting objects in the parent would cost about as much
    as indexing itself, so the Index unpacks a term only when it is first read.
    """

    __slots__ = ("doc_ids", "tfs", "positions")

    def __init__(self) -> None:
        # array('I') while indexing; numpy uint32/int32 views when read from a file (storage.load)
        self.doc_ids: array[int] | npt.NDArray[np.integer] = array("I")
        self.tfs: array[int] | npt.NDArray[np.integer] = array("I")
        # all positions of all postings, one after another; empty if not stored
        self.positions: array[int] | npt.NDArray[np.integer] = array("I")

    @classmethod
    def from_arrays(cls, doc_ids: npt.NDArray[np.integer], tfs: npt.NDArray[np.integer],
                    positions: npt.NDArray[np.integer]) -> PackedPostings:
        packed = cls()
        packed.doc_ids, packed.tfs, packed.positions = doc_ids, tfs, positions
        return packed

    def __len__(self) -> int:
        return len(self.doc_ids)

    def append(self, doc_id: int, positions: list[int], keep_positions: bool) -> None:
        assert isinstance(self.doc_ids, array) and isinstance(self.tfs, array) and isinstance(self.positions, array)
        self.doc_ids.append(doc_id)
        self.tfs.append(len(positions))
        if keep_positions:
            self.positions.extend(positions)

    def extend(self, other: PackedPostings) -> None:
        assert isinstance(self.doc_ids, array) and isinstance(self.tfs, array) and isinstance(self.positions, array)
        self.doc_ids.extend(other.doc_ids)
        self.tfs.extend(other.tfs)
        self.positions.extend(other.positions)

    def unpack(self) -> list[Posting]:
        ids: list[int] = self.doc_ids.tolist()
        tfs: list[int] = self.tfs.tolist()
        if not len(self.positions):
            return [Posting(d, tf) for d, tf in zip(ids, tfs, strict=True)]
        out: list[Posting] = []
        pos: list[int] = self.positions.tolist()
        i = 0
        for d, tf in zip(ids, tfs, strict=True):
            out.append(Posting(d, tf, tuple(pos[i:i + tf])))
            i += tf
        return out


class Index(Mapping[str, list[Posting]]):
    """Read-only mapping term -> postings, plus document metadata.

    Behaves like a dict: ``len(index)`` is the vocabulary size, ``"async" in index``,
    ``index["async"]``, ``for term in index``.
    """

    def __init__(self, postings: Mapping[str, list[Posting] | PackedPostings], docs: list[DocMeta],
                 with_positions: bool = False) -> None:
        self._postings: dict[str, list[Posting] | PackedPostings] = dict(postings)
        self.docs = docs
        self.with_positions = with_positions
        self._arrays: dict[str, tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]] = {}

    def __getitem__(self, term: str) -> list[Posting]:
        plist = self._postings[term]
        if isinstance(plist, PackedPostings):
            plist = self._postings[term] = plist.unpack()
        return plist

    def term_arrays(self, term: str) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """doc_ids and tfs of a term as int32 arrays (lab 8: the form the vectorised scorers read)."""
        cached = self._arrays.get(term)
        if cached is not None:
            return cached
        raw = self._postings.get(term)
        if raw is None:
            ids = tfs = np.empty(0, dtype=np.int32)
        elif isinstance(raw, PackedPostings):
            ids = np.asarray(raw.doc_ids).astype(np.int32, copy=False)
            tfs = np.asarray(raw.tfs).astype(np.int32, copy=False)
        else:
            ids = np.fromiter((p.doc_id for p in raw), dtype=np.int32, count=len(raw))
            tfs = np.fromiter((p.tf for p in raw), dtype=np.int32, count=len(raw))
        self._arrays[term] = (ids, tfs)
        return ids, tfs

    @cached_property
    def doc_lengths(self) -> npt.NDArray[np.int32]:
        """Length of every document in tokens, indexed by doc_id."""
        return np.fromiter((d.length for d in self.docs), dtype=np.int32, count=len(self.docs))

    def raw(self, term: str) -> list[Posting] | PackedPostings:
        """Postings as they are stored, without unpacking (for merge and storage)."""
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
        return (self.docs == other.docs and self.with_positions == other.with_positions
                and self.keys() == other.keys() and all(self[t] == other[t] for t in self))

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
        return {term: self[term] for term in self}

    def close(self) -> None:
        """Drop the data so that memory is freed even if references to the object remain."""
        self._postings = {}
        self._arrays = {}
        self.docs = []
        self.__dict__.pop("avg_doc_length", None)
        self.__dict__.pop("doc_lengths", None)


def index_documents(documents: Iterable[Document], positions: bool = False,
                    on_document: Callable[[Document], None] | None = None) -> Index:
    """Index documents that arrive in increasing doc_id order.

    Appending keeps every postings list sorted without an extra sort.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    docs: list[DocMeta] = []
    for doc in documents:
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


@timed
def build_index(root: str | Path, positions: bool = False,
                on_document: Callable[[Document], None] | None = None) -> Index:
    """One pass over the lazy pipeline of lab 1, on a single core."""
    return index_documents(iter_documents(root), positions, on_document)


def index_documents_packed(documents: Iterable[Document], positions: bool = False) -> Index:
    """Same as index_documents, but postings go straight into PackedPostings."""
    postings: defaultdict[str, PackedPostings] = defaultdict(PackedPostings)
    docs: list[DocMeta] = []
    for doc in documents:
        term_pos: defaultdict[str, list[int]] = defaultdict(list)
        length = 0
        for pos, tok in enumerate(tokenize(doc.text)):
            term_pos[tok].append(pos)
            length += 1
        for term, plist in term_pos.items():
            postings[term].append(doc.doc_id, plist, positions)
        docs.append(DocMeta(doc.doc_id, str(doc.path), doc.title, length))
    return Index(postings, docs, positions)
