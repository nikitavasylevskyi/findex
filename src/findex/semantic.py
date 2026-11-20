"""Semantic search (lab 8): embeddings of document fragments, cosine similarity, RRF.

``findex embed`` splits every document into fragments of about 200 words, turns
each fragment into a vector with a small embedding model, normalises the
vectors to length 1 and saves them as one matrix in ``.npy``. A query becomes a
vector with the same model; since all vectors have length 1, the dot product is
the cosine similarity, and one matrix-vector product scores every fragment. A
document gets the score of its best fragment.

The model is ``minishlab/potion-base-8M`` (model2vec): static embeddings
distilled from a sentence-transformer, 256 dimensions, about 30 MB, no PyTorch.
It is much faster and lighter than all-MiniLM-L6-v2, which matters on a free
512 MB instance, at the price of some quality.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from findex.corpus import Document, iter_documents

DEFAULT_MODEL = "minishlab/potion-base-8M"
CHUNK_WORDS = 200
CHUNK_STRIDE = 150  # fragments overlap by 50 words, so a passage cut in two is still whole in one of them
RRF_K = 60  # the constant of Reciprocal Rank Fusion; 60 is the value from the original paper
FUSION_DEPTH = 100  # how deep each ranking goes into the fusion

Matrix = npt.NDArray[np.float32]
_WORD_SPAN = re.compile(r"\S+")


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> Matrix: ...


class Model2VecEncoder:
    """Loads the model on first use (a few hundred ms, from the Hugging Face cache)."""

    def __init__(self, name: str = DEFAULT_MODEL) -> None:
        self.name = name
        self._model: Any = None  # model2vec.StaticModel, imported only when needed

    def encode(self, texts: Sequence[str]) -> Matrix:
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.name, force_download=False)
        vectors = cast("npt.NDArray[np.floating]", self._model.encode(list(texts), use_multiprocessing=False))
        return vectors.astype(np.float32)


def normalize_rows(m: npt.NDArray[np.floating]) -> Matrix:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.maximum(norms, 1e-12)).astype(np.float32)


def chunk_text(text: str, words: int = CHUNK_WORDS, stride: int = CHUNK_STRIDE) -> list[tuple[int, str]]:
    """Fragments of ``words`` words every ``stride`` words, each with its character offset in ``text``."""
    spans = [(m.start(), m.end()) for m in _WORD_SPAN.finditer(text)]
    if not spans:
        return []
    out: list[tuple[int, str]] = []
    start = 0
    while True:
        end = min(start + words, len(spans))
        out.append((spans[start][0], text[spans[start][0]:spans[end - 1][1]]))
        if end == len(spans):
            return out
        start += stride


@dataclass
class SemanticIndex:
    vectors: Matrix  # (fragments, dim), rows of length 1
    chunk_doc: npt.NDArray[np.int32]  # document of each fragment
    chunk_start: npt.NDArray[np.int32]  # character offset of each fragment in its document
    model: str
    doc_paths: list[str]

    @property
    def num_docs(self) -> int:
        return len(self.doc_paths)

    def save(self, path: str | Path) -> None:
        """``x.npy`` holds the matrix; ``x.chunks.npy`` and ``x.json`` what each row belongs to."""
        path = Path(path).with_suffix(".npy")
        np.save(path, self.vectors)
        np.save(path.with_suffix(".chunks.npy"), np.stack([self.chunk_doc, self.chunk_start]))
        path.with_suffix(".json").write_text(json.dumps({
            "model": self.model, "dim": int(self.vectors.shape[1]), "fragments": int(len(self.vectors)),
            "chunk_words": CHUNK_WORDS, "chunk_stride": CHUNK_STRIDE, "doc_paths": self.doc_paths,
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> SemanticIndex:
        path = Path(path).with_suffix(".npy")
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        chunks = np.load(path.with_suffix(".chunks.npy"))
        vectors = np.load(path, mmap_mode="r")  # mapped, not read: the OS pages it in as needed
        return cls(vectors, chunks[0].astype(np.int32), chunks[1].astype(np.int32), meta["model"],
                   meta["doc_paths"])

    def check_matches(self, doc_paths: Sequence[str]) -> None:
        if list(doc_paths) != self.doc_paths:
            raise ValueError("embeddings were built for another corpus or index; run findex embed again")

    def rank(self, query_vector: Matrix, k: int) -> list[tuple[int, float, int]]:
        """Top ``k`` documents as (doc_id, cosine of the best fragment, that fragment's offset)."""
        if k <= 0 or not len(self.vectors):
            return []
        sims = self.vectors @ query_vector.reshape(-1)  # one product scores every fragment
        order = np.argsort(-sims, kind="stable")
        # the first time a document appears in this order is its best fragment
        docs, first = np.unique(self.chunk_doc[order], return_index=True)
        by_score = np.lexsort((docs, first))  # documents in the order of their best fragment
        best = order[first[by_score][:k]]
        return [(int(self.chunk_doc[i]), float(sims[i]), int(self.chunk_start[i])) for i in best]


def build_embeddings(documents: Iterable[Document], encoder: Encoder, batch: int = 512,
                     on_document: Callable[[Document], None] | None = None) -> SemanticIndex:
    vectors: list[Matrix] = []
    chunk_doc: list[int] = []
    chunk_start: list[int] = []
    paths: list[str] = []
    pending: list[str] = []

    def flush() -> None:
        if pending:
            vectors.append(normalize_rows(encoder.encode(pending)))
            pending.clear()

    for doc in documents:
        paths.append(str(doc.path))
        for start, text in chunk_text(doc.text):
            chunk_doc.append(doc.doc_id)
            chunk_start.append(start)
            pending.append(text)
            if len(pending) >= batch:
                flush()
        if on_document is not None:
            on_document(doc)
    flush()
    matrix = np.concatenate(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
    return SemanticIndex(matrix, np.asarray(chunk_doc, dtype=np.int32), np.asarray(chunk_start, dtype=np.int32),
                         encoder.name, paths)


def embed_corpus(root: str | Path, encoder: Encoder,
                 on_document: Callable[[Document], None] | None = None) -> SemanticIndex:
    return build_embeddings(iter_documents(root), encoder, on_document=on_document)


def rrf(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank of d in each list), ranks from 1.

    Only ranks are used, so BM25 scores and cosines, which live on different scales,
    never have to be compared. Ties go to the smaller doc_id.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
