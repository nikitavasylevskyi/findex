"""One entry point for the three search modes: lexical (BM25/TF-IDF), semantic and hybrid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from findex.corpus import display_path, load_text
from findex.query import QuerySyntaxError
from findex.ranking import Scorer, Searcher, SearchResult, make_snippet, snippet_from_text
from findex.semantic import FUSION_DEPTH, Encoder, SemanticIndex, normalize_rows, rrf
from findex.tokenize import tokenize

Mode = Literal["lexical", "semantic", "hybrid"]
_SYNTAX = ('"', "(", ")")
_OPERATORS = {"AND", "OR", "NOT"}


class SemanticUnavailable(RuntimeError):
    """The service was started without embeddings."""


@dataclass
class ResultPage:
    total: int
    results: list[SearchResult]


def bag_of_words(query: str) -> str:
    """A plain query as OR of its words, so that BM25 ranks partial matches for the fusion.

    A query that uses the syntax (quotes, AND/OR/NOT, brackets) is kept as written.
    """
    if any(c in query for c in _SYNTAX) or any(w in _OPERATORS for w in query.split()):
        return query
    return " OR ".join(dict.fromkeys(tokenize(query))) or query


class ModeSearcher:
    def __init__(self, searcher: Searcher, semantic: SemanticIndex | None = None,
                 encoder: Encoder | None = None) -> None:
        self.searcher = searcher
        self.semantic = semantic
        self.encoder = encoder
        if semantic is not None:
            semantic.check_matches([d.path for d in searcher.index.docs])

    @property
    def has_semantic(self) -> bool:
        return self.semantic is not None and self.encoder is not None

    def search(self, query: str, mode: Mode = "lexical", scorer: Scorer | None = None, k: int = 10,
               offset: int = 0, mark: tuple[str, str] = ("**", "**"), snippets: bool = True) -> ResultPage:
        if mode == "lexical":
            results = self.searcher.search(query, scorer, k=k, offset=offset, mark=mark, snippets=snippets)
            return ResultPage(self.searcher.count(query), results)
        semantic = self._semantic_ranking(query)
        if mode == "semantic":
            ranked = [(doc_id, score) for doc_id, score, _ in semantic]
        else:
            lexical = self.searcher.search(bag_of_words(query), scorer, k=FUSION_DEPTH, snippets=False)
            ranked = rrf([[r.doc_id for r in lexical], [doc_id for doc_id, _, _ in semantic]])
        starts = {doc_id: start for doc_id, _, start in semantic}
        terms = self._terms(query)
        page: list[SearchResult] = []
        for doc_id, score in ranked[offset:offset + k]:
            meta = self.searcher.index.docs[doc_id]
            snip = self._snippet(meta.path, terms, starts.get(doc_id), mark) if snippets else ""
            page.append(SearchResult(score, doc_id, meta.title, display_path(meta.path), snip))
        return ResultPage(len(ranked), page)

    def _semantic_ranking(self, query: str) -> list[tuple[int, float, int]]:
        if self.semantic is None or self.encoder is None:
            raise SemanticUnavailable("semantic search needs embeddings (findex embed)")
        vector = normalize_rows(self.encoder.encode([query]))[0]
        return self.semantic.rank(vector, FUSION_DEPTH)

    def _terms(self, query: str) -> tuple[str, ...]:
        try:
            return self.searcher.matching(query)[1]
        except QuerySyntaxError:
            return tuple(dict.fromkeys(tokenize(query)))

    @staticmethod
    def _snippet(path: str, terms: tuple[str, ...], start: int | None, mark: tuple[str, str]) -> str:
        """The best fragment if the semantic side found the document, else the usual snippet."""
        if start is None:
            return make_snippet(path, terms, mark=mark)
        try:
            text = load_text(path)
        except (OSError, ValueError, KeyError):
            return ""
        return snippet_from_text(text[start:start + 1500], terms, mark=mark)
