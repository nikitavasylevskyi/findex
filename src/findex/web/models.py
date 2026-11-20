"""Pydantic models live only at the edge: they check what comes in and shape what goes out.

Inside, the code keeps working with the dataclasses of labs 2–3
(SearchResult, DocMeta); the conversion happens here, once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from findex.corpus import display_path
from findex.index import DocMeta
from findex.ranking import SearchResult


class SearchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=200, description="Query: words, \"phrases\", AND/OR/NOT, ( )")
    k: int = Field(10, ge=1, le=50, description="Results per page")
    scorer: Literal["bm25", "tfidf"] = "bm25"
    page: int = Field(1, ge=1, le=1000)

    @field_validator("q")
    @classmethod
    def not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must contain something besides spaces")
        return value


class Hit(BaseModel):
    doc_id: int
    title: str
    path: str
    score: float
    snippet: str = Field(description="±80 characters around the best match, terms marked with **")

    @classmethod
    def from_result(cls, r: SearchResult) -> Hit:
        return cls(doc_id=r.doc_id, title=r.title, path=r.path, score=round(r.score, 4), snippet=r.snippet)


class SearchResponse(BaseModel):
    query: str
    scorer: str
    total: int = Field(description="Documents that match, on all pages")
    page: int
    pages: int
    k: int
    took_ms: float
    hits: list[Hit]


class DocResponse(BaseModel):
    doc_id: int
    title: str
    path: str
    length: int = Field(description="Number of tokens")
    text: str

    @classmethod
    def from_meta(cls, meta: DocMeta, text: str) -> DocResponse:
        return cls(doc_id=meta.doc_id, title=meta.title, path=display_path(meta.path), length=meta.length,
                   text=text)


class StatsResponse(BaseModel):
    documents: int
    terms: int
    avg_doc_length: float
    with_positions: bool
    load_seconds: float
    query_cache_hits: int
    query_cache_misses: int


class Health(BaseModel):
    status: Literal["ok", "loading", "failed"]
    documents: int | None = None
    detail: str | None = None
