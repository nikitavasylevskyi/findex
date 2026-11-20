"""The FastAPI application.

A request goes: socket → uvicorn (parses HTTP, builds the ASGI ``scope``) →
the request-id middleware → FastAPI routing → dependencies (``Depends``) →
the route → a Pydantic model → JSON, or a Jinja2 page.

Search is CPU-bound pure Python, so the routes that search are plain ``def``:
FastAPI runs them in its thread pool and the event loop keeps serving other
requests (health checks, static files) in the meantime. ``async def`` with the
same body would hold the loop for the whole search (measured in the README).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from markupsafe import Markup, escape
from pydantic import ValidationError

from findex import __version__, storage
from findex.corpus import display_path, load_text
from findex.ranking import SCORERS, Searcher
from findex.web.models import DocResponse, Health, Hit, SearchParams, SearchResponse, StatsResponse
from findex.web.settings import Settings

log = logging.getLogger("findex.web")

HERE = Path(__file__).parent
MARK = ("\x02", "\x03")  # snippet highlight markers for the HTML page; cannot occur in escaped text


@dataclass
class Engine:
    """What the lifespan loads once and every request shares (read-only)."""

    status: Literal["loading", "ok", "failed"] = "loading"
    searcher: Searcher | None = None
    load_seconds: float = 0.0
    error: str = ""


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def get_searcher(engine: Annotated[Engine, Depends(get_engine)]) -> Searcher:
    if engine.searcher is None:
        raise HTTPException(503, f"index is {engine.status}")
    return engine.searcher


SearcherDep = Annotated[Searcher, Depends(get_searcher)]


def run_search(searcher: Searcher, params: SearchParams, mark: tuple[str, str] = ("**", "**")) -> SearchResponse:
    t0 = time.perf_counter()
    total = searcher.count(params.q)
    pages = max(1, math.ceil(total / params.k))
    results = searcher.search(params.q, SCORERS[params.scorer], k=params.k,
                              offset=(params.page - 1) * params.k, mark=mark)
    took = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=params.q, scorer=params.scorer, total=total, page=params.page, pages=pages,
                          k=params.k, took_ms=round(took, 2), hits=[Hit.from_result(r) for r in results])


def highlight(snippet: str) -> Markup:
    return Markup(str(escape(snippet)).replace(MARK[0], "<mark>").replace(MARK[1], "</mark>"))


def create_app(settings: Settings | None = None, *, load_index: bool = True) -> FastAPI:
    settings = settings or Settings()  # type: ignore[call-arg]
    if not logging.getLogger().handlers:
        logging.basicConfig(level=settings.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        engine = app.state.engine = Engine()
        task = asyncio.create_task(load(engine)) if load_index else None
        yield
        if task is not None:
            task.cancel()
        if engine.searcher is not None:
            engine.searcher.index.close()

    async def load(engine: Engine) -> None:
        """Load in a worker thread: the server already answers /health with 503 while it runs."""
        t0 = time.perf_counter()
        try:
            index = await asyncio.to_thread(storage.load, settings.index_path)
        except Exception as e:  # missing file, wrong format: the service stays up and says so
            engine.status, engine.error = "failed", f"{e.__class__.__name__}: {e}"
            log.error("could not load index %s: %s", settings.index_path, engine.error)
            return
        engine.searcher = Searcher(index)
        engine.load_seconds = time.perf_counter() - t0
        engine.status = "ok"
        log.info("index %s loaded: %d documents, %d terms in %.2f s", settings.index_path,
                 index.num_docs, len(index), engine.load_seconds)

    app = FastAPI(title="findex", version=__version__, lifespan=lifespan, docs_url="/api", redoc_url=None,
                  description="Full-text search over Python PEPs and the CPython documentation.")
    app.state.engine = Engine()
    app.state.settings = settings
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["highlight"] = highlight
    templates.env.filters["display_path"] = display_path
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def request_id(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = rid
        try:
            response = await call_next(request)
        except Exception:
            # the stack goes to the log, the client only gets the id to quote
            log.exception("request %s %s %s failed", rid, request.method, request.url.path)
            response = JSONResponse({"detail": "internal server error", "request_id": rid}, status_code=500)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/search", response_model=SearchResponse, tags=["api"])
    def search(params: Annotated[SearchParams, Query()], searcher: SearcherDep) -> SearchResponse:
        """Ranked results, one page at a time."""
        return run_search(searcher, params)

    @app.get("/docs/{doc_id}", response_model=DocResponse, tags=["api"],
             responses={404: {"description": "No document with this id"}})
    def document(doc_id: int, searcher: SearcherDep) -> DocResponse:
        docs = searcher.index.docs
        if not 0 <= doc_id < len(docs):
            raise HTTPException(404, f"no document {doc_id}")
        meta = docs[doc_id]
        return DocResponse.from_meta(meta, load_text(meta.path))

    @app.get("/stats", response_model=StatsResponse, tags=["api"])
    def stats(searcher: SearcherDep, engine: Annotated[Engine, Depends(get_engine)]) -> StatsResponse:
        index = searcher.index
        cache = searcher.matching.cache_info()
        return StatsResponse(documents=index.num_docs, terms=len(index), avg_doc_length=round(index.avg_doc_length, 1),
                             with_positions=index.with_positions, load_seconds=round(engine.load_seconds, 2),
                             query_cache_hits=cache.hits, query_cache_misses=cache.misses)

    @app.get("/health", response_model=Health, response_model_exclude_none=True, tags=["api"],
             responses={503: {"model": Health, "description": "Index not loaded (yet)"}})
    async def health(engine: Annotated[Engine, Depends(get_engine)]) -> Response | Health:
        if engine.searcher is None:
            body = Health(status=engine.status, detail=engine.error or None)
            return JSONResponse(body.model_dump(exclude_none=True), status_code=503)
        return Health(status="ok", documents=engine.searcher.index.num_docs)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request, engine: Annotated[Engine, Depends(get_engine)], q: str = "",
             page: int = 1, scorer: str = "bm25") -> HTMLResponse:
        context: dict[str, object] = {"q": q.strip(), "scorer": scorer, "engine": engine, "error": None,
                                      "result": None}
        if q.strip() and engine.searcher is not None:
            try:
                params = SearchParams(q=q, page=max(1, min(page, 1000)), k=settings.page_size,
                                      scorer=scorer if scorer in SCORERS else "bm25")  # type: ignore[arg-type]
            except ValidationError as e:
                context["error"] = e.errors()[0]["msg"]
            else:
                context["result"] = run_search(engine.searcher, params, mark=MARK)
        return templates.TemplateResponse(request, "search.html", context)

    @app.get("/view/{doc_id}", response_class=HTMLResponse, include_in_schema=False)
    def view(request: Request, doc_id: int, searcher: SearcherDep, q: str = "") -> HTMLResponse:
        docs = searcher.index.docs
        if not 0 <= doc_id < len(docs):
            raise HTTPException(404, f"no document {doc_id}")
        meta = docs[doc_id]
        return templates.TemplateResponse(request, "document.html", {
            "doc": DocResponse.from_meta(meta, load_text(meta.path)), "q": q})

    return app
