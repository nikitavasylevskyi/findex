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
from findex.hybrid import ModeSearcher, SemanticUnavailable
from findex.ranking import SCORERS, Searcher
from findex.semantic import Model2VecEncoder, SemanticIndex
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
    modes: ModeSearcher | None = None
    load_seconds: float = 0.0
    error: str = ""


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def get_modes(engine: Annotated[Engine, Depends(get_engine)]) -> ModeSearcher:
    if engine.modes is None:
        raise HTTPException(503, f"index is {engine.status}")
    return engine.modes


def get_searcher(modes: Annotated[ModeSearcher, Depends(get_modes)]) -> Searcher:
    return modes.searcher


SearcherDep = Annotated[Searcher, Depends(get_searcher)]
ModesDep = Annotated[ModeSearcher, Depends(get_modes)]


def run_search(modes: ModeSearcher, params: SearchParams, mark: tuple[str, str] = ("**", "**")) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        page = modes.search(params.q, params.mode, SCORERS[params.scorer], k=params.k,
                            offset=(params.page - 1) * params.k, mark=mark)
    except SemanticUnavailable as e:
        raise HTTPException(400, str(e)) from None
    took = (time.perf_counter() - t0) * 1000
    return SearchResponse(query=params.q, scorer=params.scorer, mode=params.mode, total=page.total,
                          page=params.page, pages=max(1, math.ceil(page.total / params.k)), k=params.k,
                          took_ms=round(took, 2), hits=[Hit.from_result(r) for r in page.results])


def load_semantic(searcher: Searcher, path: Path) -> ModeSearcher:
    semantic = SemanticIndex.load(path)
    encoder = Model2VecEncoder(semantic.model)
    encoder.encode(["warm up"])  # load the model now, not on the first query
    return ModeSearcher(searcher, semantic, encoder)


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
            searcher = Searcher(index)
            modes = ModeSearcher(searcher)
            if settings.embeddings_path is not None:
                modes = await asyncio.to_thread(load_semantic, searcher, settings.embeddings_path)
        except Exception as e:  # missing file, wrong format: the service stays up and says so
            engine.status, engine.error = "failed", f"{e.__class__.__name__}: {e}"
            log.error("could not load index %s: %s", settings.index_path, engine.error)
            return
        engine.searcher, engine.modes = searcher, modes
        engine.load_seconds = time.perf_counter() - t0
        engine.status = "ok"
        log.info("index %s loaded: %d documents, %d terms, semantic %s, in %.2f s", settings.index_path,
                 index.num_docs, len(index), "on" if modes.has_semantic else "off", engine.load_seconds)

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
    def search(params: Annotated[SearchParams, Query()], modes: ModesDep) -> SearchResponse:
        """Ranked results, one page at a time. ``mode=semantic|hybrid`` needs embeddings on the server."""
        return run_search(modes, params)

    @app.get("/docs/{doc_id}", response_model=DocResponse, tags=["api"],
             responses={404: {"description": "No document with this id"}})
    def document(doc_id: int, searcher: SearcherDep) -> DocResponse:
        docs = searcher.index.docs
        if not 0 <= doc_id < len(docs):
            raise HTTPException(404, f"no document {doc_id}")
        meta = docs[doc_id]
        return DocResponse.from_meta(meta, load_text(meta.path))

    @app.get("/stats", response_model=StatsResponse, tags=["api"])
    def stats(modes: ModesDep, engine: Annotated[Engine, Depends(get_engine)]) -> StatsResponse:
        index = modes.searcher.index
        cache = modes.searcher.matching.cache_info()
        sem = modes.semantic
        return StatsResponse(documents=index.num_docs, terms=len(index), avg_doc_length=round(index.avg_doc_length, 1),
                             with_positions=index.with_positions, load_seconds=round(engine.load_seconds, 2),
                             query_cache_hits=cache.hits, query_cache_misses=cache.misses,
                             semantic_model=sem.model if sem else None,
                             semantic_fragments=len(sem.vectors) if sem else None)

    @app.get("/health", response_model=Health, response_model_exclude_none=True, tags=["api"],
             responses={503: {"model": Health, "description": "Index not loaded (yet)"}})
    async def health(engine: Annotated[Engine, Depends(get_engine)]) -> Response | Health:
        if engine.modes is None:
            body = Health(status=engine.status, detail=engine.error or None)
            return JSONResponse(body.model_dump(exclude_none=True), status_code=503)
        modes = ["lexical", "semantic", "hybrid"] if engine.modes.has_semantic else ["lexical"]
        return Health(status="ok", documents=engine.modes.searcher.index.num_docs, modes=modes)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request, engine: Annotated[Engine, Depends(get_engine)], q: str = "",
             page: int = 1, scorer: str = "bm25", mode: str = "lexical") -> HTMLResponse:
        semantic_on = engine.modes is not None and engine.modes.has_semantic
        if mode not in ("lexical", "semantic", "hybrid") or not semantic_on:
            mode = "lexical"
        context: dict[str, object] = {"q": q.strip(), "scorer": scorer, "mode": mode, "engine": engine,
                                      "semantic_on": semantic_on, "error": None, "result": None}
        if q.strip() and engine.modes is not None:
            try:
                params = SearchParams(q=q, page=max(1, min(page, 1000)), k=settings.page_size, mode=mode,  # type: ignore[arg-type]
                                      scorer=scorer if scorer in SCORERS else "bm25")  # type: ignore[arg-type]
                context["result"] = run_search(engine.modes, params, mark=MARK)
            except ValidationError as e:
                context["error"] = e.errors()[0]["msg"]
            except HTTPException as e:
                context["error"] = e.detail
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
