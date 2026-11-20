"""Web API through TestClient. The index comes from dependency_overrides, never from a file."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import doc_id
from fastapi import FastAPI
from fastapi.testclient import TestClient

from findex import storage
from findex.index import Index
from findex.ranking import Searcher
from findex.web.app import Engine, create_app, get_engine, get_searcher
from findex.web.settings import Settings


@pytest.fixture
def app(index: Index) -> FastAPI:
    app = create_app(Settings(index_path=Path("does-not-exist.bin")), load_index=False)
    searcher = Searcher(index)
    engine = Engine(status="ok", searcher=searcher)
    app.dependency_overrides[get_searcher] = lambda: searcher
    app.dependency_overrides[get_engine] = lambda: engine
    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_search_returns_ranked_page(client: TestClient, index: Index) -> None:
    r = client.get("/search", params={"q": "walrus", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and body["pages"] == 2 and body["page"] == 1
    assert [h["doc_id"] for h in body["hits"]][0] == doc_id(index, "c.rst")  # the shortest document wins in BM25
    assert "**walrus**" in body["hits"][0]["snippet"]
    assert r.headers["x-request-id"]


def test_search_pages_do_not_overlap(client: TestClient) -> None:
    pages = [client.get("/search", params={"q": "walrus", "k": 2, "page": p}).json() for p in (1, 2)]
    ids = [h["doc_id"] for page in pages for h in page["hits"]]
    assert len(ids) == len(set(ids)) == 3


def test_search_tfidf_and_boolean(client: TestClient, index: Index) -> None:
    body = client.get("/search", params={"q": "walrus AND operator", "scorer": "tfidf"}).json()
    assert [h["doc_id"] for h in body["hits"]] == [doc_id(index, "pep-0572.rst")]
    assert body["scorer"] == "tfidf"


@pytest.mark.parametrize("params", [
    {},
    {"q": ""},
    {"q": "   "},
    {"q": "x" * 201},
    {"q": "walrus", "k": 0},
    {"q": "walrus", "k": 51},
    {"q": "walrus", "page": 0},
    {"q": "walrus", "scorer": "pagerank"},
    {"q": "walrus", "k": "ten"},
    {"q": "walrus", "extra": 1},
])
def test_search_rejects_bad_params(client: TestClient, params: dict[str, object]) -> None:
    r = client.get("/search", params=params)  # type: ignore[arg-type]
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"][0] == "query"


def test_document(client: TestClient, index: Index) -> None:
    did = doc_id(index, "pep-0572.rst")
    body = client.get(f"/docs/{did}").json()
    assert body["title"] == "PEP 572: Assignment Expressions"
    assert "walrus operator" in body["text"]


@pytest.mark.parametrize("bad", ["999", "-1"])
def test_unknown_document_is_404(client: TestClient, bad: str) -> None:
    assert client.get(f"/docs/{bad}").status_code == 404
    assert client.get(f"/view/{bad}").status_code == 404


def test_stats(client: TestClient, index: Index) -> None:
    body = client.get("/stats").json()
    assert body["documents"] == index.num_docs and body["terms"] == len(index) and body["with_positions"]


def test_health_ok(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "documents": 8}


def test_health_is_503_while_index_is_not_loaded(app: FastAPI) -> None:
    app.dependency_overrides.clear()  # the real dependencies: the lifespan did not load anything
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "loading"
        assert client.get("/search", params={"q": "walrus"}).status_code == 503
        assert "ще завантажується" in client.get("/", params={"q": "walrus"}).text


def test_health_reports_failed_load(tmp_path: Path) -> None:
    app = create_app(Settings(index_path=tmp_path / "missing.bin"))
    with TestClient(app) as client:
        for _ in range(100):  # loading runs in a worker thread
            if client.get("/health").json()["status"] != "loading":
                break
        r = client.get("/health")
        assert r.status_code == 503 and r.json()["status"] == "failed"
        assert "missing.bin" in r.json()["detail"]


def test_lifespan_loads_index_file(tmp_path: Path, index: Index) -> None:
    path = tmp_path / "index.bin"
    storage.save(index, path)
    with TestClient(create_app(Settings(index_path=path))) as client:
        for _ in range(200):
            if client.get("/health").status_code == 200:
                break
        assert client.get("/search", params={"q": "walrus"}).json()["total"] == 3


def test_unexpected_error_is_500_without_stack(app: FastAPI, caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def count(self, query: str) -> int:
            raise RuntimeError("secret internals")

    app.dependency_overrides[get_searcher] = lambda: Broken()
    with TestClient(app, raise_server_exceptions=False) as client:
        r = client.get("/search", params={"q": "walrus"}, headers={"x-request-id": "abc123"})
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error", "request_id": "abc123"}
    assert "secret internals" not in r.text
    assert any("abc123" in rec.getMessage() and rec.exc_info for rec in caplog.records)


def test_html_page(client: TestClient) -> None:
    home = client.get("/")
    assert home.status_code == 200 and "<form" in home.text
    page = client.get("/", params={"q": 'walrus OR "<script>"'})
    assert "<mark>walrus</mark>" in page.text
    assert "<script>" not in page.text and "&lt;script&gt;" in page.text  # the query is escaped
    assert "/view/" in page.text
    assert "Запит не прийнято" in client.get("/", params={"q": "x" * 300}).text


def test_document_page(client: TestClient, index: Index) -> None:
    r = client.get(f"/view/{doc_id(index, 'pep-0572.rst')}", params={"q": "walrus"})
    assert r.status_code == 200 and "Assignment Expressions" in r.text and "до результатів" in r.text
