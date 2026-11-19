"""Crawler tests: everything goes through httpx.MockTransport, no live network."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import pytest

from findex.corpus import display_path, iter_documents, load_text
from findex.crawl import crawl, fetch
from findex.crawl import fetching as fetch_module
from findex.crawl.crawler import CrawlStats, Page, parse_html
from findex.crawl.fetching import retry_after
from findex.crawl.robots import RobotsCache
from findex.crawl.run import run_crawl
from findex.crawl.urls import canonicalize
from findex.index import build_index
from findex.ranking import Searcher

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def client_for(handler: Handler, **kw: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), max_redirects=5, **kw)  # type: ignore[arg-type]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record backoff pauses instead of waiting for them."""
    waits: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay: float, *args: object) -> None:
        waits.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(fetch_module.asyncio, "sleep", fake_sleep)
    return waits


@pytest.mark.parametrize(("raw", "expected"), [
    ("HTTP://Example.COM", "http://example.com/"),
    ("https://example.com:443/a/./b/../c", "https://example.com/a/c"),
    ("http://example.com:8080//x//y/", "http://example.com:8080/x/y/"),
    ("https://example.com/p?b=2&a=1#frag", "https://example.com/p?a=1&b=2"),
    ("https://example.com/docs/index.html", "https://example.com/docs/"),
    ("mailto:someone@example.com", None),
    ("javascript:void(0)", None),
])
def test_canonicalize(raw: str, expected: str | None) -> None:
    assert canonicalize(raw) == expected


def test_canonicalize_relative() -> None:
    assert canonicalize("../pep-0008/#intro", base="https://peps.python.org/pep-0001/") == \
        "https://peps.python.org/pep-0008/"


def test_retry_after_seconds_and_date() -> None:
    assert retry_after(httpx.Response(429, headers={"Retry-After": "3"})) == 3.0
    later = retry_after(httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}))
    assert later == 0.0  # a date in the past means "now"
    assert retry_after(httpx.Response(429)) is None


def test_fetch_retries_429_with_retry_after(no_sleep: list[float]) -> None:
    calls = Counter[str]()

    async def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(200, html="<p>ok</p>")

    async def go() -> None:
        async with client_for(handler) as client:
            res = await fetch(client, "https://x.test/")
        assert res.ok and res.attempts == 2 and res.nbytes > 0
        assert no_sleep == [2.0]

    asyncio.run(go())


def test_fetch_gives_up_after_three_retries(no_sleep: list[float]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async def go() -> None:
        async with client_for(handler) as client:
            res = await fetch(client, "https://x.test/", backoff_base=0.5)
        assert res.status == 503 and res.attempts == 4
        assert len(no_sleep) == 3
        assert all(0 <= w <= 0.5 * 2 ** i for i, w in enumerate(no_sleep))  # backoff with jitter

    asyncio.run(go())


@pytest.mark.parametrize("status", [400, 403, 404, 410])
def test_fetch_does_not_retry_other_4xx(status: int, no_sleep: list[float]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    async def go() -> None:
        async with client_for(handler) as client:
            res = await fetch(client, "https://x.test/")
        assert res.status == status and res.attempts == 1 and no_sleep == []

    asyncio.run(go())


def test_fetch_timeout_is_retried_then_reported(no_sleep: list[float]) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.Event().wait()  # never answers
        raise AssertionError

    async def go() -> None:
        async with client_for(handler) as client:
            res = await fetch(client, "https://x.test/", timeout=0.01)
        assert res.status is None and res.error == "timeout" and res.attempts == 4

    asyncio.run(go())


def test_fetch_redirect_loop() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        target = "/b" if request.url.path == "/a" else "/a"
        return httpx.Response(302, headers={"Location": target})

    async def go() -> None:
        async with client_for(handler) as client:
            res = await fetch(client, "https://x.test/a")
        assert res.status is None and res.error == "redirect loop" and res.attempts == 1

    asyncio.run(go())


@pytest.mark.parametrize(("status", "body", "allowed"), [
    (200, "User-agent: *\nDisallow: /private/\n", False),
    (404, "", True),
    (503, "", False),
])
def test_robots(status: int, body: str, allowed: bool) -> None:
    requests = Counter[str]()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests[request.url.path] += 1
        return httpx.Response(status, text=body)

    async def go() -> None:
        async with client_for(handler) as client:
            robots = RobotsCache(client, "findex-test")
            results = await asyncio.gather(*(robots.can_fetch("https://x.test/private/p") for _ in range(5)))
            assert results == [allowed] * 5
            assert await robots.can_fetch("https://x.test/public") is (status != 503)
        assert requests["/robots.txt"] == 1  # cached per host, even for concurrent first calls

    asyncio.run(go())


def test_parse_html() -> None:
    title, text, links = parse_html(
        "<html><head><title> Hi </title><script>var x = 1</script></head>"
        "<body><p>Hello <b>world</b></p><a href='/a'>A</a><a>no href</a></body></html>")
    assert title == "Hi"
    assert "Hello" in text and "world" in text and "var x" not in text
    assert links == ["/a"]


def test_parse_html_prefers_main_content() -> None:
    _, text, links = parse_html("<body><div>Menu » Index</div><article><p>Real text</p>"
                                "<a href='/next'>next</a></article><a href='/side'>side</a></body>")
    assert text == "Real text next"
    assert links == ["/next", "/side"]  # links are still taken from the whole page


def site(pages: int, *, delay: float = 0.0) -> tuple[Handler, Counter[str], dict[str, int]]:
    """Pages /p0../pN that link to their neighbours, an external link and a disallowed one."""
    hits = Counter[str]()
    load = {"now": 0, "max": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        hits[path] += 1
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /secret\n")
        load["now"] += 1
        load["max"] = max(load["max"], load["now"])
        try:
            await asyncio.sleep(delay)
            if path == "/secret":
                return httpx.Response(200, html="<title>secret</title>")
            n = int(path.removeprefix("/p")) if path.startswith("/p") else 0
            targets = range(1, pages) if n == 0 else (n + 1, n + 2, n)  # p0 is a table of contents
            links = "".join(f'<a href="/p{m}#x">p{m}</a>' for m in targets if m < pages)
            return httpx.Response(200, html=f"<title>page {n}</title><body>word{n} common "
                                            f'{links}<a href="https://other.test/">x</a>'
                                            f'<a href="/secret">s</a></body>')
        finally:
            load["now"] -= 1

    return handler, hits, load


async def collect(handler: Handler, **kw: object) -> tuple[list[Page], CrawlStats]:
    stats = CrawlStats()
    pages: list[Page] = []
    async with client_for(handler) as client:
        async for page in crawl(["https://site.test/p0"], client=client, stats=stats, **kw):  # type: ignore[arg-type]
            pages.append(page)
    return pages, stats


def test_crawl_visits_each_page_once_and_respects_robots() -> None:
    handler, hits, _ = site(30)
    pages, stats = asyncio.run(collect(handler, max_pages=100, concurrency=5))
    assert sorted(p.url for p in pages) == sorted(f"https://site.test/p{n}" for n in range(30))
    assert all(hits[f"/p{n}"] == 1 for n in range(30))
    assert hits["/secret"] == 0 and stats.robots_blocked == 1
    assert stats.pages == 30 and stats.errors == 0


def test_crawl_stops_at_max_pages() -> None:
    handler, hits, _ = site(500)
    pages, stats = asyncio.run(collect(handler, max_pages=20, concurrency=8))
    assert len(pages) == 20 == stats.pages
    assert sum(hits.values()) < 60  # workers were cancelled, not left crawling


@pytest.mark.parametrize(("concurrency", "per_host"), [(10, 3), (2, 8)])
def test_crawl_limits_concurrency(concurrency: int, per_host: int) -> None:
    handler, _, load = site(40, delay=0.01)
    asyncio.run(collect(handler, max_pages=40, concurrency=concurrency, per_host=per_host))
    assert load["max"] == min(concurrency, per_host)


def test_crawl_delay_between_requests_to_one_host() -> None:
    handler, _, _ = site(6)

    async def go() -> float:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await collect(handler, max_pages=6, concurrency=6, delay=0.02)
        return loop.time() - t0

    assert asyncio.run(go()) >= 5 * 0.02


def test_run_crawl_writes_jsonl_that_can_be_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _, _ = site(12)
    real_client = httpx.AsyncClient

    def fake_client(*args: object, **kw: object) -> httpx.AsyncClient:
        kw.pop("limits", None)
        return real_client(transport=httpx.MockTransport(handler), **kw)  # type: ignore[arg-type]

    monkeypatch.setattr("findex.crawl.crawler.httpx.AsyncClient", fake_client)
    out, log = tmp_path / "crawl.jsonl", tmp_path / "crawl.log"
    stats = asyncio.run(run_crawl("https://site.test/p0", out, max_pages=12, concurrency=4, per_host=4,
                                  delay=0, timeout=5, log_path=log))
    lines = out.read_text(encoding="utf-8").splitlines()
    assert stats.pages == len(lines) == 12
    assert {json.loads(line)["url"] for line in lines} == {f"https://site.test/p{n}" for n in range(12)}
    assert "ROBOTS https://site.test/secret" in log.read_text(encoding="utf-8")

    docs = list(iter_documents(out))
    assert [d.doc_id for d in docs] == list(range(12))
    assert load_text(str(docs[3].path)) == docs[3].text
    assert display_path(str(docs[3].path)).startswith("https://site.test/p")

    index = build_index(out, positions=True)
    hits = Searcher(index).search("word7", k=3)
    assert [h.path for h in hits] == ["https://site.test/p7"]
    assert "**word7**" in hits[0].snippet
