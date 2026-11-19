"""crawl(): an async generator of pages.

A queue of URLs, N workers inside a TaskGroup, one semaphore for all requests
and one per host, plus a minimal pause between two requests to the same host.
Pages are handed to the caller as soon as they are parsed, so nothing keeps
the whole crawl in memory.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field

import httpx
from selectolax.lexbor import LexborHTMLParser

from findex.crawl.fetching import FetchResult, fetch
from findex.crawl.robots import RobotsCache
from findex.crawl.urls import canonicalize, host_of

log = logging.getLogger("findex.crawl")
status_log = logging.getLogger("findex.crawl.status")  # one line per URL, goes to crawl.log

USER_AGENT = "findex-crawler/0.6 (student project; https://github.com/nikitavasylevskyi/findex)"


@dataclass
class Page:
    url: str
    status: int
    title: str
    text: str
    depth: int
    nbytes: int
    elapsed: float


@dataclass
class CrawlStats:
    pages: int = 0
    errors: int = 0
    in_flight: int = 0
    queued: int = 0
    robots_blocked: int = 0
    started: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def rate(self) -> float:
        return self.pages / self.elapsed if self.elapsed > 0 else 0.0


def parse_html(html: str) -> tuple[str, str, list[str]]:
    """Title, visible text and raw href values."""
    tree = LexborHTMLParser(html)
    node = tree.css_first("title")
    title = node.text(strip=True) if node is not None else ""
    links = [href for a in tree.css("a[href]") if (href := a.attributes.get("href"))]
    for junk in tree.css("script, style, noscript, nav, header, footer"):
        junk.decompose()
    # the main content if the page marks it, so that menus and breadcrumbs do not end up in snippets
    main = tree.css_first("main, article, [role=main]")
    body = main if main is not None else tree.body if tree.body is not None else tree.root
    text = body.text(separator=" ", strip=True) if body is not None else ""
    return title, text, links


class _Host:
    """Per-host politeness: at most ``per_host`` requests at once and ``delay`` seconds between starts."""

    def __init__(self, per_host: int) -> None:
        self.sem = asyncio.Semaphore(per_host)
        self.lock = asyncio.Lock()
        self.next_start = 0.0

    async def wait_turn(self, delay: float) -> None:
        if delay <= 0:
            return
        loop = asyncio.get_running_loop()
        async with self.lock:  # book a start time, then sleep outside the lock
            now = loop.time()
            start = max(now, self.next_start)
            self.next_start = start + delay
        await asyncio.sleep(start - now)


async def crawl(seeds: Iterable[str], *, max_pages: int = 200, concurrency: int = 10, per_host: int = 4,
                delay: float = 0.0, timeout: float = 10.0, same_host: bool = True,
                user_agent: str = USER_AGENT, client: httpx.AsyncClient | None = None,
                stats: CrawlStats | None = None) -> AsyncGenerator[Page]:
    """Yield up to ``max_pages`` HTML pages reachable from ``seeds``.

    When the limit is reached (or the generator is closed early) the workers
    are cancelled and the HTTP client is closed.
    """
    stats = stats if stats is not None else CrawlStats()
    own_client = client is None
    if client is None:
        client = httpx.AsyncClient(headers={"user-agent": user_agent}, max_redirects=5,
                                   limits=httpx.Limits(max_connections=concurrency))
    robots = RobotsCache(client, user_agent)
    frontier: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    out: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    hosts: defaultdict[str, _Host] = defaultdict(lambda: _Host(per_host))
    limit = asyncio.Semaphore(concurrency)
    allowed_hosts: set[str] = set()

    for seed in seeds:
        url = canonicalize(seed)
        if url is None:
            raise ValueError(f"not an http(s) URL: {seed}")
        if url not in seen:
            seen.add(url)
            allowed_hosts.add(host_of(url))
            frontier.put_nowait((url, 0))

    def enqueue(url: str, depth: int) -> None:
        if url in seen or (same_host and host_of(url) not in allowed_hosts):
            return
        if len(seen) >= max_pages * 50:  # the frontier cannot grow without bound
            return
        seen.add(url)
        frontier.put_nowait((url, depth))

    async def handle(url: str, depth: int) -> None:
        if not await robots.can_fetch(url):
            stats.robots_blocked += 1
            status_log.info("ROBOTS %s", url)
            return
        host = hosts[host_of(url)]
        crawl_delay = await robots.crawl_delay(url) or 0.0
        async with host.sem:  # the host first: a busy host must not hold global slots
            await host.wait_turn(max(delay, crawl_delay))
            async with limit:
                stats.in_flight += 1
                try:
                    res: FetchResult = await fetch(client, url, timeout=timeout)
                finally:
                    stats.in_flight -= 1
        status_log.info("%s %s %d B %.0f ms attempts=%d%s", res.status or "ERR", url, res.nbytes,
                        res.elapsed * 1000, res.attempts, f" {res.error}" if res.error else "")
        if not res.ok:
            stats.errors += 1
            return
        final = canonicalize(res.final_url) or url
        if final != url:  # redirected: remember where we ended up, so links to it are not fetched again
            if final in seen and depth > 0:
                return
            seen.add(final)
        if "html" not in res.content_type:
            return
        title, text, links = parse_html(res.text)
        for href in links:
            link = canonicalize(href, base=final)
            if link is not None:
                enqueue(link, depth + 1)
        out.put_nowait(Page(final, res.status or 0, title, text, depth, res.nbytes, res.elapsed))

    async def worker() -> None:
        while True:
            url, depth = await frontier.get()
            try:
                await handle(url, depth)
            except Exception:
                stats.errors += 1
                log.exception("worker failed on %s", url)
            finally:
                frontier.task_done()
                stats.queued = frontier.qsize()

    async def run() -> None:
        try:
            async with asyncio.TaskGroup() as tg:
                workers = [tg.create_task(worker()) for _ in range(concurrency)]
                await frontier.join()  # every queued URL handled and nothing new was added
                for w in workers:
                    w.cancel()
        finally:
            out.put_nowait(None)

    runner = asyncio.create_task(run())
    try:
        while True:
            page = await out.get()
            if page is None:
                break
            stats.pages += 1
            yield page
            if stats.pages >= max_pages:
                break
    finally:
        runner.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner
        if own_client:
            await client.aclose()
