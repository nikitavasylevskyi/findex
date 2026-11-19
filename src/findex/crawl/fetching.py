"""One page: timeout, retries with exponential backoff and jitter, Retry-After."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from dataclasses import dataclass

import httpx

log = logging.getLogger("findex.crawl")

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER = 30.0


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int | None  # None: no HTTP answer at all (timeout, connection error, redirect loop)
    nbytes: int
    elapsed: float
    attempts: int
    content_type: str = ""
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300


def retry_after(resp: httpx.Response) -> float | None:
    """Retry-After as seconds: either a number or an HTTP date."""
    value = resp.headers.get("retry-after")
    if value is None:
        return None
    if value.strip().isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def backoff(attempt: int, base: float, cap: float = 10.0) -> float:
    """Full jitter: uniform in [0, base * 2**attempt], so retrying clients do not come back together."""
    return random.uniform(0, min(cap, base * 2 ** attempt))


async def fetch(client: httpx.AsyncClient, url: str, *, timeout: float = 10.0, retries: int = 3,
                backoff_base: float = 0.5) -> FetchResult:
    """GET with at most ``retries`` repeats, only for 429, 5xx and timeouts.

    Other 4xx are final. Redirects are followed by httpx (max_redirects on the
    client); a redirect loop ends as TooManyRedirects and is not retried.
    """
    t0 = time.perf_counter()
    attempt = 0
    while True:
        attempt += 1
        wait: float | None = None
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url, follow_redirects=True)
                body = await resp.aread()
        except TimeoutError:
            if attempt > retries:
                return FetchResult(url, url, None, 0, time.perf_counter() - t0, attempt, error="timeout")
            log.info("retry %s: timeout after %.1f s (attempt %d)", url, timeout, attempt)
        except httpx.TooManyRedirects:
            return FetchResult(url, url, None, 0, time.perf_counter() - t0, attempt, error="redirect loop")
        except httpx.HTTPError as e:
            return FetchResult(url, url, None, 0, time.perf_counter() - t0, attempt, error=e.__class__.__name__)
        else:
            if resp.status_code not in RETRY_STATUS or attempt > retries:
                ctype = resp.headers.get("content-type", "")
                text = resp.text if "html" in ctype or "text/" in ctype else ""
                return FetchResult(url, str(resp.url), resp.status_code, len(body), time.perf_counter() - t0,
                                   attempt, ctype, text)
            wait = retry_after(resp)
            if wait is not None:
                wait = min(wait, MAX_RETRY_AFTER)
            log.info("retry %s: %d%s (attempt %d)", url, resp.status_code,
                     f", Retry-After {wait:.1f} s" if wait is not None else "", attempt)
        await asyncio.sleep(wait if wait is not None else backoff(attempt - 1, backoff_base))
