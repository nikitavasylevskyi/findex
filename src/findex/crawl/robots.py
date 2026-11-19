"""robots.txt, fetched once per host and kept for the whole crawl."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger("findex.crawl")


@dataclass
class Rules:
    verdict: Literal["allow all", "disallow all", "rules"]
    parser: RobotFileParser | None = None

    def can_fetch(self, agent: str, url: str) -> bool:
        if self.parser is None:
            return self.verdict == "allow all"
        return self.parser.can_fetch(agent, url)

    def crawl_delay(self, agent: str) -> float | None:
        if self.parser is None:
            return None
        delay = self.parser.crawl_delay(agent)
        return float(delay) if delay is not None else None


class RobotsCache:
    """Answers can_fetch(url) for any host; each host's robots.txt is downloaded once.

    Rules follow RFC 9309: 4xx (no file) means everything is allowed, a server
    error or a network failure means nothing is allowed for that host.
    """

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self._rules: dict[str, Rules] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def rules(self, url: str) -> Rules:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:  # the first request for a host fetches, the others wait for its answer
            if origin not in self._rules:
                self._rules[origin] = await self._load(origin)
        return self._rules[origin]

    async def _load(self, origin: str) -> Rules:
        try:
            resp = await self.client.get(origin + "/robots.txt", follow_redirects=True, timeout=10)
        except httpx.HTTPError as e:
            log.warning("robots %s: %s, host skipped", origin, e.__class__.__name__)
            return Rules("disallow all")
        if resp.status_code >= 500:
            rules = Rules("disallow all")
        elif resp.status_code >= 400:
            rules = Rules("allow all")
        else:
            parser = RobotFileParser(origin + "/robots.txt")
            parser.parse(resp.text.splitlines())
            rules = Rules("rules", parser)
        log.info("robots %s: %d, %s", origin, resp.status_code, rules.verdict)
        return rules

    async def can_fetch(self, url: str) -> bool:
        return (await self.rules(url)).can_fetch(self.user_agent, url)

    async def crawl_delay(self, url: str) -> float | None:
        return (await self.rules(url)).crawl_delay(self.user_agent)
