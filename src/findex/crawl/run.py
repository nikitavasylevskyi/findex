"""findex crawl: pages go to JSONL as they arrive, the status of every URL to crawl.log."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from findex.crawl.crawler import USER_AGENT, CrawlStats, crawl

status_log = logging.getLogger("findex.crawl.status")


def counters(stats: CrawlStats, max_pages: int) -> Text:
    return Text.from_markup(
        f"[bold]pages[/] {stats.pages}/{max_pages} │ [bold]{stats.rate:.1f}[/] pages/s │ "
        f"[bold]in flight[/] {stats.in_flight} │ [bold]errors[/] {stats.errors} │ "
        f"[bold]queue[/] {stats.queued} │ {stats.elapsed:.1f} s")


async def run_crawl(seed: str, out_path: Path, *, max_pages: int, concurrency: int, per_host: int,
                    delay: float, timeout: float, log_path: Path | None, user_agent: str = USER_AGENT,
                    console: Console | None = None, debug: bool = False) -> CrawlStats:
    if debug:
        loop = asyncio.get_running_loop()
        loop.set_debug(True)
        loop.slow_callback_duration = 0.05  # report any step that holds the loop for more than 50 ms
    handler: logging.Handler | None = None
    if log_path is not None:
        handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        crawl_logger = logging.getLogger("findex.crawl")  # findex.crawl.status propagates into it
        crawl_logger.addHandler(handler)
        crawl_logger.setLevel(logging.INFO)
        crawl_logger.propagate = debug  # per-URL lines go to the file; to the terminal only with -vv
    stats = CrawlStats()
    try:
        with (out_path.open("w", encoding="utf-8") as out,
              Live(counters(stats, max_pages), console=console or Console(quiet=True), refresh_per_second=8,
                   auto_refresh=False) as live):

            async def refresh() -> None:
                while True:
                    live.update(counters(stats, max_pages), refresh=True)
                    await asyncio.sleep(0.125)

            ticker = asyncio.create_task(refresh())
            try:
                pages = crawl([seed], max_pages=max_pages, concurrency=concurrency, per_host=per_host,
                              delay=delay, timeout=timeout, user_agent=user_agent, stats=stats)
                async with contextlib.aclosing(pages):
                    async for page in pages:
                        record = {"url": page.url, "title": page.title, "status": page.status,
                                  "depth": page.depth, "text": page.text}
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                ticker.cancel()
                live.update(counters(stats, max_pages), refresh=True)
    finally:
        if handler is not None:
            logging.getLogger("findex.crawl").removeHandler(handler)
            logging.getLogger("findex.crawl").propagate = True
            handler.close()
    return stats
