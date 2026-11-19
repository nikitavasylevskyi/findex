"""Lab 6: the same crawl with 1, 5 and 20 requests at once.

Each run is a fresh interpreter, so the peak RSS belongs to that crawl only::

    uv run python scripts/bench_lab6.py http://127.0.0.1:8765/ --pages 200 --out bench/lab6-mirror.json
    uv run python scripts/bench_lab6.py https://peps.python.org/ --pages 200 --out bench/lab6-peps.json

``--blocking-demo`` shows what asyncio's debug mode reports when CPU-bound work
(building the index) runs inside a coroutine instead of outside the loop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_lab5 import cpu_model, peak_rss_mib  # noqa: E402

RUNS = [(1, 1), (5, 5), (20, 20), (20, 4)]  # (concurrency, per host)


def one(url: str, pages: int, concurrency: int, per_host: int) -> dict[str, float]:
    from findex.crawl.run import run_crawl

    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        stats = asyncio.run(run_crawl(url, Path(tmp) / "c.jsonl", max_pages=pages, concurrency=concurrency,
                                      per_host=per_host, delay=0.0, timeout=10.0,
                                      log_path=Path(tmp) / "c.log"))
        wall = time.perf_counter() - t0
    return {"wall": wall, "pages": stats.pages, "rate": stats.pages / wall, "errors": stats.errors,
            "rss_mib": peak_rss_mib()}


def run(args: argparse.Namespace) -> None:
    rows = []
    print(f"{args.url}, {args.pages} pages; {cpu_model()}")
    for concurrency, per_host in RUNS:
        proc = subprocess.run([sys.executable, __file__, "--one", args.url, str(args.pages),
                               str(concurrency), str(per_host)], capture_output=True, text=True)
        if proc.returncode != 0:
            sys.exit(f"concurrency {concurrency} failed:\n{proc.stderr}")
        row = {"concurrency": concurrency, "per_host": per_host} | json.loads(proc.stdout)
        rows.append(row)
        print(f"| {concurrency} | {per_host} | {row['wall']:.1f} | {row['rate']:.1f} | {row['errors']:.0f} "
              f"| {row['rss_mib']:.0f} |", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"url": args.url, "pages": args.pages, "cpu": cpu_model(),
                                          "rows": rows}, indent=2))


def blocking_demo(corpus: str) -> None:
    """Index building inside a coroutine: the loop cannot run anything else meanwhile."""
    from findex.index import build_index

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    async def ticker() -> None:
        last = time.perf_counter()
        for _ in range(30):
            await asyncio.sleep(0.1)
            now = time.perf_counter()
            if now - last > 0.5:
                print(f"ticker: expected 0.1 s between ticks, got {now - last:.2f} s")
            last = now

    async def main() -> None:
        asyncio.get_running_loop().slow_callback_duration = 0.1
        tick = asyncio.create_task(ticker())
        await asyncio.sleep(0.25)
        build_index(corpus, positions=True)  # wrong: CPU-bound work on the event loop
        await tick

    asyncio.run(main(), debug=True)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        url, pages, concurrency, per_host = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
        print(json.dumps(one(url, pages, concurrency, per_host)))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("url", nargs="?")
    parser.add_argument("--pages", type=int, default=200)
    parser.add_argument("--out", default="bench/lab6.json")
    parser.add_argument("--blocking-demo", metavar="CORPUS")
    args = parser.parse_args()
    if args.blocking_demo:
        blocking_demo(args.blocking_demo)
    else:
        run(args)


if __name__ == "__main__":
    main()
