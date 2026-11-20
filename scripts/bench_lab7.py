"""Lab 7: def vs async def under 20 concurrent requests, then oha.

    uv run python scripts/bench_lab7.py --index index.bin --oha path/to/oha --out bench/lab7.json
    uv run python scripts/bench_lab7.py --deployed https://findex.onrender.com --oha path/to/oha

The ``async`` variant is built only here (``async_app``): the application
itself keeps the plain ``def`` route.
"""


import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx

sys.path.insert(0, str(Path(__file__).parent))
from bench_lab5 import cpu_model  # noqa: E402

QUERIES = ["assignment expressions", "type hints", "asyncio event loop", "generator", "pattern matching",
           "garbage collection", "unicode strings", "import system", "context manager", "decorators",
           "f-strings", "dataclasses", "exception groups", "free threading", "packaging metadata",
           "async generators", "keyword arguments", "dictionary order", "module attributes", "iterators"]
OHA_URL = "/search?q=asyncio+event+loop&k=10"


def async_app():  # type: ignore[no-untyped-def]  # uvicorn factory
    """The same app with /search as ``async def``: the search then runs on the event loop itself."""
    from fastapi import Query
    from fastapi.routing import APIRoute

    from findex.web.app import SearcherDep, create_app, run_search
    from findex.web.models import SearchParams, SearchResponse

    app = create_app()
    app.router.routes = [r for r in app.router.routes if not (isinstance(r, APIRoute) and r.path == "/search")]

    @app.get("/search", response_model=SearchResponse)
    async def search(params: Annotated[SearchParams, Query()], searcher: SearcherDep) -> SearchResponse:
        return run_search(searcher, params)

    app.router.routes.insert(0, app.router.routes.pop())  # before the HTML catch-alls, like the original
    return app


def start(variant: str, workers: int, port: int, index: str) -> "subprocess.Popen[bytes]":
    env = os.environ | {"INDEX_PATH": index, "PORT": str(port), "LOG_LEVEL": "warning",
                        "PYTHONPATH": str(Path(__file__).parent)}
    target = "bench_lab7:async_app" if variant == "async" else "findex.web.app:create_app"
    cmd = [sys.executable, "-m", "uvicorn", target, "--factory", "--port", str(port), "--workers", str(workers),
           "--log-level", "warning"]
    proc = subprocess.Popen(cmd, env=env)
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                # every worker has its own copy of the index: give them all time to load
                time.sleep(2 if workers > 1 else 0)
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    proc.kill()
    raise SystemExit(f"{variant} x{workers} did not become healthy")


async def burst(base: str) -> dict[str, float]:
    """20 different searches at once, and /health asked in the middle of them."""
    async with httpx.AsyncClient(base_url=base, timeout=60) as client:
        await client.get("/search", params={"q": "warm up"})

        async def one(q: str) -> float:
            t0 = time.perf_counter()
            r = await client.get("/search", params={"q": q, "k": 10})
            r.raise_for_status()
            return time.perf_counter() - t0

        async def health() -> float:
            await asyncio.sleep(0.2)  # the searches are already running
            t0 = time.perf_counter()
            await client.get("/health")
            return time.perf_counter() - t0

        t0 = time.perf_counter()
        times, health_time = await asyncio.gather(asyncio.gather(*(one(q) for q in QUERIES)), health())
        wall = time.perf_counter() - t0
    return {"wall_ms": wall * 1000, "median_ms": statistics.median(times) * 1000,
            "max_ms": max(times) * 1000, "health_ms": health_time * 1000}


def oha(binary: str, url: str, seconds: int, connections: int) -> dict[str, float]:
    out = subprocess.run([binary, "-z", f"{seconds}s", "-c", str(connections), "--no-tui",
                          "--output-format", "json", url], check=True, capture_output=True, text=True).stdout
    data = json.loads(out)
    pct = data["latencyPercentiles"]
    return {"rps": data["summary"]["requestsPerSec"], "p50_ms": pct["p50"] * 1000, "p95_ms": pct["p95"] * 1000,
            "p99_ms": pct["p99"] * 1000, "success": data["summary"]["successRate"],
            "statuses": data.get("statusCodeDistribution", {})}


def local(args: argparse.Namespace) -> None:
    rows = []
    print(f"local, {cpu_model()}, {os.cpu_count()} CPUs")
    for variant, workers in (("async", 1), ("def", 1), ("def", 4)):
        proc = start(variant, workers, args.port, args.index)
        try:
            base = f"http://127.0.0.1:{args.port}"
            bursts = [asyncio.run(burst(base)) for _ in range(3)]
            b = {k: statistics.median(x[k] for x in bursts) for k in bursts[0]}
            o = oha(args.oha, base + OHA_URL, args.seconds, args.connections) if args.oha else {}
        finally:
            proc.terminate()
            proc.wait(timeout=30)
        row = {"variant": variant, "workers": workers, "burst": b, "oha": o}
        rows.append(row)
        print(f"{variant:5} x{workers}: 20 at once {b['wall_ms']:.0f} ms (median {b['median_ms']:.0f}, "
              f"max {b['max_ms']:.0f}, /health {b['health_ms']:.0f} ms)"
              + (f"; oha {o['rps']:.0f} req/s p50 {o['p50_ms']:.0f} p95 {o['p95_ms']:.0f} p99 {o['p99_ms']:.0f} ms"
                 if o else ""), flush=True)
    save(args.out, {"cpu": cpu_model(), "cpus": os.cpu_count(), "rows": rows})


def deployed(args: argparse.Namespace) -> None:
    base = args.deployed.rstrip("/")
    for _ in range(30):  # a free instance may be asleep: wake it first
        try:
            if httpx.get(base + "/health", timeout=60).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(5)
    b = asyncio.run(burst(base))
    o = oha(args.oha, base + OHA_URL, args.seconds, args.connections)
    print(f"deployed {base}: 20 at once {b['wall_ms']:.0f} ms; oha {o['rps']:.1f} req/s "
          f"p50 {o['p50_ms']:.0f} p95 {o['p95_ms']:.0f} p99 {o['p99_ms']:.0f} ms, statuses {o['statuses']}")
    save(args.out, {"url": base, "burst": b, "oha": o})


def save(path: str, data: object) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index.bin")
    parser.add_argument("--port", type=int, default=8077)
    parser.add_argument("--oha", help="path to the oha binary; without it only the 20-request burst runs")
    parser.add_argument("--seconds", type=int, default=15)
    parser.add_argument("--connections", type=int, default=20)
    parser.add_argument("--deployed", metavar="URL")
    parser.add_argument("--out", default="bench/lab7.json")
    args = parser.parse_args()
    if args.deployed:
        deployed(args)
    else:
        local(args)


if __name__ == "__main__":
    main()
