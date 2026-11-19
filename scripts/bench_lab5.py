"""Lab 5: index build time for every executor and number of workers.

Every measurement runs in a fresh interpreter, so the peak RSS of the parent
process belongs to that one build only. Usage::

    uv run python scripts/bench_lab5.py data/ --repeat 3 --out bench/lab5.json
    uv run --group bench python scripts/bench_lab5.py --plot bench/lab5.json bench/lab5-gil.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def one(corpus: str, executor: str, workers: int) -> dict[str, float]:
    from findex.parallel import build_parallel

    t0 = time.perf_counter()
    c0 = time.process_time()
    _, timing = build_parallel(corpus, workers, executor, positions=True)  # type: ignore[arg-type]
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0  # the whole parent process, worker threads included
    if executor == "processes":
        cpu += timing.worker_cpu  # work done in the worker processes (their start-up is not counted)
    return {"wall": wall, "cpu": cpu, "rss_mib": peak_rss_mib(), "merge": timing.merge}


def peak_rss_mib() -> float:
    """Peak resident set size of this process (the parent; workers are not counted)."""
    try:
        import resource
    except ImportError:  # Windows
        import psutil

        return psutil.Process().memory_info().peak_wset / 2**20  # type: ignore[attr-defined]
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def run(args: argparse.Namespace) -> None:
    cpus = os.cpu_count() or 1
    workers = sorted({1, 2, 4, 8, cpus})
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"{sys.version.split()[0]} {'(GIL on)' if gil else '(GIL off)'}; {cpu_model()}, {cpus} logical CPUs")
    rows: list[dict[str, object]] = []
    for executor in args.executors:
        for w in [1] if executor == "serial" else workers:
            runs: list[dict[str, float]] = []
            for _ in range(args.repeat):
                out = subprocess.run([sys.executable, __file__, "--one", args.corpus, executor, str(w)],
                                     check=True, capture_output=True, text=True).stdout
                runs.append(json.loads(out))
            row: dict[str, object] = {"executor": executor, "workers": w}
            row |= {k: statistics.median(r[k] for r in runs) for k in runs[0]}
            rows.append(row)
            print(f"{executor:9} x{w}: wall {row['wall']:.2f} s, cpu {row['cpu']:.2f} s, "
                  f"rss {row['rss_mib']:.0f} MiB, merge {row['merge']:.2f} s", flush=True)
    result = {"python": sys.version.split()[0], "gil": gil, "cpu": cpu_model(), "cpus": cpus, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))


def table(results: list[dict[str, object]]) -> str:
    lines = ["| Python | executor | workers | wall, s | CPU, s | RSS, MiB | merge, s | speedup |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    base = None
    for res in results:
        rows: list[dict[str, float | str]] = res["rows"]  # type: ignore[assignment]
        label = f"{res['python']}{'' if res['gil'] else 't'}"
        for r in rows:
            if base is None and r["executor"] == "serial":
                base = float(r["wall"])
            speed = base / float(r["wall"]) if base else 1.0
            lines.append(f"| {label} | {r['executor']} | {r['workers']} | {r['wall']:.2f} | {r['cpu']:.2f} "
                         f"| {r['rss_mib']:.0f} | {r['merge']:.2f} | {speed:.2f}× |")
    return "\n".join(lines)


def plot(paths: list[str], png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = [json.loads(Path(p).read_text()) for p in paths]
    print(table(results))
    base = next(float(r["wall"]) for r in results[0]["rows"] if r["executor"] == "serial")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = max(r["workers"] for res in results for r in res["rows"])
    ax.plot([1, top], [1, top], "--", color="grey", label="ideal (linear)")
    for res in results:
        suffix = "" if res["gil"] else " (3.13t, no GIL)"
        for executor in ("threads", "processes"):
            pts = [(r["workers"], base / r["wall"]) for r in res["rows"] if r["executor"] == executor]
            if pts:
                ax.plot(*zip(*pts, strict=True), marker="o", label=executor + suffix)
    ax.axvline(results[0]["cpus"], color="lightgrey", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("workers")
    ax.set_ylabel("speedup vs serial (3.13, GIL)")
    ax.set_title(f"findex index: {results[0]['cpu']}, {results[0]['cpus']} CPUs")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        corpus, executor, workers = sys.argv[2], sys.argv[3], int(sys.argv[4])
        print(json.dumps(one(corpus, executor, workers)))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", default="data")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--executors", nargs="+", default=["serial", "threads", "processes"])
    parser.add_argument("--out", default="bench/lab5.json")
    parser.add_argument("--plot", nargs="+", metavar="JSON")
    parser.add_argument("--png", default="docs/lab5-speedup.png")
    args = parser.parse_args()
    if args.plot:
        plot(args.plot, args.png)
    else:
        run(args)


if __name__ == "__main__":
    main()
