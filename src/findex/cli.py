"""Command line interface: ``findex index | search | stats``."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex import __version__, boolean, storage
from findex.corpus import iter_paths
from findex.parallel import ExecutorKind, build_parallel
from findex.query import QuerySyntaxError
from findex.ranking import SCORERS, Searcher
from findex.search import open_index
from findex.stats import stats_greedy, stats_lazy
from findex.storage import IndexFormatError

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Full-text search over a corpus of Python documents.")
out = Console()
err = Console(stderr=True)
log = logging.getLogger("findex")

Scorer = Literal["bm25", "tfidf"]
Engine = Literal["merge", "set"]
Fmt = Literal["binary", "pickle"]


def setup_logging(verbose: int) -> None:
    """Configured once, here; library modules only call logging.getLogger."""
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    logging.basicConfig(level=level, stream=sys.stderr,
                        format="%(levelname)s %(name)s: %(message)s", force=True)


def fail(message: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {message}", highlight=False)
    return typer.Exit(code)


def version_callback(value: bool) -> None:
    if value:
        out.print(f"findex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    verbose: Annotated[int, typer.Option("--verbose", "-v", count=True,
                                         help="-v: timings, -vv: debug details")] = 0,
    version: Annotated[bool, typer.Option("--version", callback=version_callback, is_eager=True,
                                          help="Show the version and exit.")] = False,
) -> None:
    setup_logging(verbose)


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Directory with documents")],
    out_path: Annotated[Path, typer.Option("--out", "-o", help="Where to write the index")] = Path("index.bin"),
    fmt: Annotated[Fmt, typer.Option("--format")] = "binary",
    positions: Annotated[bool, typer.Option("--positions/--no-positions",
                                            help="Store token positions (needed for phrases)")] = True,
    workers: Annotated[int, typer.Option("--workers", "-w", min=1, help="Parallel workers")] = os.cpu_count() or 1,
    executor: Annotated[ExecutorKind, typer.Option("--executor", "-e")] = "processes",
) -> None:
    """Build an index from a directory of documents."""
    if not corpus.is_dir():
        raise fail(f"corpus directory not found: {corpus}")
    total = sum(1 for _ in iter_paths(corpus))
    t0 = time.perf_counter()
    with Progress(TextColumn("[bold]indexing"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
                  console=err, transient=True) as progress:
        task = progress.add_task("index", total=total)
        index, _ = build_parallel(corpus, workers, executor, positions,
                                  on_chunk=lambda n: progress.advance(task, n))
    storage.save(index, out_path, fmt)
    err.print(f"{index.num_docs} documents, {len(index)} terms -> {out_path} "
              f"({out_path.stat().st_size / 2**20:.1f} MiB) in {time.perf_counter() - t0:.1f} s")


@app.command()
def search(
    query: Annotated[str, typer.Argument(help='Query, e.g. \'python AND (async OR await) "event loop"\'')],
    index_path: Annotated[Path, typer.Option("--index", "-i", help="Index file")] = Path("index.bin"),
    k: Annotated[int, typer.Option("-k", min=1, help="Number of results")] = 10,
    scorer: Annotated[Scorer, typer.Option("--scorer", "-s")] = "bm25",
    as_json: Annotated[bool, typer.Option("--json", help="Print results as JSON to stdout")] = False,
    boolean_only: Annotated[bool, typer.Option("--boolean", help="Unranked Boolean search")] = False,
    engine: Annotated[Engine, typer.Option("--engine")] = "merge",
) -> None:
    """Search the index and show ranked results with snippets."""
    try:
        with open_index(index_path) as index:
            if boolean_only:
                hits = boolean.search(index, query, engine)
                rows = [{"doc_id": d, "title": index.docs[d].title, "path": index.docs[d].path}
                        for d in hits[:k]]
                if as_json:
                    print(json.dumps(rows, ensure_ascii=False, indent=2))
                else:
                    for r in rows:
                        out.print(f"{r['doc_id']:5}  {r['title']}", highlight=False)
                    err.print(f"{len(hits)} documents")
                return
            results = Searcher(index).search(query, SCORERS[scorer], k)
    except (FileNotFoundError, IndexFormatError, QuerySyntaxError) as e:
        raise fail(str(e)) from None

    if as_json:
        print(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))
        return
    if not results:
        err.print("no results")
        return
    table = Table(title=f"{query}  [{scorer}]", show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("document", style="bold", max_width=34)
    table.add_column("snippet")
    for rank, r in enumerate(results, 1):
        snippet = r.snippet.replace("[", r"\[").replace("**", "\0")
        parts = snippet.split("\0")
        snippet = "".join(f"[yellow bold]{p}[/]" if i % 2 else p for i, p in enumerate(parts))
        table.add_row(str(rank), f"{r.score:.2f}", r.title, snippet)
    out.print(table)


@app.command()
def stats(
    corpus: Annotated[Path, typer.Argument(help="Directory with documents")],
    top: Annotated[int, typer.Option("--top", min=1)] = 20,
    greedy: Annotated[bool, typer.Option("--greedy", help="List-based version of lab 1")] = False,
) -> None:
    """Corpus statistics: documents, tokens, vocabulary, most frequent terms."""
    if not corpus.is_dir():
        raise fail(f"corpus directory not found: {corpus}")
    t0 = time.perf_counter()
    s = (stats_greedy if greedy else stats_lazy)(corpus, top)
    table = Table(title=f"{corpus} — {s.documents} documents, {s.tokens} tokens, "
                        f"{s.vocabulary} terms, {s.bytes / 2**20:.1f} MiB")
    table.add_column("#", justify="right")
    table.add_column("term")
    table.add_column("count", justify="right")
    for i, (term, n) in enumerate(s.top, 1):
        table.add_row(str(i), term, str(n))
    out.print(table)
    err.print(f"{time.perf_counter() - t0:.2f} s")
