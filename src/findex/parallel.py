"""Building the index on several cores: build_partial + merge.

The corpus is split into contiguous chunks of files. Each chunk gets its own
range of doc_ids in advance, so the partial indexes never clash and merging is
just concatenation of postings lists in chunk order.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from findex.corpus import iter_paths, read_document
from findex.index import DocMeta, Index, PackedPostings, Posting, index_documents_packed

log = logging.getLogger("findex.parallel")

ExecutorKind = Literal["serial", "threads", "processes"]


def build_partial(paths: Sequence[str], first_id: int, positions: bool = False) -> Index:
    """Index one chunk. Module-level and takes only paths, so it can be sent to another process.

    The worker reads the files itself: sending file names costs a few bytes,
    sending the texts would mean pickling megabytes through a pipe. The result
    is packed (see PackedPostings) for the same reason on the way back.
    """
    docs = (read_document(first_id + i, Path(p)) for i, p in enumerate(paths))
    return index_documents_packed(docs, positions)


def merge(parts: Sequence[Index]) -> Index:
    """Join partial indexes whose doc_id ranges follow each other."""
    if len(parts) == 1:
        return parts[0]
    postings: dict[str, list[Posting] | PackedPostings] = {}
    docs: list[DocMeta] = []
    expected = 0
    for part in sorted(parts, key=lambda p: p.docs[0].doc_id if p.docs else 0):
        if part.docs and part.docs[0].doc_id != expected:
            raise ValueError(f"doc_id ranges do not follow each other at {part.docs[0].doc_id}")
        for term in part:
            plist = part.raw(term)
            have = postings.get(term)
            if have is None:
                postings[term] = plist  # first chunk with this term: take it as is, the part is not used again
            elif isinstance(have, PackedPostings) and isinstance(plist, PackedPostings):
                have.extend(plist)
            else:
                postings[term] = [*_unpacked(have), *part[term]]
        docs.extend(part.docs)
        expected += len(part.docs)
    with_positions = all(p.with_positions for p in parts) if parts else False
    return Index(postings, docs, with_positions)


def _unpacked(plist: list[Posting] | PackedPostings) -> list[Posting]:
    return plist.unpack() if isinstance(plist, PackedPostings) else plist


def chunks(paths: list[str], n: int) -> Iterator[tuple[list[str], int]]:
    """Split into n contiguous chunks of almost equal size, with their first doc_id."""
    n = max(1, min(n, len(paths)))
    size, extra = divmod(len(paths), n)
    start = 0
    for i in range(n):
        end = start + size + (1 if i < extra else 0)
        yield paths[start:end], start
        start = end


def make_executor(kind: ExecutorKind, workers: int) -> Executor | None:
    if kind == "threads":
        return ThreadPoolExecutor(max_workers=workers)
    if kind == "processes":
        # spawn: a fresh interpreter that imports this module — the same on Linux, macOS and Windows
        return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))
    return None


@dataclass
class BuildTiming:
    total: float
    merge: float
    worker_cpu: float  # CPU seconds spent inside build_partial, summed over chunks


def _build_chunk(paths: Sequence[str], first_id: int, positions: bool) -> tuple[Index, float]:
    c0 = time.thread_time()  # per thread, so it is right for threads and for a single-threaded worker process
    part = build_partial(paths, first_id, positions)
    return part, time.thread_time() - c0


def build_parallel(root: str | Path, workers: int = os.cpu_count() or 1,
                   executor: ExecutorKind = "processes", positions: bool = False,
                   chunks_per_worker: int = 4,
                   on_chunk: Callable[[int], None] | None = None) -> tuple[Index, BuildTiming]:
    """Build with a pool. A crash in a worker is re-raised here with the worker's traceback."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    t0 = time.perf_counter()
    paths = [str(p) for p in iter_paths(root)]
    n_chunks = 1 if executor == "serial" else workers * chunks_per_worker
    parts_in: list[tuple[list[str], int]] = list(chunks(paths, n_chunks))
    pool = make_executor(executor, workers)
    parts: list[Index] = []
    worker_cpu = 0.0
    if pool is None:
        for chunk, first in parts_in:
            part, cpu = _build_chunk(chunk, first, positions)
            parts.append(part)
            worker_cpu += cpu
            if on_chunk:
                on_chunk(len(chunk))
    else:
        with pool:
            futures = [pool.submit(_build_chunk, chunk, first, positions) for chunk, first in parts_in]
            for fut, (chunk, _) in zip(futures, parts_in, strict=True):
                part, cpu = fut.result()  # raises the worker's exception, never waits forever on a dead worker
                parts.append(part)
                worker_cpu += cpu
                if on_chunk:
                    on_chunk(len(chunk))
    t1 = time.perf_counter()
    index = merge(parts)
    t2 = time.perf_counter()
    log.info("%s x%d: %d chunks, build %.2f s, merge %.2f s", executor, workers, len(parts_in), t1 - t0, t2 - t1)
    return index, BuildTiming(total=t2 - t0, merge=t2 - t1, worker_cpu=worker_cpu)
