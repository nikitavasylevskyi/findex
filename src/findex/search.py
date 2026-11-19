"""Opening a saved index as a context manager."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from findex import storage
from findex.index import Index


@contextmanager
def open_index(path: str | Path) -> Generator[Index]:
    """``with open_index("index.bin") as index:`` — data is released on exit, also after an exception."""
    index = storage.load(path)
    try:
        yield index
    finally:
        index.close()
