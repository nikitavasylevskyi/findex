"""Reading the corpus lazily, one document at a time.

A corpus is either a directory of text files or a JSONL file written by
``findex crawl`` (one page per line). A crawled page is addressed as
``<file.jsonl>#<byte offset of its line>``, so its text can be read back for
snippets without keeping the pages anywhere else.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = (".rst", ".txt", ".md")

_PEP_TITLE = re.compile(r"^Title:\s*(.+)$", re.MULTILINE)
_ROLE = re.compile(r":[\w:]+:`!?~?([^`<]+?)(?:\s*<[^>]*>)?`")
_UNDERLINE = re.compile(r"^([=\-~^\"'`#*+]){3,}\s*$")


@dataclass
class Document:
    doc_id: int
    path: Path
    title: str
    text: str


def guess_title(path: Path, text: str) -> str:
    """PEPs carry a ``Title:`` header; other rst files start with an underlined heading."""
    m = _PEP_TITLE.search(text, 0, 2000)
    if m:
        number = path.stem.removeprefix("pep-").lstrip("0") or "0"
        return f"PEP {number}: {m.group(1).strip()}"
    lines = text.splitlines()
    for i in range(1, min(len(lines), 40)):
        if _UNDERLINE.match(lines[i]) and lines[i - 1].strip() and not _UNDERLINE.match(lines[i - 1]):
            return _ROLE.sub(r"\1", lines[i - 1].strip()).strip("*`")
    return path.stem


def iter_paths(root: Path) -> Iterator[Path]:
    """All document files under root in a stable order."""
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in SUFFIXES:
            yield path


def read_document(doc_id: int, path: Path) -> Document:
    text = path.read_text(encoding="utf-8", errors="replace")
    return Document(doc_id, path, guess_title(path, text), text)


def is_jsonl(path: str | Path) -> bool:
    return Path(path).suffix == ".jsonl"


def iter_jsonl(path: Path) -> Iterator[Document]:
    """Pages from a crawl, read line by line."""
    with path.open("rb") as f:
        doc_id = 0
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            page = json.loads(line)
            yield Document(doc_id, Path(f"{path}#{offset}"), page.get("title") or page["url"], page["text"])
            doc_id += 1


def _jsonl_record(ref: str) -> dict[str, str] | None:
    file, _, offset = ref.rpartition("#")
    if not is_jsonl(file) or not offset.isdigit():
        return None
    with open(file, "rb") as f:
        f.seek(int(offset))
        record: dict[str, str] = json.loads(f.readline())
        return record


def load_text(ref: str) -> str:
    """Text of a document by its stored path (a file or a crawled page)."""
    record = _jsonl_record(ref)
    if record is not None:
        return record["text"]
    return Path(ref).read_text(encoding="utf-8", errors="replace")


def display_path(ref: str) -> str:
    """What to show for a document: the file path, or the URL of a crawled page."""
    try:
        record = _jsonl_record(ref)
    except (OSError, ValueError):
        return ref
    return record["url"] if record is not None else ref


def iter_documents(root: str | Path) -> Iterator[Document]:
    """Yield documents one by one; only the current file is held in memory."""
    root = Path(root)
    if root.is_file() and is_jsonl(root):
        yield from iter_jsonl(root)
        return
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    for doc_id, path in enumerate(iter_paths(root)):
        yield read_document(doc_id, path)
