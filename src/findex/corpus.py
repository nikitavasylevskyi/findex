"""Reading the corpus lazily, one document at a time."""

from __future__ import annotations

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


def iter_documents(root: str | Path) -> Iterator[Document]:
    """Yield documents one by one; only the current file is held in memory."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {root}")
    for doc_id, path in enumerate(iter_paths(root)):
        text = path.read_text(encoding="utf-8", errors="replace")
        yield Document(doc_id, path, guess_title(path, text), text)
