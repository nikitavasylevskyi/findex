"""Saving and loading the index in two formats.

pickle
    One call, keeps the dataclasses as they are. **Never unpickle a file you did
    not create yourself**: pickle can store a reference to any callable plus its
    arguments (``__reduce__``), and ``pickle.load`` calls it — loading a foreign
    index file is the same as running foreign code.

binary
    Own format that stores only numbers and strings, so loading cannot execute
    anything. Layout::

        b"FIDX" | version:u32 | header_len:u32 | header (UTF-8 JSON) | data (array of u32)

    The JSON header has the document metadata and, for every term, its document
    frequency and how many u32 values it occupies. The data block contains, term
    after term, the doc_ids (delta-encoded), the term frequencies and, if
    positions were stored, the positions for each posting.
"""

from __future__ import annotations

import json
import pickle
import struct
from array import array
from pathlib import Path
from typing import Any, Literal

import numpy as np

from findex.index import DocMeta, Index, PackedPostings
from findex.timing import timed

MAGIC = b"FIDX"
VERSION = 1
Format = Literal["pickle", "binary"]


class IndexFormatError(ValueError):
    pass


@timed
def save(index: Index, path: str | Path, fmt: Format = "binary") -> None:
    path = Path(path)
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    data = array("I")
    terms: list[tuple[str, int, int]] = []
    for term in index:
        start = len(data)
        prev = 0
        raw = index.raw(term)
        if isinstance(raw, PackedPostings):  # straight from a parallel build: no need to unpack
            for d in raw.doc_ids.tolist():
                data.append(d - prev)
                prev = d
            data.extend(raw.tfs.tolist())
            if index.with_positions:
                data.extend(raw.positions.tolist())
        else:
            for p in raw:
                data.append(p.doc_id - prev)
                prev = p.doc_id
            data.extend(p.tf for p in raw)
            if index.with_positions:
                for p in raw:
                    data.extend(p.positions)
        terms.append((term, len(raw), len(data) - start))
    header = json.dumps({
        "positions": index.with_positions,
        "docs": [[d.path, d.title, d.length] for d in index.docs],
        "terms": terms,
    }, ensure_ascii=False).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(header)))
        f.write(header)
        data.tofile(f)


@timed
def load(path: str | Path, fmt: Format | None = None) -> Index:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"index file not found: {path}")
    with path.open("rb") as f:
        magic = f.read(4)
        if fmt is None:
            fmt = "binary" if magic == MAGIC else "pickle"
        if fmt == "pickle":
            f.seek(0)
            try:
                obj: object = pickle.load(f)  # only for files we wrote ourselves, see module docstring
            except Exception as e:
                raise IndexFormatError(f"{path} is not a findex index ({e.__class__.__name__})") from e
            if not isinstance(obj, Index):
                raise IndexFormatError(f"{path} does not contain an index")
            return obj
        if magic != MAGIC:
            raise IndexFormatError(f"{path} is not a findex binary index")
        version, header_len = struct.unpack("<II", f.read(8))
        if version != VERSION:
            raise IndexFormatError(f"unsupported index version {version}")
        header: dict[str, Any] = json.loads(f.read(header_len).decode("utf-8"))
        raw = f.read()

    # The whole body is one u32 array; every term becomes views into it. Gaps are turned back
    # into doc_ids with one cumsum per term instead of a Python loop over every posting (lab 8).
    data = np.frombuffer(raw, dtype="<u4")
    with_pos = bool(header["positions"])
    docs = [DocMeta(i, str(p), str(t), int(n)) for i, (p, t, n) in enumerate(header["docs"])]
    postings: dict[str, PackedPostings] = {}
    off = 0
    for term, df, size in header["terms"]:
        ids = np.cumsum(data[off:off + df], dtype=np.int32)
        tfs = data[off + df:off + 2 * df].astype(np.int32)
        positions = data[off + 2 * df:off + int(size)] if with_pos else data[0:0]
        postings[str(term)] = PackedPostings.from_arrays(ids, tfs, positions)
        off += int(size)
    return Index(postings, docs, with_pos)
