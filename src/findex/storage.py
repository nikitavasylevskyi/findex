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
from typing import Literal

from findex.index import DocMeta, InvertedIndex, Posting

MAGIC = b"FIDX"
VERSION = 1
Format = Literal["pickle", "binary"]


def save(index: InvertedIndex, path: str | Path, fmt: Format = "binary") -> None:
    path = Path(path)
    if fmt == "pickle":
        with path.open("wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
        return
    data = array("I")
    terms: list[list[object]] = []
    for term, plist in index.postings.items():
        start = len(data)
        prev = 0
        for p in plist:
            data.append(p.doc_id - prev)
            prev = p.doc_id
        data.extend(p.tf for p in plist)
        if index.with_positions:
            for p in plist:
                data.extend(p.positions)
        terms.append([term, len(plist), len(data) - start])
    header = json.dumps({
        "positions": index.with_positions,
        "docs": [[d.path, d.title, d.length] for d in index.docs],
        "terms": terms,
    }, ensure_ascii=False).encode("utf-8")
    with path.open("wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(header)))
        f.write(header)
        data.tofile(f)


def load(path: str | Path, fmt: Format | None = None) -> InvertedIndex:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(4)
        if fmt is None:
            fmt = "binary" if magic == MAGIC else "pickle"
        if fmt == "pickle":
            f.seek(0)
            index = pickle.load(f)  # only for files we wrote ourselves, see module docstring
            if not isinstance(index, InvertedIndex):
                raise ValueError(f"{path} does not contain an index")
            return index
        if magic != MAGIC:
            raise ValueError(f"{path} is not a findex binary index")
        version, header_len = struct.unpack("<II", f.read(8))
        if version != VERSION:
            raise ValueError(f"unsupported index version {version}")
        header = json.loads(f.read(header_len).decode("utf-8"))
        data = array("I")
        data.frombytes(f.read())

    with_pos: bool = header["positions"]
    docs = [DocMeta(i, p, t, n) for i, (p, t, n) in enumerate(header["docs"])]
    postings: dict[str, list[Posting]] = {}
    off = 0
    for term, df, size in header["terms"]:
        ids = data[off:off + df]
        tfs = data[off + df:off + 2 * df]
        doc_id = 0
        plist: list[Posting] = []
        pos_off = off + 2 * df
        for gap, tf in zip(ids, tfs):
            doc_id += gap
            if with_pos:
                plist.append(Posting(doc_id, tf, tuple(data[pos_off:pos_off + tf])))
                pos_off += tf
            else:
                plist.append(Posting(doc_id, tf))
        postings[term] = plist
        off += size
    return InvertedIndex(postings, docs, with_pos)
