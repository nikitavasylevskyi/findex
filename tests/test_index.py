from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import doc_id

from findex import boolean, storage
from findex.index import Index, build_index


def test_postings_sorted_and_counted(index: Index) -> None:
    b = doc_id(index, "b.rst")
    the = {p.doc_id: p for p in index["the"]}
    assert the[b].tf == 5
    for plist in index.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids))


def test_positions(index: Index) -> None:
    a = doc_id(index, "a.rst")
    loop = next(p for p in index["loop"] if p.doc_id == a)
    assert loop.positions == (1, 4, 9)


def test_mapping_protocol(index: Index) -> None:
    assert "walrus" in index and "java" not in index
    assert index.get("java") is None
    assert len(index) == len(list(index))
    assert index.df("the") == 4
    assert repr(index).startswith("Index(terms=")
    assert index.avg_doc_length > 0


@pytest.mark.parametrize(("a", "b", "op", "expected"), [
    ([1, 3, 5, 7], [3, 4, 5], boolean.intersect, [3, 5]),
    ([3, 4, 999, 2000], list(range(0, 1000, 3)), boolean.intersect, [3, 999]),  # bisect branch
    ([1, 3], [2, 3, 9], boolean.union, [1, 2, 3, 9]),
    ([], [2], boolean.union, [2]),
    ([1, 2, 3, 4], [2, 4, 6], boolean.difference, [1, 3]),
])
def test_merge_helpers(a: list[int], b: list[int], op: Callable[[list[int], list[int]], list[int]],
                       expected: list[int]) -> None:
    assert op(a, b) == expected


@pytest.mark.parametrize("query", [
    "async", "async generators", "walrus OR match", "the NOT walrus", "NOT the",
    "missing", "async OR missing", "the NOT event OR python", "read-only",
])
def test_engines_agree(index: Index, query: str) -> None:
    assert boolean.search(index, query, "merge") == boolean.search(index, query, "set")


@pytest.mark.parametrize("fmt", ["pickle", "binary"])
@pytest.mark.parametrize("positions", [True, False])
def test_save_load_roundtrip(corpus_dir: Path, tmp_path: Path, fmt: storage.Format, positions: bool) -> None:
    idx = build_index(corpus_dir, positions=positions)
    path = tmp_path / f"index.{fmt}"
    storage.save(idx, path, fmt)
    assert storage.load(path) == idx


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        storage.load(tmp_path / "missing.bin")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not an index at all")
    with pytest.raises(storage.IndexFormatError):
        storage.load(junk)
