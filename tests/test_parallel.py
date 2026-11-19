from pathlib import Path

import pytest

from findex import parallel
from findex.corpus import iter_paths
from findex.index import Index, PackedPostings, build_index
from findex.parallel import build_parallel, build_partial, chunks, merge


def test_serial_merge_equals_lab4_index(corpus_dir: Path, index: Index) -> None:
    paths = [str(p) for p in iter_paths(corpus_dir)]
    assert merge([build_partial(paths, 0, positions=True)]) == index


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 3])
def test_executors_give_same_index(corpus_dir: Path, index: Index, executor: str, workers: int) -> None:
    built, timing = build_parallel(corpus_dir, workers, executor, positions=True)  # type: ignore[arg-type]
    assert built == index
    assert timing.merge <= timing.total


def test_chunks_cover_everything() -> None:
    paths = [str(i) for i in range(10)]
    parts = list(chunks(paths, 3))
    assert [first for _, first in parts] == [0, 4, 7]
    assert sum((c for c, _ in parts), []) == paths
    assert len(list(chunks(paths, 50))) == 10


def test_merge_rejects_gaps(corpus_dir: Path) -> None:
    paths = [str(p) for p in iter_paths(corpus_dir)]
    with pytest.raises(ValueError, match="do not follow"):
        merge([build_partial(paths[:2], 0), build_partial(paths[2:4], 5)])


@pytest.mark.parametrize("executor", ["threads", "processes"])
def test_worker_error_is_raised(tmp_path: Path, executor: str, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "ok.txt").write_text("fine", encoding="utf-8")
    missing = tmp_path / "vanished.txt"  # listed, but gone by the time the worker reads it
    monkeypatch.setattr(parallel, "iter_paths", lambda root: iter([tmp_path / "ok.txt", missing]))
    with pytest.raises(FileNotFoundError):
        build_parallel(tmp_path, 2, executor)  # type: ignore[arg-type]


def test_parallel_matches_single_core_on_more_files(tmp_path: Path) -> None:
    for i in range(40):
        (tmp_path / f"{i:02}.txt").write_text(f"doc {i} " + "word " * (i % 7), encoding="utf-8")
    built, _ = build_parallel(tmp_path, 4, "processes")
    assert built == build_index(tmp_path)


def test_packed_postings_roundtrip(index: Index) -> None:
    for term in ("the", "async"):
        if term not in index:
            continue
        packed = PackedPostings()
        for p in index[term]:
            packed.append(p.doc_id, list(p.positions), keep_positions=True)
        assert len(packed) == index.df(term)
        assert packed.unpack() == index[term]


def test_merge_of_packed_and_unpacked_parts(corpus_dir: Path, index: Index) -> None:
    paths = [str(p) for p in iter_paths(corpus_dir)]
    first, second = build_partial(paths[:3], 0, True), build_partial(paths[3:], 3, True)
    _ = [first[t] for t in first]  # unpack one side completely
    assert merge([second, first]) == index
