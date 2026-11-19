"""Property-based tests: Hypothesis generates many small corpora."""

from pathlib import Path
from tempfile import TemporaryDirectory

from hypothesis import given, settings
from hypothesis import strategies as st

from findex import boolean, storage
from findex.index import Index, build_index

WORDS = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta"]
docs_strategy = st.lists(st.lists(st.sampled_from(WORDS), min_size=0, max_size=12), min_size=1, max_size=8)
sorted_ids = st.lists(st.integers(0, 200), max_size=60).map(lambda xs: sorted(set(xs)))


def build(docs: list[list[str]], tmp: Path, positions: bool = True) -> Index:
    for i, words in enumerate(docs):
        (tmp / f"{i:03}.txt").write_text(" ".join(words), encoding="utf-8")
    return build_index(tmp, positions=positions)


@settings(max_examples=40, deadline=None)
@given(docs=docs_strategy, positions=st.booleans(), fmt=st.sampled_from(["binary", "pickle"]))
def test_load_save_roundtrip(docs: list[list[str]], positions: bool, fmt: storage.Format) -> None:
    with TemporaryDirectory() as t:
        tmp = Path(t)
        index = build(docs, tmp, positions)
        storage.save(index, tmp / "i.bin", fmt)
        assert storage.load(tmp / "i.bin") == index


@settings(max_examples=40, deadline=None)
@given(docs=docs_strategy)
def test_postings_sorted_and_tf_matches(docs: list[list[str]]) -> None:
    with TemporaryDirectory() as t:
        index = build(docs, Path(t))
        for term, plist in index.items():
            ids = [p.doc_id for p in plist]
            assert ids == sorted(set(ids))
            for p in plist:
                assert p.tf == docs[p.doc_id].count(term) == len(p.positions)


@given(a=sorted_ids, b=sorted_ids)
def test_merge_matches_set_operations(a: list[int], b: list[int]) -> None:
    assert boolean.intersect(a, b) == sorted(set(a) & set(b))
    assert boolean.union(a, b) == sorted(set(a) | set(b))
    assert boolean.difference(a, b) == sorted(set(a) - set(b))
