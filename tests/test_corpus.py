from pathlib import Path

import pytest

from findex.corpus import guess_title, iter_documents


def test_ids_and_order(corpus_dir: Path) -> None:
    docs = list(iter_documents(corpus_dir))
    assert [d.doc_id for d in docs] == list(range(len(docs)))
    assert [d.path.name for d in docs] == sorted(d.path.name for d in docs)


@pytest.mark.parametrize(("name", "text", "title"), [
    ("pep-0008.rst", "PEP: 8\nTitle: Style Guide for Python Code\n", "PEP 8: Style Guide for Python Code"),
    ("intro.rst", "Introduction\n============\n\ntext", "Introduction"),
    ("ast.rst", ":mod:`!ast` --- Abstract syntax trees\n=====\n", "ast --- Abstract syntax trees"),
    ("plain.rst", "no heading here", "plain"),
])
def test_titles(name: str, text: str, title: str) -> None:
    assert guess_title(Path(name), text) == title


def test_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        next(iter_documents(tmp_path / "nope"))
