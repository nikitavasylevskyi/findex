from pathlib import Path

import pytest

from findex.index import Index, build_index

TEXTS = {
    "a.rst": "Event loop\n==========\n\nthe event loop runs async tasks; the loop is single threaded",
    "b.rst": "the the the the generators and the iterators",
    "c.rst": "walrus",
    "d.rst": "walrus " + " ".join(f"word{i}" for i in range(300)),
    "e.rst": " ".join(["python"] * 20),
    "f.rst": "python and the event of the year",
    "pep-0572.rst": "PEP: 572\nTitle: Assignment Expressions\n\nthe walrus operator := assigns inside expressions",
    "g.rst": "match statement and structural pattern matching; async generators yield values",
}


@pytest.fixture(scope="session")
def corpus_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("corpus")
    for name, text in TEXTS.items():
        (root / name).write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="session")
def index(corpus_dir: Path) -> Index:
    return build_index(corpus_dir, positions=True)


def doc_id(index: Index, filename: str) -> int:
    return next(d.doc_id for d in index.docs if d.path.endswith(filename))
