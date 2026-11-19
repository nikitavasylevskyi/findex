"""Download the corpus into data/.

Corpus: all Python Enhancement Proposals (github.com/python/peps) and the
reStructuredText sources of the CPython documentation (github.com/python/cpython,
directory Doc/). Both are published under open licences (PEPs are public domain
or CC0, the docs under the PSF licence).

Only git is needed. Files are copied flat into data/peps/ and data/docs/.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"


def run(*args: str, cwd: Path | None = None) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def fetch_peps(tmp: Path) -> int:
    repo = tmp / "peps"
    run("git", "clone", "--depth", "1", "https://github.com/python/peps", str(repo))
    out = DATA / "peps"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted((repo / "peps").glob("pep-*.rst")):
        shutil.copyfile(src, out / src.name)
        n += 1
    return n


def fetch_docs(tmp: Path) -> int:
    repo = tmp / "cpython"
    run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
        "https://github.com/python/cpython", str(repo))
    run("git", "sparse-checkout", "set", "Doc", cwd=repo)
    out = DATA / "docs"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for src in sorted((repo / "Doc").rglob("*.rst")):
        rel = src.relative_to(repo / "Doc").with_suffix("")
        name = "__".join(rel.parts) + ".rst"
        shutil.copyfile(src, out / name)
        n += 1
    return n


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        peps = fetch_peps(tmp)
        docs = fetch_docs(tmp)
    print(f"peps: {peps} files, docs: {docs} files -> {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
