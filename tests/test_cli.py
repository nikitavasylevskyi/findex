import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


@pytest.fixture(scope="module")
def index_file(corpus_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("idx") / "index.bin"
    result = runner.invoke(app, ["index", str(corpus_dir), "--out", str(path)])
    assert result.exit_code == 0, result.output
    return path


def test_search_table(index_file: Path) -> None:
    result = runner.invoke(app, ["search", "walrus", "--index", str(index_file)])
    assert result.exit_code == 0
    assert "Assignment Expressions" in result.stdout


def test_search_json(index_file: Path) -> None:
    result = runner.invoke(app, ["search", '"event loop"', "-i", str(index_file), "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.stdout)
    assert rows[0]["title"] == "Event loop"
    assert set(rows[0]) == {"score", "doc_id", "title", "path", "snippet"}


def test_boolean_json(index_file: Path) -> None:
    result = runner.invoke(app, ["search", "async", "-i", str(index_file), "--boolean", "--json"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 2


def test_missing_index_is_one_line(tmp_path: Path) -> None:
    result = runner.invoke(app, ["search", "x", "-i", str(tmp_path / "nope.bin")])
    assert result.exit_code == 1
    assert "error: index file not found" in result.stderr
    assert "Traceback" not in result.output


def test_bad_query(index_file: Path) -> None:
    result = runner.invoke(app, ["search", "(walrus", "-i", str(index_file)])
    assert result.exit_code == 1
    assert "missing ')'" in result.stderr


def test_stats(corpus_dir: Path) -> None:
    result = runner.invoke(app, ["stats", str(corpus_dir), "--top", "3"])
    assert result.exit_code == 0
    assert "the" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "findex" in result.stdout
