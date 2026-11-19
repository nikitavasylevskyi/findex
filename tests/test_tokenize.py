import pytest

from findex.tokenize import tokenize


@pytest.mark.parametrize(("text", "expected"), [
    ("Hello World", ["hello", "world"]),
    ("FILE ﬁle Ｐｙｔｈｏｎ", ["file", "file", "python"]),          # NFKC: ligature, full-width
    ("Straße", ["strasse"]),                                        # casefold, not lower
    ("Ǆ", ["dž"]),                                                  # NFKC splits the digraph, casefold lowers it
    ("café café", ["café", "café"]),                          # composed and decomposed é
    ("don't Python’s", ["dont", "pythons"]),                        # ASCII and typographic apostrophe
    ("'quoted' word", ["quoted", "word"]),                          # quotes are not apostrophes
    ("read-only", ["read", "only"]),
    ("PEP 572, Python 3.13", ["pep", "572", "python", "3", "13"]),
    ("__init__ sys_path", ["init", "sys_path"]),
    ("Привіт, світе! Слово’є", ["привіт", "світе", "словоє"]),
    ("  ... --- ", []),
    ("", []),
])
def test_tokenize(text: str, expected: list[str]) -> None:
    assert list(tokenize(text)) == expected


def test_tokenize_is_lazy() -> None:
    gen = tokenize("a b c")
    assert next(gen) == "a"
