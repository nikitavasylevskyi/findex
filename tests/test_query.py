import pytest
from conftest import doc_id

from findex.index import Index
from findex.query import And, Node, Not, Or, Phrase, QuerySyntaxError, Term, parse


@pytest.mark.parametrize(("query", "tree"), [
    ('python AND (async OR await) NOT java "event loop"',
     And(And(And(Term("python"), Or(Term("async"), Term("await"))), Not(Term("java"))),
         Phrase(("event", "loop")))),
    ("a b OR c", Or(And(Term("a"), Term("b")), Term("c"))),
    ("a OR b c", Or(Term("a"), And(Term("b"), Term("c")))),
    ("NOT NOT a", Not(Not(Term("a")))),
    ("a AND NOT b OR c", Term("a") & ~Term("b") | Term("c")),
    ("Async", Term("async")),
    ("read-only", And(Term("read"), Term("only"))),
    ('"Read-Only"', Phrase(("read", "only"))),
    ('"walrus"', Term("walrus")),
])
def test_parse_trees(query: str, tree: Node) -> None:
    assert parse(query) == tree


@pytest.mark.parametrize("bad", ["", "(a", "a)", "OR a", "a OR", '"open', "NOT", "()", '""'])
def test_syntax_errors(bad: str) -> None:
    with pytest.raises(QuerySyntaxError):
        parse(bad)


def test_phrase_needs_consecutive_positions(index: Index) -> None:
    a = doc_id(index, "a.rst")
    assert parse('"event loop"').evaluate(index) == [a]
    assert parse('"loop event"').evaluate(index) == []
