"""Query language and its syntax tree.

Grammar (recursive descent, one function per rule)::

    query  := or_expr
    or_expr  := and_expr ("OR" and_expr)*
    and_expr := not_expr (["AND"] not_expr)*        # juxtaposition means AND
    not_expr := "NOT" not_expr | atom
    atom   := "(" or_expr ")" | '"' words '"' | word

Example: ``python AND (async OR await) NOT java "event loop"``.

Nodes can be combined in Python as well: ``Term("a") & ~Term("b") | Phrase(("c", "d"))``.
``evaluate(index)`` returns the sorted list of matching doc_ids.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass

from findex.boolean import difference, intersect, union
from findex.index import Index
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    pass


class Node:
    """Base of all query nodes. Subclasses are frozen dataclasses, so trees compare by value."""

    def __and__(self, other: Node) -> Node:
        return And(self, other)

    def __or__(self, other: Node) -> Node:
        return Or(self, other)

    def __invert__(self) -> Node:
        return Not(self)

    def evaluate(self, index: Index) -> list[int]:
        raise NotImplementedError

    def terms(self) -> Iterator[str]:
        """Positive terms of the query (used for scoring and snippets)."""
        return iter(())


@dataclass(frozen=True)
class Term(Node):
    term: str

    def evaluate(self, index: Index) -> list[int]:
        return [p.doc_id for p in index.get(self.term, ())]

    def terms(self) -> Iterator[str]:
        yield self.term


@dataclass(frozen=True)
class Phrase(Node):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> list[int]:
        if not index.with_positions:
            raise QuerySyntaxError("phrase search needs an index built with --positions")
        plists = [index.get(w, []) for w in self.words]
        if not all(plists):
            return []
        candidates = [p.doc_id for p in plists[0]]
        for pl in plists[1:]:
            candidates = intersect(candidates, [p.doc_id for p in pl])
        by_doc = [{p.doc_id: p.positions for p in pl} for pl in plists]
        result: list[int] = []
        for doc_id in candidates:
            starts = set(by_doc[0][doc_id])
            for offset, positions in enumerate(by_doc[1:], 1):
                starts &= {pos - offset for pos in positions[doc_id]}
                if not starts:
                    break
            if starts:
                result.append(doc_id)
        return result

    def terms(self) -> Iterator[str]:
        yield from self.words


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        # "a NOT b" is evaluated as a difference instead of complementing b
        if isinstance(self.right, Not):
            return difference(self.left.evaluate(index), self.right.child.evaluate(index))
        if isinstance(self.left, Not):
            return difference(self.right.evaluate(index), self.left.child.evaluate(index))
        return intersect(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index) -> list[int]:
        return union(self.left.evaluate(index), self.right.evaluate(index))

    def terms(self) -> Iterator[str]:
        yield from self.left.terms()
        yield from self.right.terms()


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index) -> list[int]:
        return difference(list(range(index.num_docs)), self.child.evaluate(index))


_LEX = re.compile(r'\s*(?:(\()|(\))|"([^"]*)"|([^\s()"]+))')


def lex(query: str) -> list[str]:
    tokens: list[str] = []
    pos = 0
    query = query.rstrip()
    while pos < len(query):
        m = _LEX.match(query, pos)
        if not m:
            raise QuerySyntaxError(f"unclosed quote at position {pos}")
        lpar, rpar, phrase, word = m.group(1, 2, 3, 4)
        if lpar:
            tokens.append("(")
        elif rpar:
            tokens.append(")")
        elif phrase is not None:
            tokens.append(f'"{phrase}"')
        else:
            tokens.append(word)
        pos = m.end()
    return tokens


class Parser:
    def __init__(self, query: str) -> None:
        self.tokens = lex(query)
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> Node:
        if not self.tokens:
            raise QuerySyntaxError("empty query")
        node = self.or_expr()
        if self.peek() is not None:
            raise QuerySyntaxError(f"unexpected {self.peek()!r}")
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self.peek() == "OR":
            self.take()
            node = Or(node, self.and_expr())
        return node

    def and_expr(self) -> Node:
        node = self.not_expr()
        while self.peek() not in (None, ")", "OR"):
            if self.peek() == "AND":
                self.take()
            node = And(node, self.not_expr())
        return node

    def not_expr(self) -> Node:
        if self.peek() == "NOT":
            self.take()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("query ends too early")
        if tok in ("AND", "OR"):
            raise QuerySyntaxError(f"operator {tok} without a left operand")
        if tok == ")":
            raise QuerySyntaxError("unexpected ')'")
        self.take()
        if tok == "(":
            node = self.or_expr()
            if self.peek() != ")":
                raise QuerySyntaxError("missing ')'")
            self.take()
            return node
        if tok.startswith('"'):
            return _words_node(tok[1:-1], phrase=True)
        return _words_node(tok, phrase=False)


def _words_node(text: str, phrase: bool) -> Node:
    words = tuple(tokenize(text))
    if not words:
        raise QuerySyntaxError(f"no searchable words in {text!r}")
    if len(words) == 1:
        return Term(words[0])
    if phrase:
        return Phrase(words)
    node: Node = Term(words[0])  # "read-only" outside quotes: read AND only
    for w in words[1:]:
        node = And(node, Term(w))
    return node


def parse(query: str) -> Node:
    return Parser(query).parse()
