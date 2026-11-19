import unittest

from findex.query import And, Not, Or, Phrase, QuerySyntaxError, Term, parse


class ParserTest(unittest.TestCase):
    def test_example_from_assignment(self):
        tree = parse('python AND (async OR await) NOT java "event loop"')
        expected = And(And(And(Term("python"), Or(Term("async"), Term("await"))),
                           Not(Term("java"))), Phrase(("event", "loop")))
        self.assertEqual(tree, expected)

    def test_implicit_and_and_precedence(self):
        self.assertEqual(parse("a b OR c"), Or(And(Term("a"), Term("b")), Term("c")))
        self.assertEqual(parse("a OR b c"), Or(Term("a"), And(Term("b"), Term("c"))))
        self.assertEqual(parse("NOT NOT a"), Not(Not(Term("a"))))

    def test_operators_build_same_tree(self):
        self.assertEqual(parse("a AND NOT b OR c"), Term("a") & ~Term("b") | Term("c"))

    def test_words_are_normalised(self):
        self.assertEqual(parse("Async"), Term("async"))
        self.assertEqual(parse("read-only"), And(Term("read"), Term("only")))
        self.assertEqual(parse('"Read-Only"'), Phrase(("read", "only")))

    def test_errors(self):
        for bad in ["", "(a", "a)", "OR a", "a OR", '"open', "NOT", "()", '""']:
            with self.assertRaises(QuerySyntaxError, msg=bad):
                parse(bad)


if __name__ == "__main__":
    unittest.main()
