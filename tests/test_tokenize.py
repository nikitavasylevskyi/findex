import unittest

from findex.tokenize import tokenize


def toks(text):
    return list(tokenize(text))


class TokenizeTest(unittest.TestCase):
    def test_lowercase_and_nfkc(self):
        self.assertEqual(toks("FILE ﬁle Ｐｙｔｈｏｎ"), ["file", "file", "python"])

    def test_casefold(self):
        self.assertEqual(toks("Straße"), ["strasse"])

    def test_apostrophes(self):
        self.assertEqual(toks("don't Python’s"), ["dont", "pythons"])

    def test_quotes_are_not_apostrophes(self):
        self.assertEqual(toks("'quoted' word"), ["quoted", "word"])

    def test_hyphen_splits(self):
        self.assertEqual(toks("read-only"), ["read", "only"])

    def test_numbers(self):
        self.assertEqual(toks("PEP 572, Python 3.13"), ["pep", "572", "python", "3", "13"])

    def test_underscore_identifiers(self):
        self.assertEqual(toks("__init__ sys_path"), ["init", "sys_path"])

    def test_other_scripts(self):
        self.assertEqual(toks("Привіт, світе!"), ["привіт", "світе"])

    def test_is_lazy(self):
        gen = tokenize("a b c")
        self.assertEqual(next(gen), "a")

    def test_empty(self):
        self.assertEqual(toks("  ... --- "), [])


if __name__ == "__main__":
    unittest.main()
