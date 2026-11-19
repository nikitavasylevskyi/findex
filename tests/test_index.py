import tempfile
import unittest
from pathlib import Path

from findex import boolean, storage
from findex.index import build_index


def make_corpus(root: Path) -> None:
    texts = {
        "a.rst": "async functions and await expressions",
        "b.rst": "generators yield values; async generators too",
        "c.rst": "the walrus operator := assigns inside expressions",
        "d.rst": "match statement and structural pattern matching",
    }
    for name, text in texts.items():
        (root / name).write_text(text, encoding="utf-8")


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_corpus(self.root)
        self.index = build_index(self.root, positions=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_postings_sorted_and_counted(self):
        self.assertEqual([p.doc_id for p in self.index.postings["async"]], [0, 1])
        self.assertEqual(self.index.postings["generators"][0].tf, 2)
        self.assertEqual(self.index.postings["generators"][0].positions, (0, 4))
        for plist in self.index.postings.values():
            ids = [p.doc_id for p in plist]
            self.assertEqual(ids, sorted(set(ids)))

    def test_boolean_engines_agree(self):
        queries = ["async", "async await", "async OR match", "expressions NOT walrus",
                   "NOT async", "missing", "async OR missing", "and NOT match OR yield"]
        expected = {
            "async": [0, 1], "async await": [0], "async OR match": [0, 1, 3],
            "expressions NOT walrus": [0], "NOT async": [2, 3], "missing": [],
            "async OR missing": [0, 1], "and NOT match OR yield": [0, 1],
        }
        for q in queries:
            for engine in ("merge", "set"):
                self.assertEqual(boolean.search(self.index, q, engine), expected[q], (q, engine))

    def test_merge_helpers(self):
        self.assertEqual(boolean.intersect([1, 3, 5, 7], [3, 4, 5]), [3, 5])
        self.assertEqual(boolean.union([1, 3], [2, 3, 9]), [1, 2, 3, 9])
        self.assertEqual(boolean.difference([1, 2, 3, 4], [2, 4, 6]), [1, 3])

    def test_save_load_roundtrip(self):
        for fmt in ("pickle", "binary"):
            path = self.root / f"index.{fmt}"
            storage.save(self.index, path, fmt)
            loaded = storage.load(path)
            self.assertEqual(loaded.postings, self.index.postings, fmt)
            self.assertEqual(loaded.docs, self.index.docs, fmt)


if __name__ == "__main__":
    unittest.main()


class GallopTest(unittest.TestCase):
    def test_asymmetric_intersect(self):
        long = list(range(0, 1000, 3))
        self.assertEqual(boolean.intersect([3, 4, 999, 2000], long), [3, 999])
        self.assertEqual(boolean.intersect(long, [3, 4, 999]), [3, 999])
