import tempfile
import unittest
from pathlib import Path

from findex import storage
from findex.index import Index, build_index
from findex.query import parse
from findex.ranking import BM25, Searcher, TfIdf, make_snippet
from findex.search import open_index
from findex import timing

timing.ENABLED = False


class RankingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        filler = " ".join(f"word{i}" for i in range(300))
        texts = {
            "a.rst": "the event loop runs async tasks; the loop is single threaded",
            "b.rst": "the the the the generators and the iterators",
            "c.rst": "walrus",
            "d.rst": "walrus " + filler,
            "e.rst": "python python python python python python python python python python "
                     "python python python python python python python python python python",
            "f.rst": "python and the event of the year",
        }
        for name, text in texts.items():
            (root / name).write_text(text, encoding="utf-8")
        self.root = root
        self.index = build_index(root, positions=True)
        self.searcher = Searcher(self.index)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mapping_protocol(self):
        idx = self.index
        self.assertIn("walrus", idx)
        self.assertNotIn("java", idx)
        self.assertEqual(idx.df("the"), 3)
        self.assertEqual(len(idx), len(list(idx)))
        self.assertEqual(idx["walrus"][0].doc_id, 2)
        self.assertEqual(idx.get("java"), None)
        self.assertTrue(repr(idx).startswith("Index(terms="))

    def test_phrase(self):
        self.assertEqual(parse('"event loop"').evaluate(self.index), [0])
        self.assertEqual(parse("event loop").evaluate(self.index), [0])
        self.assertEqual(parse('"loop event"').evaluate(self.index), [])

    def test_rare_term_scores_higher(self):
        from findex.index import Posting
        p = Posting(5, 1)  # same document, one occurrence each
        for scorer in (TfIdf(), BM25()):
            rare = scorer.score(self.index, self.index.df("walrus"), p)
            frequent = scorer.score(self.index, self.index.df("the"), p)
            self.assertGreater(rare, frequent, scorer)

    def test_bm25_saturation(self):
        bm = BM25()
        idx = self.index
        from findex.index import Posting
        s1 = bm.score(idx, 2, Posting(4, 19))
        s2 = bm.score(idx, 2, Posting(4, 20))
        self.assertLess(s2 - s1, 0.01 * s2)

    def test_short_document_wins(self):
        res = self.searcher.search("walrus", BM25(), snippets=False)
        self.assertEqual([r.doc_id for r in res], [2, 3])

    def test_results_sorted(self):
        res = self.searcher.search("the OR python OR event", BM25(), snippets=False)
        self.assertEqual(res, sorted(res, reverse=True))

    def test_cache(self):
        self.searcher.search("walrus", snippets=False)
        self.searcher.search("walrus", snippets=False)
        self.assertEqual(self.searcher.matching.cache_info().hits, 1)

    def test_snippet_highlight(self):
        snip = make_snippet(str(self.root / "a.rst"), ("event", "loop"))
        self.assertIn("**event** **loop**", snip)

    def test_open_index_releases_on_error(self):
        path = self.root / "i.bin"
        storage.save(self.index, path)
        with self.assertRaises(RuntimeError):
            with open_index(path) as idx:
                kept = idx
                self.assertEqual(len(idx), len(self.index))
                raise RuntimeError("boom")
        self.assertEqual(len(kept), 0)
        self.assertIsInstance(kept, Index)


if __name__ == "__main__":
    unittest.main()
