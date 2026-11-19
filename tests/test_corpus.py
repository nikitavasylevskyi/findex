import tempfile
import unittest
from pathlib import Path

from findex.corpus import iter_documents


class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "pep-0008.rst").write_text("PEP: 8\nTitle: Style Guide for Python Code\n\nbody", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "intro.rst").write_text("Introduction\n============\n\ntext", encoding="utf-8")
        (root / "image.png").write_bytes(b"\x89PNG")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_titles_and_ids(self):
        docs = list(iter_documents(self.root))
        self.assertEqual([d.doc_id for d in docs], [0, 1])
        self.assertEqual(docs[0].title, "PEP 8: Style Guide for Python Code")
        self.assertEqual(docs[1].title, "Introduction")

    def test_generator(self):
        it = iter_documents(self.root)
        self.assertIs(iter(it), it)


if __name__ == "__main__":
    unittest.main()
