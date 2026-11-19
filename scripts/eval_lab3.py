"""Ranking checks and precision@5 for lab 3.

    uv run python scripts/eval_lab3.py index.bin
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from findex import timing
from findex.index import Posting
from findex.ranking import BM25, SCORERS, Searcher, TfIdf
from findex.search import open_index

timing.ENABLED = False
QUERIES = Path(__file__).resolve().parent.parent / "eval" / "queries.json"


def rel_path(path: str) -> str:
    return "/".join(Path(path).parts[-2:])


def main(index_path: str) -> None:
    with open_index(index_path) as index:
        s = Searcher(index)
        tfidf, bm25 = TfIdf(), BM25()

        print("## 1. rare term vs frequent term (one occurrence each, same document)")
        doc = next(d for d in index.docs if d.path.endswith("pep-0572.rst"))
        for term in ("the", "python", "walrus"):
            p = Posting(doc.doc_id, 1)
            print(f"  {term:8} df={index.df(term):5}  tfidf={tfidf.score(index, index.df(term), p):.2f}"
                  f"  bm25={bm25.score(index, index.df(term), p):.2f}")
        top = s.search("the OR walrus", bm25, 3, snippets=False)
        print("  top-3 for 'the OR walrus':", [r.title for r in top])

        print("\n## 2. repetitions (term 'walrus', average-length document)")
        avg_doc = min(index.docs, key=lambda d: abs(d.length - index.avg_doc_length))
        prev_b = prev_t = 0.0
        df = index.df("walrus")
        for tf in (1, 2, 3, 5, 10, 19, 20):
            p = Posting(avg_doc.doc_id, tf)
            b, t = bm25.score(index, df, p), tfidf.score(index, df, p)
            print(f"  tf={tf:2}  bm25={b:5.2f} (+{b - prev_b:.3f})  tfidf={t:5.2f} (+{t - prev_t:.3f})")
            prev_b, prev_t = b, t

        print("\n## 3. short vs long document, one occurrence each")
        term = "walrus"
        lengths = sorted(index.docs, key=lambda d: d.length)
        short, long_ = lengths[len(lengths) // 10], lengths[-1]
        for d in (short, long_):
            p = Posting(d.doc_id, 1)
            print(f"  {d.title[:45]:45} length={d.length:6}  bm25={bm25.score(index, index.df(term), p):.2f}"
                  f"  tfidf={tfidf.score(index, index.df(term), p):.2f}")

        print("\n## precision@5")
        queries = json.loads(QUERIES.read_text(encoding="utf-8"))
        totals = {"tfidf": 0.0, "bm25": 0.0}
        print("| # | query | TF-IDF | BM25 |")
        for i, q in enumerate(queries, 1):
            rel = set(q["relevant"])
            row = []
            for name in ("tfidf", "bm25"):
                res = s.search(q["query"], SCORERS[name], 5, snippets=False)
                p5 = sum(rel_path(r.path) in rel for r in res) / 5
                totals[name] += p5
                row.append(p5)
            print(f"| {i} | `{q['query']}` | {row[0]:.1f} | {row[1]:.1f} |")
        n = len(queries)
        print(f"| | **mean** | **{totals['tfidf'] / n:.2f}** | **{totals['bm25'] / n:.2f}** |")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "index.bin")
