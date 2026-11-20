"""Lab 8: precision@5 of lexical (BM25), semantic and hybrid search.

The 10 labelled queries of lab 3 plus 5 paraphrases written without the words of
the relevant documents (eval/queries_lab8.json).

    uv run python scripts/eval_lab8.py --index index.bin --embeddings embeddings.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findex import storage, timing
from findex.hybrid import ModeSearcher, bag_of_words
from findex.ranking import Searcher
from findex.semantic import Model2VecEncoder, SemanticIndex

timing.ENABLED = False
QUERIES = Path(__file__).resolve().parent.parent / "eval" / "queries_lab8.json"
MODES = ("lexical", "semantic", "hybrid")


def rel_path(path: str) -> str:
    return "/".join(Path(path).parts[-2:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="index.bin")
    parser.add_argument("--embeddings", default="embeddings.npy")
    parser.add_argument("--out", default="bench/lab8-precision.json")
    args = parser.parse_args()
    searcher = Searcher(storage.load(args.index))
    semantic = SemanticIndex.load(args.embeddings)
    modes = ModeSearcher(searcher, semantic, Model2VecEncoder(semantic.model))
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    rows = []
    print("| query | kind | lexical | semantic | hybrid |")
    print("|---|---|---:|---:|---:|")
    for q in queries:
        relevant = set(q["relevant"])
        row = {"text": q["text"], "kind": q["kind"]}
        for mode in MODES:
            query = bag_of_words(q["text"]) if mode == "lexical" else q["text"]
            top = modes.search(query, mode, k=5, snippets=False).results
            row[mode] = sum(rel_path(searcher.index.docs[r.doc_id].path) in relevant for r in top) / 5
            row[mode + "_top"] = [rel_path(searcher.index.docs[r.doc_id].path) for r in top]
        rows.append(row)
        print(f"| {q['text']} | {q['kind']} | {row['lexical']:.1f} | {row['semantic']:.1f} | {row['hybrid']:.1f} |")
    for kind in ("lab 3", "paraphrase", None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        means = " | ".join(f"**{sum(r[m] for r in sel) / len(sel):.2f}**" for m in MODES)
        print(f"| **mean, {kind or 'all 15'}** | | {means} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"model": semantic.model, "rows": rows}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
