import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bm25s
from common import CORPUS_DIR, load_corpus, demo

INDEX = CORPUS_DIR / "bm25"


def build():
    passages = load_corpus()
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize([p["rendered"] for p in passages], stopwords="en"))
    bm25.save(str(INDEX))
    print("BM25 index saved:", INDEX)


class BM25Retriever:
    name = "BM25"
    threshold_key = "score"

    def __init__(self, passages=None):
        self.passages = passages or load_corpus()
        self.bm25 = bm25s.BM25.load(str(INDEX))

    def search(self, claims, k=5):
        ids, scores = self.bm25.retrieve(bm25s.tokenize(claims, stopwords="en"), k=k)
        return [[{**self.passages[int(i)], "score": float(s)} for i, s in zip(row_i, row_s)]
                for row_i, row_s in zip(ids, scores)]


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        demo(BM25Retriever())