import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import load_corpus
from bm25_retriever import BM25Retriever
from embedding_retriever import EmbeddingRetriever

POOL = 50


class HybridRetriever:
    name = "Hybrid"
    threshold_key = "emb_score"  # rank points can't measure relevance, cosine can
    threshold = None  # minimum threshold_key for enough evidence. set from the validation sweep

    def __init__(self, passages=None, emb=None, bm25=None):
        self.passages = passages or load_corpus()
        self.emb = emb or EmbeddingRetriever(self.passages)
        self.bm25 = bm25 or BM25Retriever(self.passages)

    def search(self, claims, k=5):
        from_emb = self.emb.search(claims, POOL)
        from_bm25 = self.bm25.search(claims, POOL)
        vectors = self.emb.encode(claims)
        out = []
        for vec, a, b in zip(vectors, from_emb, from_bm25):
            points = {}
            for ranking in (a, b):
                for rank, hit in enumerate(ranking):
                    points[hit["pid"]] = points.get(hit["pid"], 0) + 1 / (60 + rank)
            best = sorted(points, key=points.get, reverse=True)[:k]
            out.append([{**self.passages[pid], "score": points[pid],
                         "emb_score": self.emb.cosine(vec, pid)} for pid in best])
        return out