import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
from sentence_transformers import SentenceTransformer
from common import CORPUS_DIR, load_corpus, demo

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX = CORPUS_DIR / "embeddings.faiss"


def build():
    passages = load_corpus()
    model = SentenceTransformer(MODEL)
    vectors = model.encode([p["rendered"] for p in passages], batch_size=256,
                           show_progress_bar=True, normalize_embeddings=True,
                           convert_to_numpy=True).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on normalised vectors = cosine
    index.add(vectors)
    faiss.write_index(index, str(INDEX))
    print("Embedding index saved:", INDEX)


class EmbeddingRetriever:
    name = "Embeddings"
    threshold_key = "score"

    def __init__(self, passages=None):
        self.passages = passages or load_corpus()
        self.model = SentenceTransformer(MODEL)
        self.index = faiss.read_index(str(INDEX))

    def encode(self, claims):
        return self.model.encode(claims, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def search(self, claims, k=5):
        scores, ids = self.index.search(self.encode(claims), k)
        return [[{**self.passages[int(i)], "score": float(s)}
                 for s, i in zip(row_s, row_i) if i != -1]
                for row_s, row_i in zip(scores, ids)]

    def cosine(self, claim_vector, pid):
        return float(self.index.reconstruct(int(pid)) @ claim_vector)


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        demo(EmbeddingRetriever())