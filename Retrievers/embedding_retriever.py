import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
from sentence_transformers import SentenceTransformer
from common import CORPUS_DIR, load_corpus, demo

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
INDEX = CORPUS_DIR / "embeddings.faiss"


def build():
    passages = load_corpus()
    model = SentenceTransformer(MODEL, revision=MODEL_REVISION)
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
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION)
        self.index = faiss.read_index(str(INDEX))

    def encode(self, claims):
        return self.model.encode(claims, normalize_embeddings=True,
                                 convert_to_numpy=True).astype("float32")

    def search(self, claims, k=5):
        return self.search_vectors(self.encode(claims), k)

    def search_vectors(self, vectors, k=5):
        scores, ids = self.index.search(vectors, k)
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
