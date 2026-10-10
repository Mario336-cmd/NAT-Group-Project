import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import CORPUS_DIR, load_corpus, demo

INDEX = CORPUS_DIR / "tfidf.joblib"
BATCH = 64  # claims scored at once. keeps the dense score matrix small


def build():
    passages = load_corpus()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.5,
                                 stop_words="english", sublinear_tf=True, dtype=np.float32)
    matrix = vectorizer.fit_transform([p["rendered"] for p in passages])  # rows are L2-normalised
    joblib.dump({"vectorizer": vectorizer, "matrix": matrix}, INDEX)
    print(f"TF-IDF index saved: {INDEX} ({matrix.shape[0]:,} passages, {matrix.shape[1]:,} terms)")


class TfidfRetriever:
    name = "TF-IDF"
    threshold_key = "score"

    def __init__(self, passages=None):
        self.passages = passages or load_corpus()
        saved = joblib.load(INDEX)
        self.vectorizer, self.matrix = saved["vectorizer"], saved["matrix"].T.tocsr()
        if self.matrix.shape[1] != len(self.passages):
            raise ValueError("TF-IDF index is stale. rerun with --build")

    def search(self, claims, k=5):
        out = []
        for start in range(0, len(claims), BATCH):
            # dot product of L2-normalised vectors = cosine similarity, 0 to 1
            scores = (self.vectorizer.transform(claims[start:start + BATCH]) @ self.matrix).toarray()
            for row in scores:
                top = np.argsort(-row)[:k]
                # score 0 means no shared words, so it is never evidence
                out.append([{**self.passages[int(i)], "score": float(row[i])} for i in top if row[i] > 0])
        return out


if __name__ == "__main__":
    if "--build" in sys.argv:
        build()
    else:
        demo(TfidfRetriever())
