# Read-only application adapter for the existing hybrid retriever.

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Backend.settings import Settings


class EvidenceRetriever:
    def __init__(self, settings: Settings, progress=None):
        completed = progress if progress is not None else lambda step: None
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"

        source = str(Path(settings.root) / "Retrievers")
        if source not in sys.path:
            sys.path.insert(0, source)
        from common import CORPUS_DIR, load_corpus

        # The original retrievers load their saved files from this fixed directory.
        # Refuse an incompatible path instead of silently reading different assets.
        index_dir = Path(settings.index_dir).resolve()
        if index_dir != Path(CORPUS_DIR).resolve():
            raise ValueError(f"The existing retrievers require index_dir={CORPUS_DIR}.")
        required = (index_dir / "passages.jsonl", index_dir / "embeddings.faiss",
                    index_dir / "bm25" / "params.index.json")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(f"Saved retrieval asset is missing: {path}")
        completed("assets")

        from hybrid_retriever import HybridRetriever
        from embedding_retriever import EmbeddingRetriever
        from bm25_retriever import BM25Retriever
        completed("dependencies")
        passages = load_corpus()
        if not passages:
            raise ValueError("The saved retrieval corpus is empty.")
        completed("corpus")
        embedding = EmbeddingRetriever(passages)
        completed("embeddings")
        bm25 = BM25Retriever(passages)
        completed("bm25")
        self.retriever = HybridRetriever(passages=passages, emb=embedding, bm25=bm25)
        if any(passage.get("pid") != index for index, passage in enumerate(passages)):
            raise ValueError("Saved corpus passage IDs do not match index row order.")
        if self.retriever.emb.index.ntotal != len(passages):
            raise ValueError("The saved embedding index does not match the corpus length.")
        params = json.loads((index_dir / "bm25" / "params.index.json").read_text("utf-8"))
        if params.get("num_docs") != len(passages):
            raise ValueError("The saved BM25 index does not match the corpus length.")
        if self.retriever.bm25.bm25.scores.get("num_docs") != len(passages):
            raise ValueError("The loaded BM25 index does not match the corpus length.")
        dimension = self.retriever.emb.model.get_embedding_dimension()
        if dimension != self.retriever.emb.index.d:
            raise ValueError("The embedding model and saved FAISS dimensions differ.")
        self.top_k = settings.top_k
        if self.top_k < 1:
            raise ValueError("top_k must be positive.")
        completed("validation")

    def search(self, claim: str, k: int | None = None) -> list[dict]:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("Enter a nonempty claim to retrieve evidence.")
        limit = self.top_k if k is None else k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("The number of evidence passages must be positive.")
        # Keep the source hybrid's RRF ranking and both score fields unchanged.
        return self.retriever.search([claim], k=limit)[0]
