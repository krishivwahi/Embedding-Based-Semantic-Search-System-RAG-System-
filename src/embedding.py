import os

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .ingest import Chunk


class EmbeddingRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2",
                 index_path: str | None = None):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # shortcut: only checks the count; delete the file after changing the model or chunking
            if self.index.ntotal == len(chunks):
                return
        embeddings = self.model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        if index_path:
            faiss.write_index(self.index, index_path)

    def search(self, query: str, k: int = 5):
        q = self.model.encode([query], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idx = self.index.search(q, k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], idx[0])]
