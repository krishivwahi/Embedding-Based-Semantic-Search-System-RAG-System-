class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (rrf_k + rank)."""

    def __init__(self, retrievers, depth: int = 100, rrf_k: int = 60):
        self.retrievers, self.depth, self.rrf_k = retrievers, depth, rrf_k

    def search(self, query: str, k: int = 5):
        scores, chunks = {}, {}
        for r in self.retrievers:
            for rank, (chunk, _s) in enumerate(r.search(query, k=self.depth), start=1):
                key = (chunk.doc_id, chunk.chunk_id)
                chunks[key] = chunk
                scores[key] = scores.get(key, 0.0) + 1 / (self.rrf_k + rank)
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [(chunks[key], scores[key]) for key in top]


class RerankRetriever:
    """Re-scores the base retriever's top `depth` chunks with a cross-encoder."""

    def __init__(self, base, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", depth: int = 50):
        from sentence_transformers import CrossEncoder  # lazy: keeps torch out of light imports

        self.base, self.depth = base, depth
        self.model = CrossEncoder(model_name)

    def search(self, query: str, k: int = 5):
        candidates = self.base.search(query, k=max(k, self.depth))
        scores = self.model.predict([(query, c.text) for c, _s in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:k]
        return [(c, float(s)) for (c, _old), s in ranked]
