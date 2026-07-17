def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / k

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant)

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def evaluate(retriever, queries: dict, qrels: dict, k: int = 10) -> dict:
    p, r, m = [], [], []
    for qid, query in queries.items():
        results = retriever.search(query, k=k * 2)
        doc_ids = []
        for chunk, _score in results:
            if chunk.doc_id not in doc_ids:
                doc_ids.append(chunk.doc_id)
        relevant = qrels[qid]
        p.append(precision_at_k(doc_ids, relevant, k))
        r.append(recall_at_k(doc_ids, relevant, k))
        m.append(mrr(doc_ids, relevant))
    n = len(queries)
    return {
        f"P@{k}": sum(p) / n,
        f"Recall@{k}": sum(r) / n,
        "MRR": sum(m) / n,
    }
