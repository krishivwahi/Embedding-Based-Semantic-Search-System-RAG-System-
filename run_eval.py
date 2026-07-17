import time
from src.dataset import load_beir
from src.baseline import TfidfRetriever
from src.embedding import EmbeddingRetriever
from src.evaluate import evaluate

K = 10

print("Loading SciFact...")
chunks, queries, qrels = load_beir("data/scifact")
print(f"{len(chunks)} docs, {len(queries)} labeled queries\n")

results = {}
for name, cls in [("TF-IDF", TfidfRetriever), ("Embeddings", EmbeddingRetriever)]:
    print(f"Building {name} index...")
    t0 = time.time()
    retriever = cls(chunks)
    build_s = time.time() - t0
    
    t0 = time.time()
    metrics = evaluate(retriever, queries, qrels, k=K)
    eval_s = time.time() - t0
    
    metrics["build (s)"] = round(build_s, 1)
    metrics["eval (s)"] = round(eval_s, 1)
    results[name] = metrics

# Printing the comparison table
cols = list(next(iter(results.values())).keys())
print(f"\n{'System':<12}" + "".join(f"{c:>12}" for c in cols))
for name, m in results.items():
    print(f"{name:<12}" + "".join(f"{m[c]:>12.4f}" if isinstance(m[c], float) else f"{m[c]:>12}" for c in cols))
