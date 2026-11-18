from src.dataset import load_beir
from src.embedding import EmbeddingRetriever
from src.pgstore import load

chunks, _queries, _qrels = load_beir("data/scifact")
dense = EmbeddingRetriever(chunks, index_path="data/scifact.faiss")
embeddings = dense.index.reconstruct_n(0, dense.index.ntotal)  # reuse cached vectors, no re-embedding
load(chunks, embeddings, {"train": "data/scifact_claims/claims_train.jsonl",
                          "dev": "data/scifact_claims/claims_dev.jsonl"})
print(f"Loaded {len(chunks)} chunks + SciFact claims into Postgres")
