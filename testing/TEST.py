from src.ingest import load_corpus
from src.baseline import TfidfRetriever
from src.embedding import EmbeddingRetriever

docs = {
    "doc1": "The cat sat on the mat. Cats are great pets and love to nap.",
    "doc2": "Python is a programming language used for machine learning.",
    "doc3": "FAISS is a library for efficient similarity search over vectors.",
}

chunks = load_corpus(docs)

print("=== TF-IDF ===")
for chunk, score in TfidfRetriever(chunks).search("vector search library", k=2):
    print(f"{score:.3f}  {chunk.doc_id}: {chunk.text[:50]}")

print("\n=== Embeddings ===")
for chunk, score in EmbeddingRetriever(chunks).search("vector search library", k=2):
    print(f"{score:.3f}  {chunk.doc_id}: {chunk.text[:50]}")
