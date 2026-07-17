from src.dataset import load_beir
from src.embedding import EmbeddingRetriever
from src.rag import answer

print("Loading corpus + building index (few minutes on first run)...")
chunks, _queries, _qrels = load_beir("data/scifact")
retriever = EmbeddingRetriever(chunks)

print("\nReady. Ask a question (or 'quit'):")
while True:
    q = input("\n> ").strip()
    if q.lower() in {"quit", "exit", ""}:
        break
        
    text, retrieved = answer(q, retriever, k=5)
    print("\n--- Answer ---")
    print(text)
    
    print("\n--- Sources ---")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"[{i}] ({score:.3f}) {chunk.doc_id}: {chunk.text[:80]}...")
