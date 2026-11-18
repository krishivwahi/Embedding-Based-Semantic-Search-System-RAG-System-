from src.pipeline import build_retriever
from src.rag import answer
from src.telemetry import setup_tracing

setup_tracing()
print("Loading corpus + index (first run builds it, takes a few minutes)...")
retriever = build_retriever()

print("\nReady. Ask a question (or 'quit'):")
while True:
    q = input("\n> ").strip()
    if q.lower() in {"quit", "exit", ""}:
        break

    text, retrieved, stats = answer(q, retriever, k=5)
    print("\n--- Answer ---")
    print(text)

    print("\n--- Sources ---")
    for i, (chunk, score) in enumerate(retrieved, 1):
        print(f"[{i}] ({score:.3f}) {chunk.doc_id}: {chunk.text[:80]}...")
    print(f"\n{stats['tokens_in']}+{stats['tokens_out']} tokens | TTFT {stats['ttft_s']}s | "
          f"{stats['latency_s']}s total | {stats['tokens_per_s']} tok/s | ${stats['cost_usd']:.6f}")
