import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL"),
)

_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

def build_prompt(query: str, retrieved) -> str:
    context = "\n\n".join(
        f"[{i + 1}] {chunk.text}" for i, (chunk, _score) in enumerate(retrieved)
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite sources by their [number]. "
        "If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

def answer(query: str, retriever, k: int = 5):
    retrieved = retriever.search(query, k=k)
    prompt = build_prompt(query, retrieved)
    response = _client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content, retrieved
