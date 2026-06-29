from dataclasses import dataclass
@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(text: str, size: int = 500, overlap: int = 50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks


def load_corpus(docs: dict[str, str]) -> list[Chunk]:
    out = []
    for doc_id, raw in docs.items():
        for ci, c in enumerate(chunk_text(raw)):
            out.append(Chunk(doc_id, ci, c))
    return out
