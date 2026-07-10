import json
from pathlib import Path
from .ingest import Chunk

def load_beir(data_dir: str):
    data_dir = Path(data_dir)
    # Corpus: each abstract becomes one Chunk (they're short — no splitting needed)
    chunks = []
    with open(data_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + ". " + d["text"]).strip(". ")
            chunks.append(Chunk(doc_id=d["_id"], chunk_id=0, text=text))
    
    # Queries
    all_queries = {}
    with open(data_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            all_queries[d["_id"]] = d["text"]
            
    # Qrels: ground-truth relevant doc ids per query (test split)
    qrels = {}
    with open(data_dir / "qrels" / "test.tsv", encoding="utf-8") as f:
        next(f)  # skip header row
        for line in f:
            qid, did, score = line.strip().split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, set()).add(did)
                
    # Keep only queries that have test labels (300 of them)
    queries = {qid: all_queries[qid] for qid in qrels}
    return chunks, queries, qrels
