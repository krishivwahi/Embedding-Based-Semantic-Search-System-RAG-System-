"""Postgres + pgvector store: unstructured text and vectors next to SciFact's structured claim labels."""
import json
import os

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .ingest import Chunk

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")

SCHEMA = """
DROP TABLE IF EXISTS claim_evidence, claims, chunks;
CREATE TABLE chunks (
    doc_id    TEXT,
    chunk_id  INT,
    text      TEXT NOT NULL,
    embedding vector({dim}) NOT NULL,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (doc_id, chunk_id)
);
CREATE TABLE claims (claim_id INT PRIMARY KEY, split TEXT NOT NULL, claim TEXT NOT NULL);
CREATE TABLE claim_evidence (
    claim_id INT REFERENCES claims,
    doc_id   TEXT,
    label    TEXT NOT NULL CHECK (label IN ('SUPPORT', 'CONTRADICT')),
    PRIMARY KEY (claim_id, doc_id)
);
"""
INDEXES = """
CREATE INDEX ON chunks USING hnsw (embedding vector_ip_ops);
CREATE INDEX ON chunks USING gin (tsv);
"""


def connect():
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def load(chunks: list[Chunk], embeddings: np.ndarray, claim_files: dict[str, str]):
    """claim_files maps split name -> SciFact claims_*.jsonl path."""
    with connect() as conn, conn.transaction(), conn.cursor() as cur:
        cur.execute(SCHEMA.format(dim=embeddings.shape[1]))
        doc_ids = {c.doc_id for c in chunks}
        with cur.copy("COPY chunks (doc_id, chunk_id, text, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
            copy.set_types(["text", "int4", "text", "vector"])
            for c, e in zip(chunks, embeddings):
                copy.write_row((c.doc_id, c.chunk_id, c.text, e))
        for split, path in claim_files.items():
            with open(path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            cur.executemany("INSERT INTO claims VALUES (%s, %s, %s)", [(r["id"], split, r["claim"]) for r in rows])
            evidence = [(r["id"], doc_id, sents[0]["label"])
                        for r in rows for doc_id, sents in r["evidence"].items() if doc_id in doc_ids]
            cur.executemany("INSERT INTO claim_evidence VALUES (%s, %s, %s)", evidence)
        cur.execute(INDEXES)


class PgVectorRetriever:
    """Same interface as EmbeddingRetriever, but the ANN search runs inside Postgres (HNSW)."""

    def __init__(self, model):
        self.model = model
        self.conn = connect()
        self.conn.execute("SET hnsw.ef_search = 100")

    def search(self, query: str, k: int = 5):
        q = np.asarray(self.model.encode(query, normalize_embeddings=True), dtype="float32")
        rows = self.conn.execute(
            "SELECT doc_id, chunk_id, text, -(embedding <#> %s) FROM chunks ORDER BY embedding <#> %s LIMIT %s",
            (q, q, k),
        ).fetchall()
        return [(Chunk(d, ci, t), float(s)) for d, ci, t, s in rows]
