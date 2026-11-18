import math

import pytest

from src.baseline import BM25Retriever
from src.evaluate import mrr, ndcg_at_k, precision_at_k, recall_at_k
from src.fusion import HybridRetriever
from src.ingest import Chunk

RETRIEVED = ["a", "x", "b", "y"]
RELEVANT = {"a", "b", "c"}


def test_precision_recall():
    assert precision_at_k(RETRIEVED, RELEVANT, 4) == 2 / 4
    assert recall_at_k(RETRIEVED, RELEVANT, 4) == 2 / 3
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == 1 / 3


def test_mrr():
    assert mrr(["x", "y", "b"], RELEVANT) == 1 / 3
    assert mrr(["x"], RELEVANT) == 0.0


def test_ndcg():
    # hits at ranks 1 and 3; ideal puts 3 hits at ranks 1..3
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 4) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b", "c"], RELEVANT, 3) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y"], RELEVANT, 2) == 0.0


CHUNKS = [
    Chunk("d1", 0, "the cat sat on the mat"),
    Chunk("d2", 0, "vitamin d deficiency raises fracture risk"),
    Chunk("d3", 0, "faiss performs vector similarity search"),
]


def test_bm25_ranks_matching_doc_first():
    hits = BM25Retriever(CHUNKS).search("vitamin deficiency", k=3)
    assert hits[0][0].doc_id == "d2"
    assert hits[0][1] > 0 and hits[-1][1] == 0


class Fixed:
    def __init__(self, order):
        self.order = order

    def search(self, query, k=5):
        return [(CHUNKS[i], 0.0) for i in self.order][:k]


def test_rrf_scores():
    # rrf_k=0: d1 and d3 get 1/1 + 1/3, d2 (2nd in both) gets 1/2 + 1/2
    hybrid = HybridRetriever([Fixed([0, 1, 2]), Fixed([2, 1, 0])], rrf_k=0)
    hits = hybrid.search("q", k=3)
    assert [round(s, 4) for _c, s in hits] == [round(1 + 1 / 3, 4)] * 2 + [1.0]
    assert hits[-1][0].doc_id == "d2"
