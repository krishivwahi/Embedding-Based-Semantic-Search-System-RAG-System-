from .baseline import BM25Retriever
from .dataset import load_beir
from .embedding import EmbeddingRetriever
from .fusion import HybridRetriever


def build_retriever(data_dir: str = "data/scifact"):
    """Best configuration from the ablation: BM25 + dense fused with RRF.
    The cross-encoder reranker is left out: it added no nDCG on SciFact and costs far more latency."""
    chunks, _queries, _qrels = load_beir(data_dir)
    dense = EmbeddingRetriever(chunks, index_path=f"{data_dir}.faiss")
    return HybridRetriever([BM25Retriever(chunks), dense])
