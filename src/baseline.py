import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .ingest import Chunk
class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words = "english")
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])
        
    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).flatten()
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


class BM25Retriever:
    """Okapi BM25 (k1=0.9, b=0.4, Anserini/Pyserini defaults) as one sparse matrix product."""

    def __init__(self, chunks: list[Chunk], k1: float = 0.9, b: float = 0.4):
        self.chunks = chunks
        self.vectorizer = CountVectorizer(stop_words="english")
        tf = self.vectorizer.fit_transform([c.text for c in chunks]).tocsr().astype("float64")
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
        dl = np.asarray(tf.sum(axis=1)).ravel()
        row_norm = k1 * (1 - b + b * dl / dl.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + row_norm[rows])
        self.weights = tf

    def search(self, query: str, k: int = 5):
        q = self.vectorizer.transform([query])
        scores = (self.weights @ q.T).toarray().ravel()
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]
