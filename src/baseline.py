from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .ingest import Chunk #chunky boy
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
