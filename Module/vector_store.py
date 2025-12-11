"""
Mock vector store for semantic search.
A toy semantic index. Use actual embeddings + vector DB in production.
"""

import math
import random
from typing import Dict, List, Tuple
from .models import Recipe


class MockVectorStore:
    """A toy semantic index. Use actual embeddings + vector DB in prod."""
    def __init__(self):
        # store mapping id -> "embedding" (here a random vector) and metadata
        self.vectors: Dict[str, List[float]] = {}

    def index(self, recipe: Recipe):
        # naive: create a deterministic pseudo-random vector from recipe title
        r = abs(hash(recipe.title)) % (10**8)
        random.seed(r)
        vec = [random.random() for _ in range(64)]
        self.vectors[recipe.id] = vec

    def query(self, text: str, top_k=10) -> List[Tuple[str, float]]:
        # return ids with 'distance' (lower = more similar)
        qh = abs(hash(text)) % (10**8)
        random.seed(qh)
        qvec = [random.random() for _ in range(64)]
        
        def sim(a, b):
            # cosine similarity
            num = sum(x*y for x, y in zip(a, b))
            lena = math.sqrt(sum(x*x for x in a))
            lenb = math.sqrt(sum(x*x for x in b))
            return num/(lena*lenb+1e-9)
        
        scores = [(rid, sim(qvec, vec)) for rid, vec in self.vectors.items()]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
