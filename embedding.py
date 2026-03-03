import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingGenerator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        print(
            f"Embedding model loaded (dim={self.model.get_sentence_embedding_dimension()})"
        )

    def generate(self, texts: List[str]) -> np.ndarray:
        print(f"Generating embeddings for {len(texts)} texts")
        return self.model.encode(texts, show_progress_bar=True)