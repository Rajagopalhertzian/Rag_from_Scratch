from typing import List, Dict, Any


class RAGRetriever:
    def __init__(self, vector_store, embedding_model):
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k=3) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.generate([query])[0]

        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]

        retrieved_docs = []

        for i, (doc_id, document, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances)
        ):
            retrieved_docs.append(
                {
                    "id": doc_id,
                    "content": document,
                    "metadata": metadata,
                    "distance": distance,
                    "rank": i + 1,
                }
            )

        return retrieved_docs