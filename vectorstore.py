import uuid
import chromadb
import numpy as np
from typing import List, Any


class VectorStore:
    def __init__(self, collection_name="pdf_documents", persist_directory="./vectorstore"):
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset_collection(self):
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        print("Collection reset")

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Documents and embeddings length mismatch")

        ids = []
        docs = []
        metadatas = []
        embedding_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            docs.append(doc.page_content)
            metadatas.append(dict(doc.metadata))
            embedding_list.append(embedding.tolist())

        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

        print(f"Added {len(ids)} documents to vector store")