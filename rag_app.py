import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from dataloader import load_pdfs
from text_splitting import split_documents
from embedding import EmbeddingGenerator
from vectorstore import VectorStore
from retriever import RAGRetriever


def rag(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k)

    if not results:
        return "No relevant context found."

    context = "\n\n".join([doc["content"] for doc in results])

    prompt = f"""
Use the following context to answer the question concisely.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    load_dotenv()

    print("Loading documents...")
    documents = load_pdfs("./pdfs")

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Generating embeddings...")
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate([doc.page_content for doc in chunks])

    print("Building vector store...")
    vectorstore = VectorStore()
    vectorstore.reset_collection()
    vectorstore.add_documents(chunks, embeddings)

    retriever = RAGRetriever(vectorstore, embedder)

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1024,
    )

    print("\nRAG system ready!")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower().strip() == "exit":
            print("Exiting RAG system...")
            break

        print("\nSearching and generating answer...\n")
        answer = rag(query, retriever, llm)

        print("Answer:\n")
        print(answer)
        print("\n" + "-" * 60 + "\n")