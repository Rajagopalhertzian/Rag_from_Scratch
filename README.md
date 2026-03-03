# 🚀 RAG From Scratch

An end-to-end **Retrieval-Augmented Generation (RAG)** system built from scratch using:

- **LangChain**
- **SentenceTransformers**
- **ChromaDB (Persistent Vector Store)**
- **LLaMA 3 (via Groq API)**

This project demonstrates how to build a fully modular RAG pipeline including:

- Document loading
- Text chunking
- Embedding generation
- Vector database storage
- Semantic retrieval
- LLM-based answer generation

---

# 📂 Project Structure

```
rag-from-scratch/
│
├── rag_app.py            # Main application entry point
├── requirements.txt      # Project dependencies
├── README.md
│
├── pdfs/                 # Place your PDF documents here
│
├── dataloader.py         # Loads PDF documents
├── text_splitting.py     # Splits documents into chunks
├── embedding.py          # Embedding model wrapper
├── vectorstore.py        # ChromaDB vector store handler
└── retriever.py          # Semantic retriever logic
```

---

# 🧠 System Architecture

```
PDF Documents
      ↓
Document Loader
      ↓
Text Splitter
      ↓
Embedding Model (MiniLM)
      ↓
Chroma Vector Store
      ↓
Semantic Retriever
      ↓
LLaMA 3 (Groq)
      ↓
Final Answer
```

---

# ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/rag-from-scratch.git
cd rag-from-scratch
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Setup API Key

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

Or export directly:

```bash
export GROQ_API_KEY=your_api_key_here
```

---

# 📄 Add Your Documents

Place your PDF files inside:

```
pdfs/
```

Example:

```
pdfs/
 ├── word2vec.pdf
 ├── blip.pdf
```

---

# ▶️ Running the Application

```bash
python rag_app.py
```

You will see:

```
RAG system ready!
Ask a question:
```

Now you can type:

```
What is BLIP?
Explain word2vec.
What is vector space?
```

---

# 🔄 Indexing Behavior

On first run, the system:

1. Loads PDFs
2. Splits documents into chunks
3. Generates embeddings
4. Stores them in ChromaDB

Because ChromaDB uses **persistent storage**, embeddings remain saved unless you explicitly reset the collection.

---

# 🧩 Core Components

## 📄 dataloader.py
- Loads PDFs using `PyPDFLoader`
- Attaches metadata

## ✂️ text_splitting.py
- Uses `RecursiveCharacterTextSplitter`
- Chunk size: 1000
- Overlap: 200

## 🧠 embedding.py
- Uses `all-MiniLM-L6-v2`
- 384-dimensional embeddings

## 🗂 vectorstore.py
- ChromaDB persistent client
- Handles document insertion

## 🔎 retriever.py
- Performs semantic similarity search
- Returns ranked results

## 🚀 rag_app.py
- Combines retriever + LLM
- Generates final answer

---

# 📈 Example Output

```
Question:
What is BLIP?

Answer:
BLIP stands for Bootstrapping Language-Image Pre-training, a method for unified vision-language understanding and generation.
```

---

# 🚀 Features

- Modular architecture
- Persistent vector database
- Semantic search
- Context-aware LLM answers
- Clean CLI interface
- Production-style structure

---

# 📌 Future Improvements

- Streamlit Web UI
- Source citation display
- Reranking layer
- Evaluation pipeline
- FastAPI REST API
- Docker deployment

---

# 🛠 Tech Stack

- Python
- LangChain
- SentenceTransformers
- ChromaDB
- Groq API (LLaMA 3)

---

# 🎯 Why This Project?

This project demonstrates strong understanding of:

- Embeddings & semantic similarity
- Vector databases
- Retrieval-Augmented Generation
- Prompt engineering
- Modular system design
- End-to-end LLM application architecture

Built as a portfolio-ready RAG system.

---
