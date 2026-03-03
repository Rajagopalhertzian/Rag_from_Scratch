from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(pdf_directory: str):
    """Load all PDFs from a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    for pdf in pdf_dir.glob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages from {pdf.name}")

        except Exception as e:
            print(f"Error loading {pdf.name}: {e}")

    return all_documents