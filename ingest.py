import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_pdf(pdf_path="Chunking_RAG.pdf", persist_dir="./chroma_db"):
    """Load a single PDF, split into chunks, and store embeddings in ChromaDB."""

    if not os.path.exists(pdf_path):
        print(f"⚠️ PDF file {pdf_path} not found.")
        return

    load_dotenv()
    embeddings = OpenAIEmbeddings()

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Initialize persistent Chroma
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    # Add documents
    vectordb.add_documents(chunks)
    print(f"✅ Ingested {pdf_path} into ChromaDB with {len(chunks)} chunks.")


if __name__ == "__main__":
    ingest_pdf()
