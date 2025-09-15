import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load .env
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# CONFIG
PDF_PATH = "Chunking_RAG.pdf" 
PERSIST_DIR = "persist_chroma"
COLLECTION = "pdf_docs"

def main():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} pages, split into {len(chunks)} chunks")

    # Embed + store in Chroma
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)
    print("✅ PDF ingested and stored in Chroma")

if __name__ == "__main__":
    main()
