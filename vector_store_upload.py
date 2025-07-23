import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
import argparse

# === Configuration ===
PDF_FOLDER = "./hr_docs"
COLLECTION_NAME = "hr_policy"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# === Load and Split PDFs ===
def load_documents():
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            filepath = os.path.join(PDF_FOLDER, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load_and_split()
            for page in pages:
                page.metadata["source"] = filename  # Track the source
            documents.extend(pages)
    return documents

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# === Build Qdrant Collection ===
def build_qdrant_collection(key,port):
    os.environ["OPENAI_API_KEY"] = key
    # === Initialize Embedding Model ===
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"📄 Loaded {len(docs)} documents → 🔹 {len(chunks)} chunks")

    Qdrant.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url=f"http://localhost:{port}",
    collection_name=COLLECTION_NAME)

    print(f"✅ Uploaded embeddings to Qdrant collection: {COLLECTION_NAME}")


def main():
    parser = argparse.ArgumentParser(description="Run HR chatbot with OpenAI key and config.")
    parser.add_argument('--openai_key', type=str, required=True, help="Your OpenAI API key")
    parser.add_argument('--port', type=int, default=8501, help="Port to run Streamlit app on")
    
    args = parser.parse_args()

    print(f"Running with OpenAI Key: {args.openai_key[:5]}... (hidden)")
    print(f"Running on port: {args.port}")
    build_qdrant_collection(args.openai_key,args.port)

if __name__ == "__main__":
    main()