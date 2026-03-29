from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

def ingest_documents():
    docs_path = "docs"
    all_documents = []

    print("Loading documents...")
    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)

        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"  Loaded PDF: {filename} ({len(documents)} pages)")

        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"  Loaded Word: {filename}")

        else:
            print(f"  Skipped: {filename} (unsupported format)")

    if not all_documents:
        print("No documents found in docs/ folder!")
        return

    print(f"\nTotal pages/sections loaded: {len(all_documents)}")

    print("Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")

    print("\nCreating embeddings and saving to FAISS...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")

    print("Vectorstore saved to vectorstore/")
    print("\nDone! Your documents are ready to be searched.")

if __name__ == "__main__":
    ingest_documents()