import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore():
    if os.path.exists("vectorstore"):
        return
    
    docs_path = "docs"
    all_documents = []

    for filename in os.listdir(docs_path):
        filepath = os.path.join(docs_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            all_documents.extend(loader.load())
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            all_documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore")
    print("Vectorstore built successfully")

if __name__ == "__main__":
    build_vectorstore()