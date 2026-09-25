"""Optional offline FAISS index builder.

The Streamlit app now builds a FAISS index directly from the PDF uploaded by the user.
This file is kept for the original/local workflow when you want to build an index from
PDF files stored in data/.
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()


def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def build_index():
    documents = load_pdf_files(DATA_PATH)
    chunks = create_chunks(documents)
    if not chunks:
        raise ValueError("No readable text was found in the PDF files in data/.")

    db = FAISS.from_documents(chunks, get_embedding_model())
    db.save_local(DB_FAISS_PATH)
    print(f"Indexed {len(documents)} pages into {len(chunks)} chunks.")


if __name__ == "__main__":
    build_index()
