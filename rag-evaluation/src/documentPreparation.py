"""We will preparer rag data by chunking them into smaller pieces wrt 
model max-length.
"""
import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document as LangchainDocument
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

def read_pdf(list_file_path: List[str]) -> List[LangchainDocument]:
    """Read pdf file and return list of LangchainDocument object.
    list_file_path: List of file path.
    """
    raw_documents = []
    for file_path in list_file_path:
        doc = PyPDFLoader(file_path).load()
        raw_documents.extend(doc)
    
    return raw_documents

def text_chunking(documents: List[LangchainDocument], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Chunking the documents into smaller pieces wrt model max-length.
    documents: List of LangchainDocument object.
    max_length: Maximum length of the model.
    """
    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        add_start_index=True,
                        separators=["\n\n", "\n", ".", " ", ""],
                    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def initialize_embeddings_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    embeddings_model = HuggingFaceEmbeddings(model_name = model_name, encode_kwargs={
            "normalize_embeddings": True
        })
    return embeddings_model

def create_vectorial_db(
        chunks: List[LangchainDocument], 
        embeddings_model: HuggingFaceEmbeddings,
        save_local_path: str = "data/indexes/"
        ):
    db = FAISS.from_documents(chunks, embeddings_model, distance_strategy=DistanceStrategy.COSINE,)
    db.save_local(save_local_path)
    return db

def prepare_rag_data(
        list_file_path: List[str], 
        chunk_size: int, 
        chunk_overlap: int, 
        embedding_model,
        model_name: str = "sentence-transformers/all-mpnet-base-v2"
        ):
    """Prepare RAG data by chunking them into smaller pieces wrt model max-length.
    list_file_path: List of file path.
    chunk_size: Chunk size.
    chunk_overlap: Chunk overlap.
    model_name: Model name.
    """
    
    print("Reading pdf files...")
    documents = read_pdf(list_file_path)
    print("Chunking the documents...")
    chunks = text_chunking(documents, chunk_size, chunk_overlap)
    print("Initializing embeddings model...")
    print("Creating vectorial db...")
    index_name = (
        f"index_chunk-{chunk_size}_embeddings-{model_name.replace('/', '~')}"
    )
    index_folder_path = f"data/indexes/{index_name}//"

    if os.path.exists(index_folder_path):
        db = FAISS.load_local(index_folder_path, 
                              embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                                allow_dangerous_deserialization=True)
    else:
        print("Index not found, generating it...")
        db = create_vectorial_db(chunks, embedding_model, index_folder_path)
    return db