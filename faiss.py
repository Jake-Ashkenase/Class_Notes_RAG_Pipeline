import os
import numpy as np
import faiss
import pickle
from redis.commands.search.query import Query
from sentence_transformers import SentenceTransformer
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding

VECTOR_DIM = 768
DOC_STORE_PATH = "faiss_doc_store.pkl"
FAISS_INDEX_PATH = "faiss_index.index"

# In-memory doc store for metadata
doc_store = {}

# ----------------------
# FAISS Setup
# ----------------------

def clear_faiss_store():
    """Clear FAISS index and metadata store."""
    print("Clearing FAISS store...")
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(DOC_STORE_PATH):
        os.remove(DOC_STORE_PATH)
    global doc_store
    doc_store = {}
    print("FAISS store cleared.")

def create_faiss_index():
    """Create an empty FAISS index."""
    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(VECTOR_DIM)  # Use cosine similarity (normalize vectors first!)
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("FAISS index created.")

# ----------------------
# Embedding Storage
# ----------------------

def store_embedding(file: str, page: str, chunk: str, embedding: list):
    """Store vector and metadata in FAISS and local doc store."""
    global doc_store

    # Normalize for cosine similarity
    vec = np.array(embedding, dtype=np.float32)
    vec /= np.linalg.norm(vec)

    index = faiss.read_index(FAISS_INDEX_PATH)
    index.add(np.array([vec]))
    faiss.write_index(index, FAISS_INDEX_PATH)

    vector_id = len(doc_store)
    doc_store[vector_id] = {
        "file": file,
        "page": page,
        "chunk": chunk
    }

    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(doc_store, f)

# ----------------------
# PDF Processing
# ----------------------

def process_pdfs(data_dir, chunk_size=100, overlap=50, embedding_model="nomic-embed-text"):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def faiss_index_pipeline(data_dir: str, chunk_size: int, overlap: int, embedding_model: str):
    clear_faiss_store()
    create_faiss_index()
    process_pdfs(data_dir, chunk_size, overlap, embedding_model)
    print("\n---Done processing PDFs into FAISS---\n")

# ----------------------
# Querying FAISS
# ----------------------

def query_faiss(query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    embedding = np.array(embedding, dtype=np.float32)
    embedding /= np.linalg.norm(embedding)

    index = faiss.read_index(FAISS_INDEX_PATH)
    D, I = index.search(np.array([embedding]), top_k)

    with open(DOC_STORE_PATH, "rb") as f:
        doc_store = pickle.load(f)

    results = []
    for idx in I[0]:
        if idx in doc_store:
            results.append(doc_store[idx])
    
    return results[0] if results else None
