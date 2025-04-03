import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding

# -------------------------
# Embedding Dimension Map
# -------------------------
EMBEDDING_DIM_MAP = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}

COLLECTION_NAME = "ds4300_embeddings"
client = QdrantClient(host="localhost", port=6333)

def clear_qdrant_store():
    print("Clearing Qdrant collection...")
    if COLLECTION_NAME in client.get_collections().collections:
        client.delete_collection(COLLECTION_NAME)

# qdrnt collection 
def create_qdrant_index(embedding_model: str):
    dim = EMBEDDING_DIM_MAP.get(embedding_model)
    if dim is None:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")

    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print(f"Qdrant collection created with dimension {dim}")

def store_embedding(file: str, page: str, chunk: str, embedding: list, original_text: str, point_id: int):
    metadata = {
        "file": file,
        "page": page,
        "chunk": chunk,
        "text": original_text
    }

    point = PointStruct(
        id=point_id,
        vector=embedding,
        payload=metadata
    )

    client.upsert(collection_name=COLLECTION_NAME, points=[point])

#preprocess call
def process_pdfs(data_dir, chunk_size=100, overlap=50, embedding_model="nomic-embed-text"):
    point_id = 0
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
                        original_text=chunk,
                        point_id=point_id
                    )
                    point_id += 1
            print(f" -----> Processed {file_name}")

def qdrant_index_pipeline(data_dir: str, chunk_size: int, overlap: int, embedding_model: str):
    clear_qdrant_store()
    create_qdrant_index(embedding_model)
    process_pdfs(data_dir, chunk_size, overlap, embedding_model)
    print(f"\n✅ Qdrant indexing complete with model: {embedding_model}\n")

def query_qdrant(query_text: str, embedding_model: str = "nomic-embed-text"):
    embedding = get_embedding(query_text, embedding_model)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=1
    )
    return results[0].payload["text"] if results else None
