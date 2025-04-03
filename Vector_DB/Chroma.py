import chromadb
import os
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding

client = chromadb.HttpClient(
    host="localhost",
    port=6378,
    ssl=False
)

DISTANCE_METRIC = "cosine"
INDEX_NAME = "embedding_index"

# -------------------------
# Embedding Model Dimension Mapping
# -------------------------
EMBEDDING_DIM_MAP = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}

# -------------------------
# Clear and Create Collection
# -------------------------

def clear_chroma_store():
    print("Clearing the existing Chroma store")
    try:
        client.delete_collection(INDEX_NAME)
    except chromadb.errors.NotFoundError:
        pass
    print("Chroma store cleared.")

def create_chroma_index(embedding_dim: int):
    collection = client.create_collection(
        name=INDEX_NAME,
        metadata={"hnsw:space": DISTANCE_METRIC}
    )
    print(f"Chroma index created with dimension {embedding_dim}")

# -------------------------
# Store Embeddings
# -------------------------

def store_embedding(file: str, page: str, chunk: str, embedding: list, original_text: str):
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    collection = client.get_collection(INDEX_NAME)

    collection.add(
        embeddings=[embedding],
        metadatas=[{
            "file": file,
            "page": page,
            "chunk": chunk,
            "text": original_text
        }],
        ids=[doc_id]
    )

# -------------------------
# PDF Processing Pipeline
# -------------------------

def process_pdfs(data_dir, chunk_size=100, overlap=50, embedding_model="nomic-embed-text"):
    '''
    Process all PDFs in the data directory and add them to Chroma with the correct index dimension.
    '''
    clear_chroma_store()

    # Get expected dimension
    embedding_dim = EMBEDDING_DIM_MAP.get(embedding_model)
    if embedding_dim is None:
        raise ValueError(f"Unknown embedding model: {embedding_model}")

    create_chroma_index(embedding_dim)

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
                        original_text=chunk
                    )
            print(f" -----> Processed {file_name}")


def chroma_index_pipeline(data_dir: str, chunk_size: int, overlap: int, embedding_model: str):
    clear_chroma_store()
    create_chroma_index(embedding_model)
    process_pdfs(data_dir, chunk_size, overlap, embedding_model)
    print(f"\n✅ Chroma indexing complete with model: {embedding_model}\n")

# Query Chroma
# -------------------------

def query_chroma(query_text: str, embedding_model: str = "nomic-embed-text"):
    embedding = get_embedding(query_text, embedding_model)

    collection = client.get_collection(INDEX_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas", "documents"]
    )

    if results["metadatas"] and results["metadatas"][0] and "text" in results["metadatas"][0][0]:
        return results["metadatas"][0][0]["text"]
    elif results["documents"]:
        return results["documents"][0][0]
    else:
        return ""
