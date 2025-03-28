import redis
import numpy as np
from redis.commands.search.query import Query
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding
import os

# ----------------------
# Initialize Redis connection
# ----------------------

redis_client = redis.Redis(host="localhost", port=6379, db=0)

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# clear anything stored in the current redis store
def clear_redis_store():
    print("Clearing the existing Redis store")
    redis_client.flushdb()
    print("Redis store cleared.")

# Create an HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# ----------------------
# Embedding Generation
# ----------------------

# store the embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )

# Process all PDF files in a given directory
def process_pdfs(data_dir, chunk_size=100, overlap=50, embedding_model="nomic-embed-text"):
    '''
    Go through all pdf's in the data directory and process them

    data_dir: the directory containing the pdf's (str)
    chunk_size: the # of tokens per chunk (int)
    overlap: the # of tokens to overlap between chunks (int)
    embedding_model: the model to use for embedding (str)
    
    '''

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
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

def redis_index_pipeline(data_dir: str, chunk_size: int, overlap: int, embedding_model: str):
    '''
    This function will clear the redis store, create a new HNSW index, and process all documents
    in the data directory. 

    data_dir: the directory containing the pdf's (str)
    chunk_size: the # of tokens per chunk (int)
    overlap: the # of tokens to overlap between chunks (int)
    embedding_model: the model to use for embedding (str)
    
    '''
    clear_redis_store()
    create_hnsw_index()

    process_pdfs(data_dir, chunk_size, overlap, embedding_model)
    print("\n---Done processing PDFs---\n")


# ----------------------
# Query the Redis store
# ----------------------


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    
    return res.docs[0].id