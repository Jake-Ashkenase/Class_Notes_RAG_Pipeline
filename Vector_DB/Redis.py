import redis
import numpy as np
from redis.commands.search.query import Query
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding
import os

# initialize redis connection
redis_client = redis.Redis(host="localhost", port=6379, db=0)

EMBEDDING_DIM_MAP = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
}

def create_hnsw_index(embedding_model: str):
    dim = EMBEDDING_DIM_MAP[embedding_model]
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {dim} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print(f"Redis index created with dimension {dim}")
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
# but cosine works here???? what
DISTANCE_METRIC = "COSINE"


# clear anything stored in the current redis store
def clear_redis_store():
    print("Clearing the existing Redis store")
    redis_client.flushdb()
    print("Redis store cleared.")

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
            ).tobytes(),  
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
    clear_redis_store()
    create_hnsw_index(embedding_model)
    process_pdfs(data_dir, chunk_size, overlap, embedding_model)


def query_redis(query_text: str, embedding_model: str = None):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )

    embedding = get_embedding(query_text, embedding_model) 
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    if not res.docs:
        return None
    return res.docs[0].id


def get_all_documents():
    """Retrieve all documents from Redis store.
    
    specifically for the BM25 Search
    """
    results = []
    for key in redis_client.keys(f"{DOC_PREFIX}:*"):
        doc = redis_client.hgetall(key)
        if doc:
            # Skip the embedding field as it's not needed for BM25
            results.append({
                "text": doc[b'chunk'].decode('utf-8'), 
                "metadata": {
                    "file": doc[b'file'].decode('utf-8'),
                    "page": doc[b'page'].decode('utf-8')
                }
            })
    return results