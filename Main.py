import time
import psutil
import csv
from LLM_Call import local_LLM_call
from Vector_DB.Redis import redis_index_pipeline, query_redis
from Vector_DB.Chroma import chroma_index_pipeline, query_chroma
from Vector_DB.qdrant import qdrant_index_pipeline, query_qdrant

CSV_PATH = "main_query_log.csv"
csv_fields = [
    "vector_db", "embedding_model", "chunk_size", "overlap",
    "llm_model", "index_time_sec", "memory_mb",
    "retrieval_time_sec", "llm_response_time_sec",
    "question", "response"
]

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def write_to_csv(row):
    file_exists = False
    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            file_exists = True
    except FileNotFoundError:
        pass

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# Chroma + Snowflake-Arctic-Embed + 100 chunk size + 50 overlap + Llama3.2

# Qdrant + Nomic-Embed-Text + 500 chunk size + 100 overlap + Llama3.2



# Chroma + MXBAI-Embed-Large + 500 chunk size + 100 overlap + Mistral

# Redis + Snowflake-Arctic-Embed + 100 chunk size + 50 overlap + Mistral

def main():
    # set config
    vector_db = "redis"
    embedding_model = "snowflake-arctic-embed"
    llm_model = "mistral:latest"
    chunk_size = 100
    overlap = 50

    mem_before = get_memory_usage()
    start_index_time = time.time()

    # index type
    redis_index_pipeline("data", chunk_size, overlap, embedding_model)
    # chroma_index_pipeline("data", chunk_size, overlap, embedding_model)
    # qdrant_index_pipeline("data", chunk_size, overlap, embedding_model)

    query = input("Enter your query: ")

    index_time = time.time() - start_index_time
    mem_after = get_memory_usage()

    retrieval_start = time.time()
    redis_top_embedding = query_redis(query, embedding_model)
    # chroma_top_embedding = query_chroma(query, embedding_model)
    # qdrant_top_embedding = query_qdrant(query, embedding_model)

    retrieval_time = time.time() - retrieval_start

    llm_start = time.time()
    response = local_LLM_call(query, llm_model, redis_top_embedding)
    # response = local_LLM_call(query, llm_model, chroma_top_embedding)
    # response = local_LLM_call(query, llm_model, qdrant_top_embedding)
    llm_time = time.time() - llm_start
    mem_after = get_memory_usage()
    print(response)

    write_to_csv({
        "vector_db": vector_db,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "llm_model": llm_model,
        "index_time_sec": round(index_time, 2),
        "memory_mb": round(mem_after - mem_before, 2),
        "retrieval_time_sec": round(retrieval_time, 2),
        "llm_response_time_sec": round(llm_time, 2),
        "question": query,
        "response": response
    })

if __name__ == "__main__":
    main()
