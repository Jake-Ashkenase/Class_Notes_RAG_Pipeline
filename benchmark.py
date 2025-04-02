import time
import psutil
import csv
from LLM_Call import local_LLM_call
from preprocess import get_embedding
from Vector_DB import Redis, Chroma, qdrant

sample_questions = [
    "When was NoSQL first used? in 20 words.",
    "Show me an example of what a JSON Object might look like in 20 words.",
    "What are the benefits of using transactions in relational database systems? in 20 words."
]

VECTOR_DB_PIPELINES = {
    "redis": (Redis.redis_index_pipeline, Redis.query_redis),
    "chroma": (Chroma.process_pdfs, Chroma.query_chroma),
    "qdrant": (qdrant.qdrant_index_pipeline, qdrant.query_qdrant),
}

EMBEDDING_MODELS = ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"]
LLM_MODELS = ["llama3.2:latest", "mistral:latest"]

FIXED_CHUNK_SIZE = 100
FIXED_OVERLAP = 50

CSV_PATH = "benchmark_vector_all.csv"
csv_fields = [
    "vector_db", "embedding_model", "chunk_size", "overlap",
    "llm_model", "index_time_sec", "memory_mb",
    "retrieval_time_sec", "llm_response_time_sec"
]

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)

def write_result_csv(row):
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

def run_full_benchmark():
    for vector_db, (index_func, query_func) in VECTOR_DB_PIPELINES.items():
        for embedding_model in EMBEDDING_MODELS:
            print(f"\n🔄 Indexing with {vector_db.upper()} | {embedding_model}")

            mem_before = get_memory_usage()
            start_time = time.time()

            # Indexing
            index_func("data", FIXED_CHUNK_SIZE, FIXED_OVERLAP, embedding_model)

            index_time = time.time() - start_time
            mem_after = get_memory_usage()

            for llm_model in LLM_MODELS:
                for question in sample_questions:
                    try:
                        retrieval_start = time.time()
                        if vector_db == "redis":
                            top_context = query_func(question)
                        else:
                            top_context = query_func(question, embedding_model)
                        retrieval_time = time.time() - retrieval_start

                        if top_context is None:
                            print(f"[!] No result for {question} using {vector_db}/{embedding_model}")
                            continue

                        llm_start = time.time()
                        response = local_LLM_call(question, llm_model, top_context)
                        llm_time = time.time() - llm_start

                        write_result_csv({
                            "vector_db": vector_db,
                            "embedding_model": embedding_model,
                            "chunk_size": FIXED_CHUNK_SIZE,
                            "overlap": FIXED_OVERLAP,
                            "llm_model": llm_model,
                            "index_time_sec": round(index_time, 2),
                            "memory_mb": round(mem_after - mem_before, 2),
                            "retrieval_time_sec": round(retrieval_time, 2),
                            "llm_response_time_sec": round(llm_time, 2),
                        })

                    except Exception as e:
                        print(f"[!] Error with {vector_db}/{embedding_model}/{llm_model} - {e}")

if __name__ == "__main__":
    run_full_benchmark()
