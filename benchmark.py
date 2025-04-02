import time
from Vector_DB.qdrant import qdrant_index_pipeline, query_qdrant
from Vector_DB.Redis import redis_index_pipeline, query_redis
from Vector_DB.Chroma import process_pdfs as chroma_process_pdfs, clear_chroma_store, create_chroma_index
from Vector_DB.Chroma import query_chroma
from LLM_Call import local_LLM_call
import csv

# csv store
csv_file = open("benchmark_results.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["DB", "Embedding Model", "Chunk Size", "Overlap", "LLM", "Time (s)"])

# Parameters
chunk_sizes = [200, 500, 1000]
overlaps = [0, 50, 100]
embedding_models = ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed"]
vector_dbs = ["redis", "qdrant", "chroma"]
llm_models = ["llama3:latest", "mistral:latest"]  # add more if needed
sample_question = "What is a vector database?"

data_dir = "data"  # path to your PDFs

# Benchmark function
def run_benchmark():
    for chunk_size in chunk_sizes:
        for overlap in overlaps:
            for embed_model in embedding_models:
                for db in vector_dbs:
                    for llm in llm_models:
                        print(f"\n===== Testing: {db}, {embed_model}, chunk={chunk_size}, overlap={overlap}, LLM={llm} =====")
                        start = time.time()

                        if db == "redis":
                            redis_index_pipeline(data_dir, chunk_size, overlap, embed_model)
                            top_embedding = query_redis(sample_question)
                        elif db == "qdrant":
                            qdrant_index_pipeline(data_dir, chunk_size, overlap, embed_model)
                            top_embedding = query_qdrant(sample_question, embed_model)
                        elif db == "chroma":
                            clear_chroma_store()
                            create_chroma_index()
                            chroma_process_pdfs(data_dir, chunk_size, overlap, embed_model)
                            top_embedding = query_chroma(sample_question, embed_model)

                        end = time.time()
                        print(f"Index + Query time: {end - start:.2f}s")

                        csv_writer.writerow([db, embed_model, chunk_size, overlap, llm, round(end - start, 2)])

                        print("\n--- LLM Response ---")
                        local_LLM_call(sample_question, llm, top_embedding)


if __name__ == "__main__":
    run_benchmark()
