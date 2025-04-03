from LLM_Call import local_LLM_call
from Vector_DB.Redis import redis_index_pipeline, query_redis
from Vector_DB.Chroma import chroma_index_pipeline, query_chroma
from Vector_DB.qdrant import qdrant_index_pipeline, query_qdrant
from BM25 import BM25
import numpy as np
#from Chroma import query_chroma

def main():
    # Redis Index 
    redis_index_pipeline("data", 100, 50, "nomic-embed-text")
    # Chroma Index 
    # chroma_index_pipeline("data", 100, 50, "nomic-embed-text")
    # qdrant Index
    # qdrant_index_pipeline("data", 100, 50, "nomic-embed-text")


    query = input("Enter your query: ")

    # Redis query
    redis_top_embedding = query_redis(query, "nomic-embed-text")

    # Chroma query 
    # chroma_top_embedding = query_chroma(query, "nomic-embed-text")

    # qdrant Query
    # qdrant_top_embedding = query_qdrant(query, "nomic-embed-text")


    # response
    response = local_LLM_call(query, "llama3.2:latest", redis_top_embedding)
    # response = local_LLM_call(query, "llama3.2:latest", chroma_top_embedding)
    # response = local_LLM_call(query, "llama3.2:latest", qdrant_top_embedding)

    # mistral
    # response = local_LLM_call(query, "mistral:latest", redis_top_embedding)


    # Call the LLM
    print("\n Response:\n")
    print(response)

if __name__ == "__main__":
    main()

