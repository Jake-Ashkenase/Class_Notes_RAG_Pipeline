from LLM_Call import local_LLM_call
# from Vector_DB.Redis import redis_index_pipeline
# from Vector_DB.Redis import query_redis
# from Vector_DB.Chroma import chroma_index_pipeline
# from Vector_DB.Chroma import query_chroma
from Vector_DB.qdrant import qdrant_index_pipeline, query_qdrant

def main():
    # Index the data
    # redis_index_pipeline("data", 100, 50, "nomic-embed-text")
    # chroma_index_pipeline("data", 100, 50, "nomic-embed-text")
    # hqdrant_index_pipeline("data", 100, 50, "snowflake-arctic-embed")

    # Get user query from terminal
    query = input("Enter your query: ")

    # Retrieval 
    # redis_top_embedding = query_redis(query)
    # chroma_top_embedding = query_chroma(query)
    qdrant_top_embedding = query_qdrant(query, "snowflake-arctic-embed")

    # Call the LLM
    # local_LLM_call(query, "llama3.2:latest", redis_top_embedding)
    local_LLM_call(query, "llama3.2:latest", qdrant_top_embedding)

if __name__ == "__main__":
    main()