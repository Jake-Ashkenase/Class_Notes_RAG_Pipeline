from LLM_Call import local_LLM_call
from Redis import redis_index_pipeline
from Redis import query_redis
#from Chroma import chroma_index_pipeline
#from Chroma import query_chroma

def main():
    # Index the data
    redis_index_pipeline("data", 100, 50, "nomic-embed-text")
    # chroma_index_pipeline("data", 100, 50, "nomic-embed-text")

    # Get user query from terminal
    query = input("Enter your query: ")

    # Retrieval 
    redis_top_embedding = query_redis(query)
    # chroma_top_embedding = query_chroma(query)

    # Call the LLM
    local_LLM_call(query, "llama3.2:latest", redis_top_embedding)

if __name__ == "__main__":
    main()