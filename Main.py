from LLM_Call import local_LLM_call
from Vector_DB.Redis import redis_index_pipeline
from Vector_DB.Redis import query_redis
from Chroma import chroma_index_pipeline
from Chroma import query_chroma


def main():

    # # index the data
    # redis_index_pipeline("data", 100, 50, "nomic-embed-text")
    # chroma_index_pipeline("data", 100, 50, "nomic-embed-text")

    # Retrieval 
    # redis_top_embedding = query_redis("Who is the professor of the course?")
    chroma_top_embedding = query_chroma("Who is the professor of the course?")

    # # call the LLM
    local_LLM_call("Who is the professor of the course?", "llama3.2:latest", chroma_top_embedding)


if __name__ == "__main__":
    main()

