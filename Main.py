from LLM_Call import local_LLM_call
from Vector_DB.Redis import index_pipeline
from Vector_DB.Redis import query_redis


def main():

    # index the data
    index_pipeline("data", 100, 50, "nomic-embed-text")
    top_embedding = query_redis("What is the capital of France?")


    # call the LLM
    local_LLM_call("Who is the professor of the course?", "llama3.2:latest", top_embedding)


if __name__ == "__main__":
    main()

