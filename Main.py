from LLM_Call import local_LLM_call
from Vector_DB.Redis import index_pipeline


def main():

    # index the data
    index_pipeline("data", 100, 50)

    # call the LLM
    local_LLM_call("What is the capital of France?", "llama3.2:latest")


if __name__ == "__main__":
    main()

