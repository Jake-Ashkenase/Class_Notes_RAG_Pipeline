import ollama


# Get Ollama Running 
def local_LLM_call(query, model):
    '''
    Create a call to the local LLM using the given query and specified model.

    query: the final query that is being passed to the LLM(str) 
    model: the name of the ollama specific model being used (str)
    '''

    # Generate response using Ollama
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": query}]
    )

    print(response["message"]["content"])
