import ollama

# Get Ollama Running 
def local_LLM_call(query, model, top_embedding, system_prompt):
    '''
    Create a call to the local LLM using the given query and specified model.

    query: the final query that is being passed to the LLM(str) 
    model: the name of the ollama specific model being used (str)
    '''

    input = f'''

        You are a helpful assistant that can answer questions about the following text:
        {top_embedding}

        You are also given a question:
        {query}

        Answer the question based on the text.
        
    '''

    # Generate response using Ollama
    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": input}]
    )

    return response["message"]["content"]