import ollama

def local_LLM_call(query, model, top_embedding, system_prompt=None):
    '''
    Create a call to the local LLM using the given query and specified model.
    '''

    if system_prompt is None:
        system_prompt = "You are a helpful assistant that answers questions about database systems using the retrieved context below."

    input = f'''
{system_prompt}

Retrieved Context:
{top_embedding}

Question:
{query}

Answer the question based on the context above.
'''

    response = ollama.chat(
        model=model, messages=[{"role": "user", "content": input}]
    )

    return response["message"]["content"]
