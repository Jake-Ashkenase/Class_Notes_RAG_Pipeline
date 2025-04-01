from LLM_Call import local_LLM_call
from Redis import redis_index_pipeline, query_redis, get_all_documents
from BM25 import BM25
import numpy as np
#from Chroma import chroma_index_pipeline
#from Chroma import query_chroma

def fusion_retrieval(query: str, alpha: float = 0.5, top_k: int = 5):
    """Perform fusion retrieval by combining vector-based and BM25 search results."""
    
    # Get vector search results from Redis
    vector_result_id = query_redis(query)
    
    # Get all documents for BM25
    all_docs = get_all_documents()
    doc_texts = [doc["text"] for doc in all_docs]
    
    # Initialize and fit BM25
    bm25 = BM25()
    bm25.fit(doc_texts)
    
    # Get BM25 search results
    bm25_results = bm25.search(query, top_k=len(all_docs))
    
    # Create score dictionaries
    # For vector search, give score 1.0 to the top result, 0 to others
    vector_scores = {}
    if vector_result_id:
        # Extract index from the document ID (assuming format includes "_chunk_X" at the end)
        try:
            top_index = int(vector_result_id.split("_chunk_")[-1])
            vector_scores[top_index] = 1.0
        except (ValueError, IndexError):
            pass
    
    # Get BM25 scores
    bm25_scores = {result["index"]: result["bm25_score"] for result in bm25_results}
    
    # Normalize BM25 scores (with safety check for zero scores)
    max_bm25_score = max(bm25_scores.values()) if bm25_scores else 1
    if max_bm25_score > 0: 
        bm25_scores = {k: v/max_bm25_score for k, v in bm25_scores.items()}
    else:
        bm25_scores = {k: 0.0 for k in bm25_scores.keys()} 

    # Combine scores
    combined_scores = []
    for i in range(len(all_docs)):
        vector_score = vector_scores.get(i, 0)
        bm25_score = bm25_scores.get(i, 0)
        combined_score = alpha * vector_score + (1 - alpha) * bm25_score
        combined_scores.append({
            "index": i,
            "text": all_docs[i]["text"],
            "metadata": all_docs[i]["metadata"],
            "score": combined_score
        })
    
    # Sort by combined score and return top k results
    return sorted(combined_scores, key=lambda x: x["score"], reverse=True)[:top_k]

def main():
    # Index the data
    # redis_index_pipeline("data", 100, 50, "nomic-embed-text")
    

    # ----------------------------------------------------------

    # Get user query from terminal
    query = input("Enter your query: ")

    # ----------------------------------------------------------

    # Retrieval

    # fusion version
    fusion_results = fusion_retrieval(query, alpha=0.5, top_k=5)
    # standerd version
    # redis_top_embedding = query_redis(query)

    # ----------------------------------------------------------
    
    # Get the top result for LLM
    top_result = fusion_results[0]

    # ----------------------------------------------------------

    # Call the LLM with the fused result
    local_LLM_call(query, "llama3.2:latest", top_result["text"])

if __name__ == "__main__":
    main()


# def main():
#     # Index the data
#     # redis_index_pipeline("data", 100, 50, "nomic-embed-text")
#     # chroma_index_pipeline("data", 100, 50, "nomic-embed-text")

#     # Get user query from terminal
#     query = input("Enter your query: ")

#     # Retrieval 
#     redis_top_embedding = query_redis(query)
#     # chroma_top_embedding = query_chroma(query)

#     # Call the LLM
#     local_LLM_call(query, "llama3.2:latest", redis_top_embedding)

# if __name__ == "__main__":
#     main()