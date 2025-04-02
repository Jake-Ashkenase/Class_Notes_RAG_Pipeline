from LLM_Call import local_LLM_call
from Vector_DB.Redis import redis_index_pipeline, query_redis, get_all_documents
from BM25 import BM25
import numpy as np
from Vector_DB.Chroma import chroma_index_pipeline, query_chroma
import time
import psutil
import os
from typing import Dict, List


# sample questions to be used for all of our tests. Questions picked of ranging complexity and expected response type
sample_questions = ["When was NoSQL first used?", 
                    "Show me an example of what a JSON Object might look like", 
                    "What are the benefits of using transactions in relational database systems?"]


## Chunk Size Tests ------------------------------------------------------------------------------------------


def test_chunk_size():
    # Test with different chunk sizes
    # Using Redis as the vector database and nomic-embed-text as the embedding model
    # Using 40 as the overlap size

    # looking just at the chunk that is returned since the LLM we are using is not relevant 
    chunk_sizes = [50, 150, 300]
    for chunk_size in chunk_sizes:
        print(f"Testing chunk size: {chunk_size}")
        redis_index_pipeline("data", chunk_size, 40, "nomic-embed-text")

        for question in sample_questions:
            print(f"Question: {question}")
            redis_top_embedding = query_redis(question)
            print(f"Redis top embedding: {redis_top_embedding}")


'''
Chunk Size Results:

Size = 50:

Question: When was NoSQL first used?
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::04 - Data Replication.pdf_page_13_chunk_upgrades dif difficult. Logical (row-based) Log For relational DBs: Inserted rows, modified rows (before and after), deleted rows. A transaction log will identify all the rows that changed in each transaction and how they changed. Logical logs are decoupled from the storage engine and easier to parse. Trigger-based Changes are logged

Size = 150:

Question: When was NoSQL first used?
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured web-based data. 5 https://www.dataversity.net/a-brief-history-of-non-relational-databases/
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::03 - Moving Beyond the Relational Model.pdf_page_3_chunk_Transaction Processing - Transaction - a sequence of one or more of the CRUD operations performed as a single, logical unit of work - Either the entire sequence succeeds (COMMIT) - OR the entire sequence fails (ROLLBACK or ABORT) - Help ensure - Data Integrity - Error Recovery - Concurrency Control - Reliable Data Storage - Simplified Error Handling 4


Size = 300:

Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured web-based data. 5 https://www.dataversity.net/a-brief-history-of-non-relational-databases/
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::03 - Moving Beyond the Relational Model.pdf_page_3_chunk_Transaction Processing - Transaction - a sequence of one or more of the CRUD operations performed as a single, logical unit of work - Either the entire sequence succeeds (COMMIT) - OR the entire sequence fails (ROLLBACK or ABORT) - Help ensure - Data Integrity - Error Recovery - Concurrency Control - Reliable Data Storage - Simplified Error Handling 4


'''

## Chunk Overlap Tests ------------------------------------------------------------------------------------------


def test_chunk_overlap():
    # Test with different chunk overlap 
    # Using Redis as the vector database and nomic-embed-text as the embedding model
    # using a chunk size of 100

    # looking just at the chunk that is returned since the LLM we are using is not relevant 
    chunk_overlap = [0, 50, 90]
    for chunk_overlap in chunk_overlap:
        print(f"Testing chunk overlap: {chunk_overlap}")
        redis_index_pipeline("data", 100, chunk_overlap, "nomic-embed-text")

        for question in sample_questions:
            print(f"Question: {question}")
            redis_top_embedding = query_redis(question)
            print(f"Redis top embedding: {redis_top_embedding}")


## Embedding Model Tests ------------------------------------------------------------------------------------------

def test_embedding_model():
    # testing 3 different embedding models from Ollama
    # using a chunk size of 100 and a chunk overlap of 50
    
    embed_models = ["snowflake-arctic-embed", "nomic-embed-text", "mxbai-embed-large",]
    results: Dict[str, Dict] = {}
    
    for embed_model in embed_models:
        print(f"\nTesting embedding model: {embed_model}")
        results[embed_model] = {
            "indexing_time": 0,
            "query_times": [],
            "memory_usage": 0,
            "retrieved_chunks": []
        }
        
        # Measure indexing time and memory
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  
        start_time = time.time()
        
        chroma_index_pipeline("data", 100, 50, embed_model)
        
        indexing_time = time.time() - start_time
        end_memory = process.memory_info().rss / 1024 / 1024
        memory_used = end_memory - start_memory
        
        results[embed_model]["indexing_time"] = indexing_time
        results[embed_model]["memory_usage"] = memory_used
        
        print(f"embedding model: {embed_model}")
        print(f"Indexing time: {indexing_time:.2f} seconds")
        print(f"Memory usage: {memory_used:.2f} MB")
        
        # Test queries and measure performance
        for question in sample_questions:
            print(f"\nQuestion: {question}")
            query_start = time.time()
            
            redis_top_embedding = query_redis(question)
            
            query_time = time.time() - query_start
            results[embed_model]["query_times"].append(query_time)
            results[embed_model]["retrieved_chunks"].append(redis_top_embedding)
            
            print(f"Query time: {query_time:.2f} seconds")
            print(f"Retrieved chunk: {redis_top_embedding[:200]}...")  # Show first 200 chars
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print(f"Average indexing time: {metrics['indexing_time']:.2f} seconds")
        print(f"Average query time: {np.mean(metrics['query_times']):.2f} seconds")
        print(f"Memory usage: {metrics['memory_usage']:.2f} MB")
        
        # Qualitative assessment of retrieval quality
        print("\nRetrieval Quality Assessment:")
        for i, chunk in enumerate(metrics['retrieved_chunks']):
            print(f"Q{i+1}: {chunk[:100]}...")  # Show first 100 chars of each retrieved chunk

'''

Embedding Model Results:

embedding model: nomic-embed-text
Indexing time: 29.53 seconds
Memory usage: 1.06 MB

Question: When was NoSQL first used?
Query time: 0.05 seconds
Retrieved chunk: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - “NoSQL” ﬁrst used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is “No...

Question: Show me an example of what a JSON Object might look like
Query time: 0.06 seconds
Retrieved chunk: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-st...

Question: What are the benefits of using transactions in relational database systems?
Query time: 0.06 seconds
Retrieved chunk: doc::04 - Data Replication.pdf_page_13_chunk_all followers must implement the same storage engine and makes upgrades difﬁcult. Logical (row-based) Log For relational DBs: Inserted rows, modiﬁed rows (...





'''


## System Prompt Tests ------------------------------------------------------------------------------------------

def test_system_prompt():
    # Testing with different system prompts
    # using the same emebdding and processing setup for each of the samples 

    # looking just at the chunk that is returned since the LLM we are using is not relevant 
    chunk_overlap = [0, 50, 90]
    for chunk_overlap in chunk_overlap:
        print(f"Testing chunk overlap: {chunk_overlap}")
        redis_index_pipeline("data", 100, chunk_overlap, "nomic-embed-text")

        for question in sample_questions:
            print(f"Question: {question}")
            redis_top_embedding = query_redis(question)
            print(f"Redis top embedding: {redis_top_embedding}")




test_embedding_model()
            
        
    

