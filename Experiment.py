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


"""

Chunk Overlap Results:

Overlap = 0:

Question: When was NoSQL first used?
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured web-based data. 5 https://www.dataversity.net/a-brief-history-of-non-relational-databases/
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::03 - Moving Beyond the Relational Model.pdf_page_3_chunk_Transaction Processing - Transaction - a sequence of one or more of the CRUD operations performed as a single, logical unit of work - Either the entire sequence succeeds (COMMIT) - OR the entire sequence fails (ROLLBACK or ABORT) - Help ensure - Data Integrity - Error Recovery - Concurrency Control - Reliable Data Storage - Simplified Error Handling 4

Overlap = 50:

Question: When was NoSQL first used?
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured web-based data. 5 https://www.dataversity.net/a-brief-history-of-non-relational-databases/
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::04 - Data Replication.pdf_page_13_chunk_all followers must implement the same storage engine and makes upgrades difﬁcult. Logical (row-based) Log For relational DBs: Inserted rows, modiﬁed rows (before and after), deleted rows. A transaction log will identify all the rows that changed in each transaction and how they changed. Logical logs are decoupled from the storage engine and easier to parse. Trigger-based Changes are logged to a separate table whenever a trigger ﬁres in response to an insert, update, or delete. Flexible because you can have application speciﬁc replication, but also more error prone.

Overlap = 90:

Question: When was NoSQL first used?
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured web-based data. 5 https://www.dataversity.net/a-brief-history-of-non-relational-databases/
Question: Show me an example of what a JSON Object might look like
Redis top embedding: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33
Question: What are the benefits of using transactions in relational database systems?
Redis top embedding: doc::04 - Data Replication.pdf_page_13_chunk_difﬁculty in handling concurrent transactions. Write-ahead Log (WAL) A byte-level speciﬁc log of every change to the database. Leader and all followers must implement the same storage engine and makes upgrades difﬁcult. Logical (row-based) Log For relational DBs: Inserted rows, modiﬁed rows (before and after), deleted rows. A transaction log will identify all the rows that changed in each transaction and how they changed. Logical logs are decoupled from the storage engine and easier to parse. Trigger-based Changes are logged to a separate table whenever a trigger ﬁres in response to an insert, update, or delete. Flexible because you can
"""


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
        
        redis_index_pipeline("data", 100, 50, embed_model)
        
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
Retrieved chunk: doc::05 - NoSQL Intro + KV DBs.pdf_page_4_chunk_NoSQL - "NoSQL" first used in 1998 by Carlo Strozzi to describe his relational database system that did not use SQL. - More common, modern meaning is "Not Only SQL" - But, sometimes thought of as non-relational DBs - Idea originally developed, in part, as a response to processing unstructured

Question: Show me an example of what a JSON Object might look like
Query time: 0.06 seconds
Retrieved chunk: doc::05 - NoSQL Intro + KV DBs.pdf_page_32_chunk_JSON Type - Full support of the JSON standard - Uses JSONPath syntax for parsing/navigating a JSON document - Internally, stored in binary in a tree-structure → fast access to sub elements 33

Question: What are the benefits of using transactions in relational database systems?
Query time: 0.06 seconds
Retrieved chunk: doc::04 - Data Replication.pdf_page_13_chunk_all followers must implement the same storage engine and makes upgrades difﬁcult. Logical (row-based) Log For relational DBs: Inserted rows, modiﬁed rows (before and after), deleted rows. A transaction log will identify all the rows that changed in each transaction and how they changed. Logical logs are decoupled from the storage engine and easier to parse. Trigger-based Changes are logged


Model: mxbai-embed-large
Average indexing time: 71.51 seconds
Average query time: 0.06 seconds
Memory usage: -0.15 MB

Retrieval Quality Assessment:
Q1: ...
Q2: ...
Q3: ...


Model: snowflake-arctic-embed
Average indexing time: 66.65 seconds
Average query time: 0.06 seconds
Memory usage: 2.43 MB

Retrieval Quality Assessment:
Q1: ...
Q2: ...
Q3: ...


'''


## System Prompt Tests ------------------------------------------------------------------------------------------

def test_system_prompt():
    # Testing with different system prompts
    # using the same emebdding and processing setup for each of the samples 

    # looking just at the chunk that is returned since the LLM we are using is not relevant 


    redis_index_pipeline("data", 100, 50, "nomic-embed-text")

    question = sample_questions[2]

    redis_top_embedding = query_redis(question)

    system_prompt1 = f"You are a helpful assistant that can answer questions about the following text: {redis_top_embedding} You are also given a question: {question} Answer the question based on the text."
        
    system_prompt2 = f"You are an expert database instructor with deep knowledge of database systems. Your task is to provide clear, accurate answers based on the following context: {redis_top_embedding} When answering the question: {question} Please follow these guidelines: 1. Only use information present in the provided context 2. If the context doesn't fully answer the question, acknowledge this limitation 3. Use technical terms accurately and explain them when necessary 4. If relevant, provide examples to illustrate your points 5. Keep your response concise and focused If the context is insufficient or unclear, state this explicitly rather than making assumptions."
    
    system_prompt3 = f"You are a knowledgeable teaching assistant helping students understand database concepts. You have access to the following relevant information: {redis_top_embedding} For the question: {question} Please provide: - A clear, direct answer using the context provided - Simple explanations that break down complex concepts - Real-world analogies when helpful - Acknowledgment of any gaps in the provided context Remember to: - Stay within the scope of the provided context - Use student-friendly language while maintaining technical accuracy - Highlight key concepts that are crucial for understanding"
    
    system_prompts = [system_prompt1, system_prompt2, system_prompt3]
    for i, system_prompt in enumerate(system_prompts):
        print(f"System Prompt #{i+1}: {system_prompt}")
        local_LLM_call(question, "llama3.2:latest", redis_top_embedding, system_prompt)



'''

System Prompt Results:

System Prompt 1: --------------------------

Based on the provided text, I'll answer the question about the benefits of using transactions in relational database systems:

The benefits of using transactions in relational database systems (specifically mentioned in the context of relational DBs) are not explicitly stated in the given text. However, we can infer that using logical logs (row-based) might provide some benefits.

But, one benefit of using transactions is to "maintain data consistency and integrity". Unfortunately, this detail was not explicitly provided in the text.

System Prompt 2: --------------------------

Based on the provided context, I can answer the question about the benefits of using transactions in relational database systems.

According to the context, a transaction log will identify all the rows that changed in each transaction and how they changed. This implies that one benefit of using transactions is that it allows for auditing and tracking changes made to data within a database.

Additionally, transactions enable atomicity, which means that either all or none of the operations are committed, ensuring data consistency and reducing errors caused by partial updates or deletions.

However, the context does not provide information on other potential benefits of using transactions. For example, it is known that one of the limitations of transactions in relational database systems is that they can make upgrades difficult if everyone must implement the same storage engine (Section 4.1 of the provided document).

Therefore, I can only partially answer this question based on the available information.

System Prompt 3: --------------------------

Let's dive into the benefits of using transactions in relational database systems.

**What are transactions?**

In the context of relational databases, a transaction is a sequence of operations (inserts, updates, deletes) performed on one or more tables as a single, all-or-nothing unit. Think of it like a shopping cart: you add items, make changes, and then either confirm your purchases or cancel everything.

**Benefits of using transactions:**

1. **Atomicity**: A transaction ensures that either all operations are applied successfully, or none are. This prevents partial updates, which can lead to inconsistencies in the data.
2. **Consistency**: Transactions maintain data consistency by ensuring that changes are made in a way that preserves the integrity of the database.
3. **Isolation**: Transactions allow multiple transactions to run concurrently without interfering with each other's operations. This means you can have multiple users updating different parts of the database simultaneously, and the results will be consistent.

**Real-world analogy:** Imagine you're working on a project with multiple team members. Each team member is updating their part of the project (e.g., adding new features). If they all work together in a transactional environment, each update is done as a single, atomic unit. Either everything gets updated correctly, or nothing does.

**Gaps in the provided context:** The document doesn't explicitly mention the benefits of transactions in terms of concurrency control, which is an important aspect of relational database systems. However, based on our understanding of transactions and their properties, we can infer that using transactions helps ensure data consistency and allows for concurrent updates without compromising the integrity of the database.

**Key concepts:**

* Atomicity
* Consistency
* Isolation

These three properties are fundamental to the success of transactions in relational database systems. By ensuring atomicity, consistency, and isolation, transactions provide a robust way to manage changes to the database, making it easier to maintain data integrity and accuracy.

'''

test_system_prompt()
            
        
    

