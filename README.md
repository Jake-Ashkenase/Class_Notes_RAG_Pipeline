# DS4300 Practical 02 — Local RAG Pipeline

Group: Class_Notes_RAG_Pipeline

Jake Ashkenase: ashkenase.j@northeastern.edu
Benjamin Rice: rice.be@northeastern.edu
Markus Zaba: zaba.m@northeastern.edu

## Overview

This project implements a local Retrieval-Augmented Generation (RAG) system that allows users to query a collection of DS4300 course notes using locally hosted vector databases and language models. The goal was to evaluate how different components—vector stores, embedding models, and LLMs—impact retrieval accuracy and response quality. Our strategy involved building a modular pipeline to ingest and chunk documents, index them with multiple vector databases, generate embeddings with various models, and benchmark responses using different local LLMs through Ollama.


## Vector Databases Used

- Redis Vector DB
- Chroma
- Qdrant

## Embedding Models Used

- nomic-embed-text
- mxbai-embed-large
- snowflake-arctic-embed

## LLMs Used

- llama3.2:latest
- mistral:latest

## Installation and Setup

### Vector Database Setup (via Docker)

You must run the following vector databases locally using Docker.

To run Redis with vector search:

```
docker run -p 6379:6379 redis/redis-stack-server
```

To install Chroma or run the docker container:

```
docker run -p 6378:8000 ghcr.io/chroma-core/chroma
```

To run Qdrant:

```
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama Setup (for Embeddings and LLMs)

Install and start Ollama locally if not already have:

```
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

Then pull the required models:

```
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull snowflake-arctic-embed
ollama pull llama3.2:latest
ollama pull mistral:latest
```

### Python Dependencies

Install all required Python packages:

```
pip install -r requirements.txt
```

Sample `requirements.txt`:

```
fitz
numpy
psutil
redis
chromadb
qdrant-client
scikit-learn
ollama
```

## How to Run the Project

### Run Benchmarking (Full Pipeline Test)

To run the full benchmark that tests combinations of:
- 3 vector databases
- 3 embedding models
- 2 local LLMs

Run:

```
python benchmark.py
```

This will create a file called:

(Note: the combination of a chunksize:500 and overlay:100 for snowflake may not work since snowflake on certain devices can't withstand that chunksize)

```
benchmark_vector_all.csv
```

Each row will include:
- The vector DB used
- The embedding model used
- The LLM used
- Index time, retrieval time, and LLM response time
- Memory usage
- The question and response

### Run Interactive CLI Mode

To run the system in a manual query mode:

```
python Main.py
```

This is how we queried it with our customer questions along with gathered unique resposne and data from select response variables we found from our benchmarking data. 

This will:
- Index the course notes
- Ask the user to enter a question
- Retrieve the most relevant chunk from Redis
- Generate a response using the configured local LLM
- Output the answer in the terminal
- Log all query/response and timing data to:

```
main_query_log.csv
```



