# DS4300 - Spring 2025 - Practical #2

Group: Class_Notes_RAG_Pipeline

Jake Ashkenase: ashkenase.j@northeastern.edu
Benjamin Rice: rice.be@northeastern.edu
Markus Zaba: zaba.m@northeastern.edu

# DS4300 Practical 02 — Local RAG Pipeline

This project implements a local Retrieval-Augmented Generation (RAG) system that allows users to query course notes through a combination of vector databases, embedding models, and local LLMs. It benchmarks multiple configurations and supports interactive querying through the command line.

## Objective

The goal of this project is to:
- Ingest course notes in PDF format
- Chunk the text and generate embeddings
- Store and retrieve these embeddings using vector databases
- Query the system with natural language questions
- Use a local LLM to answer questions based on retrieved context
- Benchmark performance of the system under various configurations

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

To install Chroma (runs in Python, no container needed):

```
pip install chromadb
```

To run Qdrant:

```
docker run -p 6333:6333 qdrant/qdrant
```

### Ollama Setup (for Embeddings and LLMs)

Install and start Ollama locally:

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



