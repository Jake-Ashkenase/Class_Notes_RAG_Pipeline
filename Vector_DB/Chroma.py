import chromadb
import os
from preprocess import extract_text_from_pdf, split_text_into_chunks, get_embedding

client = chromadb.HttpClient(
    host="localhost",  # Change this to your desired host
    port=6378,        # Change this to your desired port
    ssl=False         # Set to True if using HTTPS
)

DISTANCE_METRIC = "COSINE"
INDEX_NAME = "embedding_index"


# clear anything stored in the current chroma store
def clear_chroma_store():
    print("Clearing the existing Chroma store")
    try:
        # Try to delete the existing collection
        client.delete_collection(INDEX_NAME)
    except chromadb.errors.NotFoundError:
        # Collection doesn't exist, which is fine
        pass
    print("Chroma store cleared.") 

# Create a chroma index
def create_chroma_index():
    collection = client.create_collection(INDEX_NAME)
    print("Index created successfully.")


# ----------------------
# Embedding Generation
# ----------------------

def store_embedding(file: str, page: str, chunk: str, embedding: list, original_text: str):
    doc_id = f"{file}_page_{page}_chunk_{chunk}"
    
    collection = client.get_collection(INDEX_NAME)

    collection.add(
        embeddings=[embedding],
        metadatas=[{
            "file": file,
            "page": page,
            "chunk": chunk,
            "text": original_text  # 👈 Add this line
        }],
        ids=[doc_id]
    )

# Process all PDF files in a given directory
def process_pdfs(data_dir, chunk_size=100, overlap=50, embedding_model="nomic-embed-text"):
    '''
    Go through all pdf's in the data directory and process them

    data_dir: the directory containing the pdf's (str)
    chunk_size: the # of tokens per chunk (int)
    overlap: the # of tokens to overlap between chunks (int)
    embedding_model: the model to use for embedding (str)
    
    '''

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, embedding_model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk),
                        embedding=embedding,
                        original_text=chunk
                    )
            print(f" -----> Processed {file_name}")


def chroma_index_pipeline(data_dir: str, chunk_size: int, overlap: int, embedding_model: str):
    '''
    This function will clear the redis store, create a new HNSW index, and process all documents
    in the data directory. 

    data_dir: the directory containing the pdf's (str)
    chunk_size: the # of tokens per chunk (int)
    overlap: the # of tokens to overlap between chunks (int)
    embedding_model: the model to use for embedding (str)
    
    '''
    clear_chroma_store()
    create_chroma_index()

    process_pdfs(data_dir, chunk_size, overlap, embedding_model)


# Query Chroma
def query_chroma(query_text: str, embedding_model: str = "nomic-embed-text"):
    embedding = get_embedding(query_text, embedding_model)

    collection = client.get_collection(INDEX_NAME)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas", "documents"]  # Include both metadata and documents
    )

    # Return either from metadata (if exists) or from documents
    if results["metadatas"] and results["metadatas"][0] and "text" in results["metadatas"][0][0]:
        return results["metadatas"][0][0]["text"]
    elif results["documents"]:  # Fallback to documents if text not in metadata
        return results["documents"][0][0]
    else:
        return ""  # Return empty string if nothing found
