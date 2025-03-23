import fitz
import ollama
from sentence_transformers import SentenceTransformer

# Load SentenceTransformer models once
transformer_models = {
    "all-MiniLM-L6-v2": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": SentenceTransformer("sentence-transformers/all-mpnet-base-v2"),
}

# Generate an embedding based on the selected model
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    if model == "nomic-embed-text":
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    elif model in transformer_models:
        return transformer_models[model].encode(text).tolist()
    else:
        raise ValueError(f"Unsupported embedding model: {model}")

# Extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=100, overlap=50):
    """
    Turn the text into chunks given the size and overlap.
    
    chunk_size: the # of tokens per chunk (int)
    overlap: the # of tokens to overlap between chunks (int)
    """
    words = text.split()
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    return chunks
