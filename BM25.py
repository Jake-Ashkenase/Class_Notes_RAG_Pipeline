import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Any

class BM25:

    # k and b are tuning parameters reccomended for BM25
    def __init__(self, k1=1.5, b=0.75):
        # how much impact multiple occurences of a word should have
        self.k1 = k1
        # how much importance we give to the document length
        self.b = b
        # vectorizer to convert text to TF-IDF vectors
        self.vectorizer = TfidfVectorizer()
        # number of documents in the collection
        self.doc_count = 0
        # average length of documents in the collection
        self.avg_doc_length = 0
        # document frequencies for each term
        self.doc_freqs = {}
        # lengths of documents
        self.doc_lengths = []
        # TF-IDF vectors of documents
        self.doc_vectors = None
        self.doc_texts = []
        

    def fit(self, documents: List[str]):
        """Fit the BM25 model on a collection of documents."""
        self.doc_texts = documents
        self.doc_count = len(documents)
        
        # Calculate document lengths
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Fit TF-IDF vectorizer
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        
        # Calculate document frequencies
        feature_names = self.vectorizer.get_feature_names_out()
        for i, term in enumerate(feature_names):
            self.doc_freqs[term] = self.vectorizer.idf_[i]


    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for documents using BM25 scoring."""
        if self.doc_vectors is None:
            raise ValueError("BM25 model not fitted. Call fit() first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate BM25 scores
        scores = []
        for i, doc_vector in enumerate(self.doc_vectors):
            score = 0
            # Get non-zero elements (terms that appear in both query and document)
            query_terms = query_vector.indices
            for term in query_terms:
                if term in self.vectorizer.vocabulary_:
                    term_id = self.vectorizer.vocabulary_[term]
                    tf = doc_vector[0, term_id]
                    df = self.doc_freqs[term]
                    
                    # BM25 scoring formula
                    numerator = df * (self.k1 + 1) * tf
                    denominator = self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_length) + tf
                    score += numerator / denominator
            
            scores.append({
                "index": i,
                "text": self.doc_texts[i],
                "bm25_score": score
            })
        
        # Sort by score and return top k results
        return sorted(scores, key=lambda x: x["bm25_score"], reverse=True)[:top_k] 