import numpy as np
from typing import List, Tuple, Any
import faiss
from rank_bm25 import BM25Okapi
import re
from sentence_transformers import SentenceTransformer
import logging
from app.config import EMBEDDINGS_MODEL_NAME, BM25_WEIGHT, SEMANTIC_WEIGHT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define English stopwords list (common words to filter out)
ENGLISH_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
    'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
    'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
    'to', 'from', 'in', 'out', 'on', 'off', 'again', 'further', 'then', 'once',
    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'i', 'me', 'my', 'myself', 'we',
    'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'can', 'could', 'should', 'would', 'ought', 'will',
    'shall', 'may', 'might'
}

# Simple word tokenizer
def simple_word_tokenize(text):
    """Simple word tokenizer that splits on spaces and punctuation"""
    # Remove punctuation and replace with spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    # Split on whitespace and filter empty strings
    return [word for word in text.split() if word]

class HybridSearcher:
    """Class for performing hybrid search (BM25 + Dense embeddings)"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
        self.bm25 = None
        self.tokenized_corpus = None
    
    def preprocess_for_bm25(self, texts: List[str]) -> List[List[str]]:
        """
        Preprocess text for BM25 indexing.
        
        Args:
            texts: List of text chunks
            
        Returns:
            Tokenized and preprocessed text
        """
        tokenized_corpus = []
        
        for text in texts:
            # Tokenize, lowercase, remove stopwords and punctuation
            tokens = simple_word_tokenize(text)
            tokens = [token for token in tokens if token and token not in ENGLISH_STOPWORDS]
            tokenized_corpus.append(tokens)
        
        return tokenized_corpus
    
    def setup_bm25(self, chunks: List[str]):
        """
        Setup BM25 index.
        
        Args:
            chunks: List of text chunks to index
        """
        self.tokenized_corpus = self.preprocess_for_bm25(chunks)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
    def bm25_search(self, query: str, top_k: int = 5) -> Tuple[List[int], List[float]]:
        """
        Perform BM25 search.
        
        Args:
            query: The search query
            top_k: Number of top results to return
            
        Returns:
            Tuple of (document indices, scores)
        """
        # Preprocess the query
        tokenized_query = simple_word_tokenize(query)
        tokenized_query = [token for token in tokenized_query if token and token not in ENGLISH_STOPWORDS]
        
        # Get BM25 scores
        if not tokenized_query:
            # Handle empty query case
            return [], []
            
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        top_scores = [bm25_scores[i] for i in top_indices]
        
        # Normalize scores to 0-1 range
        max_score = max(bm25_scores) if bm25_scores.size > 0 else 1.0
        if max_score > 0:
            top_scores = [score / max_score for score in top_scores]
        
        return top_indices.tolist(), top_scores

# Global instance for reuse
_hybrid_searcher = None

def get_hybrid_searcher():
    """Singleton pattern to get or create the hybrid searcher"""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher

def hybrid_search(query: str, chunks: List[str], faiss_index: faiss.Index, top_k: int = 5) -> Tuple[List[str], List[int], List[float]]:
    """
    Perform hybrid search combining BM25 and dense embeddings.
    
    Args:
        query: The search query
        chunks: List of text chunks
        faiss_index: FAISS index for dense retrieval
        top_k: Number of results to return
        
    Returns:
        Tuple containing:
        - List of retrieved text chunks
        - List of chunk indices
        - List of hybrid scores
    """
    logger.info(f"Performing hybrid search for query: {query}")
    
    # Get or initialize the hybrid searcher
    searcher = get_hybrid_searcher()
    
    # Setup BM25 if not already setup
    if searcher.bm25 is None or searcher.tokenized_corpus is None:
        logger.info("Setting up BM25 index")
        searcher.setup_bm25(chunks)
    
    # Get BM25 results
    bm25_indices, bm25_scores = searcher.bm25_search(query, top_k=top_k*2)  # Get more results for potential reranking
    logger.info(f"BM25 search returned {len(bm25_indices)} results")
    
    # Get dense embedding results
    query_embedding = searcher.embedding_model.encode([query])[0]
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
    
    # Search the FAISS index
    semantic_scores, semantic_indices = faiss_index.search(query_embedding, top_k*2)
    semantic_scores = semantic_scores[0]  # Flatten
    semantic_indices = semantic_indices[0]  # Flatten
    logger.info(f"Dense search returned {len(semantic_indices)} results")
    
    # Combine results (simple approach - weighted average of normalized scores)
    combined_results = {}
    
    # Add BM25 results
    for idx, score in zip(bm25_indices, bm25_scores):
        if idx not in combined_results:
            combined_results[idx] = {"bm25": score, "dense": 0.0}
        else:
            combined_results[idx]["bm25"] = score
    
    # Add dense results
    for idx, score in zip(semantic_indices, semantic_scores):
        if idx not in combined_results:
            combined_results[idx] = {"bm25": 0.0, "dense": float(score)}
        else:
            combined_results[idx]["dense"] = float(score)
    
    # Calculate hybrid scores
    hybrid_results = []
    for idx, scores in combined_results.items():
        hybrid_score = (BM25_WEIGHT * scores["bm25"]) + (SEMANTIC_WEIGHT * scores["dense"])
        hybrid_results.append((idx, hybrid_score))
    
    # Sort by hybrid score and get top_k
    hybrid_results.sort(key=lambda x: x[1], reverse=True)
    top_hybrid_results = hybrid_results[:top_k]
    
    # Extract results
    result_indices = [res[0] for res in top_hybrid_results]
    result_scores = [res[1] for res in top_hybrid_results]
    result_chunks = [chunks[idx] for idx in result_indices]
    
    logger.info(f"Hybrid search returned {len(result_indices)} results")
    return result_chunks, result_indices, result_scores
