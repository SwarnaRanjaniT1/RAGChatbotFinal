import os
import numpy as np
from typing import List, Tuple, Dict, Any
import faiss
from app.preprocessing import preprocess_documents, chunk_text
from app.hybrid_search import hybrid_search
from app.llm_generator import generate_response
import pickle
import logging
from app.config import VECTOR_DB_PATH, CHUNKS_PATH, CHUNKS_METADATA_PATH, EMBEDDINGS_MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialRAG:
    def __init__(self, force_reprocess=False):
        """
        Initialize the Financial RAG system with vector database and preprocessing.
        
        Args:
            force_reprocess: If True, reprocess all documents even if a database already exists
        """
        logger.info("Initializing Financial RAG system")
        self.initialize_system(force_reprocess)
    
    def initialize_system(self, force_reprocess=False):
        """
        Load or create necessary components for the RAG system.
        
        Args:
            force_reprocess: If True, reprocess all documents even if a database already exists
        """
        # Check if vector database already exists
        if not force_reprocess and os.path.exists(VECTOR_DB_PATH) and os.path.exists(CHUNKS_PATH) and os.path.exists(CHUNKS_METADATA_PATH):
            logger.info("Loading existing vector database and chunks")
            self.load_existing_data()
        else:
            logger.info("Creating new vector database and processing documents")
            self.create_new_data()
    
    def load_existing_data(self):
        """Load existing vector database, text chunks, and metadata."""
        # Load the index
        self.index = faiss.read_index(VECTOR_DB_PATH)
        
        # Load chunks and metadata
        with open(CHUNKS_PATH, 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(CHUNKS_METADATA_PATH, 'rb') as f:
            self.chunks_metadata = pickle.load(f)
            
        logger.info(f"Loaded {len(self.chunks)} chunks from existing database")
    
    def create_new_data(self):
        """Process documents, create chunks, and build vector database."""
        # Preprocess financial documents 
        docs = preprocess_documents()
        
        # Chunk the documents and get metadata
        self.chunks, self.chunks_metadata = chunk_text(docs)
        
        # Create and save the vector database
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBEDDINGS_MODEL_NAME)
        
        # Get embeddings
        embeddings = model.encode(self.chunks, show_progress_bar=True)
        
        # Normalize the vectors before adding them to FAISS (for cosine similarity)
        faiss.normalize_L2(embeddings)
        
        # Create the FAISS index
        vector_dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(vector_dimension)  # Inner product for cosine similarity with normalized vectors
        self.index.add(embeddings)
        
        # Save the index, chunks, and metadata
        faiss.write_index(self.index, VECTOR_DB_PATH)
        
        with open(CHUNKS_PATH, 'wb') as f:
            pickle.dump(self.chunks, f)
            
        with open(CHUNKS_METADATA_PATH, 'wb') as f:
            pickle.dump(self.chunks_metadata, f)
            
        logger.info(f"Created and saved vector database with {len(self.chunks)} chunks")
    
    def answer_question(self, query: str) -> Tuple[str, float, List[str]]:
        """
        Answer a financial question using hybrid search and LLM generation.
        
        Args:
            query: The user's financial question
            
        Returns:
            Tuple containing:
            - Generated answer
            - Confidence score (0-1)
            - List of source references
        """
        logger.info(f"Processing query: {query}")
        
        # Get relevant chunks using hybrid search
        relevant_chunks, chunk_indices, hybrid_scores = hybrid_search(
            query=query,
            chunks=self.chunks,
            faiss_index=self.index,
            top_k=5  # Retrieve top 5 most relevant chunks
        )
        
        # Get metadata for the relevant chunks
        sources = [self.chunks_metadata[idx].get('source', 'Unknown source') for idx in chunk_indices]
        
        # Normalize the hybrid scores to get a confidence estimate (0-1)
        if len(hybrid_scores) > 0:
            confidence = float(np.mean(hybrid_scores))
        else:
            confidence = 0.0
        
        # Generate answer using the LLM
        context = "\n\n".join(relevant_chunks)
        answer = generate_response(query, context)
        
        return answer, confidence, sources
