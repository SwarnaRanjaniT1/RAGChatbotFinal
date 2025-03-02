import os

# Data and file paths
DATA_DIR = "data"
VECTOR_DB_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"
CHUNKS_METADATA_PATH = "chunks_metadata.pkl"

# Text chunking parameters
CHUNK_SIZE = 512  # Characters per chunk
CHUNK_OVERLAP = 128  # Character overlap between chunks

# Hybrid search parameters
BM25_WEIGHT = 0.3  # Weight for BM25 scores in hybrid search
SEMANTIC_WEIGHT = 0.7  # Weight for semantic search scores in hybrid search

# Embedding model
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Small but effective model

# Language model for response generation
# Using a smaller model for better performance on limited resources
LLM_MODEL_NAME = "distilbert/distilbert-base-uncased"  # Small open-source LLM
MAX_NEW_TOKENS = 512  # Maximum new tokens to generate
TEMPERATURE = 0.7  # Generation temperature (higher = more creative)

# Prompt template for response generation
PROMPT_TEMPLATE = """
You are a helpful financial assistant that answers questions about company financial statements.
Use only the information provided in the context to answer the question. If you don't know the
answer based on the context, admit that you don't know rather than making up information.

Context:
{context}

Question: {query}

Answer:
"""
