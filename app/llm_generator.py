import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, pipeline
import torch
import logging
import re
from app.config import LLM_MODEL_NAME, MAX_NEW_TOKENS, TEMPERATURE, PROMPT_TEMPLATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store model components
_model = None
_tokenizer = None
_qa_pipeline = None

def load_model():
    """Load the language model and tokenizer."""
    global _model, _tokenizer, _qa_pipeline
    
    if _model is None or _tokenizer is None:
        logger.info(f"Loading model: {LLM_MODEL_NAME}")
        
        try:
            # Use CPU for the model to save memory
            _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
            
            # Set padding token if not set
            if _tokenizer.pad_token is None and hasattr(_tokenizer, 'eos_token'):
                _tokenizer.pad_token = _tokenizer.eos_token
            
            # Try to load with accelerate first (more efficient)
            try:
                # Load the model with accelerate for efficient memory usage
                _model = AutoModel.from_pretrained(
                    LLM_MODEL_NAME,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
                logger.info("Model loaded with Accelerate")
            except (ImportError, Exception) as e:
                # If accelerate is not available, fall back to standard loading
                logger.warning(f"Accelerate loading failed: {str(e)}. Using standard model loading")
                _model = AutoModel.from_pretrained(LLM_MODEL_NAME)
                logger.info("Model loaded with standard configuration")
            
            # Create a simpler fill-mask pipeline for lightweight text generation
            _qa_pipeline = pipeline(
                "feature-extraction", 
                model=_model, 
                tokenizer=_tokenizer
            )
            
            logger.info("Model and pipeline loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    return _model, _tokenizer, _qa_pipeline

def generate_response(query: str, context: str) -> str:
    """
    Generate a response using embeddings from the retrieved context.
    
    Args:
        query: The user's question
        context: The context from retrieved documents
        
    Returns:
        Generated answer text
    """
    logger.info("Generating response based on retrieved context")
    
    try:
        # Load model components
        _, _, pipeline = load_model()
        
        # Since we're using a smaller model that's not designed for text generation,
        # we'll use a rule-based approach combining the retrieved context
        
        # Split context into sentences for better analysis
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Create a simple answer based on relevant sentences from context
        relevant_info = []
        query_keywords = set(query.lower().split())
        
        # Filter for most relevant sentences based on keyword matching
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in query_keywords):
                relevant_info.append(sentence)
        
        # If no direct matches, use the first few sentences of context
        if not relevant_info and sentences:
            relevant_info = sentences[:3]
        
        # Construct answer from relevant information
        if relevant_info:
            answer = "Based on the financial documents, " + " ".join(relevant_info)
        else:
            answer = "I couldn't find specific information about that in the financial documents."
        
        logger.info("Response generated successfully")
        return answer
    
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while trying to answer your question. Please try again. Error: {str(e)}"
