"""
Streamlit Cloud entry point for the Financial RAG Chatbot application.
This is the main file that Streamlit Cloud will execute.
"""

import os
import sys
import logging
import streamlit as st
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path to enable imports
# This works both on local development and in Streamlit Cloud
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

# Import the main application
try:
    from app.financial_rag import FinancialRAG
    from app.guardrails import check_input_guardrail, check_output_guardrail
    logger.info("Successfully imported application modules")
except ImportError as e:
    logger.error(f"Failed to import application modules: {e}")
    st.error(f"Error importing application modules: {e}")
    sys.exit(1)

# App title and description
st.set_page_config(
    page_title="Financial RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_rag_system(force_reinit=False):
    """
    Load the RAG system with caching to prevent reloading on each interaction.
    
    Args:
        force_reinit: Whether to force reinitialization of the system
    
    Returns:
        Initialized FinancialRAG instance
    """
    logger.info("Loading RAG system...")
    
    if "rag_system" not in st.session_state or force_reinit:
        try:
            st.session_state.rag_system = FinancialRAG(force_reprocess=force_reinit)
            logger.info("RAG system loaded successfully")
        except Exception as e:
            logger.error(f"Error loading RAG system: {e}")
            st.error(f"Error loading RAG system: {e}")
            raise e
    
    return st.session_state.rag_system

def main():
    """Main application function"""
    logger.info("Starting Financial RAG Chatbot application")
    
    # Header
    st.title("📊 Financial RAG Chatbot")
    st.markdown("""
    Ask questions about financial statements and reports using advanced RAG technology.
    This system combines keyword-based and semantic search for accurate answers.
    """)
    
    # Sidebar with app information
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This application uses:
        - Hybrid search (BM25 + Dense Embeddings)
        - Open-source language models
        - Financial document analysis
        """)
        
        # Force reprocessing button (use with caution)
        if st.button("Reload Documents"):
            st.session_state.pop("rag_system", None)
            st.success("System reset! Documents will be reprocessed.")
            st.experimental_rerun()
    
    # Initialize the RAG system
    rag_system = load_rag_system()
    
    # Chat input and history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (question, answer, confidence, sources) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(question)
        
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            # Display confidence with colored indicator
            if confidence > 0.8:
                confidence_color = "green"
            elif confidence > 0.5:
                confidence_color = "orange"
            else:
                confidence_color = "red"
            
            st.markdown(f"<span style='color:{confidence_color}'>Confidence: {confidence:.2f}</span>", unsafe_allow_html=True)
            
            # Show sources if available
            if sources:
                with st.expander("Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
    
    # Input for user question
    if question := st.chat_input("Ask about financial topics..."):
        with st.chat_message("user"):
            st.markdown(question)
        
        # Check input guardrails
        is_valid, reason = check_input_guardrail(question)
        if not is_valid:
            with st.chat_message("assistant"):
                st.markdown(f"⚠️ {reason}")
            return
        
        # Show a spinner while generating response
        with st.spinner("Thinking..."):
            try:
                # Get answer from RAG system
                answer, confidence, sources = rag_system.answer_question(question)
                
                # Check output guardrails
                is_valid_output, reason_output = check_output_guardrail(answer)
                if not is_valid_output:
                    answer = "I cannot provide a response to that question. " + reason_output
                    confidence = 0.0
                
                # Save to chat history
                st.session_state.chat_history.append((question, answer, confidence, sources))
                
                # Display the response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    # Display confidence with colored indicator
                    if confidence > 0.8:
                        confidence_color = "green"
                    elif confidence > 0.5:
                        confidence_color = "orange"
                    else:
                        confidence_color = "red"
                    
                    st.markdown(f"<span style='color:{confidence_color}'>Confidence: {confidence:.2f}</span>", unsafe_allow_html=True)
                    
                    # Show sources if available
                    if sources:
                        with st.expander("Sources"):
                            for source in sources:
                                st.markdown(f"- {source}")
            
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                with st.chat_message("assistant"):
                    st.error(f"I encountered an error: {str(e)}")

if __name__ == "__main__":
    main()