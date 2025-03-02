import streamlit as st
import os
import traceback
import sys
from app.financial_rag import FinancialRAG
from app.guardrails import check_input_guardrail, check_output_guardrail

# Set up basic logging to console
import logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

logger.info("Starting Financial RAG Chatbot application")

# Initialize RAG system (with option to force reinitialization)
def load_rag_system(force_reinit=False):
    # Initialize session state to track RAG initialization
    if "rag_initialized" not in st.session_state:
        st.session_state.rag_initialized = False
    
    # Check if we need to reinitialize the system
    if not st.session_state.rag_initialized or force_reinit:
        try:
            logger.info("Loading RAG system...")
            # Use a placeholder in session state for the RAG system
            if "rag_system" not in st.session_state or force_reinit:
                st.session_state.rag_system = FinancialRAG(force_reprocess=force_reinit)
            
            logger.info("RAG system loaded successfully")
            st.session_state.rag_initialized = True
            return st.session_state.rag_system
        except Exception as e:
            error_msg = f"Error loading RAG system: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            st.session_state.rag_initialized = False
            return None
    else:
        # Return the cached instance
        return st.session_state.rag_system

def main():
    try:
        # Set page configuration
        st.set_page_config(
            page_title="Financial RAG Chatbot",
            page_icon="💰",
            layout="wide",
        )
        
        # Initialize session state for chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Application title and description
        st.title("💰 Financial RAG Chatbot")
        st.markdown("""
        This chatbot answers questions about company financial statements using a hybrid search 
        approach combining keyword-based retrieval (BM25) and semantic search (embeddings).
        """)

        # Sidebar with information and file upload
        with st.sidebar:
            st.header("About")
            st.markdown("""
            This chatbot uses:
            - Hybrid Search (BM25 + Dense Embeddings)
            - Open-source Sentence Transformers for embeddings
            - Small Language Model from HuggingFace for generation
            - Input guardrails to filter harmful or irrelevant queries
            """)
            
            # File upload section
            st.header("Upload Financial Statements")
            st.markdown("""
            Upload your own financial statements in PDF format for analysis.
            The system will process them and use them to answer your questions.
            """)
            
            uploaded_files = st.file_uploader(
                "Upload financial statements (PDF)", 
                type=["pdf"],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                if st.button("Process Uploaded Files"):
                    with st.spinner("Processing financial statements..."):
                        try:
                            # Create data directory if it doesn't exist
                            import os
                            from app.config import DATA_DIR
                            os.makedirs(DATA_DIR, exist_ok=True)
                            
                            # Save uploaded files to data directory
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(DATA_DIR, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                            
                            # Reinitialize RAG system to process new files with force_reprocess
                            st.session_state.rag_initialized = False
                            rag_system = load_rag_system(force_reinit=True)
                            
                            st.success(f"Successfully processed {len(uploaded_files)} financial statement(s).")
                        except Exception as e:
                            st.error(f"Error processing files: {str(e)}")
            
            st.header("Sample Questions")
            st.markdown("""
            Try asking:
            - What was the revenue in the last fiscal year?
            - How did operating expenses change year-over-year?
            - What are the main risk factors mentioned?
            """)

        # Load the RAG system
        rag_system = load_rag_system()

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input field for new question
        query = st.chat_input("Ask a question about the financial statements")

        # Process the query when submitted
        if query:
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Show thinking indicator
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # Apply input guardrail
                is_valid, reason = check_input_guardrail(query)
                
                if not is_valid:
                    response = f"I'm unable to answer this question. {reason}"
                    confidence = 0.0
                    sources = []
                else:
                    # Check if rag_system is initialized properly
                    if rag_system is None:
                        response = "The RAG system failed to initialize. Please check the logs for more information."
                        confidence = 0.0
                        sources = []
                    else:
                        # Get response from RAG system
                        try:
                            response, confidence, sources = rag_system.answer_question(query)
                            
                            # Apply output guardrail
                            is_valid_output, reason_output = check_output_guardrail(response)
                            if not is_valid_output:
                                response = f"I found an answer but it may not be reliable. {reason_output}"
                                confidence = max(confidence * 0.5, 0.1)  # Reduce confidence
                        except Exception as e:
                            logger.error(f"Error generating response: {str(e)}")
                            logger.error(traceback.format_exc())
                            response = f"An error occurred while processing your question: {str(e)}"
                            confidence = 0.0
                            sources = []
                
                # Format the response with confidence score and sources
                formatted_response = f"{response}\n\n**Confidence Score:** {confidence:.2f}"
                
                if sources and len(sources) > 0:
                    formatted_response += "\n\n**Sources:**"
                    for i, source in enumerate(sources, 1):
                        formatted_response += f"\n{i}. {source}"
                
                # Update the message placeholder with the full response
                message_placeholder.markdown(formatted_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the logs for more information.")

# Run the main application
if __name__ == "__main__":
    main()