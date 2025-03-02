# Financial RAG Chatbot - Streamlit Cloud Deployment

This folder contains all necessary files for deploying the Financial RAG Chatbot to Streamlit Cloud.

## Deployment Instructions

1. Create a new Streamlit Cloud app
2. Connect to this GitHub repository
3. Set the main file path to `deploy_streamlit/streamlit_app.py`
4. Deploy the application

## Features

- Hybrid search (BM25 + Dense Embeddings)
- Open-source language models
- Financial document analysis
- Interactive query interface
- Confidence scoring

## Requirements

All dependencies are listed in the `requirements.txt` file and will be automatically installed during deployment.

## File Structure

- `streamlit_app.py`: Entry point for Streamlit Cloud
- `main.py`: Main application logic
- `app/`: Core application components
  - `financial_rag.py`: RAG system implementation
  - `hybrid_search.py`: Hybrid search logic
  - `preprocessing.py`: Document processing utilities
  - `llm_generator.py`: Response generation module
  - `guardrails.py`: Input/output security checks
  - `config.py`: Configuration parameters
- `.streamlit/`: Streamlit configuration files
- `setup.sh`: Setup script for deployment