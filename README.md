# Financial RAG Chatbot

A financial document analysis chatbot using hybrid retrieval-augmented generation (RAG) with open-source models.

## Features

- **Hybrid Search** - Combines BM25 keyword search with dense vector embeddings
- **Open-Source Models** - Uses Sentence Transformers for embeddings and DistilBERT for analysis
- **Document Upload** - Upload and analyze your own financial statements
- **Interactive UI** - Clean Streamlit interface for easy interaction
- **Input/Output Guardrails** - Safety checks for user queries and generated responses

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   streamlit run main.py
   ```

3. Open in your browser:
   - The app will be available at http://localhost:8501

## Usage

1. Upload financial statements (PDF format) through the sidebar
2. Ask questions about financial information in natural language
3. View responses with confidence scores and source references

## Sample Questions

- What was the revenue in the last fiscal year?
- How did operating expenses change year-over-year?
- What are the main risk factors mentioned?
- What is the company's cash position?

## Project Structure

```
financial-rag-chatbot/
├── .streamlit/                # Streamlit configuration
├── app/                       # Application code
│   ├── config.py              # Configuration settings
│   ├── financial_rag.py       # Main RAG implementation
│   ├── guardrails.py          # Input/output safety checks  
│   ├── hybrid_search.py       # Hybrid search implementation
│   ├── llm_generator.py       # Language model generation
│   └── preprocessing.py       # Document preprocessing
├── data/                      # Data directory for documents
├── requirements.txt           # Project dependencies
└── main.py                    # Streamlit app entry point
```

## Technologies Used

- **Streamlit** - Web application framework
- **FAISS** - Vector database for similarity search
- **Sentence Transformers** - Text embeddings
- **Transformers** - Language models for text generation
- **PyPDF2** - PDF parsing
- **BM25** - Keyword-based retrieval algorithm