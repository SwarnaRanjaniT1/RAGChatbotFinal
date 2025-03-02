#!/usr/bin/env python3
"""
Entry point script for the Financial RAG Chatbot.
This can be used as a standalone script for execution.
"""

import sys
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    """Run the Financial RAG Chatbot application."""
    
    # Get the directory containing this script
    directory = Path(__file__).parent.absolute()
    
    # Set the streamlit app entry point
    sys.argv = [
        "streamlit", "run", 
        str(directory / "streamlit_app.py"),
        "--server.port=5000",
        "--server.address=0.0.0.0",
        "--server.headless=true"
    ]
    
    # Run the streamlit CLI
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()