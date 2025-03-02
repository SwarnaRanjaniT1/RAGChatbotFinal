#!/bin/bash

# Setup script for Streamlit Cloud deployment

# Make sure the script doesn't fail silently
set -e

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download NLTK data if needed
python -c "import nltk; nltk.download('punkt')"

# Create necessary directories if they don't exist
mkdir -p data

# Echo completion message
echo "Setup completed successfully!"