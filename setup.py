"""
Setup script for the Financial RAG Chatbot application.
This is required for proper installation in the Streamlit Cloud environment.
"""

from setuptools import setup, find_packages

setup(
    name="financial-rag-chatbot",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "accelerate>=0.25.0",
        "faiss-cpu>=1.7.4",
        "nltk>=3.8.1",
        "numpy>=1.25.2",
        "pypdf2>=3.0.1",
        "rank-bm25>=0.2.2",
        "sentence-transformers>=2.2.2",
        "streamlit>=1.27.0",
        "torch>=2.0.1",
        "transformers>=4.34.0",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "financial-rag-chatbot=app.run:main",
        ],
    },
    author="Financial RAG Team",
    author_email="info@example.com",
    description="A financial chatbot using RAG technology",
    keywords="finance, rag, chatbot, nlp",
    project_urls={
        "Source Code": "https://github.com/yourusername/financial-rag-chatbot",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Topic :: Office/Business :: Financial",
    ],
)