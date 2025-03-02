import os
import re
import glob
from typing import List, Dict, Tuple, Any
import PyPDF2
import logging
from app.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple sentence tokenizer function to avoid NLTK issues
def simple_sent_tokenize(text):
    """
    A simple sentence tokenizer that splits text on common sentence boundaries.
    Works as a fallback when NLTK's tokenizer fails.
    """
    # Replace common abbreviations to avoid splitting sentences incorrectly
    text = re.sub(r'Mr\.', 'Mr', text)
    text = re.sub(r'Mrs\.', 'Mrs', text)
    text = re.sub(r'Ms\.', 'Ms', text)
    text = re.sub(r'Dr\.', 'Dr', text)
    text = re.sub(r'Inc\.', 'Inc', text)
    text = re.sub(r'Ltd\.', 'Ltd', text)
    text = re.sub(r'Corp\.', 'Corp', text)
    
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def preprocess_documents() -> List[Dict[str, Any]]:
    """
    Preprocess financial documents from the data directory.
    
    Returns:
        List of dictionaries with document text and metadata
    """
    logger.info("Starting document preprocessing")
    
    # Create data directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    processed_docs = []
    
    # Look for PDF files in the data directory
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found. Using sample financial data.")
        # Create a sample document with example financial data
        sample_doc = {
            "text": sample_financial_data(),
            "metadata": {
                "source": "Sample Financial Data",
                "year": "2022-2023",
                "type": "Annual Report"
            }
        }
        processed_docs.append(sample_doc)
    else:
        # Process each PDF file
        for pdf_file in pdf_files:
            filename = os.path.basename(pdf_file)
            logger.info(f"Processing PDF: {filename}")
            
            try:
                # Extract text from PDF
                text = extract_text_from_pdf(pdf_file)
                
                # Clean the text
                cleaned_text = clean_text(text)
                
                # Extract year from filename or content (simplistic approach)
                year_match = re.search(r'20\d{2}', filename)
                year = year_match.group(0) if year_match else "Unknown"
                
                doc = {
                    "text": cleaned_text,
                    "metadata": {
                        "source": filename,
                        "year": year,
                        "type": "Financial Statement"
                    }
                }
                processed_docs.append(doc)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
    
    logger.info(f"Preprocessed {len(processed_docs)} documents")
    return processed_docs

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def clean_text(text: str) -> str:
    """
    Clean extracted text from PDF.
    
    Args:
        text: Raw text from PDF
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Remove page numbers (simplistic approach)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    # Remove non-printable characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    return text.strip()

def chunk_text(documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Split documents into smaller, overlapping chunks for retrieval.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Tuple containing:
        - List of text chunks
        - List of metadata dictionaries for each chunk
    """
    chunks = []
    chunks_metadata = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        # Split text into sentences
        sentences = simple_sent_tokenize(text)
        
        # Create chunks with overlap
        current_chunk = []
        current_chunk_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed the chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            if current_chunk_size + sentence_size > CHUNK_SIZE and current_chunk:
                # Save the current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                chunks_metadata.append(metadata.copy())
                
                # Start a new chunk with overlap
                overlap_size = min(CHUNK_OVERLAP, len(current_chunk))
                current_chunk = current_chunk[-overlap_size:]
                current_chunk_size = sum(len(s) for s in current_chunk) + len(current_chunk) - 1
                
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_chunk_size += sentence_size + 1  # +1 for the space
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            chunks_metadata.append(metadata.copy())
    
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks, chunks_metadata

def sample_financial_data() -> str:
    """Generate sample financial data for testing when no real data is available."""
    return """
SAMPLE FINANCIAL STATEMENT
ACME CORPORATION
Annual Report 2022-2023

PART I - FINANCIAL INFORMATION

CONSOLIDATED STATEMENTS OF OPERATIONS
(In millions, except per share amounts)

                                    Year Ended         Year Ended
                                    Dec 31, 2023       Dec 31, 2022
Revenue                             $42,789            $38,542
Cost of revenue                     (22,450)           (20,121)
Gross profit                        20,339             18,421
Operating expenses:
  Research and development          (6,789)            (5,987)
  Sales and marketing               (4,932)            (4,562)
  General and administrative        (2,345)            (2,102)
Total operating expenses            (14,066)           (12,651)
Operating income                    6,273              5,770
Interest income                     587                342
Interest expense                    (245)              (198)
Other income, net                   132                89
Income before income taxes          6,747              6,003
Provision for income taxes          (1,349)            (1,201)
Net income                          $5,398             $4,802

Earnings per share:
  Basic                             $5.42              $4.86
  Diluted                           $5.37              $4.81

CONSOLIDATED BALANCE SHEETS
(In millions)
                                    Dec 31, 2023       Dec 31, 2022
ASSETS
Current assets:
  Cash and cash equivalents         $12,456            $10,234
  Short-term investments            8,765              7,543
  Accounts receivable, net          5,432              4,987
  Inventory                         2,345              2,123
  Other current assets              1,234              1,087
Total current assets                30,232             25,974

LIABILITIES AND STOCKHOLDERS' EQUITY
Current liabilities:
  Accounts payable                  $3,456             $3,234
  Accrued expenses                  2,345              2,123
  Deferred revenue                  1,234              1,087
Total current liabilities           7,035              6,444

MANAGEMENT'S DISCUSSION AND ANALYSIS

Business Overview
ACME Corporation is a leading provider of technology solutions for enterprise customers worldwide. Our products include cloud services, software applications, and hardware devices designed to improve business productivity and efficiency.

Financial Highlights
In 2023, we achieved record revenue of $42.8 billion, representing an 11% increase compared to 2022. Our operating income increased by 8.7% to $6.3 billion, while net income grew by 12.4% to $5.4 billion.

Revenue Growth
The increase in revenue was primarily driven by strong performance in our cloud services segment, which grew by 24% year-over-year to $18.5 billion. Our software applications segment contributed $15.3 billion, up 8% from the previous year, while our hardware segment generated $9.0 billion, representing a 2% increase.

Risk Factors
Our business is subject to numerous risks and uncertainties, including:
- Intense competition in all our markets
- Rapid technological change and product innovation
- Global economic conditions affecting customer spending
- Regulatory challenges in multiple jurisdictions
- Cybersecurity threats
- Supply chain disruptions
- Foreign currency exchange rate fluctuations

Liquidity and Capital Resources
As of December 31, 2023, we had $12.5 billion in cash and cash equivalents, compared to $10.2 billion at the end of 2022. Our strong cash position enables us to invest in growth opportunities, return capital to shareholders through dividends and share repurchases, and maintain financial flexibility.

Future Outlook
For fiscal year 2024, we expect continued revenue growth of 8-10%, driven by ongoing expansion in our cloud services segment. We anticipate operating margins to remain consistent with 2023 levels as we balance investments in new product development with operational efficiency initiatives.
"""

if __name__ == "__main__":
    # Test preprocessing
    docs = preprocess_documents()
    chunks, metadata = chunk_text(docs)
    print(f"Created {len(chunks)} chunks")
