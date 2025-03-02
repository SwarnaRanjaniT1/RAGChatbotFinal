import re
from typing import Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of patterns and keywords for input guardrail
INVALID_PATTERNS = [
    r'hack|exploit|illegal|porn|xxx|sex|nude|naked|steal|fraud|attack|bypass|crack|pirate'
]

# List of financial-specific keywords - queries should be related to these
FINANCIAL_KEYWORDS = [
    'revenue', 'profit', 'loss', 'income', 'expense', 'cost', 'margin', 'balance', 'asset',
    'liability', 'equity', 'cash', 'flow', 'dividend', 'stock', 'share', 'investment',
    'capital', 'tax', 'financial', 'fiscal', 'quarter', 'annual', 'report', 'statement',
    'budget', 'forecast', 'projection', 'growth', 'decline', 'increase', 'decrease',
    'performance', 'debt', 'credit', 'loan', 'interest', 'rate', 'payment', 'earnings',
    'EBITDA', 'gross', 'net', 'operating', 'liquidity', 'solvency', 'ratio', 'valuation',
    'depreciation', 'amortization', 'acquisition', 'merger', 'restructuring', 'guidance',
    'outlook', 'risk', 'audit', 'accounting', 'financial statement', 'balance sheet',
    'income statement', 'cash flow statement', 'annual report', 'quarterly report',
    'shareholder', 'stakeholder', 'investor', 'market', 'competition', 'regulation',
    'compliance', 'corporate', 'board', 'CEO', 'CFO', 'executive', 'management'
]

# List of signs of hallucination or inaccurate responses
HALLUCINATION_PATTERNS = [
    r'I do not have|I don\'t have|I cannot|I can\'t|no information|no data|not able to',
    r'specific details are not|details are not available|no specific information',
    r'it\'s unclear|it is unclear|it\'s not clear|it is not clear',
    r'I\'m unsure|I am unsure|I\'m not sure|I am not sure',
    r'without access to|I don\'t have access|I do not have access',
    r'need more information|need additional information|need further information',
    r'based on the information provided, I cannot',
    r'beyond my knowledge|beyond the information provided'
]

def check_input_guardrail(query: str) -> Tuple[bool, str]:
    """
    Check if the input query passes the guardrail.
    
    Args:
        query: The user's question
        
    Returns:
        Tuple of (is_valid, reason)
    """
    logger.info("Checking input guardrail")
    
    # Convert to lowercase for case-insensitive matching
    query_lower = query.lower()
    
    # Check for invalid patterns
    for pattern in INVALID_PATTERNS:
        if re.search(pattern, query_lower):
            reason = "This query contains inappropriate content."
            logger.warning(f"Input guardrail rejected query: {reason}")
            return False, reason
    
    # Check if query is too short
    if len(query.strip()) < 5:
        reason = "Your question is too short. Please provide a more detailed question."
        logger.warning(f"Input guardrail rejected query: {reason}")
        return False, reason
    
    # Check if query is related to financial domain
    has_financial_term = False
    for keyword in FINANCIAL_KEYWORDS:
        if keyword.lower() in query_lower:
            has_financial_term = True
            break
    
    # Allow some general questions about the company even if they don't have specific financial terms
    company_pattern = r'company|business|corporation|firm|enterprise|organization|entity'
    has_company_reference = re.search(company_pattern, query_lower) is not None
    
    if not has_financial_term and not has_company_reference:
        reason = "Please ask a question related to financial statements or company information."
        logger.warning(f"Input guardrail rejected query: {reason}")
        return False, reason
    
    # If we reach here, the query is valid
    logger.info("Input query passed guardrail check")
    return True, ""

def check_output_guardrail(response: str) -> Tuple[bool, str]:
    """
    Check if the output response passes the guardrail.
    
    Args:
        response: The generated response
        
    Returns:
        Tuple of (is_valid, reason)
    """
    logger.info("Checking output guardrail")
    
    # Convert to lowercase for case-insensitive matching
    response_lower = response.lower()
    
    # Check for signs of hallucination
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, response_lower):
            reason = "The response shows signs of uncertainty or lack of information."
            logger.warning(f"Output guardrail flagged response: {reason}")
            return False, reason
    
    # Check if response is too short
    if len(response.strip()) < 20:
        reason = "The generated response is too brief and may not be informative."
        logger.warning(f"Output guardrail flagged response: {reason}")
        return False, reason
    
    # If we reach here, the response is valid
    logger.info("Output response passed guardrail check")
    return True, ""
