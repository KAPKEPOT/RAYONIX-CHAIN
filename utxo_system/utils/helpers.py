# utxo_system/utils/helpers.py
import time
import re
from typing import Optional

def current_timestamp() -> int:
    """Get current timestamp in seconds"""
    return int(time.time())

def format_amount(amount: int, decimals: int = 8) -> str:
    """Format amount with specified decimal places"""
    return f"{amount / (10 ** decimals):.{decimals}f}"

def validate_address(address: str) -> bool:
    """
    Basic address validation.
    In a real implementation, this would validate against specific address formats.
    """
    if not address or not isinstance(address, str):
        return False
    
    # Basic length and character validation
    if len(address) < 26 or len(address) > 35:
        return False
    
    # Check for valid characters (alphanumeric, no confusing characters)
    if not re.match(r'^[1-9A-HJ-NP-Za-km-z]+$', address):
        return False
    
    return True