# utxo_system/utils/logging_config.py
import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure logging for the UTXO system"""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

# Default logger instance
logger = logging.getLogger('utxo_system')