import logging
import logging.handlers
import os
from typing import Optional

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
    """Setup logging configuration"""
    # Convert log level string to logging level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """Get logger with specific name"""
    return logging.getLogger(name)

class WalletLogger:
    """Enhanced logger for wallet operations"""
    
    def __init__(self, name: str):
        self.logger = get_logger(name)
        self.sensitive_data = set()
    
    def add_sensitive_data(self, data: str):
        """Add sensitive data to be masked in logs"""
        if data and len(data) > 4:
            self.sensitive_data.add(data)
    
    def remove_sensitive_data(self, data: str):
        """Remove sensitive data from masking"""
        self.sensitive_data.discard(data)
    
    def _mask_sensitive_data(self, message: str) -> str:
        """Mask sensitive data in log messages"""
        for sensitive in self.sensitive_data:
            if sensitive in message:
                masked = sensitive[:4] + '*' * (len(sensitive) - 4)
                message = message.replace(sensitive, masked)
        return message
    
    def debug(self, msg: str, *args, **kwargs):
        """Debug level logging with sensitive data masking"""
        masked_msg = self._mask_sensitive_data(msg)
        self.logger.debug(masked_msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Info level logging with sensitive data masking"""
        masked_msg = self._mask_sensitive_data(msg)
        self.logger.info(masked_msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Warning level logging with sensitive data masking"""
        masked_msg = self._mask_sensitive_data(msg)
        self.logger.warning(masked_msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Error level logging with sensitive data masking"""
        masked_msg = self._mask_sensitive_data(msg)
        self.logger.error(masked_msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Critical level logging with sensitive data masking"""
        masked_msg = self._mask_sensitive_data(msg)
        self.logger.critical(masked_msg, *args, **kwargs)
    
    def audit(self, event: str, details: dict):
        """Audit logging for security events"""
        import json
        audit_details = {
            'event': event,
            'timestamp': time.time(),
            'details': details
        }
        self.info(f"AUDIT: {json.dumps(audit_details)}")

# Global logger instance
logger = WalletLogger("rayonix_wallet")