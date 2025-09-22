"""
Advanced logging utilities for consensus system
"""

import logging
import logging.handlers
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time
from dataclasses import dataclass
import threading

from ..exceptions import ConsensusError

@dataclass
class LogRecord:
    """Enhanced log record with additional context"""
    timestamp: float
    level: str
    message: str
    module: str
    function: str
    line_no: int
    extra: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def __init__(self, include_extra: bool = True, include_context: bool = True):
        super().__init__()
        self.include_extra = include_extra
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            'timestamp': time.time(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_no': record.lineno,
            'logger': record.name
        }
        
        # Add extra fields if present
        if self.include_extra and hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add context information
        if self.include_context:
            log_data.update(self._get_context_info())
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)
    
    def _get_context_info(self) -> Dict[str, Any]:
        """Get context information from current thread"""
        return {
            'thread_id': threading.get_ident(),
            'thread_name': threading.current_thread().name
        }

class ContextFilter(logging.Filter):
    """Filter to add context information to log records"""
    
    def __init__(self, context: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.context = context or {}
        self.thread_local = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to log record"""
        # Add global context
        for key, value in self.context.items():
            setattr(record, key, value)
        
        # Add thread-local context
        if hasattr(self.thread_local, 'context'):
            for key, value in self.thread_local.context.items():
                setattr(record, key, value)
        
        return True
    
    def set_thread_context(self, **kwargs) -> None:
        """Set thread-local context"""
        if not hasattr(self.thread_local, 'context'):
            self.thread_local.context = {}
        self.thread_local.context.update(kwargs)
    
    def clear_thread_context(self) -> None:
        """Clear thread-local context"""
        if hasattr(self.thread_local, 'context'):
            self.thread_local.context.clear()

class LogManager:
    """Centralized log management"""
    
    def __init__(self, name: str = "consensus", level: str = "INFO", 
                 log_file: Optional[str] = None, max_size: int = 100 * 1024 * 1024,
                 backup_count: int = 5, compress: bool = True):
        self.name = name
        self.level = level
        self.log_file = log_file
        self.max_size = max_size
        self.backup_count = backup_count
        self.compress = compress
        
        self.logger = logging.getLogger(name)
        self.context_filter = ContextFilter()
        
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        # Set log level
        level = getattr(logging, self.level.upper(), logging.INFO)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler if log file specified
        if self.log_file:
            self._add_file_handler()
        
        # Add context filter
        self.logger.addFilter(self.context_filter)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def _add_file_handler(self) -> None:
        """Add file handler with rotation"""
        try:
            # Create log directory if it doesn't exist
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_size,
                backupCount=self.backup_count
            )
            
            file_handler.setLevel(self.logger.level)
            
            # Use JSON formatter for file logging
            json_formatter = JSONFormatter()
            file_handler.setFormatter(json_formatter)
            
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            raise ConsensusError(f"Failed to setup file logging: {e}")
    
    def set_level(self, level: str) -> None:
        """Set log level"""
        level_val = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(level_val)
        for handler in self.logger.handlers:
            handler.setLevel(level_val)
    
    def add_handler(self, handler: logging.Handler) -> None:
        """Add custom log handler"""
        self.logger.addHandler(handler)
    
    def set_global_context(self, **kwargs) -> None:
        """Set global context for all logs"""
        self.context_filter.context.update(kwargs)
    
    def set_thread_context(self, **kwargs) -> None:
        """Set thread-local context"""
        self.context_filter.set_thread_context(**kwargs)
    
    def clear_thread_context(self) -> None:
        """Clear thread-local context"""
        self.context_filter.clear_thread_context()
    
    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get logger instance"""
        if name:
            return logging.getLogger(f"{self.name}.{name}")
        return self.logger
    
    def enable_json_logging(self) -> None:
        """Enable JSON formatting for all handlers"""
        json_formatter = JSONFormatter()
        for handler in self.logger.handlers:
            handler.setFormatter(json_formatter)
    
    def disable_json_logging(self) -> None:
        """Disable JSON formatting"""
        text_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        for handler in self.logger.handlers:
            handler.setFormatter(text_formatter)
    
    def log_performance(self, operation: str, duration: float, 
                       extra: Optional[Dict] = None) -> None:
        """Log performance metrics"""
        log_data = {
            'operation': operation,
            'duration_ms': duration * 1000,
            'type': 'performance'
        }
        if extra:
            log_data.update(extra)
        
        self.logger.info(f"Performance: {operation} took {duration:.3f}s", extra=log_data)
    
    def log_audit(self, event: str, user: Optional[str] = None, 
                 details: Optional[Dict] = None) -> None:
        """Log audit events"""
        audit_data = {
            'event': event,
            'user': user,
            'type': 'audit'
        }
        if details:
            audit_data.update(details)
        
        self.logger.info(f"Audit: {event}", extra=audit_data)

def setup_logging(config: Dict) -> LogManager:
    """Setup logging from configuration"""
    try:
        return LogManager(
            name=config.get('name', 'consensus'),
            level=config.get('level', 'INFO'),
            log_file=config.get('file'),
            max_size=config.get('max_size', 100 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            compress=config.get('compress', True)
        )
    except Exception as e:
        raise ConsensusError(f"Failed to setup logging: {e}")

class RequestLogger:
    """Request logging middleware"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def log_request(self, request_id: str, method: str, path: str,
                         start_time: float, status: int, duration: float) -> None:
        """Log HTTP request"""
        self.logger.info(
            f"Request {request_id}: {method} {path} -> {status} ({duration:.3f}s)",
            extra={
                'request_id': request_id,
                'method': method,
                'path': path,
                'status': status,
                'duration_ms': duration * 1000,
                'type': 'http_request'
            }
        )
    
    def log_error(self, request_id: str, error: str, details: Optional[Dict] = None) -> None:
        """Log error with context"""
        error_data = {
            'request_id': request_id,
            'error': error,
            'type': 'error'
        }
        if details:
            error_data.update(details)
        
        self.logger.error(f"Error in request {request_id}: {error}", extra=error_data)