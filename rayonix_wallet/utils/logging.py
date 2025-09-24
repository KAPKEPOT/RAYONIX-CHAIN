import logging
import logging.handlers
import logging.config
import os
import time
import json
import sys
import threading
from typing import Optional, Dict, List, Any, Union, Callable
from enum import Enum
from pathlib import Path
import traceback
import inspect
from datetime import datetime
import gzip
import io

class LogLevel(Enum):
    """Log levels enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """Log format types"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    GELF = "gelf"

class LogRotationStrategy(Enum):
    """Log rotation strategies"""
    SIZE_BASED = "size"
    TIME_BASED = "time"
    SIZE_AND_TIME = "size_time"

class AuditEventType(Enum):
    """Audit event types"""
    WALLET_CREATED = "wallet_created"
    WALLET_ACCESSED = "wallet_accessed"
    TRANSACTION_SENT = "transaction_sent"
    TRANSACTION_RECEIVED = "transaction_received"
    BACKUP_CREATED = "backup_created"
    BACKUP_RESTORED = "backup_restored"
    SECURITY_EVENT = "security_event"
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILED = "auth_failed"

class StructuredFormatter(logging.Formatter):
    """Advanced structured log formatter"""
    
    def __init__(self, fmt_type: LogFormat = LogFormat.DETAILED, include_context: bool = True):
        self.fmt_type = fmt_type
        self.include_context = include_context
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data"""
        try:
            # Extract structured data if present
            structured_data = getattr(record, 'structured_data', {})
            
            if self.fmt_type == LogFormat.JSON:
                return self._format_json(record, structured_data)
            elif self.fmt_type == LogFormat.GELF:
                return self._format_gelf(record, structured_data)
            else:
                return self._format_text(record, structured_data)
                
        except Exception as e:
            return f"Log formatting error: {str(e)} - {record.getMessage()}"
    
    def _format_json(self, record: logging.LogRecord, structured_data: Dict) -> str:
        """Format as JSON"""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName or f"Thread-{record.thread}",
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        if structured_data:
            log_entry["data"] = structured_data
        
        return json.dumps(log_entry, default=str)
    
    def _format_gelf(self, record: logging.LogRecord, structured_data: Dict) -> str:
        """Format as GELF (Graylog Extended Log Format)"""
        gelf_data = {
            "version": "1.1",
            "host": os.getenv('HOSTNAME', 'unknown'),
            "short_message": record.getMessage(),
            "full_message": self._get_full_message(record),
            "timestamp": record.created,
            "level": self._gelf_level(record.levelno),
            "_logger": record.name,
            "_module": record.module,
            "_function": record.funcName,
            "_line": record.lineno,
        }
        
        # Add structured data as GELF fields
        for key, value in structured_data.items():
            gelf_data[f"_{key}"] = value
        
        return json.dumps(gelf_data, default=str)
    
    def _format_text(self, record: logging.LogRecord, structured_data: Dict) -> str:
        """Format as detailed text"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        base_msg = f"{timestamp} | {record.levelname:8} | {record.name} | {record.getMessage()}"
        
        if self.include_context and structured_data:
            base_msg += f" | {json.dumps(structured_data, default=str)}"
        
        if record.exc_info:
            base_msg += f"\n{self.formatException(record.exc_info)}"
        
        return base_msg
    
    def _gelf_level(self, levelno: int) -> int:
        """Convert Python log levels to GELF levels"""
        mapping = {
            logging.DEBUG: 7,
            logging.INFO: 6,
            logging.WARNING: 4,
            logging.ERROR: 3,
            logging.CRITICAL: 2
        }
        return mapping.get(levelno, 6)
    
    def _get_full_message(self, record: logging.LogRecord) -> str:
        """Get full message including exception info"""
        msg = record.getMessage()
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        return msg

class SensitiveDataFilter(logging.Filter):
    """Filter to mask sensitive data in logs"""
    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = set()
        self.masking_enabled = True
    
    def add_sensitive_pattern(self, pattern: str):
        """Add sensitive data pattern to mask"""
        if pattern and len(pattern) > 4:
            self.sensitive_patterns.add(pattern)
    
    def remove_sensitive_pattern(self, pattern: str):
        """Remove sensitive data pattern"""
        self.sensitive_patterns.discard(pattern)
    
    def disable_masking(self):
        """Temporarily disable masking"""
        self.masking_enabled = False
    
    def enable_masking(self):
        """Enable masking"""
        self.masking_enabled = True
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Mask sensitive data in log records"""
        if not self.masking_enabled:
            return True
            
        try:
            # Mask in message
            if hasattr(record, 'msg') and record.msg:
                record.msg = self._mask_data(record.msg)
            
            # Mask in structured data
            if hasattr(record, 'structured_data'):
                record.structured_data = self._mask_structured_data(record.structured_data)
            
            # Mask in args
            if hasattr(record, 'args') and record.args:
                record.args = tuple(self._mask_data(str(arg)) if isinstance(arg, str) else arg 
                                  for arg in record.args)
        
        except Exception:
            # Don't let masking errors break logging
            pass
        
        return True
    
    def _mask_data(self, text: str) -> str:
        """Mask sensitive data in text"""
        if not isinstance(text, str):
            return text
            
        masked_text = text
        for pattern in self.sensitive_patterns:
            if pattern in masked_text:
                masked = pattern[:4] + '*' * (len(pattern) - 4)
                masked_text = masked_text.replace(pattern, masked)
        
        return masked_text
    
    def _mask_structured_data(self, data: Any) -> Any:
        """Recursively mask sensitive data in structured data"""
        if isinstance(data, dict):
            return {k: self._mask_structured_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._mask_structured_data(item) for item in data]
        elif isinstance(data, str):
            return self._mask_data(data)
        else:
            return data

class LogManager:
    """Advanced log management system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.initialized = False
        self.loggers = {}
        self.handlers = {}
        self.sensitive_filter = SensitiveDataFilter()
        self.audit_handlers = []
        self.metrics_handlers = []
        
        # Default configuration
        self.config = {
            'log_level': LogLevel.INFO,
            'log_format': LogFormat.DETAILED,
            'rotation_strategy': LogRotationStrategy.SIZE_BASED,
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'backup_count': 10,
            'rotation_interval': 'D',  # Daily
            'compress_backups': True,
            'enable_console': True,
            'enable_file': True,
            'enable_syslog': False,
            'syslog_address': '/dev/log',
            'enable_metrics': False,
        }
        
        self._initialized = True
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure logging system"""
        try:
            self.config.update(config)
            self._setup_logging_system()
        except Exception as e:
            print(f"Logging configuration failed: {e}", file=sys.stderr)
            raise
    
    def _setup_logging_system(self) -> None:
        """Setup complete logging system"""
        # Clear existing configuration
        logging.root.handlers.clear()
        
        # Basic configuration
        level = getattr(logging, self.config['log_level'].value)
        logging.root.setLevel(level)
        
        # Create formatter
        formatter = StructuredFormatter(self.config['log_format'])
        
        # Console handler
        if self.config['enable_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(self.sensitive_filter)
            logging.root.addHandler(console_handler)
        
        # File handler
        if self.config['enable_file'] and self.config.get('log_file'):
            file_handler = self._create_file_handler()
            file_handler.setFormatter(formatter)
            file_handler.addFilter(self.sensitive_filter)
            logging.root.addHandler(file_handler)
        
        # Syslog handler
        if self.config['enable_syslog']:
            syslog_handler = self._create_syslog_handler()
            syslog_handler.setFormatter(formatter)
            syslog_handler.addFilter(self.sensitive_filter)
            logging.root.addHandler(syslog_handler)
        
        # Configure third-party loggers
        self._configure_third_party_loggers()
        
        self.initialized = True
    
    def _create_file_handler(self) -> logging.Handler:
        """Create appropriate file handler based on rotation strategy"""
        log_file = self.config['log_file']
        log_dir = os.path.dirname(log_file)
        
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        strategy = self.config['rotation_strategy']
        
        if strategy == LogRotationStrategy.TIME_BASED:
            handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when=self.config['rotation_interval'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        elif strategy == LogRotationStrategy.SIZE_AND_TIME:
            handler = ConcurrentRotatingFileHandler(
                log_file,
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        else:  # SIZE_BASED
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
        
        return handler
    
    def _create_syslog_handler(self) -> logging.Handler:
        """Create syslog handler"""
        try:
            if os.path.exists(self.config['syslog_address']):
                handler = logging.handlers.SysLogHandler(
                    address=self.config['syslog_address']
                )
            else:
                handler = logging.handlers.SysLogHandler()
            
            return handler
        except Exception:
            # Fallback to basic syslog
            return logging.handlers.SysLogHandler()
    
    def _configure_third_party_loggers(self) -> None:
        """Configure log levels for third-party libraries"""
        noisy_libraries = {
            "urllib3": logging.WARNING,
            "requests": logging.WARNING,
            "asyncio": logging.WARNING,
            "botocore": logging.WARNING,
            "boto3": logging.WARNING,
            "azure": logging.WARNING,
            "google": logging.WARNING,
        }
        
        for lib, level in noisy_libraries.items():
            logging.getLogger(lib).setLevel(level)
    
    def get_logger(self, name: str, **kwargs) -> 'AdvancedLogger':
        """Get advanced logger instance"""
        if name not in self.loggers:
            self.loggers[name] = AdvancedLogger(name, **kwargs)
        
        return self.loggers[name]
    
    def add_audit_handler(self, handler: Callable[[Dict], None]) -> None:
        """Add custom audit event handler"""
        self.audit_handlers.append(handler)
    
    def add_metrics_handler(self, handler: Callable[[Dict], None]) -> None:
        """Add custom metrics handler"""
        self.metrics_handlers.append(handler)

class AdvancedLogger:
    """Advanced logger with structured logging and audit capabilities"""
    
    def __init__(self, name: str, capture_caller: bool = True):
        self.name = name
        self.logger = logging.getLogger(name)
        self.capture_caller = capture_caller
        self.sensitive_data = set()
        self.context_data = {}
        self.audit_enabled = True
        
        # Get sensitive filter from log manager
        self.sensitive_filter = LogManager().sensitive_filter
    
    def add_sensitive_data(self, data: str) -> None:
        """Add sensitive data for masking"""
        if data:
            self.sensitive_data.add(data)
            self.sensitive_filter.add_sensitive_pattern(data)
    
    def remove_sensitive_data(self, data: str) -> None:
        """Remove sensitive data from masking"""
        self.sensitive_data.discard(data)
        self.sensitive_filter.remove_sensitive_pattern(data)
    
    def set_context(self, **context) -> None:
        """Set contextual information for all subsequent logs"""
        self.context_data.update(context)
    
    def clear_context(self) -> None:
        """Clear contextual information"""
        self.context_data.clear()
    
    def _log_with_structure(self, level: int, msg: str, **kwargs) -> None:
        """Log with structured data"""
        try:
            # Prepare structured data
            structured_data = self.context_data.copy()
            structured_data.update(kwargs)
            
            # Capture caller info if requested
            if self.capture_caller:
                frame = inspect.currentframe().f_back.f_back  # Go back two frames
                caller_info = inspect.getframeinfo(frame)
                structured_data.update({
                    'caller_file': caller_info.filename,
                    'caller_line': caller_info.lineno,
                    'caller_function': caller_info.function
                })
            
            # Create log record with structured data
            extra = {'structured_data': structured_data}
            self.logger.log(level, msg, extra=extra)
            
        except Exception as e:
            # Fallback to basic logging if structured logging fails
            self.logger.log(level, f"{msg} - Structured logging failed: {e}")
    
    def debug(self, msg: str, **kwargs) -> None:
        """Debug level logging with structured data"""
        self._log_with_structure(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs) -> None:
        """Info level logging with structured data"""
        self._log_with_structure(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs) -> None:
        """Warning level logging with structured data"""
        self._log_with_structure(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs) -> None:
        """Error level logging with structured data"""
        self._log_with_structure(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs) -> None:
        """Critical level logging with structured data"""
        self._log_with_structure(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs) -> None:
        """Exception logging with traceback"""
        kwargs['exception_info'] = traceback.format_exc()
        self._log_with_structure(logging.ERROR, msg, **kwargs)
    
    def audit(self, event_type: AuditEventType, **details) -> None:
        """Audit logging for security events"""
        if not self.audit_enabled:
            return
            
        audit_data = {
            'event_type': event_type.value,
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': details.pop('user_id', 'unknown'),
            'session_id': details.pop('session_id', 'unknown'),
            'ip_address': details.pop('ip_address', 'unknown'),
            'user_agent': details.pop('user_agent', 'unknown'),
            'details': details,
            'outcome': details.pop('outcome', 'success'),
        }
        
        # Call audit handlers
        log_manager = LogManager()
        for handler in log_manager.audit_handlers:
            try:
                handler(audit_data)
            except Exception as e:
                self.error(f"Audit handler failed: {e}")
        
        self.info(f"AUDIT: {event_type.value}", **audit_data)
    
    def metric(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Log metrics data"""
        metric_data = {
            'metric_name': name,
            'metric_value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        }
        
        # Call metrics handlers
        log_manager = LogManager()
        for handler in log_manager.metrics_handlers:
            try:
                handler(metric_data)
            except Exception as e:
                self.error(f"Metrics handler failed: {e}")
        
        self.debug(f"METRIC: {name}", **metric_data)
    
    def log_performance(self, operation: str, duration: float, **context) -> None:
        """Log performance metrics"""
        self.metric(f"performance.{operation}", duration, {
            'operation': operation,
            'duration_ms': str(duration * 1000)
        })

class ConcurrentRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Thread-safe rotating file handler"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()
    
    def emit(self, record):
        with self.lock:
            super().emit(record)

# Utility functions for backward compatibility
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, 
                 max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5) -> None:
    """Backward compatibility setup function"""
    config = {
        'log_level': LogLevel(log_level.upper()),
        'log_file': log_file,
        'max_file_size': max_bytes,
        'backup_count': backup_count,
        'enable_file': log_file is not None,
    }
    
    LogManager().configure(config)

def get_logger(name: str) -> AdvancedLogger:
    """Backward compatibility logger getter"""
    return LogManager().get_logger(name)

class WalletLogger(AdvancedLogger):
    """Enhanced wallet logger with additional security features"""
    
    def __init__(self, name: str = "rayonix_wallet"):
        super().__init__(name, capture_caller=True)
        self.security_events = []
        self.max_security_events = 1000  # Prevent memory leaks
    
    def log_security_event(self, event: str, severity: str = "medium", **details):
        """Log security event with severity rating"""
        security_data = {
            'event': event,
            'severity': severity,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details
        }
        
        # Store for analysis (rotating buffer)
        self.security_events.append(security_data)
        if len(self.security_events) > self.max_security_events:
            self.security_events.pop(0)
        
        self.audit(AuditEventType.SECURITY_EVENT, 
                  security_event=event, 
                  severity=severity, 
                  **details)
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security events report"""
        return {
            'total_events': len(self.security_events),
            'events_by_severity': self._count_events_by_severity(),
            'recent_events': self.security_events[-100:],  # Last 100 events
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _count_events_by_severity(self) -> Dict[str, int]:
        """Count events by severity level"""
        counts = {}
        for event in self.security_events:
            severity = event.get('severity', 'unknown')
            counts[severity] = counts.get(severity, 0) + 1
        return counts

# Global logger instance for backward compatibility
logger = WalletLogger("rayonix_wallet")

# Context manager for temporary context
class LoggingContext:
    """Context manager for temporary logging context"""
    
    def __init__(self, logger: AdvancedLogger, **context):
        self.logger = logger
        self.context = context
        self.previous_context = {}
    
    def __enter__(self):
        self.previous_context = self.logger.context_data.copy()
        self.logger.set_context(**self.context)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.clear_context()
        self.logger.set_context(**self.previous_context)