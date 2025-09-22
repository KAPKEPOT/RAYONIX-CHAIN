"""
Advanced error handling utilities for consensus system
"""

import time
from typing import Optional, Dict, Any, Callable, Type, TypeVar
from dataclasses import dataclass
import logging
from functools import wraps
import inspect
from contextlib import contextmanager

from ..exceptions import ConsensusError, ValidationError, CryptoError, NetworkError

logger = logging.getLogger('consensus.utils')

T = TypeVar('T')

@dataclass
class ErrorContext:
    """Context information for error handling"""
    timestamp: float
    operation: str
    module: str
    function: str
    parameters: Dict[str, Any]
    retry_count: int = 0
    correlation_id: Optional[str] = None
    extra: Dict[str, Any] = None

class ErrorHandler:
    """Advanced error handling with retry logic and circuit breaking"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0,
                 circuit_breaker_threshold: int = 5, circuit_breaker_timeout: float = 60.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, float] = {}
        self.lock = threading.RLock()
    
    def handle_operation(self, operation: str, func: Callable[..., T], 
                        *args, **kwargs) -> Optional[T]:
        """Execute operation with error handling"""
        for attempt in range(self.max_retries + 1):
            try:
                # Check circuit breaker
                if self._is_circuit_open(operation):
                    raise ConsensusError(f"Circuit breaker open for {operation}")
                
                # Execute operation
                result = func(*args, **kwargs)
                
                # Reset error count on success
                self._reset_error_count(operation)
                return result
                
            except Exception as e:
                # Log error
                self._log_error(operation, e, attempt)
                
                # Update error count
                self._increment_error_count(operation)
                
                # Check if we should retry
                if attempt == self.max_retries or not self._should_retry(e):
                    raise
                
                # Wait before retry
                time.sleep(self.retry_delay * (2 ** attempt))
        
        return None
    
    @contextmanager
    def handle_context(self, operation: str):
        """Context manager for error handling"""
        try:
            yield
            self._reset_error_count(operation)
        except Exception as e:
            self._log_error(operation, e, 0)
            self._increment_error_count(operation)
            raise
    
    def _should_retry(self, error: Exception) -> bool:
        """Determine if error should be retried"""
        # Don't retry validation errors
        if isinstance(error, ValidationError):
            return False
        
        # Don't retry certain crypto errors
        if isinstance(error, CryptoError) and "invalid signature" in str(error).lower():
            return False
        
        # Retry network errors and timeouts
        if isinstance(error, NetworkError) or "timeout" in str(error).lower():
            return True
        
        # Default: retry for other errors
        return True
    
    def _increment_error_count(self, operation: str) -> None:
        """Increment error count for operation"""
        with self.lock:
            self.error_counts[operation] = self.error_counts.get(operation, 0) + 1
            
            # Check if circuit should be opened
            if self.error_counts[operation] >= self.circuit_breaker_threshold:
                self.circuit_breakers[operation] = time.time()
                logger.warning(f"Circuit breaker opened for {operation}")
    
    def _reset_error_count(self, operation: str) -> None:
        """Reset error count for operation"""
        with self.lock:
            self.error_counts.pop(operation, None)
            self.circuit_breakers.pop(operation, None)
    
    def _is_circuit_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for operation"""
        with self.lock:
            open_time = self.circuit_breakers.get(operation)
            if open_time is None:
                return False
            
            # Check if circuit should be half-open
            if time.time() - open_time >= self.circuit_breaker_timeout:
                self.circuit_breakers.pop(operation)
                return False
            
            return True
    
    def _log_error(self, operation: str, error: Exception, attempt: int) -> None:
        """Log error with context"""
        error_context = {
            'operation': operation,
            'error_type': type(error).__name__,
            'attempt': attempt,
            'max_retries': self.max_retries,
            'should_retry': self._should_retry(error)
        }
        
        logger.error(f"Error in {operation} (attempt {attempt}): {error}", extra=error_context)
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        with self.lock:
            return self.error_counts.copy()
    
    def get_circuit_status(self) -> Dict[str, bool]:
        """Get circuit breaker status"""
        with self.lock:
            return {
                op: time.time() - open_time < self.circuit_breaker_timeout
                for op, open_time in self.circuit_breakers.items()
            }

def retry_on_error(max_retries: int = 3, retry_delay: float = 1.0):
    """Decorator for retrying operations on error"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            handler = ErrorHandler(max_retries, retry_delay)
            return handler.handle_operation(func.__name__, func, *args, **kwargs)
        return wrapper
    return decorator

def handle_errors(default: Any = None, log_errors: bool = True):
    """Decorator for handling errors with default return value"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {e}")
                return default
        return wrapper
    return decorator

def validate_arguments(validators: Dict[str, Callable[[Any], bool]] = None):
    """Decorator for argument validation"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate arguments
            if validators:
                for param_name, validator in validators.items():
                    if param_name in bound_args.arguments:
                        value = bound_args.arguments[param_name]
                        if not validator(value):
                            raise ValidationError(f"Invalid argument: {param_name}={value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class ErrorReporter:
    """Error reporting and monitoring"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reported_errors: Dict[str, int] = {}
    
    def report_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Report error for monitoring"""
        if not self.enabled:
            return
        
        error_key = f"{type(error).__name__}:{str(error)}"
        self.reported_errors[error_key] = self.reported_errors.get(error_key, 0) + 1
        
        # Log error with context
        log_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'occurrences': self.reported_errors[error_key],
            'type': 'error_report'
        }
        if context:
            log_data.update(context)
        
        logger.error(f"Error reported: {error}", extra=log_data)
    
    def get_error_stats(self) -> Dict[str, int]:
        """Get error statistics"""
        return self.reported_errors.copy()
    
    def clear_stats(self) -> None:
        """Clear error statistics"""
        self.reported_errors.clear()

@contextmanager
def error_context(operation: str, **context):
    """Context manager for error context"""
    try:
        yield
    except Exception as e:
        # Enhance error with context
        error_msg = f"{operation} failed: {e}"
        raise type(e)(error_msg) from e

def create_error_response(error: Exception, status_code: int = 500) -> Dict:
    """Create standardized error response"""
    return {
        'error': {
            'code': status_code,
            'message': str(error),
            'type': type(error).__name__,
            'timestamp': time.time()
        },
        'success': False
    }