# utils/helpers.py - Common helper functions

import os
import sys
import json
import time
import logging
import asyncio
import hashlib
import secrets
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

def configure_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Configure logging for the RAYONIX node
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Basic configuration
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # File handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Apply configuration
    logging.getLogger().handlers = handlers
    
    # Set specific log levels for noisy libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

def generate_id(length: int = 16) -> str:
    """
    Generate a random ID string
    """
    return secrets.token_hex(length)

def calculate_hash(data: Any) -> str:
    """
    Calculate SHA-256 hash of data
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    return hashlib.sha256(data_str.encode()).hexdigest()

def format_timestamp(timestamp: Union[int, float]) -> str:
    """
    Format timestamp to human-readable string
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

def parse_timestamp(timestamp_str: str) -> Optional[float]:
    """
    Parse human-readable timestamp to Unix timestamp
    """
    try:
        time_struct = time.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        return time.mktime(time_struct)
    except ValueError:
        return None

def format_bytes(size: int) -> str:
    """
    Format bytes to human-readable string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def parse_bytes(size_str: str) -> Optional[int]:
    """
    Parse human-readable size string to bytes
    """
    units = {
        'B': 1,
        'KB': 1024,
        'MB': 1024**2,
        'GB': 1024**3,
        'TB': 1024**4,
        'PB': 1024**5
    }
    
    try:
        number = float(''.join(filter(lambda x: x.isdigit() or x == '.', size_str)))
        unit = ''.join(filter(lambda x: x.isalpha(), size_str)).upper()
        
        if unit in units:
            return int(number * units[unit])
        else:
            return int(number)  # Assume bytes if no unit specified
    except ValueError:
        return None

def ensure_directory(directory: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except OSError:
        return False

def read_json_file(file_path: str) -> Optional[Dict]:
    """
    Read JSON data from file
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return None

def write_json_file(file_path: str, data: Any) -> bool:
    """
    Write JSON data to file
    """
    try:
        ensure_directory(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except IOError:
        return False

def async_retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator for retrying async functions
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    
                    logging.warning(f"Attempt {attempts} failed: {e}. Retrying in {current_delay}s...")
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            
            return None
        return wrapper
    return decorator

def timeout(seconds: float):
    """
    Decorator for adding timeout to async functions
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                logging.error(f"Function {func.__name__} timed out after {seconds} seconds")
                raise
        return wrapper
    return decorator

def rate_limit(max_calls: int, period: float):
    """
    Decorator for rate limiting function calls
    """
    def decorator(func):
        calls = []
        
        async def wrapper(*args, **kwargs):
            nonlocal calls
            
            # Remove old calls
            current_time = time.time()
            calls = [call for call in calls if current_time - call < period]
            
            if len(calls) >= max_calls:
                # Calculate wait time
                oldest_call = min(calls)
                wait_time = period - (current_time - oldest_call)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # Update calls after waiting
                    current_time = time.time()
                    calls = [call for call in calls if current_time - call < period]
            
            # Add current call and execute function
            calls.append(current_time)
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator

def format_rpc_error(error: Any) -> Dict:
    """
    Format RPC error response
    """
    if isinstance(error, dict):
        return error
    elif isinstance(error, str):
        return {"code": -32000, "message": error}
    else:
        return {"code": -32000, "message": str(error)}

def sanitize_for_logging(data: Any) -> Any:
    """
    Sanitize sensitive data for logging
    """
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in ['password', 'secret', 'key', 'token', 'mnemonic']):
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = sanitize_for_logging(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_for_logging(item) for item in data]
    else:
        return data

def get_size(obj: Any) -> int:
    """
    Get approximate memory size of object in bytes
    """
    if isinstance(obj, (str, bytes)):
        return len(obj)
    elif isinstance(obj, (int, float, bool)):
        return 8  # Approximate size for primitive types
    elif isinstance(obj, (list, tuple, set)):
        return sum(get_size(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(get_size(key) + get_size(value) for key, value in obj.items())
    else:
        return 0  # Unknown type