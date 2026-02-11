"""
Utility helpers for RAYONIX blockchain node 
Utilities for logging, configuration, file management, and system operations
"""

import os
import sys
import logging
import logging.handlers
import json
import yaml
import hashlib
import secrets
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Callable
import inspect
import threading
import socket
import psutil
import gc
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import pickle

logger = logging.getLogger("rayonix_helpers")

class SecureConfig:
    """Secure configuration management with encryption"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key
        self.fernet = None
        if encryption_key:
            self._setup_encryption(encryption_key)
    
    def _setup_encryption(self, key: str):
        """Setup Fernet encryption with key derivation"""
        try:
            # Derive a secure key from the passphrase
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'rayonix_config_salt',  # In production, use a random salt per config
                iterations=100000,
            )
            key_bytes = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            self.fernet = Fernet(key_bytes)
        except Exception as e:
            logger.error(f"Failed to setup encryption: {e}")
            self.fernet = None
    
    def encrypt_value(self, value: str) -> str:
        """Encrypt a configuration value"""
        if not self.fernet:
            return value
        try:
            return self.fernet.encrypt(value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a configuration value"""
        if not self.fernet:
            return encrypted_value
        try:
            return self.fernet.decrypt(encrypted_value.encode()).decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value

class PIDManager:
    """PID file management for process tracking"""
    
    @staticmethod
    def setup_pid_file(process_name: str, data_dir: str) -> str:
        """
        Create PID file for process tracking
        
        Args:
            process_name: Name of the process
            data_dir: Data directory path
            
        Returns:
            str: Path to PID file
        """
        try:
            pid_dir = Path(data_dir) / 'run'
            pid_dir.mkdir(parents=True, exist_ok=True)
            
            pidfile = pid_dir / f"{process_name}.pid"
            
            # Check for existing process
            if pidfile.exists():
                try:
                    with open(pidfile, 'r') as f:
                        existing_pid = int(f.read().strip())
                    
                    # Check if process is still running
                    if PIDManager._is_process_running(existing_pid):
                        raise RuntimeError(f"Process already running with PID {existing_pid}")
                    else:
                        logger.warning(f"Removing stale PID file for process {existing_pid}")
                        pidfile.unlink()
                except (ValueError, OSError) as e:
                    logger.warning(f"Could not read existing PID file: {e}")
                    pidfile.unlink()
            
            # Create new PID file
            with open(pidfile, 'w') as f:
                f.write(str(os.getpid()) + '\n')
            
            # Set secure permissions
            pidfile.chmod(0o644)
            
            logger.info(f"PID file created: {pidfile}")
            return str(pidfile)
            
        except Exception as e:
            logger.error(f"Failed to create PID file: {e}")
            raise
    
    @staticmethod
    def remove_pid_file(pidfile: str) -> bool:
        """
        Remove PID file
        
        Args:
            pidfile: Path to PID file
            
        Returns:
            bool: True if successful
        """
        try:
            pid_path = Path(pidfile)
            if pid_path.exists():
                pid_path.unlink()
                logger.info(f"PID file removed: {pidfile}")
                return True
            return True  # Already removed is considered success
        except Exception as e:
            logger.error(f"Failed to remove PID file {pidfile}: {e}")
            return False
    
    @staticmethod
    def _is_process_running(pid: int) -> bool:
        """Check if process is running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

class LoggingConfig:
    """Comprehensive logging configuration"""
    
    @staticmethod
    def configure_logging(
        level: str = "INFO",
        log_file: Optional[str] = None,
        component: str = "rayonix",
        max_bytes: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        enable_console: bool = True
    ):
        """
        Configure comprehensive logging system
        
        Args:
            level: Logging level
            log_file: Path to log file
            component: Component name for logger
            max_bytes: Maximum log file size
            backup_count: Number of backup files to keep
            enable_console: Whether to enable console output
        """
        try:
            # Convert string level to logging constant
            log_level = getattr(logging, level.upper(), logging.INFO)
            
            # Create logger
            logger = logging.getLogger(component)
            logger.setLevel(log_level)
            
            # Clear existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            # Create formatter
            formatter = logging.Formatter(
                fmt='%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            if enable_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)
            
            # File handler
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            
            # Capture warnings
            logging.captureWarnings(True)
            
            # Set specific log levels for noisy libraries
            logging.getLogger('asyncio').setLevel(logging.WARNING)
            logging.getLogger('urllib3').setLevel(logging.WARNING)
            logging.getLogger('requests').setLevel(logging.WARNING)
            
            logger.info(f"Logging configured - Level: {level}, File: {log_file}")
            
        except Exception as e:
            print(f"Failed to configure logging: {e}", file=sys.stderr)
            raise

class FileLock:
    """File-based locking for cross-process synchronization"""
    
    def __init__(self, lockfile: str, timeout: int = 30):
        self.lockfile = Path(lockfile)
        self.timeout = timeout
        self._lock_acquired = False
    
    def __enter__(self):
        self.acquire()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def acquire(self) -> bool:
        """Acquire the lock with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < self.timeout:
            try:
                # Try to create the lock file exclusively
                fd = os.open(self.lockfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                
                # Write current PID to lock file
                with open(self.lockfile, 'w') as f:
                    f.write(str(os.getpid()))
                
                self._lock_acquired = True
                return True
                
            except FileExistsError:
                # Lock file exists, check if process is still alive
                try:
                    with open(self.lockfile, 'r') as f:
                        pid = int(f.read().strip())
                    
                    if not PIDManager._is_process_running(pid):
                        # Stale lock, remove it
                        self.lockfile.unlink()
                        continue
                    
                except (ValueError, OSError):
                    # Corrupted lock file, remove it
                    self.lockfile.unlink()
                    continue
                
                # Wait and retry
                time.sleep(0.1)
        
        raise TimeoutError(f"Could not acquire lock {self.lockfile} within {self.timeout}s")
    
    def release(self):
        """Release the lock"""
        if self._lock_acquired and self.lockfile.exists():
            try:
                self.lockfile.unlink()
                self._lock_acquired = False
            except OSError as e:
                logger.error(f"Failed to release lock {self.lockfile}: {e}")

class SystemInfo:
    """System information and resource monitoring"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            info = {
                'timestamp': datetime.now().isoformat(),
                'python': {
                    'version': sys.version,
                    'implementation': sys.implementation.name,
                    'path': sys.path
                },
                'platform': {
                    'system': sys.platform,
                    'node': socket.gethostname(),
                    'release': os.uname().release if hasattr(os, 'uname') else 'unknown'
                },
                'process': {
                    'pid': os.getpid(),
                    'uid': os.getuid() if hasattr(os, 'getuid') else None,
                    'gid': os.getgid() if hasattr(os, 'getgid') else None
                },
                'resources': SystemInfo._get_resource_usage(),
                'network': SystemInfo._get_network_info()
            }
            return info
        except Exception as e:
            logger.error(f"Failed to get system info: {e}")
            return {}
    
    @staticmethod
    def _get_resource_usage() -> Dict[str, Any]:
        """Get current resource usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory_rss_mb': memory_info.rss / 1024 / 1024,
                'memory_vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cpu_percent': process.cpu_percent(),
                'threads': process.num_threads(),
                'open_files': len(process.open_files()),
                'connections': len(process.connections())
            }
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    @staticmethod
    def _get_network_info() -> Dict[str, Any]:
        """Get network interface information"""
        try:
            interfaces = {}
            for interface, addrs in psutil.net_if_addrs().items():
                interfaces[interface] = [
                    {
                        'family': str(addr.family),
                        'address': addr.address,
                        'netmask': addr.netmask
                    }
                    for addr in addrs
                ]
            
            return {
                'hostname': socket.gethostname(),
                'interfaces': interfaces
            }
        except Exception as e:
            logger.error(f"Failed to get network info: {e}")
            return {}

class RetryManager:
    """Advanced retry logic with exponential backoff"""
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Any: Function result
            
        Raises:
            Exception: Last exception after all retries
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    break
                
                delay = self._calculate_delay(attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                time.sleep(delay)
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.max_delay,
            self.base_delay * (self.exponential_base ** attempt)
        )
        
        if self.jitter:
            delay = delay * (0.5 + secrets.SystemRandom().random())
        
        return delay

class DataSerializer:
    """Secure data serialization and deserialization"""
    
    @staticmethod
    def serialize_to_file(data: Any, filepath: str, encrypt: bool = False, key: Optional[str] = None) -> bool:
        """
        Serialize data to file with optional encryption
        
        Args:
            data: Data to serialize
            filepath: Output file path
            encrypt: Whether to encrypt the data
            key: Encryption key
            
        Returns:
            bool: True if successful
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            # Serialize data
            serialized_data = pickle.dumps(data)
            
            # Encrypt if requested
            if encrypt and key:
                secure_config = SecureConfig(key)
                encrypted_data = secure_config.fernet.encrypt(serialized_data)
                data_to_write = encrypted_data
            else:
                data_to_write = serialized_data
            
            # Write to file with atomic replacement
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'wb') as f:
                f.write(data_to_write)
            
            # Atomic replace
            temp_file.replace(filepath)
            
            # Set secure permissions
            filepath.chmod(0o600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to serialize data to {filepath}: {e}")
            return False
    
    @staticmethod
    def deserialize_from_file(filepath: str, encrypted: bool = False, key: Optional[str] = None) -> Any:
        """
        Deserialize data from file with optional decryption
        
        Args:
            filepath: Input file path
            encrypted: Whether data is encrypted
            key: Decryption key
            
        Returns:
            Any: Deserialized data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If deserialization fails
        """
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(filepath, 'rb') as f:
                data = f.read()
            
            # Decrypt if needed
            if encrypted and key:
                secure_config = SecureConfig(key)
                data = secure_config.fernet.decrypt(data)
            
            return pickle.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to deserialize data from {filepath}: {e}")
            raise

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_file_path(filepath: str, check_writable: bool = False) -> bool:
        """Validate file path accessibility"""
        try:
            path = Path(filepath)
            
            if path.exists():
                # Check if it's a file and readable
                if not path.is_file():
                    return False
                if not os.access(path, os.R_OK):
                    return False
                if check_writable and not os.access(path, os.W_OK):
                    return False
            else:
                # Check if directory is writable for new file
                parent_dir = path.parent
                if not parent_dir.exists():
                    parent_dir.mkdir(parents=True, exist_ok=True)
                if check_writable and not os.access(parent_dir, os.W_OK):
                    return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_port(port: int) -> bool:
        """Validate port number"""
        return 1 <= port <= 65535
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """Validate IP address"""
        try:
            socket.inet_pton(socket.AF_INET, ip)
            return True
        except socket.error:
            try:
                socket.inet_pton(socket.AF_INET6, ip)
                return True
            except socket.error:
                return False

class PerformanceUtils:
    """Performance monitoring and optimization utilities"""
    
    @staticmethod
    def timed_function(logger: Optional[logging.Logger] = None):
        """Decorator to log function execution time"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    if logger:
                        logger.debug(f"Function {func.__name__} executed in {execution_time:.3f}s")
            return wrapper
        return decorator
    
    @staticmethod
    def optimize_memory():
        """Perform memory optimization"""
        # Force garbage collection
        collected = gc.collect()
        logger.debug(f"Garbage collection collected {collected} objects")
        
        # Clear various caches if they exist
        try:
            import linecache
            linecache.clearcache()
        except:
            pass

# Convenience functions
def setup_pid_file(process_name: str, data_dir: str) -> str:
    """Convenience wrapper for PID file setup"""
    return PIDManager.setup_pid_file(process_name, data_dir)

def remove_pid_file(pidfile: str) -> bool:
    """Convenience wrapper for PID file removal"""
    return PIDManager.remove_pid_file(pidfile)

def configure_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    component: str = "rayonix",
    **kwargs
):
    """Convenience wrapper for logging configuration"""
    LoggingConfig.configure_logging(level, log_file, component, **kwargs)

def get_system_info() -> Dict[str, Any]:
    """Convenience wrapper for system info"""
    return SystemInfo.get_system_info()

def secure_hash(data: str) -> str:
    """Generate secure hash of data"""
    return hashlib.sha256(data.encode()).hexdigest()

def generate_random_string(length: int = 32) -> str:
    """Generate cryptographically secure random string"""
    return secrets.token_urlsafe(length)

# Export main classes
__all__ = [
    'SecureConfig',
    'PIDManager', 
    'LoggingConfig',
    'FileLock',
    'SystemInfo',
    'RetryManager',
    'DataSerializer',
    'ValidationUtils',
    'PerformanceUtils',
    'setup_pid_file',
    'remove_pid_file',
    'configure_logging',
    'get_system_info',
    'secure_hash',
    'generate_random_string'
]
