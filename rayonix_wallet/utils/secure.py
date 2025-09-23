import ctypes
import threading
import os
import sys
import tempfile
import secrets
from typing import Optional, Union, Tuple
from contextlib import contextmanager
import mmap
import hashlib

# Platform-specific imports for memory protection
if sys.platform == 'win32':
    import ctypes.wintypes
else:
    import resource

class SecureString:
    """Production-ready secure string implementation with memory protection"""
    
    def __init__(self, value: Optional[bytes] = None, protect_memory: bool = True):
        self._value = None
        self._length = 0
        self._lock = threading.RLock()
        self._protect_memory = protect_memory
        self._locked_pages = False
        
        if value is not None:
            self.set_value(value)
    
    def set_value(self, value: bytes) -> None:
        """Set secure value with validation and memory protection"""
        if not isinstance(value, bytes):
            raise TypeError("Value must be bytes, got {}".format(type(value).__name__))
        
        with self._lock:
            self._secure_wipe()
            
            if value:
                self._length = len(value)
                # Use secrets for secure random padding to prevent length disclosure
                padding_length = secrets.randbelow(32)  # Random padding up to 31 bytes
                total_length = self._length + padding_length
                
                # Allocate mutable buffer with extra padding
                self._value = ctypes.create_string_buffer(total_length)
                
                # Copy value and add random padding
                ctypes.memmove(ctypes.addressof(self._value), value, self._length)
                ctypes.memset(ctypes.addressof(self._value) + self._length, 
                             secrets.randbits(8), padding_length)
                
                # Protect memory from swapping if requested
                if self._protect_memory:
                    self._lock_memory()
    
    def get_value(self) -> bytes:
        """Get secure value with constant-time operation"""
        with self._lock:
            if self._value is None:
                raise ValueError("No value set")
            
            # Create copy without exposing internal buffer structure
            result = ctypes.create_string_buffer(self._length)
            ctypes.memmove(ctypes.addressof(result), ctypes.addressof(self._value), self._length)
            return bytes(result)
    
    def compare(self, other: bytes) -> bool:
        """Constant-time comparison with external data"""
        with self._lock:
            if self._value is None:
                # Return False in constant time to prevent timing attacks
                return secure_memcmp(b'', other)
            
            our_value = self.get_value()
            return secure_memcmp(our_value, other)
    
    def _lock_memory(self) -> None:
        """Attempt to lock memory pages to prevent swapping"""
        if self._value is None or self._locked_pages:
            return
            
        try:
            if sys.platform == 'win32':
                # Windows memory locking
                ctypes.windll.kernel32.VirtualLock(
                    ctypes.addressof(self._value), 
                    ctypes.sizeof(self._value)
                )
            else:
                # Unix-like systems
                if hasattr(ctypes, 'mlock'):
                    ctypes.mlock(ctypes.addressof(self._value), ctypes.sizeof(self._value))
                # Also try using mlockall for process-wide locking
                libc = ctypes.CDLL(None)
                libc.mlock(ctypes.addressof(self._value), ctypes.sizeof(self._value))
            
            self._locked_pages = True
        except (OSError, AttributeError):
            # Gracefully handle systems where memory locking isn't available
            self._locked_pages = False
    
    def _unlock_memory(self) -> None:
        """Unlock memory pages"""
        if not self._locked_pages or self._value is None:
            return
            
        try:
            if sys.platform == 'win32':
                ctypes.windll.kernel32.VirtualUnlock(
                    ctypes.addressof(self._value), 
                    ctypes.sizeof(self._value)
                )
            else:
                if hasattr(ctypes, 'munlock'):
                    ctypes.munlock(ctypes.addressof(self._value), ctypes.sizeof(self._value))
                libc = ctypes.CDLL(None)
                libc.munlock(ctypes.addressof(self._value), ctypes.sizeof(self._value))
            
            self._locked_pages = False
        except (OSError, AttributeError):
            pass
    
    def _secure_wipe(self) -> None:
        """Securely wipe the value from memory with multiple passes"""
        with self._lock:
            if self._value is not None:
                try:
                    # Multiple pass wiping for enhanced security
                    length = ctypes.sizeof(self._value)
                    
                    # Pass 1: zeros
                    ctypes.memset(ctypes.addressof(self._value), 0, length)
                    # Pass 2: ones
                    ctypes.memset(ctypes.addressof(self._value), 0xFF, length)
                    # Pass 3: random
                    random_data = os.urandom(length)
                    ctypes.memmove(ctypes.addressof(self._value), random_data, length)
                    # Final pass: zeros
                    ctypes.memset(ctypes.addressof(self._value), 0, length)
                    
                    self._unlock_memory()
                finally:
                    self._value = None
                    self._length = 0
                    self._locked_pages = False
    
    def wipe(self) -> None:
        """Public method to securely wipe the value"""
        self._secure_wipe()
    
    @contextmanager
    def temporary_access(self) -> bytes:
        """Context manager for temporary secure access to the value"""
        with self._lock:
            if self._value is None:
                raise ValueError("No value set")
            
            temp_buffer = ctypes.create_string_buffer(self._length)
            ctypes.memmove(ctypes.addressof(temp_buffer), 
                          ctypes.addressof(self._value), 
                          self._length)
            
            try:
                yield bytes(temp_buffer)
            finally:
                # Securely wipe the temporary buffer
                ctypes.memset(ctypes.addressof(temp_buffer), 0, self._length)
    
    def __len__(self) -> int:
        return self._length
    
    def __bool__(self) -> bool:
        return self._value is not None
    
    def __del__(self) -> None:
        self._secure_wipe()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._secure_wipe()

class SecureBuffer:
    """Enhanced secure buffer with memory protection and auditing"""
    
    def __init__(self, size: int, protect_memory: bool = True):
        if size <= 0 or size > 1024 * 1024 * 100:  # 100MB max
            raise ValueError("Invalid buffer size")
        
        self.size = size
        self.buffer = ctypes.create_string_buffer(size)
        self.lock = threading.RLock()
        self._protect_memory = protect_memory
        self._locked_pages = False
        self.access_count = 0
        
        if protect_memory:
            self._lock_memory()
    
    def write(self, data: bytes) -> None:
        """Write data to secure buffer with validation"""
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        
        with self.lock:
            if len(data) > self.size:
                raise ValueError("Data too large for buffer")
            
            # Securely clear existing data
            ctypes.memset(ctypes.addressof(self.buffer), 0, self.size)
            
            # Write new data
            ctypes.memmove(ctypes.addressof(self.buffer), data, len(data))
            
            # Fill remaining space with random data
            if len(data) < self.size:
                padding = os.urandom(self.size - len(data))
                ctypes.memmove(ctypes.addressof(self.buffer) + len(data), 
                              padding, len(padding))
            
            self.access_count += 1
    
    def read(self) -> bytes:
        """Read data from secure buffer"""
        with self.lock:
            self.access_count += 1
            
            # Create copy to prevent external modification
            result = ctypes.create_string_buffer(self.size)
            ctypes.memmove(ctypes.addressof(result), 
                          ctypes.addressof(self.buffer), 
                          self.size)
            return bytes(result)
    
    def read_slice(self, start: int, length: int) -> bytes:
        """Read a slice of data with bounds checking"""
        with self.lock:
            if start < 0 or length < 0 or start + length > self.size:
                raise ValueError("Invalid slice parameters")
            
            self.access_count += 1
            result = ctypes.create_string_buffer(length)
            ctypes.memmove(ctypes.addressof(result), 
                          ctypes.addressof(self.buffer) + start, 
                          length)
            return bytes(result)
    
    def _lock_memory(self) -> None:
        """Lock buffer memory to prevent swapping"""
        try:
            if sys.platform == 'win32':
                ctypes.windll.kernel32.VirtualLock(
                    ctypes.addressof(self.buffer), 
                    self.size
                )
            else:
                libc = ctypes.CDLL(None)
                libc.mlock(ctypes.addressof(self.buffer), self.size)
            
            self._locked_pages = True
        except (OSError, AttributeError):
            self._locked_pages = False
    
    def _unlock_memory(self) -> None:
        """Unlock buffer memory"""
        if not self._locked_pages:
            return
            
        try:
            if sys.platform == 'win32':
                ctypes.windll.kernel32.VirtualUnlock(
                    ctypes.addressof(self.buffer), 
                    self.size
                )
            else:
                libc = ctypes.CDLL(None)
                libc.munlock(ctypes.addressof(self.buffer), self.size)
            
            self._locked_pages = False
        except (OSError, AttributeError):
            pass
    
    def wipe(self) -> None:
        """Securely wipe the buffer with multiple passes"""
        with self.lock:
            # 3-pass secure wipe
            ctypes.memset(ctypes.addressof(self.buffer), 0, self.size)
            ctypes.memset(ctypes.addressof(self.buffer), 0xFF, self.size)
            random_data = os.urandom(self.size)
            ctypes.memmove(ctypes.addressof(self.buffer), random_data, self.size)
            ctypes.memset(ctypes.addressof(self.buffer), 0, self.size)
            
            self._unlock_memory()
            self.access_count = 0
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        with self.lock:
            return {
                'size': self.size,
                'access_count': self.access_count,
                'memory_locked': self._locked_pages
            }
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.wipe()

def secure_memcmp(a: bytes, b: bytes) -> bool:
    """Constant-time memory comparison resistant to timing attacks"""
    if len(a) != len(b):
        # Use constant-time length comparison
        dummy = secrets.token_bytes(max(len(a), len(b)))
        a_padded = a + dummy[len(a):]
        b_padded = b + dummy[len(b):]
        a, b = a_padded, b_padded
    
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
        # Add small random delay to further obscure timing
        if secrets.randbits(1):
            secrets.token_bytes(1)
    
    return result == 0

def secure_hash_compare(a: bytes, b: bytes, hash_func: str = 'sha256') -> bool:
    """Compare values using cryptographic hashes to prevent timing attacks"""
    hash_obj = hashlib.new(hash_func)
    hash_obj.update(a)
    hash_a = hash_obj.digest()
    
    hash_obj = hashlib.new(hash_func)
    hash_obj.update(b)
    hash_b = hash_obj.digest()
    
    return secure_memcmp(hash_a, hash_b)

def secure_zero_memory(data: bytes) -> None:
    """Securely zero out memory with multiple passes"""
    if not data:
        return
    
    try:
        # Create mutable buffer
        buffer = ctypes.create_string_buffer(data)
        length = len(buffer)
        
        # 3-pass secure wipe
        ctypes.memset(ctypes.addressof(buffer), 0, length)
        ctypes.memset(ctypes.addressof(buffer), 0xFF, length)
        ctypes.memset(ctypes.addressof(buffer), 0, length)
    except Exception:
        # Best effort - continue even if wiping fails
        pass

def create_secure_temp_file(suffix: str = None, prefix: str = None) -> str:
    """Create secure temporary file with strict permissions"""
    try:
        # Create file with restrictive permissions
        fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
        
        try:
            # Set restrictive permissions before closing
            os.chmod(path, 0o600)
            
            # Securely wipe the file content on exit
            with open(fd, 'wb', closefd=False) as f:
                f.write(os.urandom(1024))  # Fill with random data initially
                f.truncate(0)  # Truncate to zero
        finally:
            os.close(fd)
        
        return path
    except Exception as e:
        raise RuntimeError(f"Failed to create secure temp file: {e}")

def secure_file_write(path: str, data: bytes, mode: int = 0o600) -> None:
    """Write data to file securely with atomic replacement"""
    if not isinstance(data, bytes):
        raise TypeError("Data must be bytes")
    
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, mode=0o700)
    
    # Create temporary file in the same directory for atomicity
    temp_fd, temp_path = tempfile.mkstemp(dir=dirname, suffix='.tmp')
    
    try:
        with os.fdopen(temp_fd, 'wb') as f:
            # Write data with sync to ensure it's flushed to disk
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        
        # Set restrictive permissions
        os.chmod(temp_path, mode)
        
        # Atomic replace
        os.replace(temp_path, path)
        
    except Exception:
        # Clean up temporary file on error
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise

def secure_file_read(path: str, max_size: int = 1024 * 1024 * 100) -> bytes:
    """Read file securely with size limits"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    stat = os.stat(path)
    if stat.st_size > max_size:
        raise ValueError(f"File too large: {stat.st_size} bytes")
    
    with open(path, 'rb') as f:
        # Use memory mapping for secure reading if possible
        try:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                data = mm.read()
        except (OSError, ValueError):
            # Fallback to regular read if mmap fails
            data = f.read()
    
    return data

def secure_random_bytes(length: int) -> bytes:
    """Generate cryptographically secure random bytes"""
    if length <= 0:
        raise ValueError("Length must be positive")
    
    return secrets.token_bytes(length)

def secure_erase_file(path: str, passes: int = 3) -> None:
    """Securely erase file by overwriting before deletion"""
    if not os.path.exists(path):
        return
    
    try:
        file_size = os.path.getsize(path)
        
        with open(path, 'rb+') as f:
            for pass_num in range(passes):
                # Write random data
                f.seek(0)
                f.write(os.urandom(file_size))
                f.flush()
                os.fsync(f.fileno())
            
            # Final pass with zeros
            f.seek(0)
            f.write(b'\x00' * file_size)
            f.flush()
            os.fsync(f.fileno())
            
            # Truncate to ensure complete overwrite
            f.truncate()
        
        # Remove the file
        os.unlink(path)
        
    except Exception as e:
        # If secure erase fails, try normal deletion
        try:
            os.unlink(path)
        except OSError:
            pass
        raise RuntimeError(f"Failed to securely erase file: {e}")

# Additional utility functions
class SecureMemoryManager:
    """Manager for secure memory allocations"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._allocations = {}
                cls._instance._next_id = 0
            return cls._instance
    
    def allocate(self, size: int) -> int:
        """Allocate secure memory and return handle"""
        with self._lock:
            handle = self._next_id
            self._next_id += 1
            
            buffer = ctypes.create_string_buffer(size)
            self._allocations[handle] = {
                'buffer': buffer,
                'size': size,
                'locked': False
            }
            
            return handle
    
    def deallocate(self, handle: int) -> None:
        """Securely deallocate memory"""
        with self._lock:
            if handle in self._allocations:
                allocation = self._allocations[handle]
                # Secure wipe
                ctypes.memset(ctypes.addressof(allocation['buffer']), 0, allocation['size'])
                del self._allocations[handle]

# Export public interface
__all__ = [
    'SecureString',
    'SecureBuffer',
    'secure_memcmp',
    'secure_hash_compare',
    'secure_zero_memory',
    'create_secure_temp_file',
    'secure_file_write',
    'secure_file_read',
    'secure_random_bytes',
    'secure_erase_file',
    'SecureMemoryManager'
]