import ctypes
import threading
from typing import Optional

class SecureString:
    """Secure string implementation that attempts to wipe memory after use"""
    
    def __init__(self, value: Optional[bytes] = None):
        self._value = None
        self._length = 0
        self._lock = threading.RLock()
        
        if value is not None:
            self.set_value(value)
    
    def set_value(self, value: bytes):
        """Set secure value"""
        with self._lock:
            self.wipe()
            if value:
                # Allocate mutable buffer
                self._length = len(value)
                self._value = ctypes.create_string_buffer(value)
    
    def get_value(self) -> bytes:
        """Get secure value"""
        with self._lock:
            if self._value is None:
                raise ValueError("No value set")
            return bytes(self._value)
    
    def wipe(self):
        """Securely wipe the value from memory"""
        with self._lock:
            if self._value is not None:
                # Overwrite with zeros
                ctypes.memset(ctypes.addressof(self._value), 0, self._length)
                self._value = None
                self._length = 
    
    def __len__(self):
        return self._length
    
    def __bool__(self):
        return self._value is not None
    
    def __del__(self):
        self.wipe()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wipe()

class SecureBuffer:
    """Secure buffer for sensitive data"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = ctypes.create_string_buffer(size)
        self.lock = threading.RLock()
    
    def write(self, data: bytes):
        """Write data to secure buffer"""
        with self.lock:
            if len(data) > self.size:
                raise ValueError("Data too large for buffer")
            ctypes.memset(ctypes.addressof(self.buffer), 0, self.size)
            self.buffer.value = data
    
    def read(self) -> bytes:
        """Read data from secure buffer"""
        with self.lock:
            return bytes(self.buffer)
    
    def wipe(self):
        """Securely wipe the buffer"""
        with self.lock:
            ctypes.memset(ctypes.addressof(self.buffer), 0, self.size)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wipe()

def secure_memcmp(a: bytes, b: bytes) -> bool:
    """Constant-time memory comparison"""
    if len(a) != len(b):
        return False
    
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    
    return result == 0

def secure_zero_memory(data: bytes) -> None:
    """Attempt to zero out memory (best effort)"""
    try:
        # Create mutable buffer and zero it
        buffer = ctypes.create_string_buffer(data)
        ctypes.memset(ctypes.addressof(buffer), 0, len(buffer))
    except:
        pass  # Best effort

def create_secure_temp_file() -> str:
    """Create secure temporary file"""
    import tempfile
    fd, path = tempfile.mkstemp()
    os.close(fd)
    
    # Set restrictive permissions
    os.chmod(path, 0o600)
    return path

def secure_file_write(path: str, data: bytes) -> None:
    """Write data to file securely"""
    # Write to temporary file first
    temp_path = path + '.tmp'
    
    with open(temp_path, 'wb') as f:
        f.write(data)
    
    # Set restrictive permissions
    os.chmod(temp_path, 0o600)
    
    # Atomically replace original file
    os.replace(temp_path, path)

def secure_file_read(path: str) -> bytes:
    """Read file securely"""
    with open(path, 'rb') as f:
        return f.read()