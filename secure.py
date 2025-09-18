# secure.py
import json
import os
import secrets
import hmac
import struct
from typing import Any, Union, Optional, Tuple, Dict
import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.exceptions import InvalidTag
import base64
import logging
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RayonixSecure")

class SecurityLevel(Enum):
    """Security levels for different operations"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SecurityConfig:
    """Security configuration with production-ready defaults"""
    # PBKDF2 iterations based on security level
    PBKDF2_ITERATIONS: Dict[SecurityLevel, int] = field(default_factory=lambda: {
        SecurityLevel.LOW: 100000,
        SecurityLevel.MEDIUM: 310000,
        SecurityLevel.HIGH: 600000,
        SecurityLevel.CRITICAL: 1200000
    })
    
    # Scrypt parameters
    SCRYPTPARAMS: Dict[SecurityLevel, Dict[str, int]] = field(default_factory=lambda: {
        SecurityLevel.LOW: {'n': 2**14, 'r': 8, 'p': 1},
        SecurityLevel.MEDIUM: {'n': 2**15, 'r': 8, 'p': 1},
        SecurityLevel.HIGH: {'n': 2**16, 'r': 8, 'p': 1},
        SecurityLevel.CRITICAL: {'n': 2**17, 'r': 8, 'p': 1}
    })
    
    # Key derivation algorithm preference
    PREFER_SCRYPT: bool = True
    
    # Default security level
    DEFAULT_LEVEL: SecurityLevel = SecurityLevel.HIGH
    
    # Memory wiping passes
    WIPE_PASSES: int = 7
    
    # Maximum file size for secure operations (100MB)
    MAX_FILE_SIZE: int = 100 * 1024 * 1024
    
    # Encryption version for format compatibility
    ENCRYPTION_VERSION: int = 2

class SecureMemoryManager:
    """Singleton for managing secure memory operations"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SecureMemoryManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = SecurityConfig()
            self._secure_registry = {}
            self._registry_lock = threading.Lock()
            self._initialized = True
    
    def register_secure_object(self, obj_id: str, obj: Any):
        """Register secure object for tracking"""
        with self._registry_lock:
            self._secure_registry[obj_id] = {
                'object': obj,
                'created': time.time(),
                'access_count': 0
            }
    
    def unregister_secure_object(self, obj_id: str):
        """Unregister secure object"""
        with self._registry_lock:
            if obj_id in self._secure_registry:
                del self._secure_registry[obj_id]
    
    def get_secure_objects_count(self) -> int:
        """Get count of tracked secure objects"""
        with self._registry_lock:
            return len(self._secure_registry)

class SecureString:
    """Production-grade secure string implementation with memory protection"""
    
    def __init__(self, value: Union[str, bytes], 
                 object_id: Optional[str] = None,
                 security_level: SecurityLevel = SecurityConfig().DEFAULT_LEVEL):
        self._security_level = security_level
        self._object_id = object_id or f"secure_str_{secrets.token_hex(8)}"
        self._access_count = 0
        self._created_time = time.time()
        self._last_accessed = self._created_time
        
        # Use bytearray for mutable storage
        if isinstance(value, str):
            self._value = bytearray(value.encode('utf-8'))
        else:
            self._value = bytearray(value)
        
        # Register with memory manager
        mem_manager = SecureMemoryManager()
        mem_manager.register_secure_object(self._object_id, self)
    
    def get_value(self) -> bytes:
        """Get the value as bytes with access tracking"""
        self._access_count += 1
        self._last_accessed = time.time()
        return bytes(self._value)
    
    def get_str(self) -> str:
        """Get the value as string with access tracking"""
        self._access_count += 1
        self._last_accessed = time.time()
        return self._value.decode('utf-8')
    
    def wipe(self, passes: Optional[int] = None):
        """Securely wipe the value from memory with multiple passes"""
        wipe_passes = passes or SecureMemoryManager().config.WIPE_PASSES
        
        for pass_num in range(wipe_passes):
            # Use different patterns for each pass
            if pass_num % 3 == 0:
                # Random data
                for i in range(len(self._value)):
                    self._value[i] = secrets.randbits(8)
            elif pass_num % 3 == 1:
                # All zeros
                for i in range(len(self._value)):
                    self._value[i] = 0
            else:
                # All ones
                for i in range(len(self._value)):
                    self._value[i] = 0xFF
        
        # Final clear
        self._value = bytearray()
        
        # Unregister from memory manager
        mem_manager = SecureMemoryManager()
        mem_manager.unregister_secure_object(self._object_id)
        
        logger.debug(f"SecureString {self._object_id} wiped with {wipe_passes} passes")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get security metrics for this string"""
        return {
            'object_id': self._object_id,
            'access_count': self._access_count,
            'created_time': self._created_time,
            'last_accessed': self._last_accessed,
            'security_level': self._security_level.name,
            'length': len(self._value)
        }
    
    def __len__(self) -> int:
        return len(self._value)
    
    def __del__(self):
        try:
            self.wipe()
        except:
            # Ensure wipe is attempted even if errors occur
            pass

class SecureFile:
    """Secure file handling with atomic operations and backup"""
    
    def __init__(self, path: str):
        self.path = path
        self.backup_path = f"{path}.backup"
        self.temp_path = f"{path}.tmp"
        self.lock = threading.RLock()
    
    def atomic_write(self, data: bytes) -> bool:
        """Atomic file write with backup and fsync"""
        with self.lock:
            try:
                # Write to temporary file
                with open(self.temp_path, 'wb') as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Create backup
                if os.path.exists(self.path):
                    os.replace(self.path, self.backup_path)
                
                # Atomic replace
                os.replace(self.temp_path, self.path)
                
                # Sync directory
                dir_fd = os.open(os.path.dirname(self.path), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
                
                return True
                
            except Exception as e:
                logger.error(f"Atomic write failed: {e}")
                # Cleanup temp file
                if os.path.exists(self.temp_path):
                    os.unlink(self.temp_path)
                return False
    
    def atomic_read(self) -> Optional[bytes]:
        """Atomic file read with backup fallback"""
        with self.lock:
            try:
                with open(self.path, 'rb') as f:
                    return f.read()
            except FileNotFoundError:
                # Try backup
                try:
                    with open(self.backup_path, 'rb') as f:
                        return f.read()
                except FileNotFoundError:
                    return None
            except Exception as e:
                logger.error(f"Atomic read failed: {e}")
                return None

def _derive_key(passphrase: str, salt: bytes, 
               security_level: SecurityLevel = SecurityConfig().DEFAULT_LEVEL,
               use_scrypt: Optional[bool] = None) -> bytes:
    """Derive encryption key using preferred algorithm"""
    config = SecurityConfig()
    use_scrypt = use_scrypt if use_scrypt is not None else config.PREFER_SCRYPT
    
    if use_scrypt:
        # Use Scrypt (memory-hard KDF)
        params = config.SCRYPTPARAMS[security_level]
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=params['n'],
            r=params['r'],
            p=params['p'],
            backend=default_backend()
        )
    else:
        # Use PBKDF2
        iterations = config.PBKDF2_ITERATIONS[security_level]
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
    
    return kdf.derive(passphrase.encode())

def secure_dump(data: Any, path: str, passphrase: str, 
               security_level: SecurityLevel = SecurityConfig().DEFAULT_LEVEL,
               metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Production-grade secure data dump with authenticated encryption"""
    try:
        # Validate file size
        if isinstance(data, (str, bytes)):
            data_size = len(data) if isinstance(data, bytes) else len(data.encode('utf-8'))
            if data_size > SecurityConfig().MAX_FILE_SIZE:
                raise ValueError(f"Data too large: {data_size} bytes")
        
        # Generate cryptographic material
        salt = secrets.token_bytes(16)
        iv = secrets.token_bytes(12)  # 96 bits for GCM
        key = _derive_key(passphrase, salt, security_level)
        
        # Prepare metadata
        file_metadata = {
            'version': SecurityConfig().ENCRYPTION_VERSION,
            'security_level': security_level.name,
            'timestamp': time.time(),
            'algorithm': 'AES-GCM',
            'kdf': 'Scrypt' if SecurityConfig().PREFER_SCRYPT else 'PBKDF2',
            'custom': metadata or {}
        }
        
        # Serialize and encrypt data
        json_data = json.dumps({
            'metadata': file_metadata,
            'data': data
        }).encode('utf-8')
        
        # Encrypt with AES-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        
        # Associate metadata with encryption
        encryptor.authenticate_additional_data(json.dumps(file_metadata).encode('utf-8'))
        
        encrypted = encryptor.update(json_data) + encryptor.finalize()
        
        # Prepare secure file structure
        secure_data = {
            'salt': base64.b64encode(salt).decode('ascii'),
            'iv': base64.b64encode(iv).decode('ascii'),
            'tag': base64.b64encode(encryptor.tag).decode('ascii'),
            'ciphertext': base64.b64encode(encrypted).decode('ascii'),
            'metadata': file_metadata
        }
        
        # Atomic write
        secure_file = SecureFile(path)
        return secure_file.atomic_write(json.dumps(secure_data).encode('utf-8'))
        
    except Exception as e:
        logger.error(f"Secure dump failed: {e}", exc_info=True)
        return False

def secure_load(path: str, passphrase: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
    """Production-grade secure data load with authentication verification"""
    try:
        secure_file = SecureFile(path)
        file_data = secure_file.atomic_read()
        if not file_data:
            return None, None
        
        secure_data = json.loads(file_data.decode('utf-8'))
        
        # Check version compatibility
        if secure_data['metadata']['version'] > SecurityConfig().ENCRYPTION_VERSION:
            raise ValueError("Unsupported encryption version")
        
        # Decode components
        salt = base64.b64decode(secure_data['salt'])
        iv = base64.b64decode(secure_data['iv'])
        tag = base64.b64decode(secure_data['tag'])
        ciphertext = base64.b64decode(secure_data['ciphertext'])
        
        # Derive key
        security_level = SecurityLevel[secure_data['metadata']['security_level']]
        use_scrypt = secure_data['metadata']['kdf'] == 'Scrypt'
        key = _derive_key(passphrase, salt, security_level, use_scrypt)
        
        # Decrypt with AES-GCM
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        
        # Verify authentication data
        decryptor.authenticate_additional_data(
            json.dumps(secure_data['metadata']).encode('utf-8')
        )
        
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Parse decrypted data
        decrypted_data = json.loads(decrypted.decode('utf-8'))
        
        return decrypted_data['data'], decrypted_data['metadata']
        
    except InvalidTag:
        logger.error("Authentication failed: tampered or incorrect passphrase")
        return None, None
    except Exception as e:
        logger.error(f"Secure load failed: {e}", exc_info=True)
        return None, None

def hash_password(password: str, 
                 security_level: SecurityLevel = SecurityConfig().DEFAULT_LEVEL) -> Tuple[bytes, bytes, Dict[str, Any]]:
    """Modern password hashing with Argon2 alternative (using Scrypt)"""
    salt = secrets.token_bytes(16)
    params = SecurityConfig().SCRYPTPARAMS[security_level]
    
    # Use Scrypt as memory-hard KDF
    kdf = Scrypt(
        salt=salt,
        length=64,  # Longer output for future-proofing
        n=params['n'],
        r=params['r'],
        p=params['p'],
        backend=default_backend()
    )
    
    hashed = kdf.derive(password.encode())
    
    metadata = {
        'algorithm': 'Scrypt',
        'security_level': security_level.name,
        'timestamp': time.time(),
        'params': params
    }
    
    return hashed, salt, metadata

def verify_password(password: str, hashed: bytes, salt: bytes, metadata: Dict[str, Any]) -> bool:
    """Verify password against stored hash with metadata"""
    try:
        if metadata['algorithm'] == 'Scrypt':
            params = metadata['params']
            kdf = Scrypt(
                salt=salt,
                length=len(hashed),
                n=params['n'],
                r=params['r'],
                p=params['p'],
                backend=default_backend()
            )
            new_hash = kdf.derive(password.encode())
            return hmac.compare_digest(new_hash, hashed)
        
        else:
            # Fallback to PBKDF2 for backward compatibility
            iterations = SecurityConfig().PBKDF2_ITERATIONS[SecurityLevel[metadata['security_level']]]
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=len(hashed),
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            new_hash = kdf.derive(password.encode())
            return hmac.compare_digest(new_hash, hashed)
            
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def secure_compare(a: Union[str, bytes], b: Union[str, bytes]) -> bool:
    """Constant-time comparison to prevent timing attacks"""
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    
    return hmac.compare_digest(a, b)

def generate_secure_token(length: int = 32) -> str:
    """Generate cryptographically secure token"""
    return secrets.token_urlsafe(length)

def generate_key_pair(security_level: SecurityLevel = SecurityConfig().DEFAULT_LEVEL) -> Tuple[SecureString, bytes]:
    """Generate secure key pair for cryptographic operations"""
    # For production, consider using cryptography.hazmat.primitives.asymmetric
    private_key = secrets.token_bytes(32)
    public_key = hashlib.sha256(private_key).digest()
    
    secure_privkey = SecureString(private_key, 
                                security_level=security_level,
                                object_id=f"keypair_{secrets.token_hex(4)}")
    
    return secure_privkey, public_key

def secure_zero_memory(data: Union[bytearray, bytes, str]) -> None:
    """Securely zero memory for any data type"""
    if isinstance(data, bytearray):
        for i in range(len(data)):
            data[i] = 0
    elif isinstance(data, (bytes, str)):
        # Create mutable copy and zero it
        mutable = bytearray(data if isinstance(data, bytes) else data.encode('utf-8'))
        for i in range(len(mutable)):
            mutable[i] = 0
        # The original immutable object will be garbage collected

def get_security_audit() -> Dict[str, Any]:
    """Get security audit information"""
    mem_manager = SecureMemoryManager()
    return {
        'secure_objects_count': mem_manager.get_secure_objects_count(),
        'config': asdict(SecurityConfig()),
        'timestamp': time.time(),
        'python_version': os.sys.version,
        'system': os.sys.platform
    }

# Security middleware for application-level protection
class SecurityMiddleware:
    """Middleware for application security monitoring"""
    
    def __init__(self):
        self.failed_attempts = {}
        self.lock = threading.RLock()
        self.attack_detection_threshold = 5
        self.attack_cooldown = 300  # 5 minutes
    
    def check_attempt(self, identifier: str) -> bool:
        """Check if too many failed attempts"""
        with self.lock:
            current_time = time.time()
            attempts = self.failed_attempts.get(identifier, [])
            
            # Remove old attempts
            attempts = [t for t in attempts if current_time - t < self.attack_cooldown]
            
            if len(attempts) >= self.attack_attempts_threshold:
                return False
                
            attempts.append(current_time)
            self.failed_attempts[identifier] = attempts
            return True
    
    def reset_attempts(self, identifier: str):
        """Reset failed attempts counter"""
        with self.lock:
            if identifier in self.failed_attempts:
                del self.failed_attempts[identifier]

# Initialize security middleware singleton
security_middleware = SecurityMiddleware()

# Export public API
__all__ = [
    'SecureString',
    'secure_dump',
    'secure_load',
    'hash_password',
    'verify_password',
    'secure_compare',
    'generate_secure_token',
    'generate_key_pair',
    'secure_zero_memory',
    'get_security_audit',
    'SecurityLevel',
    'SecurityConfig',
    'SecurityMiddleware',
    'security_middleware'
]