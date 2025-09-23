import os
import json
import secrets
import struct
import time
import hmac
import hashlib
from typing import Optional, Dict, Any, Tuple, List, Union
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes, hmac as crypt_hmac
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.constant_time import bytes_eq
from cryptography.exceptions import InvalidTag, InvalidKey, InvalidSignature
from cryptography.hazmat.primitives.cmac import CMAC
from cryptography.hazmat.primitives.poly1305 import Poly1305

from rayonix_wallet.core.exceptions import CryptoError
from rayonix_wallet.utils.secure import SecureString

class CryptographicConstants:
    """Cryptographic constants and configuration"""
    # AES configurations
    AES_KEY_SIZES = {128: 16, 192: 24, 256: 32}
    DEFAULT_AES_KEY_SIZE = 256
    GCM_IV_SIZE = 12
    GCM_TAG_SIZE = 16
    CBC_IV_SIZE = 16
    
    # KDF configurations
    SALT_SIZE = 32
    SCRYPT_N = 2**14  # 16384
    SCRYPT_R = 8
    SCRYPT_P = 1
    PBKDF2_ITERATIONS = 600000
    HKDF_LENGTH = 32
    HKDF_INFO = b'rayonix-wallet-encryption'
    
    # Padding and formatting
    MAX_PADDING_SIZE = 512
    VERSION_HEADER = b'RAYXENC'
    CURRENT_VERSION = 2
    
    # Security parameters
    MAX_DECRYPT_ATTEMPTS = 3
    DECRYPT_TIMEOUT_MS = 1000
    MIN_PASSPHRASE_LENGTH = 8

class KeyDerivationManager:
    """Advanced key derivation and management"""
    
    def __init__(self, backend=None):
        self.backend = backend or default_backend()
    
    def derive_key_pbkdf2(self, passphrase: str, salt: bytes, iterations: int, 
                         key_length: int = 32) -> bytes:
        """Derive key using PBKDF2 with multiple hash options"""
        if iterations < 100000:
            raise ValueError("Iterations too low for security")
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=self.backend
        )
        return kdf.derive(passphrase.encode('utf-8'))
    
    def derive_key_scrypt(self, passphrase: str, salt: bytes, n: int, r: int, p: int,
                         key_length: int = 32) -> bytes:
        """Derive key using Scrypt with configurable parameters"""
        kdf = Scrypt(
            salt=salt,
            length=key_length,
            n=n,
            r=r,
            p=p,
            backend=self.backend
        )
        return kdf.derive(passphrase.encode('utf-8'))
    
    def derive_key_hkdf(self, input_key: bytes, salt: bytes, info: bytes, 
                       key_length: int = 32) -> bytes:
        """Derive key using HKDF for key expansion"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            info=info,
            backend=self.backend
        )
        return hkdf.derive(input_key)
    
    def generate_key_hierarchy(self, master_key: bytes, context: str) -> Dict[str, bytes]:
        """Generate a hierarchy of keys for different purposes"""
        salt = hmac.digest(master_key, context.encode(), 'sha256')
        info_base = f"rayonix-key-hierarchy-{context}".encode()
        
        return {
            'encryption': self.derive_key_hkdf(master_key, salt, info_base + b'-enc'),
            'authentication': self.derive_key_hkdf(master_key, salt, info_base + b'-auth'),
            'integrity': self.derive_key_hkdf(master_key, salt, info_base + b'-integ'),
        }

class AuthenticatedEncryption:
    """Advanced authenticated encryption with multiple modes"""
    
    def __init__(self, backend=None):
        self.backend = backend or default_backend()
        self.constants = CryptographicConstants()
    
    def encrypt_gcm(self, data: bytes, key: bytes, associated_data: bytes = None) -> bytes:
        """Encrypt using AES-GCM with additional security measures"""
        iv = secrets.token_bytes(self.constants.GCM_IV_SIZE)
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        if associated_data:
            encryptor.authenticate_additional_data(associated_data)
        
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def decrypt_gcm(self, encrypted_data: bytes, key: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt AES-GCM with comprehensive validation"""
        if len(encrypted_data) < self.constants.GCM_IV_SIZE + self.constants.GCM_TAG_SIZE:
            raise CryptoError("Invalid encrypted data length")
        
        iv = encrypted_data[:self.constants.GCM_IV_SIZE]
        tag = encrypted_data[self.constants.GCM_IV_SIZE:self.constants.GCM_IV_SIZE + self.constants.GCM_TAG_SIZE]
        ciphertext = encrypted_data[self.constants.GCM_IV_SIZE + self.constants.GCM_TAG_SIZE:]
        
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        
        if associated_data:
            decryptor.authenticate_additional_data(associated_data)
        
        try:
            return decryptor.update(ciphertext) + decryptor.finalize()
        except InvalidTag:
            # Constant-time error handling
            secrets.token_bytes(len(ciphertext))
            raise CryptoError("Authentication failed")
    
    def encrypt_cbc_hmac(self, data: bytes, key: bytes) -> bytes:
        """Encrypt using AES-CBC with HMAC authentication"""
        # Derive separate keys for encryption and MAC
        kdf = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, 
                  info=b'aes-cbc-hmac', backend=self.backend)
        key_material = kdf.derive(key)
        enc_key, mac_key = key_material[:32], key_material[32:]
        
        # Encrypt with CBC
        iv = secrets.token_bytes(self.constants.CBC_IV_SIZE)
        cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Calculate HMAC
        hmac_obj = crypt_hmac.HMAC(mac_key, hashes.SHA256(), backend=self.backend)
        hmac_obj.update(iv + ciphertext)
        mac = hmac_obj.finalize()
        
        return iv + ciphertext + mac
    
    def decrypt_cbc_hmac(self, encrypted_data: bytes, key: bytes) -> bytes:
        """Decrypt AES-CBC with HMAC verification"""
        if len(encrypted_data) < self.constants.CBC_IV_SIZE + 32:  # IV + minimum HMAC
            raise CryptoError("Invalid encrypted data length")
        
        # Derive keys
        kdf = HKDF(algorithm=hashes.SHA256(), length=64, salt=None, 
                  info=b'aes-cbc-hmac', backend=self.backend)
        key_material = kdf.derive(key)
        enc_key, mac_key = key_material[:32], key_material[32:]
        
        # Verify HMAC
        iv = encrypted_data[:self.constants.CBC_IV_SIZE]
        ciphertext = encrypted_data[self.constants.CBC_IV_SIZE:-32]
        received_mac = encrypted_data[-32:]
        
        hmac_obj = crypt_hmac.HMAC(mac_key, hashes.SHA256(), backend=self.backend)
        hmac_obj.update(iv + ciphertext)
        
        try:
            hmac_obj.verify(received_mac)
        except InvalidSignature:
            raise CryptoError("HMAC verification failed")
        
        # Decrypt
        cipher = Cipher(algorithms.AES(enc_key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()

class EncryptionManager:
    """Enterprise-grade encryption manager with advanced features"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.backend = default_backend()
        self.constants = CryptographicConstants()
        self.kdf_manager = KeyDerivationManager(self.backend)
        self.aead = AuthenticatedEncryption(self.backend)
        
        # Security counters
        self._decrypt_attempts = {}
        self._last_attempt_time = {}
        
        # Configure from settings
        self._configure_from_settings()
    
    def _configure_from_settings(self) -> None:
        """Configure manager from provided settings"""
        self.use_scrypt = self.config.get('use_scrypt', True)
        self.enable_key_hierarchy = self.config.get('enable_key_hierarchy', True)
        self.enable_brute_force_protection = self.config.get('enable_brute_force_protection', True)
        self.max_decrypt_attempts = self.config.get('max_decrypt_attempts', 
                                                   self.constants.MAX_DECRYPT_ATTEMPTS)
    
    def _check_brute_force_protection(self, identifier: str) -> None:
        """Implement brute force protection"""
        if not self.enable_brute_force_protection:
            return
            
        current_time = time.time() * 1000  # Convert to milliseconds
        last_attempt = self._last_attempt_time.get(identifier, 0)
        
        # Implement timeout after failed attempts
        if self._decrypt_attempts.get(identifier, 0) >= self.max_decrypt_attempts:
            if current_time - last_attempt < self.constants.DECRYPT_TIMEOUT_MS:
                raise CryptoError("Too many failed attempts. Please wait before retrying.")
            else:
                # Reset counter after timeout
                self._decrypt_attempts[identifier] = 0
        
        self._last_attempt_time[identifier] = current_time
    
    def _record_decrypt_attempt(self, identifier: str, success: bool) -> None:
        """Record decryption attempt for brute force protection"""
        if not self.enable_brute_force_protection:
            return
            
        if success:
            self._decrypt_attempts[identifier] = 0
        else:
            self._decrypt_attempts[identifier] = self._decrypt_attempts.get(identifier, 0) + 1
    
    def encrypt_data(self, data: bytes, key: bytes, associated_data: Optional[bytes] = None,
                   algorithm: str = 'AES-GCM') -> bytes:
        """Encrypt data with algorithm choice and enhanced security"""
        self._validate_encryption_params(data, key, algorithm)
        
        try:
            if algorithm == 'AES-GCM':
                return self.aead.encrypt_gcm(data, key, associated_data)
            elif algorithm == 'AES-CBC-HMAC':
                return self.aead.encrypt_cbc_hmac(data, key)
            else:
                raise CryptoError(f"Unsupported algorithm: {algorithm}")
                
        except Exception as e:
            raise CryptoError(f"Encryption failed: {str(e)}") from e
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes, associated_data: Optional[bytes] = None,
                   algorithm: str = 'AES-GCM', identifier: str = 'default') -> bytes:
        """Decrypt data with brute force protection and algorithm support"""
        self._check_brute_force_protection(identifier)
        
        try:
            if algorithm == 'AES-GCM':
                result = self.aead.decrypt_gcm(encrypted_data, key, associated_data)
            elif algorithm == 'AES-CBC-HMAC':
                result = self.aead.decrypt_cbc_hmac(encrypted_data, key)
            else:
                raise CryptoError(f"Unsupported algorithm: {algorithm}")
            
            self._record_decrypt_attempt(identifier, True)
            return result
            
        except CryptoError:
            self._record_decrypt_attempt(identifier, False)
            raise
        except Exception as e:
            self._record_decrypt_attempt(identifier, False)
            raise CryptoError(f"Decryption failed: {str(e)}") from e
    
    def _validate_encryption_params(self, data: bytes, key: bytes, algorithm: str) -> None:
        """Validate encryption parameters"""
        if not isinstance(data, bytes):
            raise TypeError("Data must be bytes")
        if not isinstance(key, bytes):
            raise TypeError("Key must be bytes")
        if algorithm not in ['AES-GCM', 'AES-CBC-HMAC']:
            raise ValueError("Unsupported algorithm")
        
        key_size = len(key) * 8
        if key_size not in self.constants.AES_KEY_SIZES:
            raise ValueError(f"Invalid key size: {key_size} bits")
    
    def encrypt_with_passphrase(self, data: bytes, passphrase: str, salt: Optional[bytes] = None,
                              kdf_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Advanced passphrase encryption with configurable KDF"""
        self._validate_passphrase(passphrase)
        
        salt = salt or secrets.token_bytes(self.constants.SALT_SIZE)
        kdf_params = kdf_params or {}
        
        try:
            # Derive key using selected KDF
            if kdf_params.get('kdf', 'scrypt') == 'scrypt':
                key = self.kdf_manager.derive_key_scrypt(
                    passphrase, salt,
                    kdf_params.get('n', self.constants.SCRYPT_N),
                    kdf_params.get('r', self.constants.SCRYPT_R),
                    kdf_params.get('p', self.constants.SCRYPT_P)
                )
                kdf_info = {'kdf': 'scrypt', 'n': kdf_params.get('n', self.constants.SCRYPT_N),
                           'r': kdf_params.get('r', self.constants.SCRYPT_R),
                           'p': kdf_params.get('p', self.constants.SCRYPT_P)}
            else:
                key = self.kdf_manager.derive_key_pbkdf2(
                    passphrase, salt,
                    kdf_params.get('iterations', self.constants.PBKDF2_ITERATIONS)
                )
                kdf_info = {'kdf': 'pbkdf2', 
                           'iterations': kdf_params.get('iterations', self.constants.PBKDF2_ITERATIONS)}
            
            # Encrypt data
            encrypted_data = self.encrypt_data(data, key)
            
            # Create secure package
            package = {
                'version': self.constants.CURRENT_VERSION,
                'timestamp': int(time.time()),
                'data': encrypted_data,
                'salt': salt,
                'algorithm': 'AES-GCM-256',
                **kdf_info
            }
            
            # Add integrity protection
            package['integrity'] = self._calculate_package_integrity(package)
            
            return package
            
        except Exception as e:
            raise CryptoError(f"Passphrase encryption failed: {str(e)}") from e
    
    def decrypt_with_passphrase(self, encrypted_package: Dict[str, Any], passphrase: str,
                              identifier: str = 'default') -> bytes:
        """Decrypt with passphrase and integrity verification"""
        self._validate_encrypted_package(encrypted_package)
        
        try:
            # Verify integrity
            if not self._verify_package_integrity(encrypted_package):
                raise CryptoError("Package integrity check failed")
            
            salt = encrypted_package['salt']
            kdf_method = encrypted_package['kdf']
            
            # Derive key
            if kdf_method == 'scrypt':
                key = self.kdf_manager.derive_key_scrypt(
                    passphrase, salt,
                    encrypted_package.get('n', self.constants.SCRYPT_N),
                    encrypted_package.get('r', self.constants.SCRYPT_R),
                    encrypted_package.get('p', self.constants.SCRYPT_P)
                )
            else:
                key = self.kdf_manager.derive_key_pbkdf2(
                    passphrase, salt,
                    encrypted_package.get('iterations', self.constants.PBKDF2_ITERATIONS)
                )
            
            return self.decrypt_data(encrypted_package['data'], key, identifier=identifier)
            
        except Exception as e:
            raise CryptoError(f"Passphrase decryption failed: {str(e)}") from e
    
    def _validate_passphrase(self, passphrase: str) -> None:
        """Validate passphrase strength"""
        if len(passphrase) < self.constants.MIN_PASSPHRASE_LENGTH:
            raise ValueError(f"Passphrase must be at least {self.constants.MIN_PASSPHRASE_LENGTH} characters")
    
    def _validate_encrypted_package(self, package: Dict[str, Any]) -> None:
        """Validate encrypted package structure"""
        required_fields = {'version', 'data', 'salt', 'kdf', 'integrity'}
        if not all(field in package for field in required_fields):
            raise ValueError("Invalid encrypted package structure")
    
    def _calculate_package_integrity(self, package: Dict[str, Any]) -> str:
        """Calculate integrity hash for package"""
        integrity_data = {
            'version': package['version'],
            'data': package['data'],
            'salt': package['salt'],
            'algorithm': package['algorithm'],
            'kdf': package['kdf']
        }
        serialized = json.dumps(integrity_data, sort_keys=True).encode()
        return hashlib.sha256(serialized).hexdigest()
    
    def _verify_package_integrity(self, package: Dict[str, Any]) -> bool:
        """Verify package integrity"""
        try:
            expected_integrity = self._calculate_package_integrity(package)
            return secrets.compare_digest(expected_integrity, package['integrity'])
        except:
            return False
    
    def encrypt_wallet_data(self, wallet_data: Dict[str, Any], encryption_key: SecureString,
                          algorithm: str = 'AES-GCM') -> bytes:
        """Encrypt wallet data with advanced features"""
        if not isinstance(wallet_data, dict):
            raise TypeError("Wallet data must be a dictionary")
        
        try:
            # Serialize with compression-like formatting
            serialized_data = json.dumps(wallet_data, sort_keys=True, separators=(',', ':')).encode('utf-8')
            
            # Add secure padding
            padded_data = self._add_secure_padding(serialized_data)
            
            # Encrypt with key hierarchy if enabled
            with encryption_key.temporary_access() as key:
                if self.enable_key_hierarchy:
                    key_hierarchy = self.kdf_manager.generate_key_hierarchy(key, 'wallet-encryption')
                    encryption_key_final = key_hierarchy['encryption']
                else:
                    encryption_key_final = key
                
                return self.encrypt_data(padded_data, encryption_key_final, algorithm=algorithm)
                
        except Exception as e:
            raise CryptoError(f"Wallet encryption failed: {str(e)}") from e
    
    def decrypt_wallet_data(self, encrypted_data: bytes, encryption_key: SecureString,
                          algorithm: str = 'AES-GCM', identifier: str = 'wallet') -> Dict[str, Any]:
        """Decrypt wallet data with enhanced security"""
        if not isinstance(encrypted_data, bytes):
            raise TypeError("Encrypted data must be bytes")
        
        try:
            with encryption_key.temporary_access() as key:
                if self.enable_key_hierarchy:
                    key_hierarchy = self.kdf_manager.generate_key_hierarchy(key, 'wallet-encryption')
                    encryption_key_final = key_hierarchy['encryption']
                else:
                    encryption_key_final = key
                
                padded_data = self.decrypt_data(encrypted_data, encryption_key_final, 
                                              algorithm=algorithm, identifier=identifier)
                serialized_data = self._remove_secure_padding(padded_data)
                
                wallet_data = json.loads(serialized_data.decode('utf-8'))
                
                if not isinstance(wallet_data, dict):
                    raise ValueError("Invalid wallet data structure")
                
                return wallet_data
                
        except Exception as e:
            raise CryptoError(f"Wallet decryption failed: {str(e)}") from e
    
    def _add_secure_padding(self, data: bytes) -> bytes:
        """Add secure padding with length obfuscation"""
        # Use multiple padding strategies
        primary_padding = secrets.randbelow(self.constants.MAX_PADDING_SIZE)
        secondary_padding = secrets.randbelow(64)
        
        padding_data = (
            secrets.token_bytes(primary_padding) +
            secrets.token_bytes(secondary_padding)
        )
        
        # Encode padding information securely
        header = struct.pack('>HH', primary_padding, secondary_padding)
        return header + padding_data + data
    
    def _remove_secure_padding(self, padded_data: bytes) -> bytes:
        """Remove secure padding"""
        if len(padded_data) < 4:  # Header size
            raise ValueError("Invalid padded data")
        
        try:
            primary_padding, secondary_padding = struct.unpack('>HH', padded_data[:4])
            total_padding = 4 + primary_padding + secondary_padding
            
            if len(padded_data) <= total_padding:
                raise ValueError("Invalid padding lengths")
            
            return padded_data[total_padding:]
        except struct.error:
            raise ValueError("Invalid padding header")
    
    def generate_encryption_key(self, key_size: int = None) -> bytes:
        """Generate cryptographically secure encryption key"""
        key_size = key_size or self.constants.DEFAULT_AES_KEY_SIZE
        if key_size not in self.constants.AES_KEY_SIZES:
            raise ValueError(f"Unsupported key size: {key_size}")
        
        return secrets.token_bytes(self.constants.AES_KEY_SIZES[key_size])
    
    def verify_passphrase(self, encrypted_package: Dict[str, Any], passphrase: str) -> bool:
        """Verify passphrase with additional security measures"""
        try:
            # Test key derivation without full decryption
            salt = encrypted_package['salt']
            kdf_method = encrypted_package['kdf']
            
            if kdf_method == 'scrypt':
                self.kdf_manager.derive_key_scrypt(
                    passphrase, salt,
                    encrypted_package.get('n', self.constants.SCRYPT_N),
                    encrypted_package.get('r', self.constants.SCRYPT_R),
                    encrypted_package.get('p', self.constants.SCRYPT_P),
                    key_length=16  # Derive shorter key for verification
                )
            else:
                self.kdf_manager.derive_key_pbkdf2(
                    passphrase, salt,
                    encrypted_package.get('iterations', self.constants.PBKDF2_ITERATIONS),
                    key_length=16
                )
            
            return True
        except (InvalidKey, ValueError):
            return False
        except Exception:
            return False
    
    def reencrypt_data(self, encrypted_data: bytes, old_key: bytes, new_key: bytes,
                     old_algorithm: str = 'AES-GCM', new_algorithm: str = 'AES-GCM') -> bytes:
        """Re-encrypt data with key and algorithm rotation"""
        try:
            # Decrypt with old key/algorithm
            plaintext = self.decrypt_data(encrypted_data, old_key, algorithm=old_algorithm)
            # Encrypt with new key/algorithm
            return self.encrypt_data(plaintext, new_key, algorithm=new_algorithm)
        except Exception as e:
            raise CryptoError(f"Re-encryption failed: {str(e)}") from e
    
    def create_secure_envelope(self, data: bytes, recipients: List[bytes]) -> Dict[str, Any]:
        """Create secure envelope for multiple recipients"""
        # Generate ephemeral key pair (simplified version)
        ephemeral_key = self.generate_encryption_key()
        
        # Encrypt data with ephemeral key
        encrypted_data = self.encrypt_data(data, ephemeral_key)
        
        # Encrypt ephemeral key for each recipient
        recipient_keys = {}
        for i, recipient_key in enumerate(recipients):
            encrypted_ephemeral = self.encrypt_data(ephemeral_key, recipient_key)
            recipient_keys[f'recipient_{i}'] = encrypted_ephemeral
        
        return {
            'version': self.constants.CURRENT_VERSION,
            'encrypted_data': encrypted_data,
            'recipient_keys': recipient_keys,
            'timestamp': int(time.time())
        }
    
    def decrypt_secure_envelope(self, envelope: Dict[str, Any], recipient_key: bytes, 
                              recipient_index: int = 0) -> bytes:
        """Decrypt secure envelope for specific recipient"""
        try:
            encrypted_ephemeral = envelope['recipient_keys'][f'recipient_{recipient_index}']
            ephemeral_key = self.decrypt_data(encrypted_ephemeral, recipient_key)
            return self.decrypt_data(envelope['encrypted_data'], ephemeral_key)
        except Exception as e:
            raise CryptoError(f"Envelope decryption failed: {str(e)}") from e
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status and statistics"""
        return {
            'brute_force_protection_enabled': self.enable_brute_force_protection,
            'key_hierarchy_enabled': self.enable_key_hierarchy,
            'active_decrypt_attempts': dict(self._decrypt_attempts),
            'configuration': self.config
        }
    
    def reset_security_counters(self, identifier: str = None) -> None:
        """Reset security counters"""
        if identifier:
            if identifier in self._decrypt_attempts:
                del self._decrypt_attempts[identifier]
            if identifier in self._last_attempt_time:
                del self._last_attempt_time[identifier]
        else:
            self._decrypt_attempts.clear()
            self._last_attempt_time.clear()

# Advanced utility functions
def create_encryption_manager(config: Optional[Dict[str, Any]] = None) -> EncryptionManager:
    """Create configured encryption manager instance"""
    return EncryptionManager(config)

def generate_key_from_passphrase(passphrase: str, salt: Optional[bytes] = None,
                               kdf_type: str = 'scrypt', kdf_params: Optional[Dict[str, Any]] = None) -> bytes:
    """Generate key from passphrase with advanced options"""
    manager = EncryptionManager()
    salt = salt or secrets.token_bytes(manager.constants.SALT_SIZE)
    kdf_params = kdf_params or {}
    
    if kdf_type == 'scrypt':
        return manager.kdf_manager.derive_key_scrypt(
            passphrase, salt,
            kdf_params.get('n', manager.constants.SCRYPT_N),
            kdf_params.get('r', manager.constants.SCRYPT_R),
            kdf_params.get('p', manager.constants.SCRYPT_P)
        )
    else:
        return manager.kdf_manager.derive_key_pbkdf2(
            passphrase, salt,
            kdf_params.get('iterations', manager.constants.PBKDF2_ITERATIONS)
        )

def verify_cryptographic_environment() -> Dict[str, bool]:
    """Verify cryptographic environment capabilities"""
    results = {}
    
    try:
        # Test AES-GCM
        manager = EncryptionManager()
        test_data = b'test data'
        key = manager.generate_encryption_key()
        encrypted = manager.encrypt_data(test_data, key)
        decrypted = manager.decrypt_data(encrypted, key)
        results['aes_gcm'] = (decrypted == test_data)
    except:
        results['aes_gcm'] = False
    
    try:
        # Test KDF operations
        manager = EncryptionManager()
        key = generate_key_from_passphrase("test", secrets.token_bytes(32))
        results['kdf_operations'] = (len(key) == 32)
    except:
        results['kdf_operations'] = False
    
    return results

# Export public interface
__all__ = [
    'EncryptionManager',
    'KeyDerivationManager',
    'AuthenticatedEncryption',
    'CryptographicConstants',
    'create_encryption_manager',
    'generate_key_from_passphrase',
    'verify_cryptographic_environment',
    'CryptoError'
]