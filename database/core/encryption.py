from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from typing import Optional
import os
import logging

from ..utils.exceptions import EncryptionError

logger = logging.getLogger(__name__)

class AES256Encryption:
    """AES-256 encryption implementation"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or os.urandom(32)
        self.iv_length = 16
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-CBC"""
        try:
            iv = os.urandom(self.iv_length)
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            padder = padding.PKCS7(128).padder()
            
            padded_data = padder.update(data) + padder.finalize()
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            
            return iv + encrypted
        except Exception as e:
            raise EncryptionError(f"AES encryption failed: {e}")
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt AES-256-CBC data"""
        try:
            iv = data[:self.iv_length]
            encrypted = data[self.iv_length:]
            
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            unpadder = padding.PKCS7(128).unpadder()
            
            padded = decryptor.update(encrypted) + decryptor.finalize()
            return unpadder.update(padded) + unpadder.finalize()
        except Exception as e:
            raise EncryptionError(f"AES decryption failed: {e}")

class ChaCha20Encryption:
    """ChaCha20 encryption implementation"""
    
    def __init__(self, key: Optional[bytes] = None):
        self.key = key or os.urandom(32)
        self.nonce_length = 12
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt data using ChaCha20"""
        try:
            nonce = os.urandom(self.nonce_length)
            cipher = Cipher(
                algorithms.ChaCha20(self.key, nonce),
                mode=None,
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(data)
            
            return nonce + encrypted
        except Exception as e:
            raise EncryptionError(f"ChaCha20 encryption failed: {e}")
    
    def decrypt(self, data: bytes) -> bytes:
        """Decrypt ChaCha20 data"""
        try:
            nonce = data[:self.nonce_length]
            encrypted = data[self.nonce_length:]
            
            cipher = Cipher(
                algorithms.ChaCha20(self.key, nonce),
                mode=None,
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            return decryptor.update(encrypted)
        except Exception as e:
            raise EncryptionError(f"ChaCha20 decryption failed: {e}")