# smart_contract/utils/cryptography_utils.py
import os
import logging
import hashlib
import hmac
from typing import Optional
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import ecdsa
from ecdsa import SigningKey, VerifyingKey, SECP256k1

logger = logging.getLogger("SmartContract.Crypto")

def encrypt_data(data: bytes, key: bytes) -> bytes:
    """Encrypt data using AES-256-GCM"""
    try:
        # Generate random nonce
        nonce = get_random_bytes(12)
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        
        # Encrypt and get tag
        ciphertext, tag = cipher.encrypt_and_digest(data)
        
        # Combine nonce + tag + ciphertext
        return nonce + tag + ciphertext
        
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """Decrypt data using AES-256-GCM"""
    try:
        # Split components
        nonce = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        # Create cipher
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        
        # Decrypt and verify
        return cipher.decrypt_and_verify(ciphertext, tag)
        
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise

def derive_key(password: str, salt: Optional[bytes] = None, iterations: int = 100000) -> bytes:
    """Derive encryption key from password using PBKDF2"""
    if salt is None:
        salt = get_random_bytes(16)
    
    return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations, 32)

def generate_encryption_key() -> bytes:
    """Generate a random encryption key"""
    return get_random_bytes(32)

def validate_signature(message: bytes, signature: bytes, public_key: bytes) -> bool:
    """Validate ECDSA signature"""
    try:
        # Create verifying key
        vk = VerifyingKey.from_string(public_key, curve=SECP256k1)
        
        # Verify signature
        return vk.verify(signature, message)
        
    except Exception as e:
        logger.error(f"Signature validation failed: {e}")
        return False

def generate_key_pair() -> tuple[bytes, bytes]:
    """Generate ECDSA key pair"""
    try:
        # Generate private key
        sk = SigningKey.generate(curve=SECP256k1)
        
        # Get public key
        vk = sk.get_verifying_key()
        
        return sk.to_string(), vk.to_string()
        
    except Exception as e:
        logger.error(f"Key generation failed: {e}")
        raise

def hash_data(data: bytes, algorithm: str = 'sha256') -> bytes:
    """Hash data using specified algorithm"""
    if algorithm == 'sha256':
        return hashlib.sha256(data).digest()
    elif algorithm == 'keccak256':
        from Crypto.Hash import keccak
        k = keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def hmac_sign(data: bytes, key: bytes, algorithm: str = 'sha256') -> bytes:
    """Generate HMAC signature"""
    if algorithm == 'sha256':
        return hmac.new(key, data, hashlib.sha256).digest()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

def generate_nonce() -> bytes:
    """Generate cryptographic nonce"""
    return get_random_bytes(16)

def constant_time_compare(a: bytes, b: bytes) -> bool:
    """Constant-time comparison to prevent timing attacks"""
    return hmac.compare_digest(a, b)