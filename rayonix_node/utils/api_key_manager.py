# rayonix_node/utils/api_key_manager.py
import secrets
import hashlib
import base64
import re
from typing import Optional, Tuple
import logging

logger = logging.getLogger("rayonix_node.api_key_manager")

class APIKeyManager:
    """Cryptographically secure API key generation and validation"""
    
    # Security constants
    MIN_KEY_LENGTH = 64
    RECOMMENDED_KEY_LENGTH = 128
    MAX_KEY_LENGTH = 256
    ENTROPY_BYTES = 96  # 768 bits of entropy
    
    # Character sets
    BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    URL_SAFE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    
    @classmethod
    def generate_strong_api_key(cls, length: int = RECOMMENDED_KEY_LENGTH, 
                              url_safe: bool = True) -> str:
        """
        Generate cryptographically secure API key
        
        Args:
            length: Key length (64-256 characters)
            url_safe: Use URL-safe characters
            
        Returns:
            Strong API key string
        """
        if length < cls.MIN_KEY_LENGTH or length > cls.MAX_KEY_LENGTH:
            raise ValueError(f"Key length must be between {cls.MIN_KEY_LENGTH} and {cls.MAX_KEY_LENGTH}")
        
        # Generate high-entropy random bytes
        entropy = secrets.token_bytes(cls.ENTROPY_BYTES)
        
        # Add additional entropy sources
        system_entropy = str(secrets.randbits(256)).encode()
        time_entropy = str(secrets.token_bytes(32)).encode()
        
        # Combine entropy sources
        combined_entropy = entropy + system_entropy + time_entropy
        
        # Hash for uniform distribution
        hashed_entropy = hashlib.sha3_512(combined_entropy).digest()
        
        # Encode to base64
        if url_safe:
            api_key_bytes = base64.urlsafe_b64encode(hashed_entropy)
            api_key = api_key_bytes.decode('ascii').rstrip('=')
        else:
            api_key_bytes = base64.b64encode(hashed_entropy)
            api_key = api_key_bytes.decode('ascii').rstrip('=')
        
        # Trim or extend to desired length
        if len(api_key) > length:
            api_key = api_key[:length]
        elif len(api_key) < length:
            # Pad with additional entropy if needed
            padding = secrets.token_urlsafe(length - len(api_key))
            api_key += padding[:length - len(api_key)]
        
        logger.info(f"Generated strong API key with {len(api_key)*6} bits of entropy")
        return api_key
    
    @classmethod
    def validate_key_strength(cls, api_key: str) -> Tuple[bool, str]:
        """
        Validate API key strength
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, reason)
        """
        if not api_key:
            return False, "API key cannot be empty"
        
        # Length check
        if len(api_key) < cls.MIN_KEY_LENGTH:
            return False, f"Key too short: {len(api_key)} < {cls.MIN_KEY_LENGTH}"
        
        if len(api_key) > cls.MAX_KEY_LENGTH:
            return False, f"Key too long: {len(api_key)} > {cls.MAX_KEY_LENGTH}"
        
        # Character diversity check
        unique_chars = len(set(api_key))
        if unique_chars < (len(api_key) * 0.5):  # At least 50% unique characters
            return False, f"Insufficient character diversity: {unique_chars}/{len(api_key)} unique"
        
        # Entropy estimation (simplified)
        char_set_size = len(set(api_key))
        estimated_entropy = len(api_key) * (char_set_size.bit_length())
        
        if estimated_entropy < 256:  # Minimum 256 bits effective entropy
            return False, f"Insufficient entropy: ~{estimated_entropy} bits"
        
        # Pattern checks
        if cls._contains_common_patterns(api_key):
            return False, "Key contains common patterns or sequences"
        
        return True, f"Strong key: {len(api_key)} chars, ~{estimated_entropy} bits entropy"
    
    @classmethod
    def _contains_common_patterns(cls, key: str) -> bool:
        """Check for common weak patterns"""
        patterns = [
            r'(.)\1{3,}',  # Repeated characters (4+ times)
            r'1234|2345|3456|4567|5678|6789|7890',  # Sequences
            r'(abc|def|ghi|jkl|mno|pqr|stu|vwx|yz)',  # Keyboard rows
            r'(qwer|asdf|zxcv)',  # Keyboard columns
        ]
        
        for pattern in patterns:
            if re.search(pattern, key.lower()):
                return True
        
        return False
    
    @classmethod
    def generate_key_pair(cls) -> Tuple[str, str]:
        """
        Generate API key and its hash for storage
        
        Returns:
            Tuple of (api_key, api_key_hash)
        """
        api_key = cls.generate_strong_api_key()
        api_key_hash = cls.hash_api_key(api_key)
        
        return api_key, api_key_hash
    
    @classmethod
    def hash_api_key(cls, api_key: str) -> str:
        """Create secure hash of API key for storage"""
        # Use slow hash for brute force protection
        salt = secrets.token_bytes(32)
        iterations = 100000
        
        key_hash = hashlib.pbkdf2_hmac(
            'sha256',
            api_key.encode('utf-8'),
            salt,
            iterations
        )
        
        # Combine salt and hash for storage
        stored_hash = salt.hex() + key_hash.hex()
        return stored_hash
    
    @classmethod
    def verify_api_key(cls, provided_key: str, stored_hash: str) -> bool:
        """Verify API key against stored hash"""
        try:
            # Extract salt and hash
            salt_hex = stored_hash[:64]  # 32 bytes in hex
            expected_hash_hex = stored_hash[64:]
            
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(expected_hash_hex)
            
            # Recompute hash
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                provided_key.encode('utf-8'),
                salt,
                100000  # Same iterations as generation
            )
            
            # Constant-time comparison
            return secrets.compare_digest(computed_hash.hex(), expected_hash_hex)
            
        except Exception:
            return False

# Utility functions for easy use
def generate_api_key(length: int = 128) -> str:
    """Convenience function to generate strong API key"""
    return APIKeyManager.generate_strong_api_key(length)

def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """Convenience function to validate API key strength"""
    return APIKeyManager.validate_key_strength(api_key)