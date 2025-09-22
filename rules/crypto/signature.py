"""
Cryptographic signature operations for consensus system
"""

from typing import Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.serialization import load_der_public_key
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import hashlib
import logging

from ..exceptions import CryptoError

logger = logging.getLogger('consensus.crypto')

def sign_message(private_key, message: bytes, algorithm: str = "secp256k1") -> bytes:
    """Sign message with private key"""
    try:
        if algorithm == "secp256k1":
            signature = private_key.sign(
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return signature
        elif algorithm == "rsa":
            signature = private_key.sign(
                message,
                hashes.SHA256(),
                padding=padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                )
            )
            return signature
        else:
            raise CryptoError(f"Unsupported algorithm: {algorithm}")
            
    except Exception as e:
        raise CryptoError(f"Failed to sign message: {e}")

def verify_signature(address: str, message: bytes, signature: bytes, algorithm: str = "secp256k1") -> bool:
    """Verify signature against address and message"""
    try:
        # Recover address from signature and compare
        recovered_addr = recover_address(message, signature, algorithm)
        return recovered_addr == address
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return False

def recover_address(message: bytes, signature: bytes, algorithm: str = "secp256k1") -> Optional[str]:
    """Recover address from signature (for secp256k1)"""
    try:
        if algorithm != "secp256k1":
            raise CryptoError("Address recovery only supported for secp256k1")
        
        # This is a simplified implementation
        # In production, use proper elliptic curve recovery
        import ecdsa
        from ecdsa import VerifyingKey, SECP256k1
        
        # Create verifying key from signature
        vk = VerifyingKey.from_public_key_recovery(
            signature,
            message,
            SECP256k1,
            hashfunc=hashlib.sha256
        )[0]
        
        # Get compressed public key
        public_key = vk.to_string("compressed")
        
        # Derive address from public key
        address_hash = hashlib.sha256(public_key).hexdigest()[:40]
        return f"0x{address_hash}"
        
    except Exception as e:
        logger.error(f"Failed to recover address: {e}")
        return None

def verify_signature_with_public_key(public_key_der: bytes, message: bytes, signature: bytes, algorithm: str = "secp256k1") -> bool:
    """Verify signature using public key"""
    try:
        public_key = load_der_public_key(public_key_der, backend=default_backend())
        
        if algorithm == "secp256k1":
            public_key.verify(
                signature,
                message,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        elif algorithm == "rsa":
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        else:
            raise CryptoError(f"Unsupported algorithm: {algorithm}")
            
    except InvalidSignature:
        return False
    except Exception as e:
        logger.error(f"Signature verification error: {e}")
        return False

def hash_message(message: bytes, algorithm: str = "sha256") -> bytes:
    """Hash message using specified algorithm"""
    try:
        if algorithm == "sha256":
            return hashlib.sha256(message).digest()
        elif algorithm == "sha3_256":
            return hashlib.sha3_256(message).digest()
        elif algorithm == "blake2b":
            return hashlib.blake2b(message).digest()
        else:
            raise CryptoError(f"Unsupported hash algorithm: {algorithm}")
    except Exception as e:
        raise CryptoError(f"Failed to hash message: {e}")

def create_deterministic_nonce(message: bytes, private_key: bytes) -> bytes:
    """Create deterministic nonce for signing (RFC 6979)"""
    try:
        # Simplified implementation - in production use proper RFC 6979
        return hashlib.sha256(message + private_key).digest()
    except Exception as e:
        raise CryptoError(f"Failed to create nonce: {e}")

def verify_aggregated_signature(public_keys: list, message: bytes, signature: bytes, algorithm: str = "secp256k1") -> bool:
    """Verify aggregated signature from multiple signers"""
    try:
        # This would implement BLS or other aggregate signature verification
        # Placeholder implementation
        return all(
            verify_signature_with_public_key(pk, message, signature, algorithm)
            for pk in public_keys
        )
    except Exception as e:
        logger.error(f"Aggregated signature verification failed: {e}")
        return False