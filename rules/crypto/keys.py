"""
Cryptographic key management for consensus system
"""

import os
import json
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidKey
import logging

from rules.exceptions import CryptoError

logger = logging.getLogger('consensus.crypto')

class KeyManager:
    """Cryptographic key management"""
    
    def __init__(self, key_dir: str = "./keys", key_algorithm: str = "secp256k1"):
        self.key_dir = key_dir
        self.key_algorithm = key_algorithm
        os.makedirs(key_dir, exist_ok=True)
        
        self.private_key = None
        self.public_key = None
        self.address = None
        
        self._load_or_generate_keys()
    
    def _load_or_generate_keys(self) -> None:
        """Load existing keys or generate new ones"""
        key_file = os.path.join(self.key_dir, "validator.json")
        
        if os.path.exists(key_file):
            try:
                self._load_keys(key_file)
                logger.info("Loaded existing validator keys")
            except Exception as e:
                logger.warning(f"Failed to load keys: {e}. Generating new keys...")
                self._generate_keys()
                self._save_keys(key_file)
        else:
            self._generate_keys()
            self._save_keys(key_file)
            logger.info("Generated new validator keys")
    
    def _generate_keys(self) -> None:
        """Generate new cryptographic keys"""
        try:
            if self.key_algorithm == "secp256k1":
                self.private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            elif self.key_algorithm == "rsa":
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=default_backend()
                )
            else:
                raise CryptoError(f"Unsupported key algorithm: {self.key_algorithm}")
            
            self.public_key = self.private_key.public_key()
            self.address = self._derive_address()
            
        except Exception as e:
            raise CryptoError(f"Failed to generate keys: {e}")
    
    def _load_keys(self, key_file: str) -> None:
        """Load keys from file"""
        try:
            with open(key_file, 'r') as f:
                key_data = json.load(f)
            
            private_bytes = bytes.fromhex(key_data['private_key'])
            
            if self.key_algorithm == "secp256k1":
                self.private_key = serialization.load_der_private_key(
                    private_bytes, password=None, backend=default_backend()
                )
            elif self.key_algorithm == "rsa":
                self.private_key = serialization.load_der_private_key(
                    private_bytes, password=None, backend=default_backend()
                )
            
            self.public_key = self.private_key.public_key()
            self.address = key_data['address']
            
        except Exception as e:
            raise CryptoError(f"Failed to load keys: {e}")
    
    def _save_keys(self, key_file: str) -> None:
        """Save keys to file"""
        try:
            private_bytes = self.private_key.private_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            key_data = {
                'private_key': private_bytes.hex(),
                'address': self.address,
                'algorithm': self.key_algorithm
            }
            
            with open(key_file, 'w') as f:
                json.dump(key_data, f, indent=2)
            
            # Set secure file permissions
            os.chmod(key_file, 0o600)
            
        except Exception as e:
            raise CryptoError(f"Failed to save keys: {e}")
    
    def _derive_address(self) -> str:
        """Derive address from public key"""
        try:
            public_bytes = self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Use hash of public key as address
            import hashlib
            address_hash = hashlib.sha256(public_bytes).hexdigest()[:40]
            return f"0x{address_hash}"
            
        except Exception as e:
            raise CryptoError(f"Failed to derive address: {e}")
    
    def get_public_key_der(self) -> bytes:
        """Get public key in DER format"""
        try:
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.DER,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
        except Exception as e:
            raise CryptoError(f"Failed to get public key: {e}")
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        try:
            return self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
        except Exception as e:
            raise CryptoError(f"Failed to get public key: {e}")
    
    def get_address(self) -> str:
        """Get validator address"""
        return self.address
    
    def export_public_key(self) -> Dict:
        """Export public key information"""
        return {
            'address': self.address,
            'public_key': self.get_public_key_der().hex(),
            'public_key_pem': self.get_public_key_pem(),
            'algorithm': self.key_algorithm
        }
    
    def verify_key_pair(self) -> bool:
        """Verify that private and public keys match"""
        try:
            # Test sign/verify to validate key pair
            test_message = b"test message"
            signature = self.sign_message(test_message)
            return self.verify_signature(self.address, test_message, signature)
        except Exception:
            return False
    
    def sign_message(self, message: bytes) -> bytes:
        """Sign message with private key"""
        from .signatures import sign_message
        return sign_message(self.private_key, message, self.key_algorithm)
    
    def verify_signature(self, address: str, message: bytes, signature: bytes) -> bool:
        """Verify signature against address and message"""
        from .signatures import verify_signature
        return verify_signature(address, message, signature, self.key_algorithm)
    
    def recover_address(self, message: bytes, signature: bytes) -> Optional[str]:
        """Recover address from signature"""
        from .signatures import recover_address
        return recover_address(message, signature, self.key_algorithm)