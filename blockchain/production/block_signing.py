# blockchain/production/block_signing.py
import time
import hashlib
from typing import Dict, Any, Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

class BlockSigner:
    """Handles block signing and signature verification"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signature_algorithms = {
            'ecdsa_secp256k1': self._sign_with_ecdsa,
            'ecdsa_secp256r1': self._sign_with_ecdsa_p256,
            'schnorr': self._sign_with_schnorr
        }
        self.verification_algorithms = {
            'ecdsa_secp256k1': self._verify_ecdsa,
            'ecdsa_secp256r1': self._verify_ecdsa_p256,
            'schnorr': self._verify_schnorr
        }
    
    def sign_block(self, block: Any, private_key: bytes, 
                  algorithm: str = 'ecdsa_secp256k1') -> Optional[bytes]:
        """Sign block with specified algorithm"""
        try:
            sign_func = self.signature_algorithms.get(algorithm)
            if not sign_func:
                raise ValueError(f"Unsupported signing algorithm: {algorithm}")
            
            # Create signing data
            signing_data = self._get_signing_data(block)
            
            # Sign data
            signature = sign_func(signing_data, private_key)
            
            return signature
            
        except Exception as e:
            print(f"Block signing failed: {e}")
            return None
    
    def verify_block_signature(self, block: Any, public_key: bytes, 
                             signature: bytes, algorithm: str = 'ecdsa_secp256k1') -> bool:
        """Verify block signature"""
        try:
            verify_func = self.verification_algorithms.get(algorithm)
            if not verify_func:
                raise ValueError(f"Unsupported verification algorithm: {algorithm}")
            
            # Create signing data
            signing_data = self._get_signing_data(block)
            
            # Verify signature
            return verify_func(signing_data, public_key, signature)
            
        except Exception as e:
            print(f"Signature verification failed: {e}")
            return False
    
    def _get_signing_data(self, block: Any) -> bytes:
        """Get data to be signed from block"""
        # This should include all critical block data but exclude the signature itself
        signing_data = b"".join([
            block.header.version.to_bytes(4, 'big'),
            block.header.height.to_bytes(8, 'big'),
            bytes.fromhex(block.header.previous_hash),
            bytes.fromhex(block.header.merkle_root),
            block.header.timestamp.to_bytes(8, 'big'),
            block.header.difficulty.to_bytes(8, 'big'),
            block.header.nonce.to_bytes(8, 'big'),
            block.header.validator.encode()
        ])
        
        # Include transaction commitments for better security
        tx_hashes = [bytes.fromhex(tx.hash) for tx in block.transactions]
        for tx_hash in tx_hashes:
            signing_data += tx_hash
        
        return signing_data
    
    def _sign_with_ecdsa(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using ECDSA with secp256k1 curve"""
        # Load private key
        private_key = ec.derive_private_key(
            int.from_bytes(private_key, 'big'),
            ec.SECP256K1()
        )
        
        # Sign data
        signature = private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        
        return signature
    
    def _verify_ecdsa(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """Verify ECDSA signature with secp256k1 curve"""
        try:
            # Load public key
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                public_key
            )
            
            # Verify signature
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            
            return True
            
        except Exception as e:
            return False
    
    def _sign_with_ecdsa_p256(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using ECDSA with secp256r1 curve (P-256)"""
        # Load private key
        private_key = ec.derive_private_key(
            int.from_bytes(private_key, 'big'),
            ec.SECP256R1()
        )
        
        # Sign data
        signature = private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        
        return signature
    
    def _verify_ecdsa_p256(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """Verify ECDSA signature with secp256r1 curve"""
        try:
            # Load public key
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(),
                public_key
            )
            
            # Verify signature
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            
            return True
            
        except Exception as e:
            return False
    
    def _sign_with_schnorr(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using Schnorr signature algorithm"""
        # Placeholder implementation - would use actual Schnorr signing
        # For now, return ECDSA signature as fallback
        return self._sign_with_ecdsa(data, private_key)
    
    def _verify_schnorr(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """Verify Schnorr signature"""
        # Placeholder implementation - would use actual Schnorr verification
        # For now, use ECDSA verification as fallback
        return self._verify_ecdsa(data, public_key, signature)
    
    def generate_key_pair(self, algorithm: str = 'ecdsa_secp256k1') -> Dict[str, bytes]:
        """Generate key pair for block signing"""
        if algorithm == 'ecdsa_secp256k1':
            curve = ec.SECP256K1()
        elif algorithm == 'ecdsa_secp256r1':
            curve = ec.SECP256R1()
        else:
            raise ValueError(f"Unsupported algorithm for key generation: {algorithm}")
        
        # Generate private key
        private_key = ec.generate_private_key(curve)
        
        # Get public key
        public_key = private_key.public_key()
        
        # Serialize keys
        private_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')
        public_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        
        return {
            'private_key': private_bytes,
            'public_key': public_bytes,
            'algorithm': algorithm
        }
    
    def get_default_algorithm(self) -> str:
        """Get default signing algorithm"""
        return self.config.get('default_signing_algorithm', 'ecdsa_secp256k1')
    
    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported signing algorithms"""
        return list(self.signature_algorithms.keys())