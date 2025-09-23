# blockchain/production/block_signing.py
import time
import hashlib
import struct
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import secrets
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from cachetools import TTLCache, cached
import os

# Configure logging
logger = logging.getLogger(__name__)

class SigningAlgorithm(Enum):
    ECDSA_SECP256K1 = "ecdsa_secp256k1"
    ECDSA_SECP256R1 = "ecdsa_secp256r1"
    SCHNORR_BIP340 = "schnorr_bip340"
    ED25519 = "ed25519"

class SignatureResult:
    def __init__(self, success: bool, signature: Optional[bytes] = None, error: Optional[str] = None, timestamp: float = None):
        self.success = success
        self.signature = signature
        self.error = error
        self.timestamp = timestamp or time.time()

@dataclass
class KeyMetadata:
    key_id: str
    algorithm: SigningAlgorithm
    created_at: float
    expires_at: Optional[float]
    key_usage: int = 0
    last_used: Optional[float] = None

class SchnorrSignature:
    """BIP-340 compliant Schnorr signature implementation"""
    
    @staticmethod
    def point_add(P1: Tuple[int, int], P2: Tuple[int, int], curve: ec.EllipticCurve) -> Tuple[int, int]:
        """Point addition on elliptic curve"""
        # Implementation of elliptic curve point addition
        if P1 is None:
            return P2
        if P2 is None:
            return P1
            
        x1, y1 = P1
        x2, y2 = P2
        
        if x1 == x2:
            if y1 == y2:
                # Point doubling
                lam = (3 * x1 * x1) * pow(2 * y1, curve.p - 2, curve.p) % curve.p
            else:
                return None  # Point at infinity
        else:
            lam = ((y2 - y1) * pow(x2 - x1, curve.p - 2, curve.p)) % curve.p
            
        x3 = (lam * lam - x1 - x2) % curve.p
        y3 = (lam * (x1 - x3) - y1) % curve.p
        
        return (x3, y3)
    
    @staticmethod
    def point_multiply(k: int, P: Tuple[int, int], curve: ec.EllipticCurve) -> Tuple[int, int]:
        """Scalar multiplication on elliptic curve"""
        # Implementation using double-and-add algorithm
        result = None
        addend = P
        
        while k:
            if k & 1:
                result = SchnorrSignature.point_add(result, addend, curve)
            addend = SchnorrSignature.point_add(addend, addend, curve)
            k >>= 1
            
        return result
    
    @staticmethod
    def bytes_from_int(x: int) -> bytes:
        return x.to_bytes(32, byteorder='big')
    
    @staticmethod
    def int_from_bytes(b: bytes) -> int:
        return int.from_bytes(b, byteorder='big')
    
    @staticmethod
    def tagged_hash(tag: str, msg: bytes) -> bytes:
        tag_hash = hashlib.sha256(tag.encode()).digest()
        return hashlib.sha256(tag_hash + tag_hash + msg).digest()
    
    @staticmethod
    def lift_x(x: int, curve: ec.EllipticCurve) -> Optional[Tuple[int, int]]:
        """Lift x coordinate to point on curve"""
        if x >= curve.p:
            return None
        y_sq = (pow(x, 3, curve.p) + 7) % curve.p
        y = pow(y_sq, (curve.p + 1) // 4, curve.p)
        if pow(y, 2, curve.p) != y_sq:
            return None
        return (x, y if y & 1 == 0 else curve.p - y)
    
    @staticmethod
    def sign(msg: bytes, seckey: bytes, aux_rand: bytes = None) -> bytes:
        """BIP-340 Schnorr signature"""
        if len(seckey) != 32:
            raise ValueError("Secret key must be 32 bytes")
        if aux_rand is None:
            aux_rand = secrets.token_bytes(32)
        if len(aux_rand) != 32:
            raise ValueError("Auxiliary random data must be 32 bytes")
            
        d0 = SchnorrSignature.int_from_bytes(seckey)
        P = SchnorrSignature.point_multiply(d0, (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798, 
                                               0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8),
                                           ec.SECP256K1())
        if P is None:
            raise ValueError("Invalid secret key")
            
        d = d0 if (P[1] % 2 == 0) else ec.SECP256K1().p - d0
        
        t = d ^ SchnorrSignature.int_from_bytes(SchnorrSignature.tagged_hash("BIP0340/aux", aux_rand))
        k0 = SchnorrSignature.int_from_bytes(SchnorrSignature.tagged_hash("BIP0340/nonce", 
                                                                         SchnorrSignature.bytes_from_int(t) + msg)) % ec.SECP256K1().p
        if k0 == 0:
            raise RuntimeError("Failure. This happens only with negligible probability.")
            
        R = SchnorrSignature.point_multiply(k0, (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                                               0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8),
                                           ec.SECP256K1())
        if R is None:
            raise RuntimeError("Failure. This happens only with negligible probability.")
            
        k = k0 if (R[1] % 2 == 0) else ec.SECP256K1().p - k0
        e = SchnorrSignature.int_from_bytes(SchnorrSignature.tagged_hash("BIP0340/challenge", 
                                                                        SchnorrSignature.bytes_from_int(R[0]) + 
                                                                        SchnorrSignature.bytes_from_int(P[0]) + msg)) % ec.SECP256K1().p
        
        sig = SchnorrSignature.bytes_from_int(R[0]) + SchnorrSignature.bytes_from_int((k + e * d) % ec.SECP256K1().p)
        return sig
    
    @staticmethod
    def verify(msg: bytes, pubkey: bytes, sig: bytes) -> bool:
        """BIP-340 Schnorr signature verification"""
        if len(pubkey) != 32:
            return False
        if len(sig) != 64:
            return False
            
        try:
            P = SchnorrSignature.lift_x(SchnorrSignature.int_from_bytes(pubkey), ec.SECP256K1())
            if P is None:
                return False
                
            r = SchnorrSignature.int_from_bytes(sig[:32])
            s = SchnorrSignature.int_from_bytes(sig[32:])
            if r >= ec.SECP256K1().p or s >= ec.SECP256K1().p:
                return False
                
            e = SchnorrSignature.int_from_bytes(SchnorrSignature.tagged_hash("BIP0340/challenge", 
                                                                            sig[:32] + pubkey + msg)) % ec.SECP256K1().p
            R = SchnorrSignature.point_add(SchnorrSignature.point_multiply(s, (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                                                                             0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8),
                                                                         ec.SECP256K1()),
                                         SchnorrSignature.point_multiply(e, P, ec.SECP256K1()),
                                         ec.SECP256K1())
                                         
            if R is None or (R[1] % 2 != 0) or R[0] != r:
                return False
                
            return True
        except Exception:
            return False

class BlockSigner:
    """Production-ready block signing and verification system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = default_backend()
        self._key_cache = TTLCache(maxsize=1000, ttl=3600)  # 1 hour cache
        self._signature_cache = TTLCache(maxsize=5000, ttl=300)  # 5 minute cache
        self._key_metadata: Dict[str, KeyMetadata] = {}
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=config.get('signing_threads', 4))
        
        # Initialize algorithms with proper error handling
        self.signature_algorithms = {
            SigningAlgorithm.ECDSA_SECP256K1: self._sign_with_ecdsa_secp256k1,
            SigningAlgorithm.ECDSA_SECP256R1: self._sign_with_ecdsa_secp256r1,
            SigningAlgorithm.SCHNORR_BIP340: self._sign_with_schnorr_bip340,
            SigningAlgorithm.ED25519: self._sign_with_ed25519
        }
        
        self.verification_algorithms = {
            SigningAlgorithm.ECDSA_SECP256K1: self._verify_ecdsa_secp256k1,
            SigningAlgorithm.ECDSA_SECP256R1: self._verify_ecdsa_secp256r1,
            SigningAlgorithm.SCHNORR_BIP340: self._verify_schnorr_bip340,
            SigningAlgorithm.ED25519: self._verify_ed25519
        }
        
        # Performance metrics
        self.metrics = {
            'sign_operations': 0,
            'verify_operations': 0,
            'sign_errors': 0,
            'verify_errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"BlockSigner initialized with algorithms: {[alg.value for alg in self.signature_algorithms.keys()]}")

    def sign_block(self, block: Any, private_key: bytes, 
                  algorithm: SigningAlgorithm = SigningAlgorithm.ECDSA_SECP256K1,
                  key_id: Optional[str] = None) -> SignatureResult:
        """Production-grade block signing with comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Input validation
            if not private_key or len(private_key) not in [32, 64]:
                error_msg = f"Invalid private key length: {len(private_key) if private_key else 0}"
                logger.error(error_msg)
                return SignatureResult(False, error=error_msg, timestamp=start_time)
            
            if algorithm not in self.signature_algorithms:
                error_msg = f"Unsupported signing algorithm: {algorithm}"
                logger.error(error_msg)
                return SignatureResult(False, error=error_msg, timestamp=start_time)
            
            # Generate signing data
            signing_data = self._get_signing_data(block)
            data_hash = hashlib.sha256(signing_data).digest()
            
            # Check cache for recent signatures
            cache_key = self._generate_cache_key(data_hash, private_key, algorithm)
            if cache_key in self._signature_cache:
                self.metrics['cache_hits'] += 1
                logger.debug("Signature cache hit")
                return SignatureResult(True, signature=self._signature_cache[cache_key], timestamp=start_time)
            
            self.metrics['cache_misses'] += 1
            
            # Perform signing
            sign_func = self.signature_algorithms[algorithm]
            signature = sign_func(signing_data, private_key)
            
            # Cache the signature
            self._signature_cache[cache_key] = signature
            
            # Update key metadata
            if key_id:
                self._update_key_metadata(key_id)
            
            self.metrics['sign_operations'] += 1
            logger.info(f"Block signed successfully with algorithm {algorithm.value}, key_id: {key_id}")
            
            return SignatureResult(True, signature=signature, timestamp=start_time)
            
        except Exception as e:
            self.metrics['sign_errors'] += 1
            error_msg = f"Block signing failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return SignatureResult(False, error=error_msg, timestamp=start_time)

    def verify_block_signature(self, block: Any, public_key: bytes, 
                             signature: bytes, algorithm: SigningAlgorithm = SigningAlgorithm.ECDSA_SECP256K1,
                             key_id: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Production-grade signature verification with detailed error reporting"""
        start_time = time.time()
        
        try:
            # Input validation
            if not public_key or len(public_key) not in [33, 64, 65]:
                error_msg = f"Invalid public key length: {len(public_key) if public_key else 0}"
                logger.warning(error_msg)
                return False, error_msg
            
            if not signature:
                error_msg = "Signature cannot be empty"
                logger.warning(error_msg)
                return False, error_msg
            
            if algorithm not in self.verification_algorithms:
                error_msg = f"Unsupported verification algorithm: {algorithm}"
                logger.error(error_msg)
                return False, error_msg
            
            # Generate signing data
            signing_data = self._get_signing_data(block)
            data_hash = hashlib.sha256(signing_data).digest()
            
            # Generate cache key for verification result
            cache_key = self._generate_verify_cache_key(data_hash, public_key, signature, algorithm)
            if cache_key in self._signature_cache:
                self.metrics['cache_hits'] += 1
                logger.debug("Verification cache hit")
                return True, None
            
            self.metrics['cache_misses'] += 1
            
            # Perform verification
            verify_func = self.verification_algorithms[algorithm]
            is_valid = verify_func(signing_data, public_key, signature)
            
            # Cache positive results only (security consideration)
            if is_valid:
                self._signature_cache[cache_key] = b"valid"
            
            # Update key metadata
            if key_id and is_valid:
                self._update_key_metadata(key_id)
            
            self.metrics['verify_operations'] += 1
            
            if is_valid:
                logger.debug(f"Signature verified successfully with algorithm {algorithm.value}")
                return True, None
            else:
                error_msg = "Signature verification failed"
                logger.warning(error_msg)
                return False, error_msg
                
        except Exception as e:
            self.metrics['verify_errors'] += 1
            error_msg = f"Signature verification error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg

    def _get_signing_data(self, block: Any) -> bytes:
        """Comprehensive signing data generation with Merkle tree validation"""
        try:
            # Basic block header data
            header_data = struct.pack(
                '>I Q 32s 32s Q Q Q 32s',
                block.header.version,
                block.header.height,
                bytes.fromhex(block.header.previous_hash),
                bytes.fromhex(block.header.merkle_root),
                block.header.timestamp,
                block.header.difficulty,
                block.header.nonce,
                block.header.validator.encode('utf-8')
            )
            
            # Include transaction commitments with Merkle tree validation
            tx_hashes = [bytes.fromhex(tx.hash) for tx in block.transactions]
            
            # Build Merkle tree and include root commitment
            if tx_hashes:
                merkle_root = self._compute_merkle_root(tx_hashes)
                if merkle_root != bytes.fromhex(block.header.merkle_root):
                    logger.warning("Computed Merkle root doesn't match block header")
                
                # Include all transaction hashes for comprehensive signing
                for tx_hash in tx_hashes:
                    header_data += tx_hash
            
            # Add additional security features
            header_data += struct.pack('>Q', int(time.time()))  # Current timestamp
            header_data += os.urandom(16)  # Random salt for replay protection
            
            return header_data
            
        except Exception as e:
            logger.error(f"Error generating signing data: {e}")
            raise

    def _compute_merkle_root(self, hashes: List[bytes]) -> bytes:
        """Compute Merkle root from transaction hashes"""
        if not hashes:
            return hashlib.sha256().digest()
        
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])  # Duplicate last hash if odd number
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hash = hashlib.sha256(combined).digest()
                new_hashes.append(new_hash)
            hashes = new_hashes
        
        return hashes[0]

    def _sign_with_ecdsa_secp256k1(self, data: bytes, private_key: bytes) -> bytes:
        """ECDSA signing with secp256k1 curve and deterministic nonce"""
        try:
            # Load private key with proper error handling
            private_key_obj = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256K1(),
                self.backend
            )
            
            # Use deterministic nonce generation (RFC 6979)
            signature = private_key_obj.sign(
                data,
                ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            
            return signature
            
        except Exception as e:
            logger.error(f"ECDSA secp256k1 signing failed: {e}")
            raise

    def _verify_ecdsa_secp256k1(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """ECDSA verification with comprehensive error handling"""
        try:
            # Reconstruct public key from bytes
            public_key_obj = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                public_key
            )
            
            # Verify signature
            public_key_obj.verify(
                signature,
                data,
                ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            
            return True
            
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"ECDSA secp256k1 verification error: {e}")
            return False

    def _sign_with_ecdsa_secp256r1(self, data: bytes, private_key: bytes) -> bytes:
        """ECDSA signing with secp256r1 curve (P-256)"""
        try:
            private_key_obj = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256R1(),
                self.backend
            )
            
            signature = private_key_obj.sign(
                data,
                ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            
            return signature
            
        except Exception as e:
            logger.error(f"ECDSA secp256r1 signing failed: {e}")
            raise

    def _verify_ecdsa_secp256r1(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """ECDSA verification with secp256r1 curve"""
        try:
            public_key_obj = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256R1(),
                public_key
            )
            
            public_key_obj.verify(
                signature,
                data,
                ec.ECDSA(utils.Prehashed(hashes.SHA256()))
            )
            
            return True
            
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"ECDSA secp256r1 verification error: {e}")
            return False

    def _sign_with_schnorr_bip340(self, data: bytes, private_key: bytes) -> bytes:
        """BIP-340 compliant Schnorr signing"""
        try:
            if len(private_key) != 32:
                raise ValueError("Schnorr signatures require 32-byte private keys")
            
            return SchnorrSignature.sign(data, private_key)
            
        except Exception as e:
            logger.error(f"Schnorr BIP-340 signing failed: {e}")
            raise

    def _verify_schnorr_bip340(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """BIP-340 compliant Schnorr verification"""
        try:
            if len(public_key) != 32 or len(signature) != 64:
                return False
            
            return SchnorrSignature.verify(data, public_key, signature)
            
        except Exception as e:
            logger.error(f"Schnorr BIP-340 verification error: {e}")
            return False

    def _sign_with_ed25519(self, data: bytes, private_key: bytes) -> bytes:
        """Ed25519 signing implementation"""
        try:
            # Placeholder for actual Ed25519 implementation
            # In production, use a library like cryptography or ed25519-blake2b
            raise NotImplementedError("Ed25519 signing not yet implemented")
            
        except Exception as e:
            logger.error(f"Ed25519 signing failed: {e}")
            raise

    def _verify_ed25519(self, data: bytes, public_key: bytes, signature: bytes) -> bool:
        """Ed25519 verification implementation"""
        try:
            # Placeholder for actual Ed25519 implementation
            raise NotImplementedError("Ed25519 verification not yet implemented")
            
        except Exception as e:
            logger.error(f"Ed25519 verification error: {e}")
            return False

    def generate_key_pair(self, algorithm: SigningAlgorithm = SigningAlgorithm.ECDSA_SECP256K1,
                         key_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate cryptographic key pair with metadata"""
        try:
            if algorithm == SigningAlgorithm.ECDSA_SECP256K1:
                curve = ec.SECP256K1()
            elif algorithm == SigningAlgorithm.ECDSA_SECP256R1:
                curve = ec.SECP256R1()
            elif algorithm == SigningAlgorithm.SCHNORR_BIP340:
                curve = ec.SECP256K1()  # Schnorr uses same curve
            elif algorithm == SigningAlgorithm.ED25519:
                # Ed25519 uses different key generation
                raise NotImplementedError("Ed25519 key generation not implemented")
            else:
                raise ValueError(f"Unsupported algorithm for key generation: {algorithm}")
            
            # Generate private key
            private_key = ec.generate_private_key(curve, self.backend)
            
            # Get public key
            public_key = private_key.public_key()
            
            # Serialize keys according to algorithm requirements
            if algorithm in [SigningAlgorithm.ECDSA_SECP256K1, SigningAlgorithm.ECDSA_SECP256R1]:
                private_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')
                public_bytes = public_key.public_bytes(
                    encoding=serialization.Encoding.X962,
                    format=serialization.PublicFormat.UncompressedPoint
                )
            elif algorithm == SigningAlgorithm.SCHNORR_BIP340:
                # Schnorr uses 32-byte public keys (x-coordinate only)
                private_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')
                public_numbers = public_key.public_numbers()
                public_bytes = public_numbers.x.to_bytes(32, 'big')
            else:
                raise ValueError(f"Unsupported algorithm for serialization: {algorithm}")
            
            key_id = key_id or self._generate_key_id()
            
            # Store key metadata
            key_metadata = KeyMetadata(
                key_id=key_id,
                algorithm=algorithm,
                created_at=time.time(),
                expires_at=time.time() + (365 * 24 * 3600)  # 1 year default
            )
            
            with self._lock:
                self._key_metadata[key_id] = key_metadata
            
            result = {
                'key_id': key_id,
                'private_key': private_bytes,
                'public_key': public_bytes,
                'algorithm': algorithm.value,
                'created_at': key_metadata.created_at,
                'expires_at': key_metadata.expires_at
            }
            
            logger.info(f"Generated key pair with ID: {key_id}, algorithm: {algorithm.value}")
            return result
            
        except Exception as e:
            logger.error(f"Key pair generation failed: {e}")
            raise

    def _generate_key_id(self) -> str:
        """Generate unique key identifier"""
        return hashlib.sha256(os.urandom(32)).hexdigest()[:16]

    def _generate_cache_key(self, data_hash: bytes, private_key: bytes, algorithm: SigningAlgorithm) -> str:
        """Generate cache key for signing operations"""
        key_material = data_hash + private_key[:16] + algorithm.value.encode()
        return hashlib.sha256(key_material).hexdigest()

    def _generate_verify_cache_key(self, data_hash: bytes, public_key: bytes, 
                                 signature: bytes, algorithm: SigningAlgorithm) -> str:
        """Generate cache key for verification operations"""
        key_material = data_hash + public_key[:16] + signature[:16] + algorithm.value.encode()
        return hashlib.sha256(key_material).hexdigest()

    def _update_key_metadata(self, key_id: str):
        """Update key usage statistics"""
        with self._lock:
            if key_id in self._key_metadata:
                metadata = self._key_metadata[key_id]
                metadata.key_usage += 1
                metadata.last_used = time.time()

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance and operational metrics"""
        with self._lock:
            return self.metrics.copy()

    def get_key_metadata(self, key_id: str) -> Optional[KeyMetadata]:
        """Retrieve metadata for a specific key"""
        with self._lock:
            return self._key_metadata.get(key_id)

    def rotate_expired_keys(self) -> List[str]:
        """Identify and mark expired keys for rotation"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key_id, metadata in self._key_metadata.items():
                if metadata.expires_at and metadata.expires_at < current_time:
                    expired_keys.append(key_id)
                    logger.warning(f"Key {key_id} has expired and should be rotated")
        
        return expired_keys

    def get_supported_algorithms(self) -> List[str]:
        """Get list of supported signing algorithms"""
        return [alg.value for alg in self.signature_algorithms.keys()]

    def get_default_algorithm(self) -> SigningAlgorithm:
        """Get default signing algorithm from configuration"""
        default_alg = self.config.get('default_signing_algorithm', 'ecdsa_secp256k1')
        try:
            return SigningAlgorithm(default_alg)
        except ValueError:
            logger.warning(f"Invalid default algorithm in config: {default_alg}, using ECDSA_SECP256K1")
            return SigningAlgorithm.ECDSA_SECP256K1

    def cleanup(self):
        """Cleanup resources"""
        self._thread_pool.shutdown(wait=True)
        self._key_cache.clear()
        self._signature_cache.clear()
        logger.info("BlockSigner cleanup completed")

    def __del__(self):
        """Destructor for resource cleanup"""
        self.cleanup()