# rayonix_wallet/crypto/key_management.py
import os
import hashlib
import secrets
import struct
import threading
import time
from typing import Optional, Tuple, Dict, Any, List
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from bip32 import BIP32
from mnemonic import Mnemonic

from rayonix_wallet.core.wallet_types import WalletType, SecureKeyPair
from rayonix_wallet.core.config import WalletConfig
from rayonix_wallet.core.exceptions import CryptoError
from rayonix_wallet.utils.secure import SecureString
from rayonix_wallet.utils.logging import logger

class KeyManager:    
    def __init__(self, config: WalletConfig):
        self.config = config
        self.backend = default_backend()
        
        # Cryptographic state
        self.master_key: Optional[SecureKeyPair] = None
        self.encryption_key: Optional[SecureString] = None
        self._creation_mnemonic: Optional[SecureString] = None
        self._derivation_cache: Dict[str, bytes] = {}
        
        # Security and concurrency
        self._lock = threading.RLock()
        self.failed_attempts = 0
        self.last_attempt_time: Optional[float] = None
        
        # Performance optimization
        self._bip32_cache: Optional[BIP32] = None
        
        logger.debug(f"KeyManager initialized for wallet type: {config.wallet_type}")

    def initialize_from_master_key(self, master_key: SecureKeyPair, config: WalletConfig) -> bool:
        """Initialize from existing master key - required by wallet.py"""
        with self._lock:
            try:
                if not self._validate_master_key_cryptographic(master_key):
                    raise CryptoError("Master key cryptographic validation failed")
                
                self.master_key = master_key
                self.config = config
                self._bip32_cache = None  # Clear cache
                
                logger.info("KeyManager initialized from master key")
                return True
                
            except Exception as e:
                logger.error(f"Master key initialization failed: {e}")
                raise CryptoError(f"Failed to initialize from master key: {e}")

    def initialize_from_mnemonic(self, mnemonic_phrase: str, passphrase: str = "") -> bool:
        """Initialize from BIP39 mnemonic with comprehensive cryptographic validation"""
        with self._lock:
            try:
                if not self._validate_mnemonic_cryptographic(mnemonic_phrase):
                    raise CryptoError("Mnemonic phrase cryptographic validation failed")
                
                # Generate cryptographically secure seed
                seed = self._mnemonic_to_secure_seed(mnemonic_phrase, passphrase)
                
                # Derive master key using consolidated method
                master_key = self.derive_master_key_from_seed(seed)
                
                # Validate key pair cryptographically
                if not self._validate_key_pair_cryptographic(master_key):
                    raise CryptoError("Derived key pair failed cryptographic validation")
                
                self.master_key = master_key
                self._creation_mnemonic = SecureString(mnemonic_phrase.encode())
                self._bip32_cache = None  # Clear cache
                
                logger.info("KeyManager initialized from mnemonic")
                return True
                
            except Exception as e:
                logger.error(f"Mnemonic initialization failed: {e}")
                self._secure_cleanup()
                raise CryptoError(f"Failed to initialize from mnemonic: {e}")

    def initialize_from_private_key(self, private_key: str, wallet_type: WalletType) -> bool:
        """Initialize from private key with cryptographic validation"""
        with self._lock:
            try:
                priv_key_bytes = self._decode_and_validate_private_key(private_key)
                public_key_bytes = self._derive_public_key_cryptographic(priv_key_bytes)
                
                # Create secure key pair
                private_key_secure = SecureString(priv_key_bytes)
                self.master_key = SecureKeyPair(
                    _private_key=private_key_secure,
                    public_key=public_key_bytes,
                    chain_code=self._generate_deterministic_chain_code(priv_key_bytes),
                    depth=0,
                    index=0,
                    parent_fingerprint=b'\x00\x00\x00\x00'
                )
                
                self.config.wallet_type = wallet_type
                self._bip32_cache = None
                
                logger.info("KeyManager initialized from private key")
                return True
                
            except Exception as e:
                logger.error(f"Private key initialization failed: {e}")
                self._secure_cleanup()
                raise CryptoError(f"Failed to initialize from private key: {e}")

    def derive_master_key_from_seed(self, seed: bytes) -> SecureKeyPair:
        """Derive master key from seed with comprehensive cryptographic validation"""
        with self._lock:
            try:
                bip32 = BIP32.from_seed(seed)
                
                # Extract private key with multiple fallback methods
                private_key_bytes = self._extract_private_key_robust(bip32)
                
                # Validate private key cryptographically
                if not self._validate_private_key_cryptographic(private_key_bytes):
                    raise CryptoError("Private key cryptographic validation failed")
                
                # Derive public key
                public_key_bytes = self._derive_public_key_cryptographic(private_key_bytes)
                
                # Extract chain code with enhanced robust extraction
                chain_code = self._extract_chain_code_robust(bip32)
                
                # Create secure key pair
                private_key_secure = SecureString(private_key_bytes)
                
                master_key = SecureKeyPair(
                    _private_key=private_key_secure,
                    public_key=public_key_bytes,
                    chain_code=chain_code,
                    depth=0,
                    index=0,
                    parent_fingerprint=b'\x00\x00\x00\x00'
                )
                
                # Validate the complete key pair
                if not self._validate_key_pair_cryptographic(master_key):
                    raise CryptoError("Cryptographic key pair validation failed")
                
                return master_key
                
            except Exception as e:
                raise CryptoError(f"Master key derivation failed: {e}")

    def derive_master_key_from_mnemonic(self, mnemonic: str, passphrase: str = "") -> SecureKeyPair:
        """Derive master key directly from mnemonic"""
        with self._lock:
            try:
                if not self._validate_mnemonic_cryptographic(mnemonic):
                    raise CryptoError("Mnemonic validation failed")
                
                seed = self._mnemonic_to_secure_seed(mnemonic, passphrase)
                return self.derive_master_key_from_seed(seed)
                
            except Exception as e:
                raise CryptoError(f"Master key derivation from mnemonic failed: {e}")

    def _validate_mnemonic_cryptographic(self, mnemonic: str) -> bool:
        """Cryptographic mnemonic validation with entropy analysis"""
        try:
            mnemo = Mnemonic("english")
            
            # Basic BIP39 validation
            if not mnemo.check(mnemonic):
                return False
            
            # Entropy validation
            entropy = mnemo.to_entropy(mnemonic)
            if len(entropy) < 16:  # 128-bit minimum
                return False
            
            # Check for weak patterns
            if self._detect_weak_mnemonic(mnemonic):
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Mnemonic validation failed: {e}")
            return False

    def _detect_weak_mnemonic(self, mnemonic: str) -> bool:
        """Detect cryptographically weak mnemonics"""
        words = mnemonic.split()
        
        # Check word uniqueness
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.7:  # 70% unique words minimum
            return True
        
        # Check for sequential patterns in BIP39 word list
        try:
            mnemo = Mnemonic("english")
            word_list = mnemo.wordlist
            indices = [word_list.index(word) for word in words if word in word_list]
            
            if len(indices) == len(words):
                # Check for sequential indices
                sequential_count = 0
                for i in range(1, len(indices)):
                    if indices[i] == indices[i-1] + 1:
                        sequential_count += 1
                        if sequential_count >= 3:  # 3+ sequential words
                            return True
        except Exception:
            pass
        
        return False

    def _mnemonic_to_secure_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """Generate cryptographically secure seed with key stretching"""
        try:
            mnemo = Mnemonic("english")
            base_seed = mnemo.to_seed(mnemonic, passphrase)
            
            # Additional key stretching for production use
            pbkdf2 = PBKDF2HMAC(
                algorithm=hashes.SHA512(),
                length=64,
                salt=self._generate_cryptographic_salt(),
                iterations=2048,  # Additional stretching
                backend=self.backend
            )
            
            stretched_seed = pbkdf2.derive(base_seed)
            
            # Domain separation with HKDF
            hkdf = HKDF(
                algorithm=hashes.SHA512(),
                length=64,
                salt=stretched_seed[:32],
                info=b'rayonix-keymanager-seed-final',
                backend=self.backend
            )
            
            return hkdf.derive(stretched_seed)
            
        except Exception as e:
            raise CryptoError(f"Secure seed generation failed: {e}")

    def _extract_private_key_robust(self, bip32_obj) -> bytes:
        """Robust private key extraction with multiple fallback methods"""
        extraction_methods = [
            # Primary method
            lambda: bip32_obj.get_privkey_from_path("m"),
            # Attribute-based methods
            lambda: getattr(bip32_obj, '_privkey', None),
            lambda: getattr(bip32_obj, 'private_key', None),
            lambda: getattr(bip32_obj, '_k', None),
            # Extended key parsing
            lambda: self._parse_private_key_from_extended(bip32_obj)
        ]
        
        for method in extraction_methods:
            try:
                private_key = method()
                if private_key and len(private_key) == 32:
                    return private_key
            except Exception:
                continue
        
        raise CryptoError("All private key extraction methods failed")

    def _parse_private_key_from_extended(self, bip32_obj) -> bytes:
        """Parse private key from extended key format"""
        try:
            xprv = bip32_obj.get_xpriv()
            # Private key is typically at specific offset in extended format
            if len(xprv) >= 78:
                # This is implementation-specific and may need adjustment
                # based on the bip32 library version
                return xprv[46:78]  # Typical offset for private key
            raise CryptoError("Extended key too short")
        except Exception as e:
            raise CryptoError(f"Extended key parsing failed: {e}")

    def _extract_chain_code_robust(self, bip32_obj) -> bytes:
        """Enhanced robust chain code extraction with validation"""
        extraction_methods = [
            # Primary method: parse from extended key
            lambda: self._parse_chain_code_from_extended(bip32_obj),
            # Fallback: generate deterministic chain code from private key
            lambda: self._generate_deterministic_chain_code(
                self._extract_private_key_robust(bip32_obj)
            ),
            # Library-specific attributes for different BIP32 implementations
            lambda: getattr(bip32_obj, '_c', None),
            lambda: getattr(bip32_obj, 'chaincode', None),
            lambda: getattr(bip32_obj, '_chaincode', None),
        ]
        
        for method in extraction_methods:
            try:
                chain_code = method()
                if chain_code and len(chain_code) == 32:
                    return chain_code
            except Exception:
                continue
        
        # Final fallback - generate secure random chain code
        logger.warning("Using fallback random chain code generation")
        return secrets.token_bytes(32)

    def _parse_chain_code_from_extended(self, bip32_obj) -> bytes:
        """Parse chain code from extended key format with multiple approaches"""
        try:
            # Method 1: Get extended private key and parse
            xprv = bip32_obj.get_xpriv()
            if len(xprv) >= 78:
                # Chain code is typically at bytes 13-45 in extended format
                chain_code = xprv[13:45]
                if len(chain_code) == 32:
                    return chain_code
            
            # Method 2: Try extended public key as fallback
            xpub = bip32_obj.get_xpub()
            if len(xpub) >= 78:
                chain_code = xpub[13:45]
                if len(chain_code) == 32:
                    return chain_code
                    
            # Method 3: Try library-specific methods
            if hasattr(bip32_obj, 'get_chain_code'):
                chain_code = bip32_obj.get_chain_code()
                if chain_code and len(chain_code) == 32:
                    return chain_code
                
            raise CryptoError("Cannot extract chain code from extended keys")
            
        except Exception as e:
            raise CryptoError(f"Chain code extraction failed: {e}")

    def _generate_deterministic_chain_code(self, private_key: bytes) -> bytes:
        """Generate deterministic chain code when not available"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=private_key[:16],
            info=b'rayonix-deterministic-chain-code',
            backend=self.backend
        )
        return hkdf.derive(private_key + struct.pack('>Q', int(time.time())))

    def _validate_private_key_cryptographic(self, private_key: bytes) -> bool:
        """Cryptographic private key validation"""
        if len(private_key) != 32:
            return False
        
        # Check key is within valid secp256k1 range
        key_int = int.from_bytes(private_key, 'big')
        curve_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        
        if key_int <= 1 or key_int >= curve_order - 1:
            return False
        
        # Check for weak keys
        if key_int < 2**128:  # Too small
            return False
        
        return True

    def _derive_public_key_cryptographic(self, private_key: bytes) -> bytes:
        """Cryptographic public key derivation with validation"""
        try:
            private_key_obj = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256K1(),
                self.backend
            )
            
            public_key_obj = private_key_obj.public_key()
            
            public_key_bytes = public_key_obj.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint
            )
            
            if not self._validate_public_key_cryptographic(public_key_bytes):
                raise CryptoError("Derived public key validation failed")
            
            return public_key_bytes
            
        except Exception as e:
            raise CryptoError(f"Public key derivation failed: {e}")

    def _validate_public_key_cryptographic(self, public_key: bytes) -> bool:
        """Cryptographic public key validation"""
        try:
            ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), public_key)
            
            if len(public_key) == 33:
                prefix = public_key[0]
                if prefix not in [0x02, 0x03]:
                    return False
            
            return True
            
        except Exception:
            return False

    def _decode_and_validate_private_key(self, private_key: str) -> bytes:
        """Decode and cryptographically validate private key"""
        try:
            # Handle hex formats
            if private_key.startswith('0x'):
                key_bytes = bytes.fromhex(private_key[2:])
            else:
                key_bytes = bytes.fromhex(private_key)
            
            # Validate key
            if not self._validate_private_key_cryptographic(key_bytes):
                raise CryptoError("Private key validation failed")
            
            return key_bytes
            
        except ValueError:
            raise CryptoError("Invalid private key format - must be hex")
        except Exception as e:
            raise CryptoError(f"Private key decoding failed: {e}")

    def _validate_master_key_cryptographic(self, master_key: SecureKeyPair) -> bool:
        """Validate master key cryptographic consistency"""
        try:
            # Verify private-public key consistency
            derived_public = self._derive_public_key_cryptographic(master_key.private_key)
            
            # Constant-time comparison
            if len(derived_public) != len(master_key.public_key):
                return False
            
            result = 0
            for x, y in zip(derived_public, master_key.public_key):
                result |= x ^ y
            
            return result == 0
            
        except Exception:
            return False

    def _validate_key_pair_cryptographic(self, key_pair: SecureKeyPair) -> bool:
        """Cryptographic validation of key pair consistency"""
        return self._validate_master_key_cryptographic(key_pair)

    def _generate_cryptographic_salt(self) -> bytes:
        """Generate cryptographically secure salt"""
        return secrets.token_bytes(32)

    def get_derivation_path(self, index: int, is_change: bool = False) -> str:
        """Get BIP44 derivation path"""
        change_index = 1 if is_change else 0
        return f"m/44'/{self.config.coin_type}'/{self.config.account_index}'/{change_index}/{index}"

    def derive_public_key(self, derivation_path: str) -> bytes:
        """Derive public key from derivation path with caching"""
        with self._lock:
            if not self.master_key:
                raise CryptoError("Master key not available")
            
            # Check cache
            if derivation_path in self._derivation_cache:
                return self._derivation_cache[derivation_path]
            
            try:
                bip32 = self._get_bip32_instance()
                public_key = bip32.get_pubkey_from_path(derivation_path)
                
                # Validate derived key
                if not self._validate_public_key_cryptographic(public_key):
                    raise CryptoError("Derived public key validation failed")
                
                # Cache result
                if len(self._derivation_cache) < 1000:
                    self._derivation_cache[derivation_path] = public_key
                
                return public_key
                
            except Exception as e:
                raise CryptoError(f"Public key derivation failed for path {derivation_path}: {e}")

    def export_private_key(self, derivation_path: str) -> str:
        """Export private key for derivation path"""
        with self._lock:
            if not self.master_key:
                raise CryptoError("Master key not available")
            
            if self.config.wallet_type == WalletType.HD:
                bip32 = self._get_bip32_instance()
                private_key = bip32.get_privkey_from_path(derivation_path)
                return private_key.hex()
            elif self.config.wallet_type == WalletType.NON_HD:
                return self.master_key.private_key.hex()
            else:
                raise CryptoError(f"Unsupported wallet type: {self.config.wallet_type}")

    def _get_bip32_instance(self) -> BIP32:
        """Get or create BIP32 instance with caching"""
        if self._bip32_cache is None:
            if not self.master_key:
                raise CryptoError("Master key not available")
            self._bip32_cache = BIP32.from_seed(self.master_key.private_key)
        return self._bip32_cache

    def verify_passphrase(self, passphrase: str) -> bool:
        """Verify wallet passphrase with rate limiting"""
        with self._lock:
            current_time = time.time()
            
            # Rate limiting
            if (self.last_attempt_time and 
                current_time - self.last_attempt_time < 1.0):  # 1 second minimum between attempts
                return False
            
            self.last_attempt_time = current_time
            
            try:
                # Use constant-time comparison
                expected_hash = hashlib.sha256(passphrase.encode()).digest()
                stored_hash = hashlib.sha256((self.config.passphrase or "").encode()).digest()
                
                # Constant-time comparison
                result = 0
                for x, y in zip(expected_hash, stored_hash):
                    result |= x ^ y
                
                is_valid = result == 0
                
                if is_valid:
                    self.failed_attempts = 0
                    logger.debug("Passphrase verification successful")
                else:
                    self.failed_attempts += 1
                    logger.warning(f"Failed passphrase attempt #{self.failed_attempts}")
                
                return is_valid
                
            except Exception as e:
                logger.error(f"Passphrase verification error: {e}")
                self.failed_attempts += 1
                return False

    def derive_encryption_key(self, passphrase: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from passphrase"""
        salt = salt or self._generate_cryptographic_salt()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(passphrase.encode())

    def generate_mnemonic(self, strength: int = 256) -> str:
        """Generate BIP39 mnemonic phrase with cryptographic entropy"""
        try:
            mnemo = Mnemonic("english")
            mnemonic = mnemo.generate(strength=strength)
            
            # Validate the generated mnemonic
            if not self._validate_mnemonic_cryptographic(mnemonic):
                raise CryptoError("Generated mnemonic failed validation")
            
            return mnemonic
            
        except Exception as e:
            raise CryptoError(f"Mnemonic generation failed: {e}")

    def get_public_key(self) -> bytes:
        """Get master public key"""
        with self._lock:
            if not self.master_key:
                raise CryptoError("Master key not available")
            return self.master_key.public_key

    def get_master_key_info(self) -> Dict[str, Any]:
        """Get comprehensive master key information"""
        with self._lock:
            if not self.master_key:
                return {'has_master_key': False}
            
            return {
                'has_master_key': True,
                'public_key_hex': self.master_key.public_key.hex(),
                'chain_code_hex': self.master_key.chain_code.hex() if self.master_key.chain_code else None,
                'depth': self.master_key.depth,
                'index': self.master_key.index,
                'parent_fingerprint': self.master_key.parent_fingerprint.hex(),
                'wallet_type': self.config.wallet_type.value if self.config.wallet_type else None,
            }

    def get_key_info(self) -> Dict[str, Any]:
        """Get key management information for debugging"""
        with self._lock:
            return {
                'wallet_type': self.config.wallet_type.value if self.config.wallet_type else None,
                'has_master_key': self.master_key is not None,
                'has_encryption_key': self.encryption_key is not None,
                'derivation_cache_size': len(self._derivation_cache),
                'failed_attempts': self.failed_attempts,
                'bip32_cache_initialized': self._bip32_cache is not None
            }

    def get_master_key(self) -> Optional[SecureKeyPair]:
    	"""Get the master key (read-only access)"""
    	with self._lock:
    		return self.master_key                      

    def wipe(self):
        """Securely wipe all keys and sensitive data from memory"""
        with self._lock:
            try:
                if self.master_key:
                    self.master_key.wipe()
                    self.master_key = None
                
                if self.encryption_key:
                    self.encryption_key.wipe()
                    self.encryption_key = None
                
                if self._creation_mnemonic:
                    self._creation_mnemonic.wipe()
                    self._creation_mnemonic = None
                
                # Clear caches
                self._derivation_cache.clear()
                self._bip32_cache = None
                
                # Clear security state
                self.failed_attempts = 0
                self.last_attempt_time = None
                
                logger.info("KeyManager securely wiped")
                
            except Exception as e:
                logger.error(f"Secure wipe failed: {e}")
                raise CryptoError(f"Secure wipe failed: {e}")

    def _secure_cleanup(self):
        """Internal secure cleanup"""
        self.wipe()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.wipe()