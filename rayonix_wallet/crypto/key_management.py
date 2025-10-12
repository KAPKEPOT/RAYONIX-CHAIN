import os
import hashlib
import secrets
from typing import Optional, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from bip32 import BIP32
from mnemonic import Mnemonic

from rayonix_wallet.core.types import WalletType, SecureKeyPair
from rayonix_wallet.core.exceptions import CryptoError
from rayonix_wallet.utils.secure import SecureString

class KeyManager:
    """Key management and derivation"""
    
    def __init__(self, config):
        self.config = config
        self.master_key: Optional[SecureKeyPair] = None
        self.encryption_key: Optional[SecureString] = None
        self._creation_mnemonic: Optional[SecureString] = None
    
    def initialize_from_mnemonic(self, mnemonic_phrase: str, passphrase: str = "") -> bool:
        """Initialize from BIP39 mnemonic phrase"""
        try:
            if not self._validate_mnemonic(mnemonic_phrase):
                raise CryptoError("Invalid mnemonic phrase")
            
            seed = self._mnemonic_to_seed(mnemonic_phrase, passphrase)
            bip32 = BIP32.from_seed(seed)
            
            # Get private key using correct method
            try:
            	private_key_bytes = bip32.get_privkey_from_path("m")
            except Exception:
            	private_key_bytes = bip32.private_key
            private_key_secure = SecureString(private_key_bytes)
            
            # Get public key
            public_key_bytes = bip32.get_pubkey_from_path("m")
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=public_key_bytes,
                chain_code=bip32.chain_code,
                depth=0,
                index=0,
                parent_fingerprint=b'\x00\x00\x00\x00'
            )
            self._creation_mnemonic = SecureString(mnemonic_phrase.encode())
            return True
        except Exception as e:
        	raise CryptoError(f"Failed to initialize from mnemonic: {e}")
      
    def initialize_from_private_key(self, private_key: str, wallet_type: WalletType) -> bool:
        """Initialize from private key"""
        try:
            priv_key_bytes = self._decode_private_key(private_key)
            public_key = self._private_to_public(priv_key_bytes)
            
            private_key_secure = SecureString(priv_key_bytes)
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=public_key
            )
            
            return True
            
        except Exception as e:
            raise CryptoError(f"Failed to initialize from private key: {e}")
    
    def generate_mnemonic(self, strength: int = 256) -> str:
        """Generate BIP39 mnemonic phrase"""
        mnemo = Mnemonic("english")
        return mnemo.generate(strength=strength)
    
    def _validate_mnemonic(self, mnemonic_phrase: str) -> bool:
        """Validate BIP39 mnemonic phrase"""
        mnemo = Mnemonic("english")
        return mnemo.check(mnemonic_phrase)
    
    def _mnemonic_to_seed(self, mnemonic_phrase: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed using BIP39"""
        mnemo = Mnemonic("english")
        return mnemo.to_seed(mnemonic_phrase, passphrase)
    
    def _decode_private_key(self, private_key: str) -> bytes:
        """Decode private key from various formats"""
        try:
            if private_key.startswith('0x'):
                return bytes.fromhex(private_key[2:])
            else:
                return bytes.fromhex(private_key)
        except:
            raise CryptoError("Invalid private key format")
    
    def _private_to_public(self, private_key: bytes) -> bytes:
        """Convert private key to public key using cryptography library"""
        try:
            sk = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256K1(),  
                self.backend
            )
            vk = sk.public_key()
            
            # Get compressed public key
            public_key_bytes = vk.public_bytes(
                encoding=ec.Encoding.X962,
                format=ec.PublicFormat.CompressedPoint
            )
            
            return public_key_bytes
            
        except Exception as e:
            raise CryptoError(f"Public key derivation failed: {e}")
    
    def get_derivation_path(self, index: int, is_change: bool = False) -> str:
        """Get BIP44 derivation path"""
        change_index = 1 if is_change else 0
        return f"m/44'/0'/{self.config.account_index}'/{change_index}/{index}"
    
    def derive_public_key(self, derivation_path: str) -> bytes:
        """Derive public key from derivation path"""
        if not self.master_key:
            raise CryptoError("Master key not available")
        
        bip32 = BIP32.from_seed(self.master_key.private_key)
        return bip32.get_pubkey_from_path(derivation_path)
    
    def export_private_key(self, address: str, derivation_path: str) -> Optional[str]:
        """Export private key for address"""
        if not self.master_key:
            raise CryptoError("Master key not available")
        
        if self.config.wallet_type == WalletType.HD:
            bip32 = BIP32.from_seed(self.master_key.private_key)
            private_key = bip32.get_privkey_from_path(derivation_path)
            return private_key.hex()
        elif self.config.wallet_type == WalletType.NON_HD:
            return self.master_key.private_key.hex()
        
        return None
    
    def verify_passphrase(self, passphrase: str) -> bool:
        """Verify wallet passphrase"""
        expected_hash = hashlib.sha256(passphrase.encode()).hexdigest()
        stored_hash = hashlib.sha256((self.config.passphrase or "").encode()).hexdigest()
        return expected_hash == stored_hash
    
    def derive_encryption_key(self, passphrase: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from passphrase"""
        salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(passphrase.encode())
    
    def wipe(self):
        """Securely wipe all keys from memory"""
        if self.master_key:
            self.master_key.wipe()
            self.master_key = None
        
        if self.encryption_key:
            self.encryption_key.wipe()
            self.encryption_key = None
        
        if self._creation_mnemonic:
            self._creation_mnemonic.wipe()
            self._creation_mnemonic = None
    
    def get_public_key(self) -> bytes:
        """Get master public key"""
        if not self.master_key:
            raise CryptoError("Master key not available")
        return self.master_key.public_key