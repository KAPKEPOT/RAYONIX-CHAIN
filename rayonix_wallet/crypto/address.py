import hashlib
import base58
import bech32
from typing import Optional

from rayonix_wallet.core.types import AddressType
from rayonix_wallet.core.exceptions import CryptoError

class AddressDerivation:
    """Address derivation and validation"""
    
    def __init__(self, config):
        self.config = config
    
    def derive_address(self, public_key: bytes, index: int, is_change: bool) -> str:
        """Derive address from public key based on address type"""
        if self.config.address_type == AddressType.RAYONIX:
            return self._derive_rayonix_address(public_key)
        elif self.config.address_type == AddressType.P2PKH:
            return self._derive_p2pkh_address(public_key)
        elif self.config.address_type == AddressType.P2WPKH:
            return self._derive_p2wpkh_address(public_key)
        elif self.config.address_type == AddressType.BECH32:
            return self._derive_bech32_address(public_key)
        else:
            return self._derive_rayonix_address(public_key)
    
    def _derive_rayonix_address(self, public_key: bytes) -> str:
        """Derive Rayonix-specific address"""
        sha_hash = hashlib.sha256(public_key).digest()
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        network_byte = b'\x3C' if self.config.network == "mainnet" else b'\x6F'
        payload = network_byte + ripemd_hash
        
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        address_bytes = payload + checksum
        return base58.b58encode(address_bytes).decode('ascii')
    
    def _derive_p2pkh_address(self, public_key: bytes) -> str:
        """Derive P2PKH address (legacy Bitcoin-style)"""
        sha_hash = hashlib.sha256(public_key).digest()
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        version_byte = b'\x00' if self.config.network == "mainnet" else b'\x6F'
        payload = version_byte + ripemd_hash
        
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        address_bytes = payload + checksum
        return base58.b58encode(address_bytes).decode('ascii')
    
    def _derive_p2wpkh_address(self, public_key: bytes) -> str:
        """Derive P2WPKH address (native SegWit)"""
        sha_hash = hashlib.sha256(public_key).digest()
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        hrp = "bc" if self.config.network == "mainnet" else "tb"
        return bech32.encode(hrp, 0, ripemd_hash)
    
    def _derive_bech32_address(self, public_key: bytes) -> str:
        """Derive Bech32 address"""
        sha_hash = hashlib.sha256(public_key).digest()
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        hrp = "ray" if self.config.network == "mainnet" else "tray"
        return bech32.encode(hrp, 0, ripemd_hash)
    
    def validate_address(self, address: str, address_type: AddressType, network: str) -> bool:
        """Validate cryptocurrency address"""
        try:
            if address_type == AddressType.RAYONIX:
                return self._validate_rayonix_address(address, network)
            elif address_type == AddressType.P2PKH:
                return self._validate_p2pkh_address(address, network)
            elif address_type in [AddressType.P2WPKH, AddressType.BECH32]:
                return self._validate_bech32_address(address, network)
            else:
                return self._validate_rayonix_address(address, network)
        except:
            return False
    
    def _validate_rayonix_address(self, address: str, network: str) -> bool:
        """Validate Rayonix address"""
        try:
            decoded = base58.b58decode(address)
            if len(decoded) != 25:
                return False
            
            payload = decoded[:-4]
            checksum = decoded[-4:]
            
            calculated_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            return checksum == calculated_checksum
        except:
            return False
    
    def _validate_p2pkh_address(self, address: str, network: str) -> bool:
        """Validate P2PKH address"""
        try:
            decoded = base58.b58decode(address)
            return len(decoded) == 25
        except:
            return False
    
    def _validate_bech32_address(self, address: str, network: str) -> bool:
        """Validate Bech32 address"""
        try:
            hrp, data, spec = bech32.decode(address)
            expected_hrp = "bc" if network == "mainnet" else "tb"
            return hrp == expected_hrp and data is not None
        except:
            return False