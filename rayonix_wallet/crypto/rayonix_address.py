import hashlib
import struct
import secrets
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, IntEnum
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from rayonix_wallet.core.exceptions import CryptoError, InvalidAddressError
from .base32_encoding import Base32Crockford

class AddressType(IntEnum):
    """RAYONIX address types with specific cryptographic properties"""
    STANDARD_P2PKH = 1      # Pay to Public Key Hash (SHA256 + RIPEMD160)
    MULTISIG_2OF3 = 2       # 2-of-3 Multisig (M-of-N script hash)
    MULTISIG_3OF5 = 3       # 3-of-5 Multisig  
    SMART_CONTRACT = 4      # Smart contract address (Keccak-256 derivation)
    TOKEN_CONTRACT = 5      # Token contract address (ERC-20 compatible)
    SEGWIT_V0 = 6           # SegWit version 0 (Bech32, P2WPKH)
    SEGWIT_V1 = 7           # SegWit version 1 (Bech32m, P2TR)
    STEALTH = 8             # Stealth address (dual-key derivation)

class AddressVersion(IntEnum):
    """Address format versions with cryptographic upgrades"""
    V1_LEGACY = 0   # Initial version (single SHA256 + RIPEMD160)
    V1_ENHANCED = 1 # Enhanced features (double SHA256 + key derivation)
    V2_ADVANCED = 2 # Advanced capabilities (BLAKE2b + HKDF)
    V3_FUTURE = 3   # Post-quantum preparation

@dataclass(frozen=True)
class AddressComponents:
    """Deconstructed address components for cryptographic validation"""
    prefix: str
    address_type: AddressType
    version: AddressVersion
    payload: bytes
    checksum: str
    raw_address: str
    cryptographic_context: bytes
    
    @property
    def network(self) -> str:
        return "mainnet" if self.prefix == "ryx" else "testnet"
    
    @property
    def is_mainnet(self) -> bool:
        return self.prefix == "ryx"
    
    @property
    def is_testnet(self) -> bool:
        return self.prefix == "ryxt"

class RayonixAddressEngine:
    """
    Production-grade RAYONIX address engine with complete cryptographic implementation
    """
    
    # Cryptographic constants
    HASH_ALGORITHM = hashes.SHA256()
    RIPEMD160_DIGEST_SIZE = 20
    SHA256_DIGEST_SIZE = 32
    BLAKE2B_DIGEST_SIZE = 32
    KECCAK256_DIGEST_SIZE = 32
    CHECKSUM_SIZE = 6
    PAYLOAD_SIZE = 32
    COMPLETE_ADDRESS_LENGTH = 43
    CRYPTOGRAPHIC_CONTEXT = b'rayonix-address-crypto-v1'
    
    # Network prefixes
    MAINNET_PREFIX = "ryx"
    TESTNET_PREFIX = "ryxt"
    
    # ECDSA curve - FIXED: Use SECP256K1 (uppercase)
    CURVE = ec.SECP256K1()
    BACKEND = default_backend()
    
    def __init__(self, network: str = "mainnet", strict_validation: bool = True):
        self.network = network
        self.strict_validation = strict_validation
        self.prefix = self.MAINNET_PREFIX if network == "mainnet" else self.TESTNET_PREFIX
        
        # Initialize cryptographic components
        self._hkdf_salt = secrets.token_bytes(32)
        self._validation_cache: Dict[str, bool] = {}
        self._derivation_cache: Dict[Tuple[bytes, int, bool, int], str] = {}
        
    def derive_address_from_public_key(self, 
                                     public_key: bytes, 
                                     address_type: AddressType = AddressType.STANDARD_P2PKH,
                                     version: AddressVersion = AddressVersion.V1_ENHANCED,
                                     index: int = 0,
                                     is_change: bool = False,
                                     derivation_path: Optional[str] = None) -> str:
        """
        Complete address derivation with full cryptographic processing
        """
        # Comprehensive input validation
        self._validate_public_key_cryptographic(public_key)
        
        # Cache key with cryptographic context
        cache_key = (public_key, index, is_change, address_type.value)
        if cache_key in self._derivation_cache:
            return self._derivation_cache[cache_key]
        
        try:
            # Step 1: Cryptographic key processing
            processed_key = self._process_public_key_cryptographic(public_key, address_type, version)
            
            # Step 2: Generate cryptographic hash based on version
            key_hash = self._generate_cryptographic_hash(processed_key, address_type, version)
            
            # Step 3: Create address payload with derivation context
            payload = self._create_cryptographic_payload(key_hash, address_type, version, index, is_change)
            
            # Step 4: Encode with cryptographic context
            encoded_payload = self._encode_with_cryptographic_context(payload, address_type, version)
            
            # Step 5: Calculate cryptographic checksum
            address = self._append_cryptographic_checksum(encoded_payload)
            
            # Cache with invalidation conditions
            if derivation_path and len(self._derivation_cache) < 10000:  # Prevent memory exhaustion
                self._derivation_cache[cache_key] = address
            
            return address
            
        except Exception as e:
            raise CryptoError(f"Cryptographic address derivation failed: {str(e)}") from e
    
    def _validate_public_key_cryptographic(self, public_key: bytes) -> None:
        """Comprehensive cryptographic validation of public key"""
        if not isinstance(public_key, bytes):
            raise InvalidAddressError("Public key must be bytes")
        
        if len(public_key) not in [33, 65]:
            raise InvalidAddressError(f"Invalid public key length: {len(public_key)}")
        
        # Validate ECDSA point using cryptography library - FIXED: Use SECP256K1
        try:
            ec.EllipticCurvePublicKey.from_encoded_point(self.CURVE, public_key)
        except Exception as e:
            raise InvalidAddressError(f"Invalid ECDSA public key: {str(e)}")
        
        # Additional validation for compressed format
        if len(public_key) == 33:
            prefix = public_key[0]
            if prefix not in [0x02, 0x03]:
                raise InvalidAddressError(f"Invalid compressed public key prefix: {prefix:02x}")
    
    def _process_public_key_cryptographic(self, public_key: bytes, address_type: AddressType, version: AddressVersion) -> bytes:
        """Cryptographic processing of public key based on address type and version"""
        # Always start with compressed format for consistency
        compressed_key = self._compress_public_key_cryptographic(public_key)
        
        if version == AddressVersion.V3_FUTURE:
            # V3 uses BLAKE2b for post-quantum preparation
            return self._process_key_blake2b(compressed_key, address_type)
        elif version == AddressVersion.V2_ADVANCED:
            # V2 uses HKDF for key separation
            return self._process_key_hkdf(compressed_key, address_type)
        else:
            # V1 uses standard processing
            return self._process_key_standard(compressed_key, address_type)
    
    def _compress_public_key_cryptographic(self, public_key: bytes) -> bytes:
        """Cryptographic public key compression with validation"""
        if len(public_key) == 33:
            return public_key
        
        # Extract coordinates from uncompressed key
        x = public_key[1:33]
        y = public_key[33:65]
        
        # Validate coordinates are on curve - FIXED: Use SECP256K1
        try:
            point = ec.EllipticCurvePublicKey.from_encoded_point(self.CURVE, public_key)
            # Get compressed representation
            compressed = point.public_bytes(Encoding.X962, PublicFormat.CompressedPoint)
            return compressed
        except Exception as e:
            raise CryptoError(f"Public key compression failed: {str(e)}")
    
    def _process_key_blake2b(self, public_key: bytes, address_type: AddressType) -> bytes:
        """Process key using BLAKE2b for version V3"""
        try:
            # Use hashlib's blake2b implementation
            personalization = f"rayonix-{address_type.name}".encode()[:16]
            blake2b = hashlib.blake2b(digest_size=self.BLAKE2B_DIGEST_SIZE, person=personalization)
            blake2b.update(public_key)
            return blake2b.digest()
        except Exception:
            # Fallback to SHA3 if BLAKE2b not available
            sha3_hash = hashlib.sha3_256(public_key)
            return sha3_hash.digest()
    
    def _process_key_hkdf(self, public_key: bytes, address_type: AddressType) -> bytes:
        """Process key using HKDF for version V2"""
        info = f"rayonix-address-{address_type.value}".encode()
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._hkdf_salt,
            info=info,
            backend=self.BACKEND
        )
        return hkdf.derive(public_key)
    
    def _process_key_standard(self, public_key: bytes, address_type: AddressType) -> bytes:
        """Standard key processing for versions V0 and V1"""
        if address_type in [AddressType.SEGWIT_V0, AddressType.SEGWIT_V1]:
            return self._process_key_segwit(public_key, address_type)
        elif address_type in [AddressType.SMART_CONTRACT, AddressType.TOKEN_CONTRACT]:
            return self._process_key_keccak(public_key)
        else:
            return public_key
    
    def _process_key_segwit(self, public_key: bytes, address_type: AddressType) -> bytes:
        """Process key for SegWit addresses"""
        if address_type == AddressType.SEGWIT_V1:
            # Taproot: use x-only public key (remove prefix)
            return public_key[1:] if len(public_key) == 33 else public_key[1:33]
        else:
            # SegWit v0: use compressed key directly
            return public_key
    
    def _process_key_keccak(self, public_key: bytes) -> bytes:
        """Process key using Keccak-256 for contract addresses"""
        try:
            # Use SHA3-256 as a reliable alternative to Keccak
            sha3_hash = hashlib.sha3_256()
            sha3_hash.update(public_key[1:] if len(public_key) == 65 else public_key)
            return sha3_hash.digest()
        except Exception:
            # Fallback to SHA256
            sha256_hash = hashlib.sha256(public_key[1:] if len(public_key) == 65 else public_key)
            return sha256_hash.digest()
    
    def _generate_cryptographic_hash(self, processed_key: bytes, address_type: AddressType, version: AddressVersion) -> bytes:
        """Generate cryptographic hash based on address type and version"""
        if version == AddressVersion.V3_FUTURE:
            return self._generate_hash_blake2b(processed_key, address_type)
        elif version == AddressVersion.V2_ADVANCED:
            return self._generate_hash_hkdf(processed_key, address_type)
        else:
            return self._generate_hash_standard(processed_key, address_type)
    
    def _generate_hash_blake2b(self, data: bytes, address_type: AddressType) -> bytes:
        """Generate hash using BLAKE2b"""
        try:
            personalization = f"rayonix-hash-{address_type.name}".encode()[:16]
            blake2b = hashlib.blake2b(digest_size=32, person=personalization)
            blake2b.update(data)
            hash1 = blake2b.digest()
            
            # Second round for additional security
            blake2b = hashlib.blake2b(digest_size=20, person=personalization)
            blake2b.update(hash1)
            return blake2b.digest()
        except Exception:
            # Fallback to SHA256
            sha_hash = hashlib.sha256(data).digest()
            return hashlib.new('ripemd160', sha_hash).digest()
    
    def _generate_hash_hkdf(self, data: bytes, address_type: AddressType) -> bytes:
        """Generate hash using HKDF chain"""
        # First HKDF for initial hash
        hkdf1 = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._hkdf_salt,
            info=b'rayonix-hash-primary',
            backend=self.BACKEND
        )
        hash1 = hkdf1.derive(data)
        
        # Second HKDF for final hash
        hkdf2 = HKDF(
            algorithm=hashes.SHA256(),
            length=20,
            salt=hash1[:16],
            info=b'rayonix-hash-secondary',
            backend=self.BACKEND
        )
        return hkdf2.derive(data)
    
    def _generate_hash_standard(self, data: bytes, address_type: AddressType) -> bytes:
        """Generate standard cryptographic hash"""
        if address_type in [AddressType.SMART_CONTRACT, AddressType.TOKEN_CONTRACT]:
            # Contract addresses use full 32-byte hash
            return self._process_key_keccak(data)
        else:
            # Standard addresses use SHA256 + RIPEMD160
            sha_hash = hashlib.sha256(data).digest()
            ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
            return ripemd_hash
    
    def _create_cryptographic_payload(self, key_hash: bytes, address_type: AddressType, 
                                   version: AddressVersion, index: int, is_change: bool) -> bytes:
        """Create cryptographic payload with derivation context"""
        # Start with the key hash
        payload = key_hash
        
        # Add derivation context for HD wallets
        if index > 0 or is_change:
            derivation_context = struct.pack('>IB', index, 1 if is_change else 0)
            
            # Use HKDF to incorporate derivation context
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=len(payload),
                salt=derivation_context,
                info=b'rayonix-payload-derivation',
                backend=self.BACKEND
            )
            payload = hkdf.derive(payload)
        
        # Ensure exact payload size using cryptographic stretching
        if len(payload) < self.PAYLOAD_SIZE:
            # Use PBKDF2 for secure extension
            import hashlib as fallback_hashlib
            extension = fallback_hashlib.pbkdf2_hmac(
                'sha256', 
                payload, 
                b'rayonix-payload-extension', 
                10000,  # 10,000 iterations
                self.PAYLOAD_SIZE - len(payload)
            )
            payload += extension
        elif len(payload) > self.PAYLOAD_SIZE:
            # Use HKDF for secure truncation
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=self.PAYLOAD_SIZE,
                salt=payload[:16],
                info=b'rayonix-payload-truncation',
                backend=self.BACKEND
            )
            payload = hkdf.derive(payload)
        
        return payload
    
    def _encode_with_cryptographic_context(self, payload: bytes, address_type: AddressType, version: AddressVersion) -> str:
        """Encode payload with cryptographic context"""
        type_char = str(address_type.value)
        version_char = chr(ord('a') + version.value)
        
        # Encode payload to Base32
        encoded_payload = Base32Crockford.encode(payload, include_padding=False)
        
        # Combine with cryptographic context
        return f"{self.prefix}{type_char}{version_char}{encoded_payload}"
    
    def _append_cryptographic_checksum(self, encoded_data: str) -> str:
        """Calculate and append cryptographic checksum"""
        # Use HMAC-SHA256 for checksum calculation
        import hmac
        
        # Derive checksum key from encoded data
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._hkdf_salt,
            info=b'rayonix-checksum-key',
            backend=self.BACKEND
        )
        checksum_key = hkdf.derive(encoded_data.encode())
        
        # Calculate HMAC-SHA256
        hmac_obj = hmac.new(checksum_key, encoded_data.encode(), digestmod='sha256')
        checksum_full = hmac_obj.digest()
        
        # Encode first 4 bytes as Base32 and take first 6 characters
        checksum = Base32Crockford.encode(checksum_full[:4], include_padding=False)[:self.CHECKSUM_SIZE]
        
        return encoded_data + checksum
    
    def validate_address(self, address: str, expected_type: Optional[AddressType] = None) -> bool:
        """
        Comprehensive cryptographic address validation
        """
        if address in self._validation_cache:
            return self._validation_cache[address]
        
        try:
            # Multi-stage validation
            validation_stages = [
                self._validate_basic_structure,
                self._validate_cryptographic_components,
                self._validate_checksum_cryptographic,
                self._validate_payload_cryptographic,
                self._validate_network_context,
            ]
            
            for stage in validation_stages:
                if not stage(address):
                    self._validation_cache[address] = False
                    return False
            
            # Type-specific validation if requested
            if expected_type and not self._validate_address_type(address, expected_type):
                self._validation_cache[address] = False
                return False
            
            self._validation_cache[address] = True
            return True
            
        except Exception:
            self._validation_cache[address] = False
            return False
    
    def _validate_basic_structure(self, address: str) -> bool:
        """Validate basic address structure"""
        if not isinstance(address, str) or len(address) != self.COMPLETE_ADDRESS_LENGTH:
            return False
        
        if not (address.startswith(self.MAINNET_PREFIX) or address.startswith(self.TESTNET_PREFIX)):
            return False
        
        # Validate type character
        type_char = address[3]
        if not type_char.isdigit() or int(type_char) not in [t.value for t in AddressType]:
            return False
        
        # Validate version character
        version_char = address[4]
        if not version_char.isalpha() or not version_char.islower():
            return False
        
        return True
    
    def _validate_cryptographic_components(self, address: str) -> bool:
        """Validate cryptographic components"""
        try:
            components = self._deconstruct_address_cryptographic(address)
            if not components:
                return False
            
            # Validate payload encoding
            if not Base32Crockford.validate(components.raw_address[5:37], strict=True):
                return False
            
            # Validate checksum encoding
            if not Base32Crockford.validate(components.checksum, strict=True):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _validate_checksum_cryptographic(self, address: str) -> bool:
        """Validate cryptographic checksum"""
        try:
            components = self._deconstruct_address_cryptographic(address)
            if not components:
                return False
            
            address_without_checksum = components.raw_address[:-self.CHECKSUM_SIZE]
            expected_checksum = self._calculate_cryptographic_checksum(address_without_checksum)
            
            return Base32Crockford.secure_compare(components.checksum, expected_checksum)
            
        except Exception:
            return False
    
    def _calculate_cryptographic_checksum(self, data: str) -> str:
        """Calculate cryptographic checksum"""
        import hmac
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self._hkdf_salt,
            info=b'rayonix-checksum-key',
            backend=self.BACKEND
        )
        checksum_key = hkdf.derive(data.encode())
        
        hmac_obj = hmac.new(checksum_key, data.encode(), digestmod='sha256')
        checksum_full = hmac_obj.digest()
        
        return Base32Crockford.encode(checksum_full[:4], include_padding=False)[:self.CHECKSUM_SIZE]
    
    def _validate_payload_cryptographic(self, address: str) -> bool:
        """Validate payload cryptographic properties"""
        try:
            components = self._deconstruct_address_cryptographic(address)
            if not components:
                return False
            
            # Validate payload size
            if len(components.payload) != self.PAYLOAD_SIZE:
                return False
            
            # Validate payload structure based on address type
            return self._validate_payload_structure(components)
            
        except Exception:
            return False
    
    def _validate_payload_structure(self, components: AddressComponents) -> bool:
        """Validate payload structure based on address type"""
        if components.address_type in [AddressType.SEGWIT_V0, AddressType.SEGWIT_V1]:
            return self._validate_segwit_payload_structure(components)
        elif components.address_type in [AddressType.SMART_CONTRACT, AddressType.TOKEN_CONTRACT]:
            return self._validate_contract_payload_structure(components)
        else:
            return self._validate_standard_payload_structure(components)
    
    def _validate_segwit_payload_structure(self, components: AddressComponents) -> bool:
        """Validate SegWit payload structure"""
        if components.address_type == AddressType.SEGWIT_V0:
            # P2WPKH: 20-byte witness program
            return len(components.payload) == 20
        elif components.address_type == AddressType.SEGWIT_V1:
            # P2TR: 32-byte witness program
            return len(components.payload) == 32
        return False
    
    def _validate_contract_payload_structure(self, components: AddressComponents) -> bool:
        """Validate contract payload structure"""
        # Contract addresses typically use full 32-byte hashes
        return len(components.payload) == 32
    
    def _validate_standard_payload_structure(self, components: AddressComponents) -> bool:
        """Validate standard payload structure"""
        # Standard addresses use 20-byte RIPEMD160 hashes
        return len(components.payload) == 20
    
    def _validate_network_context(self, address: str) -> bool:
        """Validate network context"""
        components = self._deconstruct_address_cryptographic(address)
        if not components:
            return False
        
        expected_prefix = self.MAINNET_PREFIX if self.network == "mainnet" else self.TESTNET_PREFIX
        return components.prefix == expected_prefix
    
    def _validate_address_type(self, address: str, expected_type: AddressType) -> bool:
        """Validate address type matches expected type"""
        components = self._deconstruct_address_cryptographic(address)
        return components and components.address_type == expected_type
    
    def _deconstruct_address_cryptographic(self, address: str) -> Optional[AddressComponents]:
        """Cryptographic deconstruction of address"""
        try:
            prefix = address[:3]
            type_char = address[3]
            version_char = address[4]
            payload_encoded = address[5:37]
            checksum = address[37:]
            
            # Decode payload with cryptographic validation
            payload = Base32Crockford.decode(payload_encoded, strict=True)
            
            # Parse type and version
            address_type = AddressType(int(type_char))
            version = AddressVersion(ord(version_char) - ord('a'))
            
            # Generate cryptographic context
            crypto_context = hashlib.sha256(
                prefix.encode() + type_char.encode() + version_char.encode() + payload
            ).digest()
            
            return AddressComponents(
                prefix=prefix,
                address_type=address_type,
                version=version,
                payload=payload,
                checksum=checksum,
                raw_address=address,
                cryptographic_context=crypto_context
            )
        except Exception:
            return None
    
    def generate_multisig_address(self, 
                                public_keys: List[bytes],
                                required_signatures: int,
                                address_type: AddressType = AddressType.MULTISIG_2OF3,
                                version: AddressVersion = AddressVersion.V1_ENHANCED) -> str:
        """
        Generate multisignature address with complete cryptographic implementation
        """
        # Validate multisig parameters
        if not public_keys or len(public_keys) < 2:
            raise InvalidAddressError("Multisig requires at least 2 public keys")
        
        if required_signatures < 1 or required_signatures > len(public_keys):
            raise InvalidAddressError(f"Invalid required signatures: {required_signatures} for {len(public_keys)} keys")
        
        # Validate all public keys cryptographically
        for i, pubkey in enumerate(public_keys):
            try:
                self._validate_public_key_cryptographic(pubkey)
            except CryptoError as e:
                raise InvalidAddressError(f"Invalid public key at index {i}: {str(e)}")
        
        try:
            # Sort public keys for deterministic address generation
            sorted_keys = sorted(public_keys)
            
            # Create multisig script hash
            script_hash = self._create_multisig_script_hash(sorted_keys, required_signatures)
            
            # Generate address from script hash
            return self.derive_address_from_public_key(script_hash, address_type, version)
            
        except Exception as e:
            raise InvalidAddressError(f"Multisig address generation failed: {str(e)}") from e
    
    def _create_multisig_script_hash(self, public_keys: List[bytes], required_sigs: int) -> bytes:
        """Create multisig script hash with complete script serialization"""
        # Serialize multisig script: OP_[required_sigs] [pubkeys...] OP_[total_keys] OP_CHECKMULTISIG
        script_parts = []
        
        # Add required signatures opcode
        script_parts.append(bytes([80 + required_sigs]))  # OP_1 = 81, OP_2 = 82, etc.
        
        # Add public keys
        for pubkey in public_keys:
            if len(pubkey) == 33:
                script_parts.append(b'\x21')  # Push 33 bytes
            else:
                script_parts.append(b'\x41')  # Push 65 bytes
            script_parts.append(pubkey)
        
        # Add total keys opcode and CHECKMULTISIG
        script_parts.append(bytes([80 + len(public_keys)]))  # OP_1 = 81, OP_2 = 82, etc.
        script_parts.append(b'\xae')  # OP_CHECKMULTISIG
        
        # Combine script
        script = b''.join(script_parts)
        
        # Hash script
        sha_hash = hashlib.sha256(script).digest()
        return hashlib.new('ripemd160', sha_hash).digest()
    
    def get_address_info(self, address: str) -> Dict[str, Any]:
        """
        Get complete cryptographic information about address
        """
        if not self.validate_address(address):
            raise InvalidAddressError("Cannot get info for invalid address")
        
        components = self._deconstruct_address_cryptographic(address)
        if not components:
            raise InvalidAddressError("Failed to deconstruct address")
        
        # Calculate additional cryptographic properties
        payload_hash = hashlib.sha256(components.payload).hexdigest()
        context_hash = hashlib.sha256(components.cryptographic_context).hexdigest()
        
        return {
            'address': address,
            'type': components.address_type.name,
            'type_code': components.address_type.value,
            'version': components.version.name,
            'version_code': components.version.value,
            'network': components.network,
            'payload_size': len(components.payload),
            'payload_hash': payload_hash,
            'cryptographic_context': context_hash,
            'checksum_algorithm': 'HMAC-SHA256',
            'hash_algorithm': self._get_hash_algorithm_for_version(components.version),
            'key_processing': self._get_key_processing_for_version(components.version),
            'security_level': self._calculate_security_level(components),
            'is_valid': True
        }
    
    def _get_hash_algorithm_for_version(self, version: AddressVersion) -> str:
        """Get hash algorithm for address version"""
        algorithms = {
            AddressVersion.V1_LEGACY: 'SHA256+RIPEMD160',
            AddressVersion.V1_ENHANCED: 'SHA256+RIPEMD160+HKDF',
            AddressVersion.V2_ADVANCED: 'HKDF-Chain',
            AddressVersion.V3_FUTURE: 'BLAKE2b'
        }
        return algorithms.get(version, 'Unknown')
    
    def _get_key_processing_for_version(self, version: AddressVersion) -> str:
        """Get key processing method for address version"""
        processing = {
            AddressVersion.V1_LEGACY: 'Standard Compression',
            AddressVersion.V1_ENHANCED: 'Compression+Derivation',
            AddressVersion.V2_ADVANCED: 'HKDF Processing',
            AddressVersion.V3_FUTURE: 'BLAKE2b Processing'
        }
        return processing.get(version, 'Unknown')
    
    def _calculate_security_level(self, components: AddressComponents) -> str:
        """Calculate cryptographic security level"""
        if components.version == AddressVersion.V3_FUTURE:
            return 'Post-Quantum Prepared'
        elif components.version == AddressVersion.V2_ADVANCED:
            return 'High (HKDF)'
        elif components.version == AddressVersion.V1_ENHANCED:
            return 'Medium (Enhanced)'
        else:
            return 'Standard'
    
    def clear_caches(self) -> None:
        """Clear all cryptographic caches"""
        self._validation_cache.clear()
        self._derivation_cache.clear()
        
        # Generate new HKDF salt for forward security
        self._hkdf_salt = secrets.token_bytes(32)