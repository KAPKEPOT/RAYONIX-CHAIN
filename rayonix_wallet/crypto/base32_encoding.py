# rayonix_wallet/crypto/base32_encoding.py
import struct
import secrets
from typing import Tuple, Optional, List
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from rayonix_wallet.core.exceptions import CryptoError

class Base32Crockford:
    """RFC-4648 compliant Base32 encoding with Crockford's alphabet modifications"""
    
    # Crockford's Base32 alphabet (no I, L, O, U for readability)
    ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
    DECODE_MAP = {
        '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
        'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
        'J': 18, 'K': 19, 'M': 20, 'N': 21, 'P': 22, 'Q': 23, 'R': 24, 'S': 25,
        'T': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31,
        # Common misreads
        'I': 18, 'L': 19, 'O': 20, 'U': 27
    }
    
    PADDING_CHAR = '='
    BACKEND = default_backend()
    
    @classmethod
    def encode(cls, data: bytes, include_padding: bool = True) -> str:
        """
        Encode bytes to Base32 string using Crockford's alphabet
        """
        if not isinstance(data, bytes):
            raise CryptoError("Input must be bytes")
        
        if len(data) == 0:
            return ""
        
        encoded_chars = []
        buffer = 0
        bits_in_buffer = 0
        total_bits = len(data) * 8
        
        for byte in data:
            buffer = (buffer << 8) | byte
            bits_in_buffer += 8
            
            while bits_in_buffer >= 5:
                bits_in_buffer -= 5
                index = (buffer >> bits_in_buffer) & 0x1F
                encoded_chars.append(cls.ALPHABET[index])
                buffer &= (1 << bits_in_buffer) - 1
        
        # Handle remaining bits
        if bits_in_buffer > 0:
            # Pad remaining bits to form a complete 5-bit group
            buffer <<= (5 - bits_in_buffer)
            index = buffer & 0x1F
            encoded_chars.append(cls.ALPHABET[index])
        
        result = ''.join(encoded_chars)
        
        # Add padding if requested
        if include_padding:
            padding_needed = (8 - len(result) % 8) % 8
            result += cls.PADDING_CHAR * padding_needed
        
        return result
    
    @classmethod
    def decode(cls, encoded: str, strict: bool = True) -> bytes:
        """
        Decode Base32 string to bytes with comprehensive validation
        """
        if not isinstance(encoded, str):
            raise CryptoError("Input must be string")
        
        # Remove padding and whitespace, convert to uppercase
        encoded = encoded.rstrip(cls.PADDING_CHAR).upper().replace(' ', '').replace('\t', '').replace('\n', '')
        
        if len(encoded) == 0:
            return b""
        
        # Validate characters and build bit string
        bit_string = ""
        for char in encoded:
            if char not in cls.DECODE_MAP:
                if strict:
                    raise CryptoError(f"Invalid Base32 character: {char}")
                else:
                    continue
            
            value = cls.DECODE_MAP[char]
            bit_string += format(value, '05b')
        
        # Convert bit string to bytes
        decoded_bytes = bytearray()
        for i in range(0, len(bit_string), 8):
            byte_bits = bit_string[i:i+8]
            if len(byte_bits) < 8:
                if strict and len(byte_bits) > 0:
                    raise CryptoError("Invalid padding - incomplete byte")
                break
            decoded_bytes.append(int(byte_bits, 2))
        
        return bytes(decoded_bytes)
    
    @classmethod
    def encode_with_checksum(cls, data: bytes, checksum_length: int = 6) -> Tuple[str, str]:
        """
        Encode data with separate checksum using HKDF for cryptographic strength
        """
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=secrets.token_bytes(16),
            info=b'rayonix-base32-checksum',
            backend=cls.BACKEND
        )
        
        # Derive checksum key
        checksum_key = hkdf.derive(data + struct.pack('>Q', len(data)))
        
        # Calculate checksum using HMAC-SHA256
        import hmac
        hmac_obj = hmac.new(checksum_key, data, digestmod='sha256')
        checksum_full = hmac_obj.digest()
        
        # Encode data and checksum
        encoded_data = cls.encode(data, include_padding=False)
        encoded_checksum = cls.encode(checksum_full[:4], include_padding=False)[:checksum_length]
        
        return encoded_data, encoded_checksum
    
    @classmethod
    def verify_with_checksum(cls, encoded_data: str, checksum: str, original_data: bytes) -> bool:
        """
        Verify data with cryptographic checksum
        """
        try:
            # Recalculate expected checksum
            _, expected_checksum = cls.encode_with_checksum(original_data, len(checksum))
            
            # Constant-time comparison
            return cls.secure_compare(checksum, expected_checksum)
        except Exception:
            return False
    
    @staticmethod
    def secure_compare(a: str, b: str) -> bool:
        """
        Constant-time string comparison to prevent timing attacks
        """
        if len(a) != len(b):
            return False
        
        # Use secrets.compare_digest for actual constant-time comparison
        return secrets.compare_digest(a.encode('utf-8'), b.encode('utf-8'))
    
    @classmethod
    def validate_encoded_string(cls, encoded: str, strict: bool = True) -> Tuple[bool, List[str]]:
        """
        Comprehensive validation of Base32 encoded string
        Returns: (is_valid, list_of_errors)
        """
        errors = []
        
        if not isinstance(encoded, str):
            errors.append("Input must be string")
            return False, errors
        
        # Check for invalid characters
        for i, char in enumerate(encoded):
            if char == cls.PADDING_CHAR:
                continue
            if char not in cls.DECODE_MAP:
                errors.append(f"Invalid character at position {i}: '{char}'")
        
        # Check padding
        if cls.PADDING_CHAR in encoded:
            padding_start = encoded.index(cls.PADDING_CHAR)
            if padding_start < len(encoded) - 1 and any(c != cls.PADDING_CHAR for c in encoded[padding_start:]):
                errors.append("Padding characters must only appear at the end")
        
        # Try decoding if strict
        if strict and not errors:
            try:
                cls.decode(encoded, strict=True)
            except CryptoError as e:
                errors.append(f"Decoding failed: {str(e)}")
        
        return len(errors) == 0, errors