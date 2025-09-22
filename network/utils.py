import asyncio
import time
import uuid
import hashlib
import secrets
import zlib
import msgpack
import struct
import logging
import json
import random
import ipaddress
from typing import Any, Dict, List, Tuple, Optional, Set
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

logger = logging.getLogger("AdvancedNetwork")

class CompressionUtils:
    """Utility class for data compression"""
    
    @staticmethod
    def compress_data(data: bytes, level: int = 6) -> bytes:
        """Compress data using zlib with specified compression level"""
        try:
            return zlib.compress(data, level=level)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            raise
    
    @staticmethod
    def decompress_data(data: bytes) -> bytes:
        """Decompress data using zlib"""
        try:
            return zlib.decompress(data)
        except zlib.error as e:
            logger.error(f"Decompression error: {e}")
            # Try to handle corrupted data gracefully
            if len(data) > 0:
                return data  # Return original data if decompression fails
            raise
        except Exception as e:
            logger.error(f"Unexpected decompression error: {e}")
            raise

class SerializationUtils:
    """Utility class for message serialization"""
    
    @staticmethod
    def serialize_message(message: Any) -> bytes:
        """Serialize message to bytes using msgpack"""
        try:
            return msgpack.packb(message, use_bin_type=True)
        except Exception as e:
            logger.error(f"Message serialization error: {e}")
            raise
    
    @staticmethod
    def deserialize_message(data: bytes) -> Any:
        """Deserialize message from bytes using msgpack"""
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception as e:
            logger.error(f"Message deserialization error: {e}")
            raise

class EncryptionUtils:
    """Utility class for data encryption"""
    
    @staticmethod
    def encrypt_data(data: bytes, key: bytes) -> bytes:
        """Encrypt data using AES-GCM"""
        try:
            iv = get_random_bytes(12)
            cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            return iv + ciphertext + tag
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    @staticmethod
    def decrypt_data(data: bytes, key: bytes) -> bytes:
        """Decrypt data using AES-GCM"""
        try:
            if len(data) < 28:  # IV (12) + tag (16) minimum
                raise ValueError("Data too short for decryption")
            
            iv = data[:12]
            ciphertext = data[12:-16]
            tag = data[-16:]
            
            cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

class NetworkUtils:
    """General network utilities"""
    
    @staticmethod
    def generate_node_id() -> str:
        """Generate unique node ID using cryptographic randomness"""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    
    @staticmethod
    def create_message_header(payload: bytes, magic: bytes, version: int = 1) -> bytes:
        """Create message header with magic, command, length, and checksum"""
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        header = struct.pack(
            '4s12sI4sI',
            magic,
            b'RAYX_MSG',
            len(payload),
            checksum,
            version
        )
        
        return header
    
    @staticmethod
    def parse_message_header(data: bytes) -> Tuple[Dict, bytes]:
        """Parse message header from data"""
        if len(data) < 28:  # Header size with version
            raise ValueError("Message too short for header")
        
        magic, command, length, checksum, version = struct.unpack('4s12sI4sI', data[:28])
        payload = data[28:28+length]
        
        if len(payload) != length:
            raise ValueError("Payload length mismatch")
        
        return {
            'magic': magic,
            'command': command,
            'length': length,
            'checksum': checksum,
            'version': version
        }, payload
    
    @staticmethod
    async def receive_data(reader: asyncio.StreamReader, header_size: int = 28) -> bytes:
        """Receive data with proper error handling"""
        try:
            header_data = await reader.readexactly(header_size)
            header = NetworkUtils.parse_message_header(header_data + b'\x00' * header_size)[0]
            payload = await reader.readexactly(header['length'])
            return header_data + payload
        except asyncio.IncompleteReadError:
            raise ConnectionError("Connection closed during data reception")
        except Exception as e:
            raise ConnectionError(f"Failed to receive data: {e}")
    
    @staticmethod
    async def send_data(writer: asyncio.StreamWriter, data: bytes):
        """Send data with proper error handling"""
        try:
            writer.write(data)
            await writer.drain()
        except Exception as e:
            raise ConnectionError(f"Failed to send data: {e}")
    
    @staticmethod
    def is_valid_ip_address(address: str) -> bool:
        """Check if the given string is a valid IP address"""
        try:
            ipaddress.ip_address(address)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def is_private_ip(address: str) -> bool:
        """Check if the IP address is private"""
        try:
            ip = ipaddress.ip_address(address)
            return ip.is_private
        except ValueError:
            return False
    
    @staticmethod
    def calculate_distance(node_id1: str, node_id2: str) -> int:
        """Calculate XOR distance between two node IDs"""
        if len(node_id1) != len(node_id2):
            raise ValueError("Node IDs must be the same length")
        
        bytes1 = bytes.fromhex(node_id1)
        bytes2 = bytes.fromhex(node_id2)
        
        distance = 0
        for b1, b2 in zip(bytes1, bytes2):
            distance = (distance << 8) | (b1 ^ b2)
        
        return distance
    
    @staticmethod
    def generate_nonce() -> str:
        """Generate a cryptographic nonce for replay protection"""
        return secrets.token_hex(16)
    
    @staticmethod
    def validate_nonce(nonce: str, max_age: int = 300) -> bool:
        """Validate a nonce (basic implementation)"""
        # In production, you'd want to track used nonces and check timestamps
        return len(nonce) == 32  # Simple length check for demo

class CryptoUtils:
    """Cryptographic utilities"""
    
    @staticmethod
    def generate_ec_keypair() -> ec.EllipticCurvePrivateKey:
        """Generate Elliptic Curve key pair using SECP256K1"""
        return ec.generate_private_key(ec.SECP256K1(), default_backend())
    
    @staticmethod
    def serialize_public_key(public_key: ec.EllipticCurvePublicKey) -> bytes:
        """Serialize public key to compressed format"""
        return public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )
    
    @staticmethod
    def deserialize_public_key(public_key_bytes: bytes) -> ec.EllipticCurvePublicKey:
        """Deserialize public key from compressed format"""
        return ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key_bytes
        )
    
    @staticmethod
    def sign_data(private_key: ec.EllipticCurvePrivateKey, data: bytes) -> bytes:
        """Sign data with private key using ECDSA"""
        return private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
    
    @staticmethod
    def verify_signature(public_key: ec.EllipticCurvePublicKey, signature: bytes, data: bytes) -> bool:
        """Verify signature with public key"""
        try:
            public_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
            return True
        except Exception:
            return False

class TimeUtils:
    """Time-related utilities"""
    
    @staticmethod
    def current_millis() -> int:
        """Get current time in milliseconds"""
        return int(time.time() * 1000)
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable form"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"
    
    @staticmethod
    def is_fresh(timestamp: float, max_age: float) -> bool:
        """Check if a timestamp is fresh (within max_age)"""
        return time.time() - timestamp <= max_age

class ValidationUtils:
    """Data validation utilities"""
    
    @staticmethod
    def validate_peer_info(peer_info: Dict) -> bool:
        """Validate peer information structure"""
        required_fields = ['address', 'port', 'protocol']
        return all(field in peer_info for field in required_fields)
    
    @staticmethod
    def validate_message_structure(message: Dict) -> bool:
        """Validate message structure"""
        required_fields = ['message_id', 'message_type', 'payload']
        return all(field in message for field in required_fields)
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 256) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            return ""
        return input_str.strip()[:max_length]