# smart_contract/utils/serialization_utils.py
import json
import pickle
import zlib
import logging
from typing import Any, Optional
import msgpack

logger = logging.getLogger("SmartContract.Serialization")

def serialize_contract(contract: Any) -> bytes:
    """Serialize contract object to bytes"""
    try:
        # Convert to dictionary first
        contract_dict = contract.to_dict()
        
        # Use msgpack for efficient serialization
        return msgpack.packb(contract_dict, use_bin_type=True)
        
    except Exception as e:
        logger.error(f"Contract serialization failed: {e}")
        raise

def deserialize_contract(data: bytes, contract_class: Any) -> Any:
    """Deserialize bytes to contract object"""
    try:
        # Deserialize from msgpack
        contract_dict = msgpack.unpackb(data, raw=False)
        
        # Reconstruct contract object
        # This would need to be implemented based on contract structure
        return contract_class(**contract_dict)
        
    except Exception as e:
        logger.error(f"Contract deserialization failed: {e}")
        raise

def compress_data(data: bytes) -> bytes:
    """Compress data using zlib"""
    try:
        return zlib.compress(data, level=9)
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        raise

def decompress_data(compressed_data: bytes) -> bytes:
    """Decompress data using zlib"""
    try:
        return zlib.decompress(compressed_data)
    except Exception as e:
        logger.error(f"Decompression failed: {e}")
        raise

def json_serialize(obj: Any) -> str:
    """Serialize object to JSON with extended support"""
    def default_serializer(o):
        if hasattr(o, 'to_dict'):
            return o.to_dict()
        elif hasattr(o, '__dict__'):
            return o.__dict__
        else:
            raise TypeError(f"Object of type {type(o)} is not JSON serializable")
    
    return json.dumps(obj, default=default_serializer, indent=2)

def json_deserialize(json_str: str) -> Any:
    """Deserialize JSON string to object"""
    return json.loads(json_str)

def binary_to_hex(data: bytes) -> str:
    """Convert binary data to hex string"""
    return data.hex()

def hex_to_binary(hex_str: str) -> bytes:
    """Convert hex string to binary data"""
    return bytes.fromhex(hex_str)

def base64_encode(data: bytes) -> str:
    """Encode binary data to base64"""
    import base64
    return base64.b64encode(data).decode('utf-8')

def base64_decode(b64_str: str) -> bytes:
    """Decode base64 string to binary data"""
    import base64
    return base64.b64decode(b64_str)

def create_checksum(data: bytes) -> str:
    """Create checksum for data validation"""
    return hashlib.sha256(data).hexdigest()

def validate_checksum(data: bytes, expected_checksum: str) -> bool:
    """Validate data against checksum"""
    actual_checksum = create_checksum(data)
    return hmac.compare_digest(actual_checksum, expected_checksum)