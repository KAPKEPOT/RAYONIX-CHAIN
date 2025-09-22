"""
Message serialization and deserialization for network communication
"""

import json
import msgpack
from typing import Dict, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

from ..exceptions import NetworkError

logger = logging.getLogger('consensus.network')

class SerializationFormat(Enum):
    JSON = "json"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"

@dataclass
class MessageHeader:
    """Message header for network communication"""
    version: str = "1.0.0"
    message_type: str = ""
    timestamp: float = 0.0
    signature: Optional[str] = None
    compression: bool = False

class MessageSerializer:
    """Message serialization and deserialization"""
    
    def __init__(self, format: SerializationFormat = SerializationFormat.JSON, 
                 compress: bool = False, sign_messages: bool = True):
        self.format = format
        self.compress = compress
        self.sign_messages = sign_messages
        
    def serialize_message(self, message_type: str, data: Dict, 
                         signer: Optional[Any] = None) -> bytes:
        """Serialize message with header"""
        try:
            header = MessageHeader(
                version="1.0.0",
                message_type=message_type,
                timestamp=time.time(),
                compression=self.compress
            )
            
            # Serialize data based on format
            if self.format == SerializationFormat.JSON:
                data_bytes = json.dumps(data).encode()
            elif self.format == SerializationFormat.MSGPACK:
                data_bytes = msgpack.packb(data)
            else:
                raise NetworkError(f"Unsupported format: {self.format}")
            
            # Compress if enabled
            if self.compress:
                import zlib
                data_bytes = zlib.compress(data_bytes)
            
            # Create message with header
            message = {
                'header': {
                    'version': header.version,
                    'type': header.message_type,
                    'timestamp': header.timestamp,
                    'compression': header.compression
                },
                'data': data_bytes.hex() if self.format == SerializationFormat.JSON else data_bytes
            }
            
            # Sign message if signer provided
            if self.sign_messages and signer:
                message_data = json.dumps(message).encode()
                signature = signer.sign_message(message_data)
                message['header']['signature'] = signature.hex()
            
            # Final serialization
            if self.format == SerializationFormat.JSON:
                return json.dumps(message).encode()
            else:
                return msgpack.packb(message)
                
        except Exception as e:
            raise NetworkError(f"Failed to serialize message: {e}")
    
    def deserialize_message(self, message_bytes: bytes, 
                           verifier: Optional[Any] = None) -> Dict:
        """Deserialize message and validate"""
        try:
            # Parse message based on format
            if self.format == SerializationFormat.JSON:
                message = json.loads(message_bytes.decode())
            elif self.format == SerializationFormat.MSGPACK:
                message = msgpack.unpackb(message_bytes)
            else:
                raise NetworkError(f"Unsupported format: {self.format}")
            
            # Verify signature if present
            if self.sign_messages and verifier and 'header' in message:
                signature_hex = message['header'].get('signature')
                if signature_hex:
                    # Remove signature for verification
                    original_signature = message['header']['signature']
                    del message['header']['signature']
                    
                    message_data = json.dumps(message).encode()
                    signature = bytes.fromhex(original_signature)
                    
                    if not verifier.verify_signature(message_data, signature):
                        raise NetworkError("Invalid message signature")
                    
                    # Restore signature
                    message['header']['signature'] = original_signature
            
            # Extract data
            data_field = message.get('data')
            if isinstance(data_field, str):
                data_bytes = bytes.fromhex(data_field)
            else:
                data_bytes = data_field
            
            # Decompress if needed
            header = message.get('header', {})
            if header.get('compression', False):
                import zlib
                data_bytes = zlib.decompress(data_bytes)
            
            # Parse data based on format
            if self.format == SerializationFormat.JSON:
                data = json.loads(data_bytes.decode())
            else:
                data = msgpack.unpackb(data_bytes)
            
            return {
                'header': header,
                'data': data
            }
            
        except Exception as e:
            raise NetworkError(f"Failed to deserialize message: {e}")
    
    def create_proposal_message(self, proposal_data: Dict, signer: Any) -> bytes:
        """Create serialized proposal message"""
        return self.serialize_message("proposal", proposal_data, signer)
    
    def create_vote_message(self, vote_data: Dict, signer: Any) -> bytes:
        """Create serialized vote message"""
        return self.serialize_message("vote", vote_data, signer)
    
    def create_peer_discovery_message(self, peer_data: Dict, signer: Any) -> bytes:
        """Create serialized peer discovery message"""
        return self.serialize_message("peer_discovery", peer_data, signer)
    
    def parse_proposal_message(self, message_bytes: bytes, verifier: Any) -> Dict:
        """Parse and validate proposal message"""
        message = self.deserialize_message(message_bytes, verifier)
        if message['header']['type'] != 'proposal':
            raise NetworkError("Not a proposal message")
        return message['data']
    
    def parse_vote_message(self, message_bytes: bytes, verifier: Any) -> Dict:
        """Parse and validate vote message"""
        message = self.deserialize_message(message_bytes, verifier)
        if message['header']['type'] != 'vote':
            raise NetworkError("Not a vote message")
        return message['data']
    
    def validate_message_timestamp(self, message: Dict, max_age: float = 30.0) -> bool:
        """Validate message timestamp"""
        header = message.get('header', {})
        timestamp = header.get('timestamp', 0)
        return time.time() - timestamp <= max_age
    
    def get_message_size(self, message_bytes: bytes) -> int:
        """Get message size in bytes"""
        return len(message_bytes)
    
    def estimate_message_size(self, data: Dict) -> int:
        """Estimate serialized message size"""
        try:
            serialized = self.serialize_message("estimate", data)
            return len(serialized)
        except:
            return 0