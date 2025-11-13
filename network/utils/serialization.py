import msgpack
import json
from typing import Any
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType

def serialize_message(message: NetworkMessage) -> bytes:
    """Enhanced serialization for blockchain standards"""
    try:
        # Convert to dict for serialization with enhanced structure
        message_dict = {
            'v': 1,  # Version for forward compatibility
            'id': message.message_id,
            'type': message.message_type.name,
            'payload': message.payload,
            'ts': message.timestamp,  # Shorter keys for efficiency
            'ttl': message.ttl,
            'sig': message.signature,
            'src': message.source_node,
            'dst': message.destination_node,
            'pri': message.priority
        }
        
        # Use msgpack with optimized settings
        return msgpack.packb(message_dict, use_bin_type=True, strict_types=True)
        
    except Exception as e:
        raise Exception(f"Message serialization error: {e}")

def deserialize_message(data: bytes) -> NetworkMessage:
    """Enhanced deserialization with validation"""
    try:
        message_dict = msgpack.unpackb(data, raw=False, strict_map_key=False)
        
        # Version check for forward compatibility
        if message_dict.get('v', 1) != 1:
            raise ValueError(f"Unsupported message version: {message_dict.get('v')}")
        
        # Convert back to NetworkMessage
        message = NetworkMessage(
            message_id=message_dict['id'],
            message_type=MessageType[message_dict['type']],
            payload=message_dict['payload'],
            timestamp=message_dict['ts'],
            ttl=message_dict['ttl'],
            signature=message_dict.get('sig'),
            source_node=message_dict.get('src'),
            destination_node=message_dict.get('dst'),
            priority=message_dict.get('pri', 1)
        )
        
        return message
        
    except Exception as e:
        raise Exception(f"Message deserialization error: {e}")