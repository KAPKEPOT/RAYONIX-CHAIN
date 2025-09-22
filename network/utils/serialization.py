import msgpack
import json
from typing import Any
from ..models.network_message import NetworkMessage
from ..config.network_types import MessageType

def serialize_message(message: NetworkMessage) -> bytes:
    """Serialize message to bytes"""
    try:
        # Convert to dict for serialization
        message_dict = {
            'message_id': message.message_id,
            'message_type': message.message_type.name,
            'payload': message.payload,
            'timestamp': message.timestamp,
            'ttl': message.ttl,
            'signature': message.signature,
            'source_node': message.source_node,
            'destination_node': message.destination_node,
            'priority': message.priority
        }
        
        # Use msgpack for efficient binary serialization
        return msgpack.packb(message_dict, use_bin_type=True)
        
    except Exception as e:
        raise Exception(f"Message serialization error: {e}")

def deserialize_message(data: bytes) -> NetworkMessage:
    """Deserialize message from bytes"""
    try:
        message_dict = msgpack.unpackb(data, raw=False)
        
        # Convert back to NetworkMessage
        message = NetworkMessage(
            message_id=message_dict['message_id'],
            message_type=MessageType[message_dict['message_type']],
            payload=message_dict['payload'],
            timestamp=message_dict['timestamp'],
            ttl=message_dict['ttl'],
            signature=message_dict.get('signature'),
            source_node=message_dict.get('source_node'),
            destination_node=message_dict.get('destination_node'),
            priority=message_dict.get('priority', 1)
        )
        
        return message
        
    except Exception as e:
        raise Exception(f"Message deserialization error: {e}")