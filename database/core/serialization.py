# database/core/serialization.py

import msgpack
import logging
from typing import Any
from decimal import Decimal
from datetime import datetime, date
import uuid
from dataclasses import is_dataclass, asdict

from database.utils.exceptions import SerializationError

logger = logging.getLogger(__name__)

class MsgPackSerializer:
    """Pure MessagePack serialization - ONE consistent format"""
    
    def __init__(self, use_bin_type=True, strict_types=False, datetime_support=True):
        self.use_bin_type = use_bin_type
        self.strict_types = strict_types
        self.datetime_support = datetime_support
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to MessagePack using ONE consistent format"""
        try:
            # Convert to serializable format first
            serializable_value = self._convert_to_serializable(value)
            
            return msgpack.packb(
                serializable_value, 
                use_bin_type=self.use_bin_type,
                strict_types=self.strict_types
            )
        except (msgpack.PackException, TypeError, ValueError) as e:
            logger.error(f"MessagePack serialization error for type {type(value)}: {e}")
            raise SerializationError(f"MessagePack serialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected MessagePack serialization error: {e}")
            raise SerializationError(f"MessagePack serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize MessagePack to value using ONE consistent format"""
        if not data:
            return None
            
        try:
            # Unpack the data
            unpacked_data = msgpack.unpackb(
                data, 
                raw=False,
                strict_map_key=False,
                use_list=True
            )
            
            # Convert back from serializable format
            return self._convert_from_serializable(unpacked_data)
            
        except (msgpack.UnpackException, ValueError) as e:
            logger.error(f"MessagePack deserialization error: {e}")
            raise SerializationError(f"MessagePack deserialization failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected MessagePack deserialization error: {e}")
            raise SerializationError(f"MessagePack deserialization failed: {e}")
    
    def _convert_to_serializable(self, value: Any) -> Any:
        """Convert complex types to MessagePack-serializable formats"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, bytes):
            return value  # MsgPack handles bytes natively
        elif isinstance(value, (list, tuple)):
            return [self._convert_to_serializable(item) for item in value]
        elif isinstance(value, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in value.items()}
        elif isinstance(value, (datetime, date)):
            if self.datetime_support:
                return {
                    '__type__': 'datetime' if isinstance(value, datetime) else 'date',
                    'isoformat': value.isoformat()
                }
            else:
                return value.isoformat()
        elif isinstance(value, Decimal):
            return {
                '__type__': 'decimal',
                'value': str(value)
            }
        elif isinstance(value, uuid.UUID):
            return {
                '__type__': 'uuid',
                'value': str(value)
            }
        elif isinstance(value, set):
            return {
                '__type__': 'set',
                'values': [self._convert_to_serializable(item) for item in value]
            }
        elif is_dataclass(value):
            return {
                '__type__': 'dataclass',
                'class': value.__class__.__name__,
                'data': asdict(value)
            }
        elif hasattr(value, '__dict__'):
            # Generic object - convert to dict
            return {
                '__type__': 'object',
                'class': value.__class__.__name__,
                'data': {k: self._convert_to_serializable(v) 
                        for k, v in value.__dict__.items() 
                        if not k.startswith('_')}
            }
        else:
            # Fallback to string representation
            logger.warning(f"Converting unsupported type {type(value)} to string")
            return str(value)
    
    def _convert_from_serializable(self, data: Any) -> Any:
        """Convert from serializable format back to original types"""
        if data is None:
            return None
        elif isinstance(data, (str, int, float, bool, bytes)):
            return data
        elif isinstance(data, list):
            return [self._convert_from_serializable(item) for item in data]
        elif isinstance(data, dict):
            # Check for special type markers
            if '__type__' in data:
                return self._restore_special_type(data)
            else:
                return {k: self._convert_from_serializable(v) for k, v in data.items()}
        else:
            return data
    
    def _restore_special_type(self, data: dict) -> Any:
        """Restore special types from serialized format"""
        type_name = data.get('__type__')
        
        if type_name == 'datetime':
            return datetime.fromisoformat(data['isoformat'])
        elif type_name == 'date':
            return date.fromisoformat(data['isoformat'])
        elif type_name == 'decimal':
            return Decimal(data['value'])
        elif type_name == 'uuid':
            return uuid.UUID(data['value'])
        elif type_name == 'set':
            return set(self._convert_from_serializable(data['values']))
        elif type_name in ['dataclass', 'object']:
            # Note: We can't fully reconstruct arbitrary objects without their classes
            # Return the data dict for the application to handle
            logger.debug(f"Returning serialized {type_name} data: {data['class']}")
            return data['data']
        else:
            logger.warning(f"Unknown special type: {type_name}")
            return data

# Global serializer instance for consistent usage
DEFAULT_SERIALIZER = MsgPackSerializer(
    use_bin_type=True,
    strict_types=False,
    datetime_support=True
)

# Convenience functions for easy usage
def serialize(value: Any) -> bytes:
    """Serialize value using the default MsgPack serializer"""
    return DEFAULT_SERIALIZER.serialize(value)

def deserialize(data: bytes) -> Any:
    """Deserialize data using the default MsgPack serializer"""
    return DEFAULT_SERIALIZER.deserialize(data)

def create_serializer(**kwargs) -> MsgPackSerializer:
    """Create a customized MsgPack serializer"""
    return MsgPackSerializer(**kwargs)