import json
import msgpack
import pickle
from typing import Any, Dict, Optional
from enum import Enum, auto
import logging
from datetime import datetime, date
from decimal import Decimal
import uuid
import struct
from io import BytesIO

# Avro imports
try:
    import avro.schema
    from avro.io import DatumWriter, DatumReader, BinaryEncoder, BinaryDecoder
    AVRO_AVAILABLE = True
except ImportError:
    AVRO_AVAILABLE = False

# Protobuf imports
try:
    from google.protobuf.message import Message as ProtobufMessage
    from google.protobuf.json_format import MessageToDict, ParseDict
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False

from database.utils.exceptions import SerializationError

logger = logging.getLogger(__name__)

class SerializationFormat(Enum):
    """Supported serialization formats"""
    JSON = auto()
    MSGPACK = auto()
    PROTOBUF = auto()
    AVRO = auto()
    PICKLE = auto()

class JSONSerializer:
    """JSON serialization with extended data type support"""
    
    def __init__(self, encoding='utf-8', ensure_ascii=False, indent=None):
        self.encoding = encoding
        self.ensure_ascii = ensure_ascii
        self.indent = indent
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to JSON bytes"""
        try:
            return json.dumps(
                value, 
                ensure_ascii=self.ensure_ascii, 
                indent=self.indent,
                default=self._default_encoder,
                separators=(',', ':') if not self.indent else None
            ).encode(self.encoding)
        except (TypeError, ValueError, OverflowError) as e:
            logger.error(f"JSON serialization error for value: {type(value)}")
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to value"""
        try:
            return json.loads(data.decode(self.encoding), parse_float=Decimal)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
            logger.error(f"JSON deserialization error for data: {data[:100]}")
            raise SerializationError(f"JSON deserialization failed: {e}")
    
    def _default_encoder(self, obj):
        """Handle non-serializable objects"""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return str(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif isinstance(obj, bytes):
            return obj.hex()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        else:
            return str(obj)

class MsgPackSerializer:
    """MessagePack serialization for efficient binary format"""
    
    def __init__(self, use_bin_type=True, strict_types=False):
        self.use_bin_type = use_bin_type
        self.strict_types = strict_types
    
    def serialize(self, value: Any) -> bytes:
        """Serialize value to MessagePack"""
        try:
            return msgpack.packb(
                value, 
                use_bin_type=self.use_bin_type,
                strict_types=self.strict_types,
                default=self._default_encoder
            )
        except (msgpack.PackException, TypeError, ValueError) as e:
            logger.error(f"MessagePack serialization error for value: {type(value)}")
            raise SerializationError(f"MessagePack serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize MessagePack to value"""
        try:
            return msgpack.unpackb(
                data, 
                raw=False,
                strict_map_key=False,
                use_list=True
            )
        except (msgpack.UnpackException, ValueError) as e:
            logger.error(f"MessagePack deserialization error for data: {data[:100]}")
            raise SerializationError(f"MessagePack deserialization failed: {e}")
    
    def _default_encoder(self, obj):
        """Handle non-serializable objects for MessagePack"""
        if isinstance(obj, (datetime, date)):
            return {'__datetime__': obj.isoformat()}
        elif isinstance(obj, Decimal):
            return {'__decimal__': str(obj)}
        elif isinstance(obj, uuid.UUID):
            return {'__uuid__': str(obj)}
        elif isinstance(obj, set):
            return list(obj)
        return str(obj)

class ProtobufSerializer:
    """Protocol Buffers serialization implementation"""
    
    def __init__(self, message_class: Optional[type] = None):
        if not PROTOBUF_AVAILABLE:
            raise SerializationError(
                "Protobuf not available. Install with: pip install protobuf"
            )
        self.message_class = message_class
    
    def serialize(self, value: Any) -> bytes:
        """Serialize using Protocol Buffers"""
        try:
            if isinstance(value, ProtobufMessage):
                return value.SerializeToString()
            elif self.message_class and isinstance(value, (dict, list)):
                # Convert dict/list to protobuf message
                message = self._dict_to_protobuf(value)
                return message.SerializeToString()
            else:
                raise SerializationError(
                    f"Unsupported value type for Protobuf serialization: {type(value)}"
                )
        except Exception as e:
            logger.error(f"Protobuf serialization error: {e}")
            raise SerializationError(f"Protobuf serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize Protocol Buffers"""
        if not self.message_class:
            raise SerializationError("Protobuf message class not specified")
        
        try:
            message = self.message_class()
            message.ParseFromString(data)
            return MessageToDict(
                message, 
                including_default_value_fields=True,
                preserving_proto_field_name=True
            )
        except Exception as e:
            logger.error(f"Protobuf deserialization error: {e}")
            raise SerializationError(f"Protobuf deserialization failed: {e}")
    
    def _dict_to_protobuf(self, data: Dict) -> ProtobufMessage:
        """Convert dictionary to protobuf message"""
        if not self.message_class:
            raise SerializationError("Protobuf message class not specified")
        
        message = self.message_class()
        ParseDict(data, message)
        return message

class AvroSerializer:
    """Apache Avro serialization implementation"""
    
    def __init__(self, schema: Optional[Any] = None):
        if not AVRO_AVAILABLE:
            raise SerializationError(
                "Avro not available. Install with: pip install avro-python3"
            )
        
        self.schema = schema
        if schema:
            self._validate_schema(schema)
    
    def serialize(self, value: Any) -> bytes:
        """Serialize using Avro"""
        try:
            if not self.schema:
                raise SerializationError("Avro schema not specified")
            
            writer = DatumWriter(self.schema)
            bytes_writer = BytesIO()
            encoder = BinaryEncoder(bytes_writer)
            writer.write(value, encoder)
            return bytes_writer.getvalue()
        except Exception as e:
            logger.error(f"Avro serialization error: {e}")
            raise SerializationError(f"Avro serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize Avro"""
        try:
            if not self.schema:
                raise SerializationError("Avro schema not specified")
            
            reader = DatumReader(self.schema)
            bytes_reader = BytesIO(data)
            decoder = BinaryDecoder(bytes_reader)
            return reader.read(decoder)
        except Exception as e:
            logger.error(f"Avro deserialization error: {e}")
            raise SerializationError(f"Avro deserialization failed: {e}")
    
    def _validate_schema(self, schema):
        """Validate Avro schema"""
        try:
            if isinstance(schema, str):
                avro.schema.parse(schema)
            elif isinstance(schema, avro.schema.Schema):
                pass  # Already valid schema object
            else:
                raise ValueError("Invalid schema type")
        except Exception as e:
            raise SerializationError(f"Invalid Avro schema: {e}")

class PickleSerializer:
    """Python pickle serialization with security considerations"""
    
    def __init__(self, protocol=None, fix_imports=True):
        self.protocol = protocol
        self.fix_imports = fix_imports
    
    def serialize(self, value: Any) -> bytes:
        """Serialize using pickle"""
        try:
            return pickle.dumps(value, protocol=self.protocol, fix_imports=self.fix_imports)
        except (pickle.PickleError, TypeError, AttributeError) as e:
            logger.error(f"Pickle serialization error: {e}")
            raise SerializationError(f"Pickle serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        """Deserialize using pickle with security restrictions"""
        try:
            return pickle.loads(data, fix_imports=self.fix_imports)
        except (pickle.PickleError, TypeError, AttributeError, ImportError) as e:
            logger.error(f"Pickle deserialization error: {e}")
            raise SerializationError(f"Pickle deserialization failed: {e}")

class SerializationFactory:
    """Factory for creating serializers"""
    
    @staticmethod
    def create_serializer(format_type: SerializationFormat, **kwargs) -> Any:
        """Create serializer instance based on format"""
        serializers = {
            SerializationFormat.JSON: JSONSerializer,
            SerializationFormat.MSGPACK: MsgPackSerializer,
            SerializationFormat.PROTOBUF: ProtobufSerializer,
            SerializationFormat.AVRO: AvroSerializer,
            SerializationFormat.PICKLE: PickleSerializer,
        }
        
        if format_type not in serializers:
            raise SerializationError(f"Unsupported serialization format: {format_type}")
        
        serializer_class = serializers[format_type]
        
        try:
            return serializer_class(**kwargs)
        except Exception as e:
            raise SerializationError(f"Failed to create {format_type.name} serializer: {e}")

# Utility functions for common serialization tasks
def serialize_to_format(value: Any, format_type: SerializationFormat, **kwargs) -> bytes:
    """Convenience function for one-off serialization"""
    serializer = SerializationFactory.create_serializer(format_type, **kwargs)
    return serializer.serialize(value)

def deserialize_from_format(data: bytes, format_type: SerializationFormat, **kwargs) -> Any:
    """Convenience function for one-off deserialization"""
    serializer = SerializationFactory.create_serializer(format_type, **kwargs)
    return serializer.deserialize(data)

def detect_serialization_format(data: bytes) -> Optional[SerializationFormat]:
    """Attempt to detect the serialization format of given data"""
    if not data:
        return None
    
    # Check for JSON (starts with { or [)
    try:
        first_char = data[:1].decode('utf-8', errors='ignore')
        if first_char in ('{', '['):
            json.loads(data.decode('utf-8'))
            return SerializationFormat.JSON
    except:
        pass
    
    # Check for MessagePack
    try:
        msgpack.unpackb(data)
        return SerializationFormat.MSGPACK
    except:
        pass
    
    # Check for Pickle (heuristic)
    try:
        if data.startswith(b'\x80'):  # Common pickle protocol marker
            return SerializationFormat.PICKLE
    except:
        pass
    
    return None