"""
Safe database wrapper with consistent type handling
"""
import logging
from typing import Any, Optional, Union
import pickle

logger = logging.getLogger(__name__)

class SafeDatabaseWrapper:
    """Wrapper for database operations with consistent type handling"""
    
    def __init__(self, database: Any):
        self.database = database
        self.encoding = 'utf-8'
    
    def put(self, key: Union[str, bytes], value: Any) -> bool:
        """Safe put operation with automatic key encoding"""
        try:
            key_bytes = self._ensure_bytes(key)
            
            # Serialize value
            if isinstance(value, (str, int, float, bool, bytes)):
                # Use native serialization for basic types
                value_data = str(value).encode(self.encoding)
            else:
                # Use pickle for complex objects
                value_data = pickle.dumps(value)
            
            return self.database.put(key_bytes, value_data)
            
        except Exception as e:
            logger.error(f"Safe put failed for key {key}: {e}")
            return False
    
    def get(self, key: Union[str, bytes]) -> Optional[Any]:
        """Safe get operation with automatic type handling"""
        try:
            key_bytes = self._ensure_bytes(key)
            value_data = self.database.get(key_bytes)
            
            if value_data is None:
                return None
            
            # Try to deserialize based on content
            try:
                # First try to decode as string
                return value_data.decode(self.encoding)
            except UnicodeDecodeError:
                # Then try pickle
                return pickle.loads(value_data)
                
        except Exception as e:
            logger.error(f"Safe get failed for key {key}: {e}")
            return None
    
    def _ensure_bytes(self, key: Union[str, bytes]) -> bytes:
        """Ensure key is bytes"""
        if isinstance(key, str):
            return key.encode(self.encoding)
        return key
    
    def delete(self, key: Union[str, bytes]) -> bool:
        """Safe delete operation"""
        try:
            key_bytes = self._ensure_bytes(key)
            return self.database.delete(key_bytes)
        except Exception as e:
            logger.error(f"Safe delete failed for key {key}: {e}")
            return False