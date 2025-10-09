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
            
            # Handle different value types
            if isinstance(value, str):
                value_data = value.encode(self.encoding)
            elif isinstance(value, int):
                value_data = str(value).encode(self.encoding)
            elif isinstance(value, bytes):
                value_data = value
            else:
                # Use pickle for complex objects
                value_data = pickle.dumps(value)
            
            # Use appropriate put method based on database type
            if hasattr(self.database, 'put'):
                return self.database.put(key_bytes, value_data)
            elif isinstance(self.database, dict):  # Memory DB
                self.database[key_bytes] = value_data
                return True
            else:
                # Fallback for SQLite
                cursor = self.database.cursor()
                cursor.execute("INSERT OR REPLACE INTO data (key, value) VALUES (?, ?)", 
                             (key_bytes, value_data))
                self.database.commit()
                return True
            
        except Exception as e:
            logger.error(f"Safe put failed for key {key}: {e}")
            return False
    
    def get(self, key: Union[str, bytes], default: Any = None) -> Optional[Any]:
        """Safe get operation with automatic type handling"""
        try:
            key_bytes = self._ensure_bytes(key)
            value_data = None
            
            # Use appropriate get method based on database type
            if hasattr(self.database, 'get'):
                value_data = self.database.get(key_bytes)
            elif isinstance(self.database, dict):  # Memory DB
                value_data = self.database.get(key_bytes)
            else:
                # Fallback for SQLite
                cursor = self.database.cursor()
                cursor.execute("SELECT value FROM data WHERE key = ?", (key_bytes,))
                result = cursor.fetchone()
                value_data = result[0] if result else None
            
            if value_data is None:
                return default
            return value_data
     
        except Exception as e:
            logger.error(f"Safe get failed for key {key}: {e}")
            return default
    
    def _ensure_bytes(self, key: Union[str, bytes]) -> bytes:
        """Ensure key is bytes"""
        if isinstance(key, str):
            return key.encode(self.encoding)
        return key
    
    def delete(self, key: Union[str, bytes]) -> bool:
        """Safe delete operation"""
        try:
            key_bytes = self._ensure_bytes(key)
            
            if hasattr(self.database, 'delete'):
                return self.database.delete(key_bytes)
            elif isinstance(self.database, dict):  # Memory DB
                if key_bytes in self.database:
                    del self.database[key_bytes]
                    return True
                return False
            else:
                # Fallback for SQLite
                cursor = self.database.cursor()
                cursor.execute("DELETE FROM data WHERE key = ?", (key_bytes,))
                self.database.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Safe delete failed for key {key}: {e}")
            return False