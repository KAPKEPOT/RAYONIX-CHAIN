import hashlib
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import logging

from database.utils.types import IndexConfig
from database.utils.exceptions import IndexError

logger = logging.getLogger(__name__)

class FunctionalBTreeIndex:
    """Functional B-Tree secondary index"""
    
    def __init__(self, name: str, config: IndexConfig, db):
        self.name = name
        self.config = config
        self.db = db
        self.prefix = f"_index_{name}_".encode()
        self.unique = config.unique
        self.sparse = config.sparse
    
    def calculate_update(self, key: bytes, new_value: Any, 
                        old_value: Any, ttl: Optional[int]) -> Optional[Any]:
        """Calculate index update for value change"""
        try:
            # Extract index fields from value
            index_values = self._extract_index_values(new_value)
            old_index_values = self._extract_index_values(old_value) if old_value else []
            
            if not index_values and self.sparse:
                return None
            
            return {
                'new_values': index_values,
                'old_values': old_index_values
            }
        except Exception as e:
            raise IndexError(f"Index calculation failed: {e}")
    
    def update(self, key: bytes, value: Any, update_data: Any):
        """Update index with new data"""
        try:
            new_values = update_data.get('new_values', []) or []
            old_values = update_data.get('old_values', []) or []  # Ensure it's never None
            
            # Remove old index entries
            for old_value in old_values:
                index_key = self._create_index_key(old_value, key)
                self.db.db.delete(index_key)
            
            # Add new index entries
            for new_value in new_values:
                index_key = self._create_index_key(new_value, key)
                self.db.db.put(index_key, b'')  # Value is empty, we only care about keys
        
        except Exception as e:
            raise IndexError(f"Index update failed: {e}")
    
    def remove(self, key: bytes, removal_data: Any):
        """Remove key from index"""
        try:
            old_values = removal_data.get('old_values', []) or []  # Ensure it's never None
            for old_value in old_values:
                index_key = self._create_index_key(old_value, key)
                self.db.db.delete(index_key)
        except Exception as e:
            raise IndexError(f"Index removal failed: {e}")
    
    def query(self, query: Any, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Query index for values"""
        try:
            results = []
            query_key = self._create_index_key(query)
            
            # Iterate over index entries
            it = self.db.db.iterator(prefix=query_key)
            for i, (index_key, _) in enumerate(it):
                if i < offset:
                    continue
                if len(results) >= limit:
                    break
                
                # Extract original key from index key
                original_key = self._extract_original_key(index_key)
                try:
                    value = self.db.get(original_key)
                    results.append(value)
                except Exception:
                    continue
            
            return results
        except Exception as e:
            raise IndexError(f"Index query failed: {e}")
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract index values from document"""
        # This is a placeholder - actual implementation depends on index configuration
        if hasattr(value, 'get') and callable(value.get):
            # Assume it's a dictionary-like object
            return [value.get(self.config.fields[0])] if self.config.fields else []
        return []
    
    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create index key from value and optional original key"""
        if isinstance(index_value, str):
            index_bytes = index_value.encode()
        elif isinstance(index_value, (int, float)):
            index_bytes = str(index_value).encode()
        else:
            index_bytes = str(index_value).encode()
        
        if original_key:
            return self.prefix + index_bytes + b'_' + original_key
        return self.prefix + index_bytes
    
    def _extract_original_key(self, index_key: bytes) -> bytes:
        """Extract original key from index key"""
        return index_key.split(b'_')[-1]

class FunctionalHashIndex(FunctionalBTreeIndex):
    """Functional Hash index implementation"""
    
    def __init__(self, name: str, config: IndexConfig, db):
        super().__init__(name, config, db)
        self.hash_function = hashlib.md5 if config.hash_function == 'md5' else hashlib.sha256
    
    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create hashed index key"""
        if isinstance(index_value, str):
            value_str = index_value
        else:
            value_str = str(index_value)
        
        hashed = self.hash_function(value_str.encode()).digest()
        
        if original_key:
            return self.prefix + hashed + b'_' + original_key
        return self.prefix + hashed

class FunctionalLSMIndex:
    """LSM Tree index for write-heavy workloads"""
    
    def __init__(self, name: str, config: IndexConfig, db):
        self.name = name
        self.config = config
        self.db = db
        self.memtable = {}
        self.sstables = []
        self.lock = threading.RLock()
    
    def update(self, key: bytes, value: Any, update_data: Any):
        """Update LSM index"""
        with self.lock:
            new_values = update_data.get('new_values', [])
            
            # Add to memtable
            for index_value in new_values:
                index_key = self._create_index_key(index_value, key)
                self.memtable[index_key] = b''
            
            # Check if memtable needs flushing
            if len(self.memtable) >= self.config.memtable_size:
                self._flush_memtable()
    
    def _flush_memtable(self):
        """Flush memtable to SSTable"""
        with self.lock:
            if not self.memtable:
                return
            
            # Create SSTable from memtable
            sstable = dict(self.memtable)
            self.sstables.append(sstable)
            self.memtable.clear()
            
            # Compact if too many SSTables
            if len(self.sstables) > self.config.max_sstables:
                self._compact_sstables()
    
    def _compact_sstables(self):
        """Compact SSTables"""
        # Implementation would merge multiple SSTables into one
        pass

class CompoundIndex(FunctionalBTreeIndex):
    """Compound index supporting multiple fields"""
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract compound index values"""
        if not self.config.fields:
            return []
        
        if hasattr(value, 'get') and callable(value.get):
            # Extract all field values
            field_values = []
            for field in self.config.fields:
                field_value = value.get(field)
                if field_value is None and not self.sparse:
                    return []
                field_values.append(field_value)
            
            return [tuple(field_values)]
        return []
    
    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create compound index key"""
        if isinstance(index_value, tuple):
            # Join tuple elements with separator
            value_str = '|'.join(str(v) for v in index_value)
        else:
            value_str = str(index_value)
        
        index_bytes = value_str.encode()
        
        if original_key:
            return self.prefix + index_bytes + b'_' + original_key
        return self.prefix + index_bytes