import hashlib
import time
import struct
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
import threading
import logging
from collections import OrderedDict, defaultdict
import heapq
from bisect import bisect_left, bisect_right
import uuid

from database.utils.types import IndexConfig
from database.utils.exceptions import IndexError, DuplicateKeyError

logger = logging.getLogger(__name__)

class IndexType(Enum):
    """Supported index types"""
    BTREE = auto()
    HASH = auto()
    LSM = auto()
    COMPOUND = auto()
    UNIQUE = auto()
    SPATIAL = auto()

@dataclass
class IndexStats:
    """Index statistics and metrics"""
    name: str
    entries_count: int = 0
    memory_usage: int = 0
    query_count: int = 0
    update_count: int = 0
    last_compaction: float = 0
    sstable_count: int = 0

class IndexEntry:
    """Index entry with metadata"""
    
    __slots__ = ('key', 'value', 'timestamp', 'version')
    
    def __init__(self, key: bytes, value: Any = None, timestamp: float = None):
        self.key = key
        self.value = value
        self.timestamp = timestamp or time.time()
        self.version = 1

class FunctionalBTreeIndex: 
    def __init__(self, name: str, config: IndexConfig, db):
        self.name = name
        self.config = config
        self.db = db
        self.prefix = f"_index_{name}_".encode()
        self.unique = config.unique
        self.sparse = config.sparse
        self.stats = IndexStats(name)
        self.lock = threading.RLock()
        self.index_data = {}
        self._cache = OrderedDict()
        self._cache_size = 1000
        
        self._build_from_existing_data()
        
        # Initialize index if it doesn't exist
        #self._initialize_index()
        
        
    def _build_from_existing_data(self):
        """Build index from existing blockchain data"""
        try:
            # Scan all data in the blockchain and index it
            for key, value in self.db.iterate():
                # Skip system keys and index metadata
                if key.startswith(b'_index_') or key.startswith(b'_system_'):
                    continue
                    
                # Extract index values and update index
                index_values = self._extract_index_values(value)
                for index_value in index_values:
                    index_key = self._create_index_key(index_value)
                    if index_key not in self.index_data:
                        self.index_data[index_key] = []
                    if key not in self.index_data[index_key]:
                        self.index_data[index_key].append(key)
                        
            self.stats.entries_count = len(self.index_data)
            logger.info(f"Index {self.name} built from {len(self.index_data)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to build index {self.name} from existing data: {e}")
    
    #def _initialize_index(self):
        #"""Initialize index metadata"""
        #try:
           # metadata_key = f"_index_meta_{self.name}".encode()
   #         existing_meta = self.db.db.get(metadata_key)
           # if not existing_meta:
              #  metadata = {
                #    'created_at': time.time(),
              #      'config': self.config.__dict__,
                 #   'version': '1.0'
              #  }
                
                #if hasattr(self.db.db, 'put'):
                	#self.db.db.put(metadata_key, pickle.dumps(metadata))
               # else:
                	# Memory database (dict)
                	#self.db.db[metadata_key] = pickle.dumps(metadata)
                	
     #   except Exception as e:
     #       logger.warning(f"Failed to initialize index metadata: {e}")
    
    def calculate_update(self, key: bytes, new_value: Any, 
                        old_value: Any, ttl: Optional[int]) -> Optional[Any]:
        """Calculate index update for value change with validation"""
        try:
            with self.lock:
                new_index_values = self._extract_index_values(new_value)
                old_index_values = self._extract_index_values(old_value) if old_value else []
                
                # Handle sparse indexes
                if not new_index_values and self.sparse:
                    return None
                
                # Validate uniqueness
                if self.unique and new_index_values:
                    for index_value in new_index_values:
                        if self._check_unique_violation(index_value, key):
                            raise DuplicateKeyError(
                                f"Unique constraint violated for index {self.name} "
                                f"with value {index_value}"
                            )
                
                return {
                    'new_values': new_index_values,
                    'old_values': old_index_values,
                    'key': key,
                    'timestamp': time.time()
                }
        except DuplicateKeyError:
            raise
        except Exception as e:
            raise IndexError(f"Index calculation failed: {e}")
    
    def update(self, key: bytes, value: Any, update_data: Any):
        """Update in-memory index from blockchain data"""
        try:
            with self.lock:
                new_values = update_data.get('new_values', [])
                old_values = update_data.get('old_values', [])
                
                # Remove old index entries from memory
                for old_value in old_values:
                    index_key = self._create_index_key(old_value)
                    if index_key in self.index_data and key in self.index_data[index_key]:
                        self.index_data[index_key].remove(key)
                        # Clean up empty lists
                        if not self.index_data[index_key]:
                            del self.index_data[index_key]
                        self.stats.entries_count -= 1
                
                # Add new index entries to memory
                for new_value in new_values:
                    index_key = self._create_index_key(new_value)
                    if index_key not in self.index_data:
                        self.index_data[index_key] = []
                    if key not in self.index_data[index_key]:
                        self.index_data[index_key].append(key)
                        self.stats.entries_count += 1
                
                self.stats.update_count += 1
                
        except Exception as e:
            raise IndexError(f"Index update failed: {e}")
    
    def remove(self, key: bytes, removal_data: Any):
        """Remove key from in-memory index """
        try:
            with self.lock:
                old_values = removal_data.get('old_values', [])
                for old_value in old_values:
                    index_key = self._create_index_key(old_value)
                    if index_key in self.index_data and key in self.index_data[index_key]:
                        self.index_data[index_key].remove(key)
                        # Clean up empty lists
                        if not self.index_data[index_key]:
                            del self.index_data[index_key]
                        self.stats.entries_count -= 1
                
                self.stats.update_count += 1
                
        except Exception as e:
            raise IndexError(f"Index removal failed: {e}")
    
    def query(self, query: Any, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Query using in-memory index, fetch from blockchain storage - BLOCKCHAIN FIX"""
        try:
            results = []
            
            # Handle range queries
            if isinstance(query, dict) and 'range' in query:
                range_query = query['range']
                start = range_query.get('start')
                end = range_query.get('end')
                results = self._range_query(start, end, limit, offset)
            else:
                # Exact match query using in-memory index
                query_key = self._create_index_key(query)
                
                if query_key in self.index_data:
                    doc_keys = self.index_data[query_key][offset:offset + limit]
                    
                    # Fetch actual documents from blockchain storage
                    for doc_key in doc_keys:
                        try:
                            value = self.db.get(doc_key, use_cache=False)
                            if value is not None:
                                results.append(value)
                        except Exception as e:
                            logger.warning(f"Failed to fetch indexed document {doc_key}: {e}")
                            continue
            
            self.stats.query_count += 1
            return results
            
        except Exception as e:
            raise IndexError(f"Index query failed: {e}")
    
    def _range_query(self, start: Any, end: Any, limit: int, offset: int) -> List[Any]:
        """Execute range query using in-memory index"""
        results = []
        count = 0
        
        # Get all index keys in sorted order
        sorted_keys = sorted(self.index_data.keys())
        
        # Find start and end positions
        start_key = self._create_index_key(start) if start is not None else None
        end_key = self._create_index_key(end) if end is not None else None
        
        for index_key in sorted_keys:
            # Check range bounds
            if start_key and index_key < start_key:
                continue
            if end_key and index_key > end_key:
                break
                
            # Process documents for this index key
            doc_keys = self.index_data[index_key]
            for doc_key in doc_keys:
                if count < offset:
                    count += 1
                    continue
                    
                if len(results) >= limit:
                    break
                    
                try:
                    value = self.db.get(doc_key, use_cache=False)
                    if value is not None:
                        results.append(value)
                except Exception as e:
                    logger.warning(f"Failed to fetch ranged document {doc_key}: {e}")
                    continue
        
        return results
    
    def _check_unique_violation(self, index_value: Any, exclude_key: bytes) -> bool:
        """Check if unique constraint would be violated"""
        query_key = self._create_index_key(index_value)
        it = self.db.db.iterator(prefix=query_key)
        
        for index_key, _ in it:
            existing_key = self._extract_original_key(index_key)
            if existing_key != exclude_key:
                return True
        return False
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract index values from document with field validation"""
        if not value:
            return []
            
        if hasattr(value, 'get') and callable(value.get):
            if not self.config.fields:
                return []
            
            field_values = []
            for field in self.config.fields:
                field_value = value.get(field)
                if field_value is None:
                    if not self.sparse:
                        return []  # Skip if sparse and field is missing
                    else:
                        field_value = "__NULL__"
            field_values.append(field_value)
                
            return field_values if not self.config.compound else [tuple(field_values)]
        
        # Handle non-dict objects
        elif hasattr(value, '__dict__'):
            if not self.config.fields:
            	return []
            	
            field_values = []
            for field in self.config.fields:
                field_value = getattr(value, field, None)
                if field_value is None:
                    if not self.sparse:
                    	return []
                    else:
                    	field_value = "__NULL__"
                field_values.append(field_value)
                
            return field_values if not self.config.compound else [tuple(field_values)]
            
        # Handle case where value is not a dict/object but we need to index it
        elif self.config.fields and len(self.config.fields) == 1:
        	return [value]
        return []


    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create index key with proper encoding"""
        # Convert value to bytes with type prefix for proper ordering
        if index_value is None:
            value_bytes = b'\x00'
        elif isinstance(index_value, str):
            value_bytes = b'\x01' + index_value.encode('utf-8')
        elif isinstance(index_value, int):
            value_bytes = b'\x02' + struct.pack('>q', index_value)
        elif isinstance(index_value, float):
            value_bytes = b'\x03' + struct.pack('>d', index_value)
        elif isinstance(index_value, bool):
            value_bytes = b'\x04' + (b'\x01' if index_value else b'\x00')
        else:
            value_bytes = b'\x05' + str(index_value).encode('utf-8')
        
        if original_key:
            return self.prefix + value_bytes + b'|' + original_key
        return self.prefix + value_bytes
    
    def _extract_original_key(self, index_key: bytes) -> bytes:
        """Extract original key from index key"""
        parts = index_key.split(b'|')
        return parts[-1] if len(parts) > 1 else b''
    
    def _serialize_entry(self, value: Any) -> bytes:
        """Serialize index entry with metadata"""
        entry = {
            'value': value,
            'indexed_at': time.time(),
            'version': 1
        }
        return pickle.dumps(entry)
    
    def _deserialize_entry(self, data: bytes) -> Any:
        """Deserialize index entry"""
        entry = pickle.loads(data)
        return entry.get('value')
    
    def _cache_put(self, key: bytes, value: bytes):
        """Add to cache"""
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = value
    
    def _cache_pop(self, key: bytes):
        """Remove from cache"""
        self._cache.pop(key, None)
    
    def get_stats(self) -> IndexStats:
        """Get index statistics"""
        return self.stats
    
    def compact(self):
        """Compact index if needed"""
        # B-Tree indexes typically don't need compaction
        pass

class FunctionalHashIndex(FunctionalBTreeIndex):
    """Functional Hash index implementation with collision handling"""
    
    def __init__(self, name: str, config: IndexConfig, db):
        super().__init__(name, config, db)
        self.hash_function = getattr(hashlib, config.hash_function, hashlib.md5)
        self.collision_map = defaultdict(list)
    
    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create hashed index key with collision detection"""
        if isinstance(index_value, str):
            value_str = index_value
        else:
            value_str = str(index_value)
        
        hashed = self.hash_function(value_str.encode()).digest()
        
        if original_key:
            # Store mapping for collision resolution
            collision_key = self.prefix + hashed
            self.collision_map[collision_key].append((index_value, original_key))
            return collision_key + b'|' + original_key
        
        return self.prefix + hashed
    
    def query(self, query: Any, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Query hash index with collision handling"""
        try:
            results = []
            query_key = self._create_index_key(query)
            
            # Handle exact matches first
            it = self.db.db.iterator(prefix=query_key)
            for i, (index_key, value_data) in enumerate(it):
                if i < offset:
                    continue
                if len(results) >= limit:
                    break
                
                # Verify this is the exact match (handle collisions)
                original_key = self._extract_original_key(index_key)
                if self._verify_exact_match(query, index_key, original_key):
                    try:
                        value = self._deserialize_entry(value_data)
                        results.append(value)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize hash index entry: {e}")
                        continue
            
            self.stats.query_count += 1
            return results
            
        except Exception as e:
            raise IndexError(f"Hash index query failed: {e}")
    
    def _verify_exact_match(self, query: Any, index_key: bytes, original_key: bytes) -> bool:
        """Verify exact match for hash index (handle collisions)"""
        collision_key = index_key.split(b'|')[0]
        mappings = self.collision_map.get(collision_key, [])
        
        for stored_value, stored_key in mappings:
            if stored_key == original_key and stored_value == query:
                return True
        return False

class FunctionalLSMIndex:
    """LSM Tree index for write-heavy workloads with compaction"""
    
    def __init__(self, name: str, config: IndexConfig, db):
        self.name = name
        self.config = config
        self.db = db
        self.memtable = OrderedDict()
        self.sstables = []
        self.lock = threading.RLock()
        self.stats = IndexStats(name)
        self.prefix = f"_index_lsm_{name}_".encode()
        
        # LSM configuration
        self.memtable_size = config.memtable_size or 10000
        self.max_sstables = config.max_sstables or 10
        self.compaction_threshold = config.compaction_threshold or 5
        
        # Background compaction thread
        self._compaction_thread = None
        self._stop_compaction = threading.Event()
        
        self._start_background_compaction()
    
    def calculate_update(self, key: bytes, new_value: Any, 
                        old_value: Any, ttl: Optional[int]) -> Optional[Any]:
        """Calculate LSM index update"""
        try:
            with self.lock:
                new_index_values = self._extract_index_values(new_value)
                old_index_values = self._extract_index_values(old_value) if old_value else []
                
                if not new_index_values and self.config.sparse:
                    return None
                
                return {
                    'new_values': new_index_values,
                    'old_values': old_index_values,
                    'key': key,
                    'timestamp': time.time()
                }
        except Exception as e:
            raise IndexError(f"LSM index calculation failed: {e}")
    
    def update(self, key: bytes, value: Any, update_data: Any):
        """Update LSM index"""
        try:
            with self.lock:
                new_values = update_data.get('new_values', [])
                old_values = update_data.get('old_values', [])
                
                # Remove old entries from memtable
                for old_value in old_values:
                    index_key = self._create_index_key(old_value, key)
                    self.memtable.pop(index_key, None)
                
                # Add new entries to memtable
                for new_value in new_values:
                    index_key = self._create_index_key(new_value, key)
                    entry_data = self._serialize_entry(value)
                    self.memtable[index_key] = entry_data
                
                self.stats.update_count += 1
                self.stats.memory_usage = len(self.memtable) * 100  # Approximate
                
                # Check if memtable needs flushing
                if len(self.memtable) >= self.memtable_size:
                    self._flush_memtable()
                    
        except Exception as e:
            raise IndexError(f"LSM index update failed: {e}")
    
    def _flush_memtable(self):
        """Flush memtable to SSTable"""
        with self.lock:
            if not self.memtable:
                return
            
            # Create sorted SSTable
            sstable = sorted(self.memtable.items(), key=lambda x: x[0])
            self.sstables.append(sstable)
            self.memtable.clear()
            
            self.stats.sstable_count = len(self.sstables)
            
            # Trigger compaction if needed
            if len(self.sstables) >= self.compaction_threshold:
                self._trigger_compaction()
    
    def _trigger_compaction(self):
        """Trigger background compaction"""
        if not self._stop_compaction.is_set():
            self._stop_compaction.set()
        
        if self._compaction_thread and self._compaction_thread.is_alive():
            self._compaction_thread.join(timeout=1.0)
        
        self._stop_compaction.clear()
        self._compaction_thread = threading.Thread(target=self._compact_sstables)
        self._compaction_thread.daemon = True
        self._compaction_thread.start()
    
    def _compact_sstables(self):
        """Compact SSTables in background"""
        try:
            with self.lock:
                if len(self.sstables) <= 1:
                    return
                
                # Merge sort all SSTables
                merged = self._merge_sstables(self.sstables)
                
                # Keep only the latest version of each key
                compacted = self._remove_duplicates(merged)
                
                # Replace old SSTables with compacted one
                self.sstables = [compacted]
                self.stats.sstable_count = 1
                self.stats.last_compaction = time.time()
                
                logger.info(f"Compacted {self.name} index: {len(compacted)} entries")
                
        except Exception as e:
            logger.error(f"Compaction failed for index {self.name}: {e}")
    
    def _merge_sstables(self, sstables: List[List[Tuple[bytes, bytes]]]) -> List[Tuple[bytes, bytes]]:
        """Merge multiple SSTables using heap merge"""
        return list(heapq.merge(*sstables, key=lambda x: x[0]))
    
    def _remove_duplicates(self, entries: List[Tuple[bytes, bytes]]) -> List[Tuple[bytes, bytes]]:
        """Remove duplicate keys, keeping the latest"""
        seen = {}
        for key, value in entries:
            seen[key] = value  # Later entries overwrite earlier ones
        return list(seen.items())
    
    def query(self, query: Any, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Query LSM index"""
        try:
            results = []
            query_key = self._create_index_key(query)
            
            # Search in memtable first
            memtable_results = self._search_memtable(query_key, limit, offset)
            results.extend(memtable_results)
            
            # Search in SSTables if needed
            if len(results) < limit:
                sstable_results = self._search_sstables(query_key, limit - len(results), offset)
                results.extend(sstable_results)
            
            self.stats.query_count += 1
            return results
            
        except Exception as e:
            raise IndexError(f"LSM index query failed: {e}")
    
    def _search_memtable(self, query_key: bytes, limit: int, offset: int) -> List[Any]:
        """Search memtable for keys"""
        results = []
        count = 0
        
        for key, value_data in self.memtable.items():
            if key.startswith(query_key):
                if count < offset:
                    count += 1
                    continue
                
                if len(results) >= limit:
                    break
                
                try:
                    value = self._deserialize_entry(value_data)
                    results.append(value)
                except Exception as e:
                    logger.warning(f"Failed to deserialize memtable entry: {e}")
        
        return results
    
    def _search_sstables(self, query_key: bytes, limit: int, offset: int) -> List[Any]:
        """Search SSTables for keys using binary search"""
        results = []
        count = 0
        
        for sstable in reversed(self.sstables):  # Search newest first
            # Binary search for the prefix
            start_idx = bisect_left(sstable, (query_key, b''))
            end_idx = bisect_right(sstable, (query_key + b'\xff', b''))
            
            for i in range(start_idx, min(end_idx, len(sstable))):
                if count < offset:
                    count += 1
                    continue
                
                if len(results) >= limit:
                    break
                
                key, value_data = sstable[i]
                try:
                    value = self._deserialize_entry(value_data)
                    results.append(value)
                except Exception as e:
                    logger.warning(f"Failed to deserialize SSTable entry: {e}")
        
        return results
    
    def _start_background_compaction(self):
        """Start background compaction thread"""
        self._compaction_thread = threading.Thread(target=self._background_compaction)
        self._compaction_thread.daemon = True
        self._compaction_thread.start()
    
    def _background_compaction(self):
        """Background compaction loop"""
        while not self._stop_compaction.is_set():
            time.sleep(300)  # Check every 5 minutes
            if len(self.sstables) >= self.compaction_threshold:
                self._compact_sstables()
    
    def close(self):
        """Cleanup LSM index"""
        self._stop_compaction.set()
        if self._compaction_thread:
            self._compaction_thread.join(timeout=5.0)
        self._flush_memtable()

class CompoundIndex(FunctionalBTreeIndex):
    """Compound index supporting multiple fields with proper ordering"""
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract compound index values with proper field ordering"""
        if not value or not self.config.fields:
            return []
        
        field_values = []
        for field in self.config.fields:
            if hasattr(value, 'get') and callable(value.get):
                field_value = value.get(field)
            elif hasattr(value, '__dict__'):
                field_value = getattr(value, field, None)
            else:
                field_value = None
            
            if field_value is None:
                if not self.config.sparse:
                    return []
                else:
                    # Use a sentinel value for sparse indexes
                    field_value = "__NULL__"
            
            field_values.append(field_value)
        
        return [tuple(field_values)]
    
    def _create_index_key(self, index_value: Any, original_key: Optional[bytes] = None) -> bytes:
        """Create compound index key with field separators"""
        if isinstance(index_value, tuple):
            # Encode each field with type prefix
            encoded_parts = []
            for field_value in index_value:
                if field_value == "__NULL__":
                    encoded_parts.append(b'\x00')
                elif isinstance(field_value, str):
                    encoded_parts.append(b'\x01' + field_value.encode('utf-8'))
                elif isinstance(field_value, int):
                    encoded_parts.append(b'\x02' + struct.pack('>q', field_value))
                elif isinstance(field_value, float):
                    encoded_parts.append(b'\x03' + struct.pack('>d', field_value))
                elif isinstance(field_value, bool):
                    encoded_parts.append(b'\x04' + (b'\x01' if field_value else b'\x00'))
                else:
                    encoded_parts.append(b'\x05' + str(field_value).encode('utf-8'))
            
            compound_key = b'|'.join(encoded_parts)
        else:
            compound_key = self._create_index_key((index_value,))
        
        if original_key:
            return self.prefix + compound_key + b'||' + original_key
        return self.prefix + compound_key

class IndexManager:
    """Manager for multiple indexes with lifecycle management"""
    
    def __init__(self, db):
        self.db = db
        self.indexes: Dict[str, Any] = {}
        self.lock = threading.RLock()
    
    def create_index(self, name: str, config: IndexConfig) -> Any:
        """Create a new index"""
        with self.lock:
            if name in self.indexes:
                raise IndexError(f"Index already exists: {name}")
            
            index_class = self._get_index_class(config.type)
            index = index_class(name, config, self.db)
            self.indexes[name] = index
            
            logger.info(f"Created index: {name} (type: {config.type})")
            return index
    
    def drop_index(self, name: str):
        """Drop an index"""
        with self.lock:
            if name not in self.indexes:
                raise IndexError(f"Index not found: {name}")
            
            index = self.indexes.pop(name)
            if hasattr(index, 'close'):
                index.close()
            
            # Clean up index data
            self._cleanup_index_data(name)
            
            logger.info(f"Dropped index: {name}")
    
    def get_index(self, name: str) -> Any:
        """Get index by name"""
        with self.lock:
            if name not in self.indexes:
                raise IndexError(f"Index not found: {name}")
            return self.indexes[name]
    
    def _get_index_class(self, index_type: str) -> type:
        """Get index class by type"""
        index_classes = {
            'btree': FunctionalBTreeIndex,
            'hash': FunctionalHashIndex,
            'lsm': FunctionalLSMIndex,
            'compound': CompoundIndex,
        }
        
        if index_type not in index_classes:
            raise IndexError(f"Unsupported index type: {index_type}")
        
        return index_classes[index_type]
    
    def _cleanup_index_data(self, name: str):
        """Clean up index data from database"""
        try:
            prefix = f"_index_{name}_".encode()
            it = self.db.db.iterator(prefix=prefix)
            
            with self.db.db.write_batch() as batch:
                for key, _ in it:
                    batch.delete(key)
            
            # Delete metadata
            metadata_key = f"_index_meta_{name}".encode()
            self.db.db.delete(metadata_key)
            
        except Exception as e:
            logger.warning(f"Failed to cleanup index data for {name}: {e}")
    
    def get_all_stats(self) -> Dict[str, IndexStats]:
        """Get statistics for all indexes"""
        with self.lock:
            return {name: index.get_stats() for name, index in self.indexes.items()}
    
    def close(self):
        """Close all indexes"""
        with self.lock:
            for name, index in self.indexes.items():
                if hasattr(index, 'close'):
                    try:
                        index.close()
                    except Exception as e:
                        logger.error(f"Error closing index {name}: {e}")
            self.indexes.clear()