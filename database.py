# database.py
import plyvel
import json
import zlib
import lz4.frame
import snappy
import msgpack
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import threading
import time
import hashlib
import struct
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import mmap
import numpy as np
from bloom_filter import BloomFilter
import crc32c
import logging
from datetime import datetime, date
import uuid
import decimal
from collections import defaultdict, OrderedDict
from functools import lru_cache
import asyncio
from concurrent.futures import ProcessPoolExecutor
import tempfile
import shutil
from sortedcontainers import SortedDict, SortedList

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

class DatabaseType(Enum):
    PLYVEL = auto()
    MEMORY = auto()
    ROCKSDB = auto()
    SQLITE = auto()

class CompressionType(Enum):
    NONE = auto()
    ZLIB = auto()
    LZ4 = auto()
    SNAPPY = auto()
    ZSTD = auto()

class EncryptionType(Enum):
    NONE = auto()
    AES256 = auto()
    CHACHA20 = auto()
    FERNET = auto()

class IndexType(Enum):
    BTREE = auto()
    HASH = auto()
    BLOOM = auto()
    LSM = auto()
    COMPOUND = auto()

class SerializationType(Enum):
    JSON = auto()
    MSGPACK = auto()
    PROTOBUF = auto()
    AVRO = auto()

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: DatabaseType = DatabaseType.PLYVEL
    compression: CompressionType = CompressionType.SNAPPY
    encryption: EncryptionType = EncryptionType.FERNET
    serialization: SerializationType = SerializationType.MSGPACK
    create_if_missing: bool = True
    error_if_exists: bool = False
    paranoid_checks: bool = False
    write_buffer_size: int = 64 * 1024 * 1024
    max_open_files: int = 1000
    block_size: int = 4096
    cache_size: int = 128 * 1024 * 1024
    bloom_filter_bits: int = 10
    compression_level: int = 6
    encryption_key: Optional[str] = None
    read_only: bool = False
    max_cache_size: int = 10000
    cache_ttl: int = 300
    background_interval: int = 60
    snapshot_interval: int = 3600
    auto_compact: bool = True
    compact_threshold: float = 0.7

@dataclass
class IndexConfig:
    """Index configuration"""
    index_type: IndexType = IndexType.BTREE
    unique: bool = False
    sparse: bool = False
    bloom_filter_size: int = 1000000
    bloom_filter_error_rate: float = 0.01
    fields: List[str] = field(default_factory=list)
    prefix: str = "idx_"

@dataclass
class BatchOperation:
    """Batch operation"""
    op_type: str  # 'put', 'delete', 'merge'
    key: bytes
    value: Optional[bytes] = None
    ttl: Optional[int] = None
    index_updates: Optional[Dict[str, Any]] = None

# Custom Exception Hierarchy
class DatabaseError(Exception):
    """Base database exception"""
    pass

class KeyNotFoundError(DatabaseError):
    """Raised when a key is not found"""
    pass

class SerializationError(DatabaseError):
    """Raised when serialization/deserialization fails"""
    pass

class IndexError(DatabaseError):
    """Base index error"""
    pass

class IndexCorruptionError(IndexError):
    """Raised when index corruption is detected"""
    pass

class TransactionError(DatabaseError):
    """Raised when transaction operations fail"""
    pass

class CompressionError(DatabaseError):
    """Raised when compression/decompression fails"""
    pass

class EncryptionError(DatabaseError):
    """Raised when encryption/decryption fails"""
    pass

class ConcurrencyError(DatabaseError):
    """Raised when concurrency issues occur"""
    pass

class IntegrityError(DatabaseError):
    """Raised when data integrity is compromised"""
    pass

class AdvancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for complex Python objects"""
    
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() 
                   if not k.startswith('_') and not callable(v)}
        elif hasattr(obj, '__slots__'):
            return {slot: getattr(obj, slot) for slot in obj.__slots__ 
                   if hasattr(obj, slot)}
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, bytes):
            return obj.hex()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class AdvancedDatabase:
    """Production-ready database layer with advanced features"""
    
    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        self.db_path = db_path
        self.config = config or DatabaseConfig()
        self.db = None
        self.cache = OrderedDict()
        self.indexes: Dict[str, Any] = {}
        self.index_configs: Dict[str, IndexConfig] = {}
        self.locks = {
            'db': threading.RLock(),
            'cache': threading.RLock(),
            'indexes': threading.RLock(),
            'stats': threading.RLock()
        }
        self.stats = DatabaseStats()
        self.encryption = None
        self.compression = None
        self.serializer = None
        self.running = True
        
        self._initialize_database()
        self._initialize_encryption()
        self._initialize_compression()
        self._initialize_serializer()
        
        # Background tasks
        self._start_background_tasks()
    
    def _initialize_database(self):
        try:
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            if self.config.db_type == DatabaseType.PLYVEL:
                self.db = plyvel.DB(
                    self.db_path,
                    create_if_missing=self.config.create_if_missing,
                    error_if_exists=self.config.error_if_exists,
                    paranoid_checks=self.config.paranoid_checks,
                    write_buffer_size=self.config.write_buffer_size,
                    max_open_files=self.config.max_open_files,
                    block_size=self.config.block_size,
                    lru_cache_size=self.config.cache_size
                )
                logger.info(f"Plyvel database initialized at {self.db_path}")
            elif self.config.db_type == DatabaseType.MEMORY:
                self.db = {}
            else:
                raise DatabaseError(f"Unsupported database type: {self.config.db_type}")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
        
        # Create default indexes
        self._create_default_indexes()
    
    def _initialize_encryption(self):
        """Initialize encryption system"""
        try:
            if self.config.encryption == EncryptionType.FERNET:
                key = self._derive_encryption_key()
                self.encryption = Fernet(key)
            elif self.config.encryption == EncryptionType.AES256:
                self.encryption = AES256Encryption(self.config.encryption_key)
            elif self.config.encryption == EncryptionType.CHACHA20:
                self.encryption = ChaCha20Encryption(self.config.encryption_key)
            elif self.config.encryption == EncryptionType.NONE:
                self.encryption = None
        except Exception as e:
            raise EncryptionError(f"Encryption initialization failed: {e}")
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from config or generate new"""
        try:
            if self.config.encryption_key:
                salt = b'rayonix_db_salt_'
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=salt,
                    iterations=100000,
                )
                return kdf.derive(self.config.encryption_key.encode())
            else:
                return Fernet.generate_key()
        except Exception as e:
            raise EncryptionError(f"Key derivation failed: {e}")
    
    def _initialize_compression(self):
        """Initialize compression system"""
        try:
            if self.config.compression == CompressionType.ZLIB:
                self.compression = ZlibCompression(self.config.compression_level)
            elif self.config.compression == CompressionType.LZ4:
                self.compression = LZ4Compression()
            elif self.config.compression == CompressionType.SNAPPY:
                self.compression = SnappyCompression()
            elif self.config.compression == CompressionType.ZSTD:
                self.compression = ZstdCompression(self.config.compression_level)
            elif self.config.compression == CompressionType.NONE:
                self.compression = None
        except Exception as e:
            raise CompressionError(f"Compression initialization failed: {e}")
    
    def _initialize_serializer(self):
        """Initialize serialization system"""
        try:
            if self.config.serialization == SerializationType.JSON:
                self.serializer = JSONSerializer()
            elif self.config.serialization == SerializationType.MSGPACK:
                self.serializer = MsgPackSerializer()
            elif self.config.serialization == SerializationType.PROTOBUF:
                self.serializer = ProtobufSerializer()
            elif self.config.serialization == SerializationType.AVRO:
                self.serializer = AvroSerializer()
        except Exception as e:
            raise SerializationError(f"Serializer initialization failed: {e}")
    
    def _create_default_indexes(self):
        """Create default indexes"""
        # Primary key index
        self.create_index("primary", IndexConfig(IndexType.BTREE, unique=True))
        
        # Timestamp index for TTL
        self.create_index("timestamp", IndexConfig(IndexType.BTREE))
        
        # Bloom filter for existence checks
        self.create_index("bloom", IndexConfig(IndexType.BLOOM))
    
    def create_index(self, index_name: str, config: IndexConfig):
        """Create a new functional index"""
        with self.locks['indexes']:
            try:
                if config.index_type == IndexType.BTREE:
                    self.indexes[index_name] = FunctionalBTreeIndex(index_name, config, self)
                elif config.index_type == IndexType.HASH:
                    self.indexes[index_name] = FunctionalHashIndex(index_name, config, self)
                elif config.index_type == IndexType.BLOOM:
                    self.indexes[index_name] = BloomFilter(
                        config.bloom_filter_size, 
                        config.bloom_filter_error_rate
                    )
                elif config.index_type == IndexType.LSM:
                    self.indexes[index_name] = FunctionalLSMIndex(index_name, config, self)
                elif config.index_type == IndexType.COMPOUND:
                    self.indexes[index_name] = CompoundIndex(index_name, config, self)
                
                self.index_configs[index_name] = config
                logger.info(f"Created index: {index_name} with type {config.index_type}")
                
            except Exception as e:
                raise IndexError(f"Failed to create index {index_name}: {e}")
    
    def put(self, key: Union[str, bytes], value: Any, ttl: Optional[int] = None, 
            use_cache: bool = True, update_indexes: bool = True) -> bool:
        """
        Store key-value pair with advanced features
        """
        key_bytes = self._ensure_bytes(key)
        
        with self.locks['db']:
            try:
                # Serialize value
                serialized_value = self._serialize_value(value)
                if serialized_value is None:
                    raise SerializationError("Serialization returned None")
                
                # Prepare value with metadata
                prepared_value = self._prepare_value_for_storage(serialized_value, ttl)
                
                # Calculate index updates if needed
                index_updates = {}
                if update_indexes:
                    index_updates = self._calculate_index_updates(key_bytes, value, None, ttl)
                
                # Store in database
                if self.config.db_type == DatabaseType.MEMORY:
                    self.db[key_bytes] = prepared_value
                else:
                    self.db.put(key_bytes, prepared_value, sync=False)
                
                # Update cache
                if use_cache:
                    with self.locks['cache']:
                        self.cache[key_bytes] = (value, time.time())
                        if len(self.cache) > self.config.max_cache_size:
                            self.cache.popitem(last=False)
                
                # Update indexes
                if update_indexes:
                    self._update_indexes(key_bytes, value, index_updates)
                
                # Update statistics
                with self.locks['stats']:
                    self.stats.put_operations += 1
                    self.stats.bytes_written += len(prepared_value)
                
                return True
                
            except Exception as e:
                with self.locks['stats']:
                    self.stats.put_errors += 1
                logger.error(f"Put operation failed for key {key_bytes}: {e}")
                raise DatabaseError(f"Put operation failed: {e}")
    
    def get(self, key: Union[str, bytes], use_cache: bool = True, 
            check_ttl: bool = True) -> Any:
        """
        Retrieve value by key
        """
        key_bytes = self._ensure_bytes(key)
        
        # Check cache first
        if use_cache:
            with self.locks['cache']:
                if key_bytes in self.cache:
                    value, timestamp = self.cache[key_bytes]
                    if time.time() - timestamp < self.config.cache_ttl:
                        with self.locks['stats']:
                            self.stats.cache_hits += 1
                        return value
                    else:
                        del self.cache[key_bytes]
        
        with self.locks['db']:
            try:
                # Retrieve from database
                if self.config.db_type == DatabaseType.MEMORY:
                    prepared_value = self.db.get(key_bytes)
                else:
                    prepared_value = self.db.get(key_bytes)
                
                if prepared_value is None:
                    with self.locks['stats']:
                        self.stats.misses += 1
                    raise KeyNotFoundError(f"Key not found: {key_bytes}")
                
                # Extract value and metadata
                value, metadata = self._extract_value_from_storage(prepared_value)
                
                # Check TTL if enabled
                if check_ttl and self._is_expired(metadata):
                    self.delete(key_bytes, update_indexes=True)
                    raise KeyNotFoundError(f"Key expired: {key_bytes}")
                
                # Verify checksum
                if not self._verify_checksum(value, metadata.get('checksum')):
                    raise IntegrityError(f"Checksum verification failed for key: {key_bytes}")
                
                # Update cache
                if use_cache:
                    with self.locks['cache']:
                        self.cache[key_bytes] = (value, time.time())
                        if len(self.cache) > self.config.max_cache_size:
                            self.cache.popitem(last=False)
                
                # Update statistics
                with self.locks['stats']:
                    self.stats.get_operations += 1
                    self.stats.bytes_read += len(prepared_value)
                
                return value
                
            except KeyNotFoundError:
                raise
            except Exception as e:
                with self.locks['stats']:
                    self.stats.get_errors += 1
                logger.error(f"Get operation failed for key {key_bytes}: {e}")
                raise DatabaseError(f"Get operation failed: {e}")
    
    def delete(self, key: Union[str, bytes], use_cache: bool = True, 
               update_indexes: bool = True) -> bool:
        """
        Delete key-value pair
        """
        key_bytes = self._ensure_bytes(key)
        
        with self.locks['db']:
            try:
                # Get current value for index updates
                current_value = None
                if update_indexes:
                    try:
                        current_value = self.get(key_bytes, use_cache=False, check_ttl=False)
                    except KeyNotFoundError:
                        current_value = None
                
                # Calculate index removals
                index_removals = {}
                if update_indexes and current_value:
                    index_removals = self._calculate_index_updates(key_bytes, None, current_value, None)
                
                # Delete from database
                if self.config.db_type == DatabaseType.MEMORY:
                    if key_bytes in self.db:
                        del self.db[key_bytes]
                    else:
                        return False
                else:
                    self.db.delete(key_bytes)
                
                # Update cache
                if use_cache:
                    with self.locks['cache']:
                        if key_bytes in self.cache:
                            del self.cache[key_bytes]
                
                # Update indexes (remove entries)
                if update_indexes and current_value:
                    self._remove_from_indexes(key_bytes, index_removals)
                
                # Update statistics
                with self.locks['stats']:
                    self.stats.delete_operations += 1
                
                return True
                
            except Exception as e:
                with self.locks['stats']:
                    self.stats.delete_errors += 1
                logger.error(f"Delete operation failed for key {key_bytes}: {e}")
                raise DatabaseError(f"Delete operation failed: {e}")
    
    def batch_write(self, operations: List[BatchOperation]) -> bool:
        """
        Execute batch operations atomically
        """
        with self.locks['db']:
            try:
                if self.config.db_type == DatabaseType.MEMORY:
                    # For memory DB, execute operations sequentially
                    for op in operations:
                        if op.op_type == 'put':
                            self.db[op.key] = op.value
                        elif op.op_type == 'delete':
                            if op.key in self.db:
                                del self.db[op.key]
                    return True
                
                # For plyvel, use write batch with index updates
                batch = self.db.write_batch()
                
                for op in operations:
                    try:
                        if op.op_type == 'put':
                            # Serialize and prepare value
                            serialized = self._serialize_value(op.value)
                            prepared = self._prepare_value_for_storage(serialized, op.ttl)
                            batch.put(op.key, prepared)
                            
                            # Update indexes if provided
                            if op.index_updates:
                                self._apply_index_updates(op.key, op.value, op.index_updates)
                                
                        elif op.op_type == 'delete':
                            batch.delete(op.key)
                            
                            # Remove from indexes
                            if op.index_updates:
                                self._remove_from_indexes(op.key, op.index_updates)
                                
                    except Exception as e:
                        logger.error(f"Batch operation failed for key {op.key}: {e}")
                        raise
                
                # Write batch atomically
                batch.write()
                
                with self.locks['stats']:
                    self.stats.batch_operations += 1
                
                return True
                
            except Exception as e:
                with self.locks['stats']:
                    self.stats.batch_errors += 1
                logger.error(f"Batch operation failed: {e}")
                raise DatabaseError(f"Batch operation failed: {e}")
    
    def iterate(self, prefix: Optional[bytes] = None, 
               reverse: bool = False, include_metadata: bool = False) -> Iterator[Tuple[bytes, Any]]:
        """
        Iterate over key-value pairs
        """
        with self.locks['db']:
            try:
                if self.config.db_type == DatabaseType.MEMORY:
                    keys = sorted(self.db.keys(), reverse=reverse)
                    for key in keys:
                        if prefix is None or key.startswith(prefix):
                            value = self._extract_value_from_storage(self.db[key])
                            yield (key, value) if not include_metadata else (key, value, {})
                else:
                    it = self.db.iterator(prefix=prefix, reverse=reverse)
                    for key, prepared_value in it:
                        value, metadata = self._extract_value_from_storage(prepared_value)
                        yield (key, value) if not include_metadata else (key, value, metadata)
                
                with self.locks['stats']:
                    self.stats.iterate_operations += 1
                
            except Exception as e:
                with self.locks['stats']:
                    self.stats.iterate_errors += 1
                logger.error(f"Iterate operation failed: {e}")
                raise DatabaseError(f"Iterate operation failed: {e}")
    
    def query(self, index_name: str, query: Any, 
              limit: int = 1000, offset: int = 0) -> List[Any]:
        """
        Query using secondary indexes
        """
        if index_name not in self.indexes:
            raise IndexError(f"Index not found: {index_name}")
        
        try:
            index = self.indexes[index_name]
            if hasattr(index, 'query'):
                results = index.query(query, limit, offset)
                with self.locks['stats']:
                    self.stats.index_queries += 1
                return results
            else:
                raise IndexError(f"Index {index_name} does not support queries")
        except Exception as e:
            with self.locks['stats']:
                self.stats.index_errors += 1
            raise IndexError(f"Query failed: {e}")
    
    def multi_get(self, keys: List[bytes], parallel: bool = True) -> Dict[bytes, Any]:
        """
        Retrieve multiple values in parallel
        """
        results = {}
        
        if parallel and len(keys) > 10:
            with ThreadPoolExecutor(max_workers=min(32, len(keys))) as executor:
                future_to_key = {
                    executor.submit(self.get, key, False, False): key 
                    for key in keys
                }
                for future in as_completed(future_to_key):
                    key = future_to_key[future]
                    try:
                        results[key] = future.result()
                    except KeyNotFoundError:
                        results[key] = None
                    except Exception as e:
                        logger.error(f"Multi-get failed for key {key}: {e}")
                        results[key] = None
        else:
            for key in keys:
                try:
                    results[key] = self.get(key, False, False)
                except KeyNotFoundError:
                    results[key] = None
                except Exception as e:
                    logger.error(f"Multi-get failed for key {key}: {e}")
                    results[key] = None
        
        return results
    
    def exists(self, key: Union[str, bytes]) -> bool:
        """Check if key exists using bloom filter"""
        key_bytes = self._ensure_bytes(key)
        
        # Check bloom filter first
        if 'bloom' in self.indexes:
            if not self.indexes['bloom'].check(key_bytes):
                return False
        
        # Fallback to actual check
        try:
            self.get(key_bytes, use_cache=False, check_ttl=False)
            return True
        except KeyNotFoundError:
            return False
        except Exception:
            return False
    
    def get_range(self, start_key: bytes, end_key: bytes, 
                 limit: int = 1000, include_metadata: bool = False) -> List[Tuple[bytes, Any]]:
        """Get range of keys"""
        results = []
        count = 0
        
        for item in self.iterate(prefix=start_key, include_metadata=include_metadata):
            key = item[0]
            if key > end_key:
                break
            if count >= limit:
                break
            results.append(item)
            count += 1
        
        return results
    
    def create_snapshot(self, snapshot_path: str) -> bool:
        """Create database snapshot"""
        try:
            if self.config.db_type == DatabaseType.PLYVEL:
                # Create snapshot directory
                Path(snapshot_path).mkdir(parents=True, exist_ok=True)
                
                # Copy database files
                for file in Path(self.db_path).iterdir():
                    if file.is_file():
                        shutil.copy2(file, Path(snapshot_path) / file.name)
                
                # Copy index data
                index_snapshot = {}
                with self.locks['indexes']:
                    for name, index in self.indexes.items():
                        if hasattr(index, 'snapshot'):
                            index_snapshot[name] = index.snapshot()
                
                # Save index snapshot
                with open(Path(snapshot_path) / 'index_snapshot.msgpack', 'wb') as f:
                    f.write(msgpack.packb(index_snapshot))
                
                return True
                
            elif self.config.db_type == DatabaseType.MEMORY:
                # For memory DB, serialize to disk
                with open(snapshot_path, 'wb') as f:
                    pickle.dump({
                        'db': self.db,
                        'indexes': self.indexes,
                        'cache': dict(self.cache)
                    }, f)
                return True
            
        except Exception as e:
            raise DatabaseError(f"Snapshot creation failed: {e}")
    
    def compact(self) -> bool:
        """Compact database"""
        try:
            if self.config.db_type == DatabaseType.PLYVEL:
                # Get database size before compaction
                original_size = self._get_database_size()
                
                # Create temporary compacted database
                with tempfile.TemporaryDirectory() as temp_dir:
                    compact_db = plyvel.DB(temp_dir, create_if_missing=True)
                    
                    # Copy all data with compression
                    with compact_db.write_batch() as batch:
                        for key, value in self.db.iterator():
                            batch.put(key, value)
                    
                    # Close and replace
                    compact_db.close()
                    self.db.close()
                    
                    # Remove original and move compacted
                    shutil.rmtree(self.db_path)
                    shutil.move(temp_dir, self.db_path)
                    
                    # Reopen database
                    self._initialize_database()
                
                # Log compaction results
                new_size = self._get_database_size()
                reduction = (original_size - new_size) / original_size * 100
                logger.info(f"Compaction completed: {reduction:.2f}% size reduction")
                
                return True
            
            return False
            
        except Exception as e:
            raise DatabaseError(f"Compaction failed: {e}")
    
    def backup(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            # Create snapshot first
            snapshot_dir = tempfile.mkdtemp()
            self.create_snapshot(snapshot_dir)
            
            # Compress backup
            shutil.make_archive(backup_path, 'zip', snapshot_dir)
            
            # Cleanup
            shutil.rmtree(snapshot_dir)
            
            return True
        except Exception as e:
            raise DatabaseError(f"Backup failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.locks['stats']:
            stats = self.stats.get_dict()
            stats['cache_size'] = len(self.cache)
            stats['index_count'] = len(self.indexes)
            stats['database_size'] = self._get_database_size()
            return stats
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value using configured serializer"""
        try:
            return self.serializer.serialize(value)
        except Exception as e:
            raise SerializationError(f"Serialization failed: {e}")
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value using configured serializer"""
        try:
            return self.serializer.deserialize(value)
        except Exception as e:
            raise SerializationError(f"Deserialization failed: {e}")
    
    def _prepare_value_for_storage(self, serialized_value: bytes, ttl: Optional[int] = None) -> bytes:
        """Prepare value for storage with metadata"""
        try:
            # Compress if enabled
            if self.compression:
                serialized_value = self.compression.compress(serialized_value)
            
            # Encrypt if enabled
            if self.encryption:
                serialized_value = self.encryption.encrypt(serialized_value)
            
            # Create metadata
            metadata = {
                'timestamp': time.time(),
                'ttl': ttl,
                'checksum': self._calculate_checksum(serialized_value),
                'version': 2,
                'compression': self.config.compression.name if self.compression else 'NONE',
                'encryption': self.config.encryption.name if self.encryption else 'NONE'
            }
            
            # Pack metadata and value
            metadata_bytes = msgpack.packb(metadata)
            return struct.pack('!I', len(metadata_bytes)) + metadata_bytes + serialized_value
            
        except Exception as e:
            raise DatabaseError(f"Value preparation failed: {e}")
    
    def _extract_value_from_storage(self, prepared_value: bytes) -> Tuple[Any, Dict]:
        """Extract value and metadata from stored data"""
        try:
            # Unpack metadata
            metadata_len = struct.unpack('!I', prepared_value[:4])[0]
            metadata_bytes = prepared_value[4:4 + metadata_len]
            value_bytes = prepared_value[4 + metadata_len:]
            
            metadata = msgpack.unpackb(metadata_bytes)
            
            # Decrypt if enabled
            if metadata['encryption'] != 'NONE' and self.encryption:
                value_bytes = self.encryption.decrypt(value_bytes)
            
            # Decompress if enabled
            if metadata['compression'] != 'NONE' and self.compression:
                value_bytes = self.compression.decompress(value_bytes)
            
            # Deserialize value
            value = self._deserialize_value(value_bytes)
            
            return value, metadata
            
        except Exception as e:
            raise DatabaseError(f"Value extraction failed: {e}")
    
    def _calculate_index_updates(self, key: bytes, new_value: Any, 
                               old_value: Any, ttl: Optional[int]) -> Dict[str, Any]:
        """Calculate index updates for a value change"""
        updates = {}
        
        for index_name, index in self.indexes.items():
            if hasattr(index, 'calculate_update'):
                update = index.calculate_update(key, new_value, old_value, ttl)
                if update:
                    updates[index_name] = update
        
        return updates
    
    def _update_indexes(self, key: bytes, value: Any, updates: Dict[str, Any]):
        """Update indexes with calculated updates"""
        with self.locks['indexes']:
            for index_name, update in updates.items():
                if index_name in self.indexes:
                    try:
                        self.indexes[index_name].update(key, value, update)
                    except Exception as e:
                        logger.error(f"Index update failed for {index_name}: {e}")
                        raise IndexError(f"Index update failed: {e}")
    
    def _remove_from_indexes(self, key: bytes, removals: Dict[str, Any]):
        """Remove key from indexes"""
        with self.locks['indexes']:
            for index_name, removal in removals.items():
                if index_name in self.indexes:
                    try:
                        self.indexes[index_name].remove(key, removal)
                    except Exception as e:
                        logger.error(f"Index removal failed for {index_name}: {e}")
    
    def _apply_index_updates(self, key: bytes, value: Any, updates: Dict[str, Any]):
        """Apply index updates directly"""
        with self.locks['indexes']:
            for index_name, update in updates.items():
                if index_name in self.indexes:
                    try:
                        self.indexes[index_name].apply_update(key, value, update)
                    except Exception as e:
                        logger.error(f"Index update application failed for {index_name}: {e}")
    
    def _ensure_bytes(self, key: Union[str, bytes]) -> bytes:
        """Ensure key is bytes"""
        if isinstance(key, str):
            return key.encode('utf-8')
        return key
    
    def _calculate_checksum(self, data: bytes) -> bytes:
        """Calculate checksum for data integrity"""
        return crc32c.crc32c(data).to_bytes(4, 'big')
    
    def _verify_checksum(self, data: bytes, checksum: bytes) -> bool:
        """Verify data checksum"""
        if checksum is None:
            return True
        return self._calculate_checksum(data) == checksum
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if value is expired based on metadata"""
        if metadata.get('ttl') is None:
            return False
        
        current_time = time.time()
        created_time = metadata.get('timestamp', current_time)
        return current_time > created_time + metadata['ttl']
    
    def _get_database_size(self) -> int:
        """Get approximate database size in bytes"""
        if self.config.db_type == DatabaseType.MEMORY:
            return sum(len(k) + len(v) for k, v in self.db.items())
        else:
            total_size = 0
            for file in Path(self.db_path).iterdir():
                if file.is_file():
                    total_size += file.stat().st_size
            return total_size
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def maintenance_worker():
            while self.running:
                try:
                    # Clean cache
                    self._clean_cache()
                    
                    # Clean expired entries
                    self._clean_expired()
                    
                    # Log statistics
                    self._log_stats()
                    
                    # Auto-compact if enabled
                    if self.config.auto_compact:
                        self._auto_compact()
                    
                    # Create snapshot if enabled
                    if self.config.snapshot_interval > 0:
                        self._create_auto_snapshot()
                    
                except Exception as e:
                    logger.error(f"Background maintenance failed: {e}")
                
                time.sleep(self.config.background_interval)
        
        threading.Thread(target=maintenance_worker, daemon=True).start()
    
    def _clean_cache(self):
        """Clean cache using LRU policy with TTL"""
        with self.locks['cache']:
            current_time = time.time()
            keys_to_remove = []
            
            for key, (value, timestamp) in self.cache.items():
                if current_time - timestamp > self.config.cache_ttl:
                    keys_to_remove.append(key)
                elif len(self.cache) > self.config.max_cache_size:
                    # Remove oldest entries if over limit
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.cache[key]
    
    def _clean_expired(self):
        """Clean expired keys from database"""
        try:
            expired_keys = []
            
            for key, prepared_value in self.db.iterator():
                try:
                    _, metadata = self._extract_value_from_storage(prepared_value)
                    if self._is_expired(metadata):
                        expired_keys.append(key)
                except:
                    continue
            
            # Delete expired keys in batches
            batch_size = 1000
            for i in range(0, len(expired_keys), batch_size):
                batch_ops = [
                    BatchOperation('delete', key, index_updates={})
                    for key in expired_keys[i:i + batch_size]
                ]
                self.batch_write(batch_ops)
                
        except Exception as e:
            logger.error(f"Expired entry cleanup failed: {e}")
    
    def _auto_compact(self):
        """Auto-compact database if fragmentation is high"""
        try:
            if self.config.db_type == DatabaseType.PLYVEL:
                # Calculate fragmentation ratio
                live_data_size = self._get_database_size()
                # This is simplified - in real implementation, you'd track writes/deletes
                fragmentation_ratio = self.stats.delete_operations / max(1, self.stats.put_operations)
                
                if fragmentation_ratio > self.config.compact_threshold:
                    logger.info("Auto-compacting database due to high fragmentation")
                    self.compact()
                    
        except Exception as e:
            logger.error(f"Auto-compaction failed: {e}")
    
    def _create_auto_snapshot(self):
        """Create automatic snapshot"""
        try:
            snapshot_dir = Path(self.db_path) / "snapshots" / f"auto_{int(time.time())}"
            self.create_snapshot(str(snapshot_dir))
            logger.info(f"Auto-snapshot created: {snapshot_dir}")
        except Exception as e:
            logger.error(f"Auto-snapshot failed: {e}")
    
    def _log_stats(self):
        """Log database statistics"""
        stats = self.get_stats()
        logger.info(f"Database stats: {stats}")
    
    def close(self):
        """Close database and cleanup"""
        self.running = False
        
        try:
            if self.config.db_type == DatabaseType.PLYVEL and self.db:
                self.db.close()
            
            # Clear cache
            with self.locks['cache']:
                self.cache.clear()
            
            logger.info("Database closed successfully")
            
        except Exception as e:
            logger.error(f"Database close failed: {e}")
            raise DatabaseError(f"Database close failed: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Index Implementations
class FunctionalBTreeIndex:
    """Functional B-Tree secondary index"""
    
    def __init__(self, name: str, config: IndexConfig, db: AdvancedDatabase):
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
            old_index_values = self._extract_index_values(old_value) if old_value else None
            
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
            new_values = update_data.get('new_values', [])
            old_values = update_data.get('old_values', [])
            
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
            old_values = removal_data.get('old_values', [])
            
            for old_value in old_values:
                index_key = self._create_index_key(old_value, key)
                self.db.db.delete(index_key)
        
        except Exception as e:
            raise IndexError(f"Index removal failed: {e}")
    
    def query(self, query_value: Any, limit: int = 1000, offset: int = 0) -> List[Any]:
        """Query index for values"""
        try:
            results = []
            index_prefix = self._create_index_prefix(query_value)
            
            # Iterate over index entries
            it = self.db.db.iterator(prefix=index_prefix)
            for i, (index_key, _) in enumerate(it):
                if i < offset:
                    continue
                if len(results) >= limit:
                    break
                
                # Extract primary key from index key
                primary_key = self._extract_primary_key(index_key)
                
                # Retrieve actual value
                try:
                    value = self.db.get(primary_key, use_cache=True)
                    results.append(value)
                except KeyNotFoundError:
                    continue
            
            return results
        
        except Exception as e:
            raise IndexError(f"Index query failed: {e}")
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract index values from data"""
        if isinstance(value, dict):
            return [value.get(field) for field in self.config.fields if field in value]
        elif hasattr(value, '__dict__'):
            return [getattr(value, field, None) for field in self.config.fields 
                   if hasattr(value, field)]
        return []
    
    def _create_index_key(self, index_value: Any, primary_key: bytes) -> bytes:
        """Create index key from value and primary key"""
        if isinstance(index_value, (str, int, float, bool)):
            value_bytes = str(index_value).encode()
        else:
            value_bytes = self.db._serialize_value(index_value)
        
        return self.prefix + value_bytes + b'_' + primary_key
    
    def _create_index_prefix(self, query_value: Any) -> bytes:
        """Create index prefix for querying"""
        if isinstance(query_value, (str, int, float, bool)):
            value_bytes = str(query_value).encode()
        else:
            value_bytes = self.db._serialize_value(query_value)
        
        return self.prefix + value_bytes + b'_'
    
    def _extract_primary_key(self, index_key: bytes) -> bytes:
        """Extract primary key from index key"""
        # Remove prefix and value to get primary key
        prefix_len = len(self.prefix)
        value_end = index_key.find(b'_', prefix_len)
        return index_key[value_end + 1:]

class FunctionalHashIndex(FunctionalBTreeIndex):
    """Hash-based index implementation"""
    
    def _create_index_key(self, index_value: Any, primary_key: bytes) -> bytes:
        """Create hashed index key"""
        if isinstance(index_value, (str, int, float, bool)):
            value_str = str(index_value)
        else:
            value_str = str(hash(index_value))
        
        # Use consistent hashing
        hash_value = hashlib.sha256(value_str.encode()).hexdigest()[:16]
        return self.prefix + hash_value.encode() + b'_' + primary_key

class FunctionalLSMIndex(FunctionalBTreeIndex):
    """LSM-tree based index for write-heavy workloads"""
    pass

class CompoundIndex(FunctionalBTreeIndex):
    """Compound index for multiple fields"""
    
    def _extract_index_values(self, value: Any) -> List[Any]:
        """Extract compound index values"""
        compound_values = []
        for field in self.config.fields:
            if isinstance(value, dict):
                field_value = value.get(field)
            elif hasattr(value, field):
                field_value = getattr(value, field)
            else:
                field_value = None
            
            compound_values.append(field_value)
        
        # Return tuple for compound key
        return [tuple(compound_values)] if None not in compound_values else []

# Serializer Implementations
class JSONSerializer:
    """JSON serializer with advanced object support"""
    
    def __init__(self):
        self.encoder = AdvancedJSONEncoder()
    
    def serialize(self, obj: Any) -> bytes:
        try:
            return self.encoder.encode(obj).encode('utf-8')
        except Exception as e:
            raise SerializationError(f"JSON serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        try:
            return json.loads(data.decode('utf-8'))
        except Exception as e:
            raise SerializationError(f"JSON deserialization failed: {e}")

class MsgPackSerializer:
    """MsgPack serializer for binary data"""
    
    def serialize(self, obj: Any) -> bytes:
        try:
            return msgpack.packb(obj, use_bin_type=True)
        except Exception as e:
            raise SerializationError(f"MsgPack serialization failed: {e}")
    
    def deserialize(self, data: bytes) -> Any:
        try:
            return msgpack.unpackb(data, raw=False)
        except Exception as e:
            raise SerializationError(f"MsgPack deserialization failed: {e}")

class ProtobufSerializer:
    """Protocol Buffers serializer (placeholder)"""
    def serialize(self, obj: Any) -> bytes:
        raise NotImplementedError("Protobuf serializer not implemented")
    
    def deserialize(self, data: bytes) -> Any:
        raise NotImplementedError("Protobuf deserializer not implemented")

class AvroSerializer:
    """Avro serializer (placeholder)"""
    def serialize(self, obj: Any) -> bytes:
        raise NotImplementedError("Avro serializer not implemented")
    
    def deserialize(self, data: bytes) -> Any:
        raise NotImplementedError("Avro deserializer not implemented")

# Compression Implementations
class ZlibCompression:
    def __init__(self, level: int = 6):
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        try:
            return zlib.compress(data, self.level)
        except Exception as e:
            raise CompressionError(f"Zlib compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        try:
            return zlib.decompress(data)
        except Exception as e:
            raise CompressionError(f"Zlib decompression failed: {e}")

class LZ4Compression:
    def compress(self, data: bytes) -> bytes:
        try:
            return lz4.frame.compress(data)
        except Exception as e:
            raise CompressionError(f"LZ4 compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            raise CompressionError(f"LZ4 decompression failed: {e}")

class SnappyCompression:
    def compress(self, data: bytes) -> bytes:
        try:
            return snappy.compress(data)
        except Exception as e:
            raise CompressionError(f"Snappy compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        try:
            return snappy.decompress(data)
        except Exception as e:
            raise CompressionError(f"Snappy decompression failed: {e}")

class ZstdCompression:
    def __init__(self, level: int = 3):
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        try:
            import zstandard as zstd
            return zstd.compress(data, self.level)
        except Exception as e:
            raise CompressionError(f"Zstd compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        try:
            import zstandard as zstd
            return zstd.decompress(data)
        except Exception as e:
            raise CompressionError(f"Zstd decompression failed: {e}")

# Encryption Implementations
class AES256Encryption:
    def __init__(self, key: Optional[str] = None):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        
        self.key = key or os.urandom(32)
        self.backend = default_backend()
    
    def encrypt(self, data: bytes) -> bytes:
        try:
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padder = padding.PKCS7(128).padder()
            padded_data = padder.update(data) + padder.finalize()
            
            encrypted = encryptor.update(padded_data) + encryptor.finalize()
            return iv + encrypted
            
        except Exception as e:
            raise EncryptionError(f"AES encryption failed: {e}")
    
    def decrypt(self, data: bytes) -> bytes:
        try:
            iv = data[:16]
            encrypted = data[16:]
            
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(encrypted) + decryptor.finalize()
            
            # Unpad data
            unpadder = padding.PKCS7(128).unpadder()
            return unpadder.update(padded_data) + unpadder.finalize()
            
        except Exception as e:
            raise EncryptionError(f"AES decryption failed: {e}")
            
class ChaCha20Encryption:
    def __init__(self, key: Optional[str] = None):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
        from cryptography.hazmat.backends import default_backend
        
        self.key = key or os.urandom(32)
        self.backend = default_backend()
    
    def encrypt(self, data: bytes) -> bytes:
        try:
            nonce = os.urandom(16)
            cipher = Cipher(
                algorithms.ChaCha20(self.key, nonce),
                mode=None,
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            encrypted = encryptor.update(data)
            return nonce + encrypted
            
        except Exception as e:
            raise EncryptionError(f"ChaCha20 encryption failed: {e}")
    
    def decrypt(self, data: bytes) -> bytes:
        try:
            nonce = data[:16]
            encrypted = data[16:]
            
            cipher = Cipher(
                algorithms.ChaCha20(self.key, nonce),
                mode=None,
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            return decryptor.update(encrypted)
            
        except Exception as e:
            raise EncryptionError(f"ChaCha20 decryption failed: {e}")

# Statistics tracking with enhanced metrics
class DatabaseStats:
    def __init__(self):
        self.put_operations = 0
        self.get_operations = 0
        self.delete_operations = 0
        self.batch_operations = 0
        self.iterate_operations = 0
        self.index_queries = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.misses = 0
        self.put_errors = 0
        self.get_errors = 0
        self.delete_errors = 0
        self.batch_errors = 0
        self.iterate_errors = 0
        self.index_errors = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self.compaction_count = 0
        self.snapshot_count = 0
        self.backup_count = 0
        self.start_time = time.time()
        self.last_operation_time = time.time()
    
    def get_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary"""
        current_time = time.time()
        uptime = current_time - self.start_time
        ops_per_sec = (self.put_operations + self.get_operations + self.delete_operations) / max(1, uptime)
        
        return {
            'put_operations': self.put_operations,
            'get_operations': self.get_operations,
            'delete_operations': self.delete_operations,
            'batch_operations': self.batch_operations,
            'iterate_operations': self.iterate_operations,
            'index_queries': self.index_queries,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'misses': self.misses,
            'put_errors': self.put_errors,
            'get_errors': self.get_errors,
            'delete_errors': self.delete_errors,
            'batch_errors': self.batch_errors,
            'iterate_errors': self.iterate_errors,
            'index_errors': self.index_errors,
            'bytes_written': self.bytes_written,
            'bytes_read': self.bytes_read,
            'compaction_count': self.compaction_count,
            'snapshot_count': self.snapshot_count,
            'backup_count': self.backup_count,
            'uptime_seconds': uptime,
            'operations_per_second': ops_per_sec,
            'last_operation_timestamp': self.last_operation_time
        }
    
    def update_operation_time(self):
        """Update last operation timestamp"""
        self.last_operation_time = time.time()

# Transaction context manager with proper rollback support
@contextmanager
def transaction(db: AdvancedDatabase):
    """Context manager for database transactions with rollback support"""
    batch_ops = []
    original_cache = dict(db.cache)
    original_indexes = {}
    
    try:
        # Snapshot index states for rollback
        with db.locks['indexes']:
            for name, index in db.indexes.items():
                if hasattr(index, 'snapshot'):
                    original_indexes[name] = index.snapshot()
        
        # Yield control to the transaction block
        yield batch_ops
        
        # Commit the transaction
        if batch_ops:
            success = db.batch_write(batch_ops)
            if not success:
                raise TransactionError("Batch write failed during transaction commit")
        
    except Exception as e:
        # Rollback changes
        try:
            # Restore cache
            with db.locks['cache']:
                db.cache.clear()
                db.cache.update(original_cache)
            
            # Restore indexes
            with db.locks['indexes']:
                for name, snapshot in original_indexes.items():
                    if hasattr(db.indexes[name], 'restore'):
                        db.indexes[name].restore(snapshot)
            
            logger.info("Transaction rolled back successfully")
            
        except Exception as rollback_error:
            logger.error(f"Rollback failed: {rollback_error}")
            raise TransactionError(f"Rollback failed: {rollback_error}") from e
        
        # Re-raise the original exception
        raise TransactionError(f"Transaction failed: {e}") from e

# Advanced query builder for complex queries
class QueryBuilder:
    """Builder for complex database queries"""
    
    def __init__(self, db: AdvancedDatabase):
        self.db = db
        self.conditions = []
        self.limit = 1000
        self.offset = 0
        self.sort_field = None
        self.sort_reverse = False
        self.index_name = None
    
    def where(self, field: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add where condition"""
        self.conditions.append(('where', field, operator, value))
        return self
    
    def limit(self, limit: int) -> 'QueryBuilder':
        """Set result limit"""
        self.limit = limit
        return self
    
    def offset(self, offset: int) -> 'QueryBuilder':
        """Set result offset"""
        self.offset = offset
        return self
    
    def order_by(self, field: str, reverse: bool = False) -> 'QueryBuilder':
        """Set sort order"""
        self.sort_field = field
        self.sort_reverse = reverse
        return self
    
    def use_index(self, index_name: str) -> 'QueryBuilder':
        """Specify index to use"""
        self.index_name = index_name
        return self
    
    def execute(self) -> List[Any]:
        """Execute the query"""
        try:
            # If using a specific index, use it for querying
            if self.index_name:
                # For simple equality queries on indexed fields
                if len(self.conditions) == 1:
                    cond = self.conditions[0]
                    if cond[0] == 'where' and cond[2] == '==':
                        return self.db.query(self.index_name, cond[3], self.limit, self.offset)
            
            # Fallback to full scan with filtering
            results = []
            count = 0
            
            for key, value in self.db.iterate():
                if self._matches_conditions(value):
                    if count >= self.offset:
                        results.append(value)
                    count += 1
                    if len(results) >= self.limit:
                        break
            
            # Apply sorting if specified
            if self.sort_field:
                results.sort(
                    key=lambda x: self._extract_field_value(x, self.sort_field),
                    reverse=self.sort_reverse
                )
            
            return results
            
        except Exception as e:
            raise DatabaseError(f"Query execution failed: {e}")
    
    def _matches_conditions(self, value: Any) -> bool:
        """Check if value matches all conditions"""
        for cond in self.conditions:
            if cond[0] == 'where':
                field_value = self._extract_field_value(value, cond[1])
                if not self._compare_values(field_value, cond[2], cond[3]):
                    return False
        return True
    
    def _extract_field_value(self, value: Any, field: str) -> Any:
        """Extract field value from object"""
        if isinstance(value, dict):
            return value.get(field)
        elif hasattr(value, field):
            return getattr(value, field)
        elif hasattr(value, '__getitem__'):
            try:
                return value[field]
            except (KeyError, TypeError):
                pass
        return None
    
    def _compare_values(self, field_value: Any, operator: str, compare_value: Any) -> bool:
        """Compare values based on operator"""
        if operator == '==':
            return field_value == compare_value
        elif operator == '!=':
            return field_value != compare_value
        elif operator == '>':
            return field_value > compare_value
        elif operator == '>=':
            return field_value >= compare_value
        elif operator == '<':
            return field_value < compare_value
        elif operator == '<=':
            return field_value <= compare_value
        elif operator == 'in':
            return field_value in compare_value
        elif operator == 'contains':
            return compare_value in field_value if field_value else False
        return False

# Database manager for multiple database instances
class DatabaseManager:
    """Manager for multiple database instances"""
    
    def __init__(self):
        self.databases: Dict[str, AdvancedDatabase] = {}
        self.lock = threading.RLock()
    
    def create_database(self, name: str, path: str, config: Optional[DatabaseConfig] = None) -> AdvancedDatabase:
        """Create a new database instance"""
        with self.lock:
            if name in self.databases:
                raise DatabaseError(f"Database '{name}' already exists")
            
            db = AdvancedDatabase(path, config)
            self.databases[name] = db
            return db
    
    def get_database(self, name: str) -> AdvancedDatabase:
        """Get database instance by name"""
        with self.lock:
            if name not in self.databases:
                raise DatabaseError(f"Database '{name}' not found")
            return self.databases[name]
    
    def close_database(self, name: str) -> bool:
        """Close database instance"""
        with self.lock:
            if name in self.databases:
                db = self.databases.pop(name)
                db.close()
                return True
            return False
    
    def close_all(self):
        """Close all database instances"""
        with self.lock:
            for name, db in list(self.databases.items()):
                try:
                    db.close()
                    del self.databases[name]
                except Exception as e:
                    logger.error(f"Failed to close database '{name}': {e}")
    
    def list_databases(self) -> List[str]:
        """List all database names"""
        with self.lock:
            return list(self.databases.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all databases"""
        stats = {}
        with self.lock:
            for name, db in self.databases.items():
                stats[name] = db.get_stats()
        return stats

# Health check and monitoring
class DatabaseHealthMonitor:
    """Monitor database health and performance"""
    
    def __init__(self, db: AdvancedDatabase, check_interval: int = 60):
        self.db = db
        self.check_interval = check_interval
        self.running = True
        self.metrics: List[Dict[str, Any]] = []
        self.max_metrics = 1000
    
    def start(self):
        """Start health monitoring"""
        def monitor_loop():
            while self.running:
                try:
                    self._collect_metrics()
                    self._check_health()
                    time.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Health monitoring failed: {e}")
        
        threading.Thread(target=monitor_loop, daemon=True).start()
    
    def stop(self):
        """Stop health monitoring"""
        self.running = False
    
    def _collect_metrics(self):
        """Collect database metrics"""
        try:
            stats = self.db.get_stats()
            metrics = {
                'timestamp': time.time(),
                'stats': stats,
                'cache_size': len(self.db.cache),
                'index_count': len(self.db.indexes)
            }
            
            self.metrics.append(metrics)
            if len(self.metrics) > self.max_metrics:
                self.metrics.pop(0)
                
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    def _check_health(self):
        """Check database health status"""
        try:
            # Check if database is responsive
            test_key = b'health_check_' + str(time.time()).encode()
            self.db.put(test_key, {'check': True}, ttl=10)
            value = self.db.get(test_key)
            
            if value is None:
                raise DatabaseError("Database health check failed")
            
            # Check memory usage
            stats = self.db.get_stats()
            if stats.get('cache_size', 0) > 1000000:  # 1 million items
                logger.warning("Database cache size is very large")
            
            # Check error rate
            total_ops = stats.get('put_operations', 0) + stats.get('get_operations', 0)
            total_errors = stats.get('put_errors', 0) + stats.get('get_errors', 0)
            
            if total_ops > 0 and total_errors / total_ops > 0.01:  # 1% error rate
                logger.warning(f"High database error rate: {total_errors/total_ops:.2%}")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def get_metrics(self, duration: int = 3600) -> List[Dict[str, Any]]:
        """Get metrics for the specified duration"""
        cutoff = time.time() - duration
        return [m for m in self.metrics if m['timestamp'] >= cutoff]