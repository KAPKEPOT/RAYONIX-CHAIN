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

from database.utils.types import (
    DatabaseType, CompressionType, EncryptionType, 
    IndexType, SerializationType, DatabaseConfig, IndexConfig, BatchOperation
)
from database.utils.exceptions import (
    DatabaseError, KeyNotFoundError, SerializationError, IndexError,
    IntegrityError, CompressionError, EncryptionError
)
from database.utils.helpers import AdvancedJSONEncoder
from database.utils.stats import DatabaseStats
from database.core.indexing import FunctionalBTreeIndex, FunctionalHashIndex, FunctionalLSMIndex, CompoundIndex
from database.core.serialization import JSONSerializer, MsgPackSerializer, ProtobufSerializer, AvroSerializer
from database.core.compression import ZlibCompression, LZ4Compression, SnappyCompression, ZstdCompression
from database.core.encryption import AES256Encryption, ChaCha20Encryption
from database.services.background_tasks import BackgroundTaskService

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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
        self.background_service = None
        
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
            	try:
            		from database.core.compression import ZlibCompression
            		self.compression = ZlibCompression(self.config.compression_level)
            		logger.info("Zlib compression initialized successfully")
            	except CompressionError as e:
            		logger.warning(f"Zlib compression initialization failed, using no compression: {e}")
            		self.compression = None
            		self.config.compression = CompressionType.NONE
            		
            elif self.config.compression == CompressionType.LZ4:
            	try:
            		import lz4.frame
            		from database.core.compression import LZ4Compression
            		self.compression = LZ4Compression()
            		logger.info("LZ4 compression initialized successfully")
            	except (CompressionError, ImportError) as e:
            		logger.warning(f"LZ4 compression initialization failed, using no compression: {e}")
            		self.compression = None
            		self.config.compression = CompressionType.NONE
            		
            elif self.config.compression == CompressionType.SNAPPY:
            	try:
            		from database.core.compression import SnappyCompression
            		self.compression = SnappyCompression()
            		logger.info("Snappy compression initialized successfully")
            	except (CompressionError, ImportError) as e:
            		logger.warning(f"Snappy compression initialization failed, using no compression: {e}")
            		self.compression = None
            		self.config.compression = CompressionType.NONE
            		
            elif self.config.compression == CompressionType.ZSTD:
            	try:
            		from database.core.compression import ZstdCompression
            		self.compression = ZstdCompression(self.config.compression_level)
            		logger.info("Zstd compression initialized successfully")
            	except (CompressionError, ImportError) as e:
            		logger.warning(f"Zstd compression initialization failed, using no compression: {e}")
            		self.compression = None
            		self.config.compression = CompressionType.NONE
            		
            elif self.config.compression == CompressionType.NONE:
            	self.compression = None
            	logger.info("Compression disabled")
            	
        except Exception as e:
        	logger.error(f"Compression initialization failed: {e}")
        	self.compression = None
        	self.config.compression = CompressionType.NONE

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
        
        # Validate inputs
        if not key_bytes:
        	raise DatabaseError("Key cannot be empty")
        	
        if value is None:
        	raise DatabaseError("Value cannot be None")
        
        with self.locks['db']:
        	try:
        		# Serialize value with error handling
        		try:
        			serialized_value = self._serialize_value(value)
        			if serialized_value is None:
        				raise SerializationError("Serialization returned None")
        		except SerializationError as e:
        			logger.error(f"Serialization failed for key {key_bytes}: {e}")
        			raise
        			
        		# Prepare value for storage
        		prepared_value = self._prepare_value_for_storage(serialized_value, ttl)
        		
        		# Calculate index updates if needed
        		index_updates = {}
        		if update_indexes:
        			try:
        				index_updates = self._calculate_index_updates(key_bytes, value, None, ttl)
        			except Exception as e:
        				logger.warning(f"Index calculation failed for key {key_bytes}: {e}")
        				
        		# Store in database
        		try:
        			if self.config.db_type == DatabaseType.PLYVEL:
        				self.db.put(key_bytes, prepared_value, sync=False)
        			else:
        				self.db.put(key_bytes, prepared_value, sync=False)
        				
        		except Exception as e:
        			raise DatabaseError(f"Database storage failed: {e}")
        		# Update cache
        		if use_cache:
        			with self.locks['cache']:
        				self.cache[key_bytes] = (value, time.time())
        				if len(self.cache) > self.config.max_cache_size:
        					self.cache.popitem(last=False)
        					
        		# Update indexes
        		if update_indexes and index_updates:
        			try:
        				self._update_indexes(key_bytes, value, index_updates)
        			except Exception as e:
        				logger.error(f"Index update failed for key {key_bytes}: {e}")
        				# Don't fail the put operation if index updates fail
        		
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
        			cache_entry = self.cache[key_bytes]
        			if isinstance(cache_entry, tuple) and len(cache_entry) == 2:
        				value, timestamp = cache_entry
        				if time.time() - timestamp < self.config.cache_ttl:
        					with self.locks['stats']:
        						self.stats.cache_hits += 1
        					return value
        				else:
        					# Remove expired cache entry
        					del self.cache[key_bytes]
        
        with self.locks['db']:
            try:
                # Retrieve from database
                if self.config.db_type == DatabaseType.PLYVEL:
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
                	try:
                		self.delete(key_bytes, update_indexes=True)
                	except Exception as e:
                		logger.warning(f"Failed to delete expired key {key_bytes}: {e}")
                	raise KeyNotFoundError(f"Key expired: {key_bytes}")
                
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
        	# Validate input
        	if serialized_value is None:
        		raise ValueError("Serialized value cannot be None")
        		
        	if not isinstance(serialized_value, bytes):
        		raise TypeError(f"Serialized value must be bytes, got {type(serialized_value)}")
        	
        	original_value = serialized_value
        	
        	# Compress if enabled and data is compressible
        	if self.compression and len(original_value) > 100:
        		try:
        			serialized_value = self.compression.compress(original_value)
        			compression_used = self.config.compression.name
        		
        		except CompressionError as e:
        			logger.warning(f"Compression failed, storing uncompressed: {e}")
        			serialized_value = original_value
        			compression_used = 'NONE'
        	else:
        		compression_used = 'NONE'
        	
        	# Encrypt if enabled
        	if self.encryption:
        		try:
        			serialized_value = self.encryption.encrypt(serialized_value)
        			encryption_used = self.config.encryption.name
        		
        		except EncryptionError as e:
        			logger.error(f"Encryption failed: {e}")
        			raise
        	else:
        		encryption_used = 'NONE'
        	
        	# Create metadata with comprehensive information
        	metadata = {
        	    
        	    'timestamp': time.time(),
        	    'ttl': ttl,
        	    'checksum': self._calculate_checksum(serialized_value),
        	    'version': 2,
        	    'compression': compression_used,
        	    'encryption': encryption_used,
        	    'original_size': len(original_value),
        	    'stored_size': len(serialized_value)
        	}
        	
        	# Pack metadata and value with error handling
        	try:
        		metadata_bytes = msgpack.packb(metadata, use_bin_type=True)
        		metadata_len = len(metadata_bytes)
        		
        		# Use struct to pack length (4 bytes for length)
        		header = struct.pack('!I', metadata_len)
        		return header + metadata_bytes + serialized_value
        		
        	except (struct.error, msgpack.PackException) as e:
        		raise DatabaseError(f"Metadata packing failed: {e}")
        		
        except Exception as e:
        	logger.error(f"Value preparation failed: {e}")
        	raise DatabaseError(f"Value preparation failed: {e}")

    def _extract_value_from_storage(self, prepared_value: bytes) -> Tuple[Any, Dict]:
        """Extract value and metadata from stored data"""
        if not prepared_value:
        	raise ValueError("Prepared value cannot be empty")
        
        # Validate minimum length
        if len(prepared_value) < 4:
        	raise DatabaseError("Invalid prepared value: too short for header")
        	
        try:
            # Unpack header with validation
            metadata_len = struct.unpack('!I', prepared_value[:4])[0]
            
            # Validate metadata length is reasonable	
            max_reasonable_size = 10 * 1024 * 1024  # 10MB max for metadata
            
            # Validate metadata length
            if (metadata_len > len(prepared_value) - 4 or metadata_len > max_reasonable_metadata_size or metadata_len == 0):
            	raise DatabaseError(f"Corrupted metadata length: {metadata_len}")
            	
            metadata_bytes = prepared_value[4:4 + metadata_len]
            value_bytes = prepared_value[4 + metadata_len:]
            
            # Unpack metadata
            try:
            	metadata = msgpack.unpackb(metadata_bytes, raw=False)
            except msgpack.UnpackException as e:
            	raise DatabaseError(f"Metadata unpacking failed: {e}")
            	
            # Validate required metadata fields
            required_fields = ['version', 'compression', 'encryption', 'checksum']
            for field in required_fields:
            	if field not in metadata:
            		raise DatabaseError(f"Missing required metadata field: {field}")
            		
            # Decrypt if enabled
            if metadata['encryption'] != 'NONE' and self.encryption:
            	try:
            		value_bytes = self.encryption.decrypt(value_bytes)
            	except EncryptionError as e:
            		raise DatabaseError(f"Decryption failed: {e}")
            	
            # Decompress if enabled
            if metadata['compression'] != 'NONE' and self.compression:
            	try:
            		value_bytes = self.compression.decompress(value_bytes)
            	except CompressionError as e:
            		raise DatabaseError(f"Decompression failed: {e}")
            # Verify checksum
            if not self._verify_checksum(value_bytes, metadata.get('checksum')):
            	raise IntegrityError("Checksum verification failed")
            	
            # Deserialize value
            try:
            	value = self._deserialize_value(value_bytes)
            except SerializationError as e:
            	raise DatabaseError(f"Deserialization failed: {e}")
            return value, metadata
            
        except struct.error as e:
        	raise DatabaseError(f"Invalid data structure: {e}")
  
    def _calculate_index_updates(self, key: bytes, new_value: Any, 
                               old_value: Any, ttl: Optional[int]) -> Dict[str, Any]:
        """Calculate index updates for a value change"""
        updates = {}
        
        for index_name, index in self.indexes.items():
            if hasattr(index, 'calculate_update'):
                try:
                	update = index.calculate_update(key, new_value, old_value, ttl)
                	if update:
                		# Ensure both values are lists, not None
                		update['new_values'] = update.get('new_values', []) or []
                		update['old_values'] = update.get('old_values', []) or []
                		updates[index_name] = update
                except Exception as e:
                	logger.error(f"Index calculation failed for {index_name}: {e}")
                	# Continue with other indexes instead of failing completely
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
        self.background_service = BackgroundTaskService(self)
        self.background_service.start()
    
    def close(self):
        """Close database and cleanup"""
        self.running = False
        
        if self.background_service:
            self.background_service.stop()
        
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