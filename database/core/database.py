# database/core/database.py 
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

from merkle_system.merkle import (
    SparseMerkleTree, MerkleTreeConfig, HashAlgorithm, 
    ProofFormat, MerkleTreeStats, global_stats
)

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
from database.core.integrity_manager import IntegrityManager
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from config.merkle_config import MerkleTreeConfig

# Configure logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class AdvancedDatabase:
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
            'stats': threading.RLock(),
            'merkle': threading.RLock()
        }
        self.stats = DatabaseStats()
        self.encryption = None
        self.compression = None
        self.serializer = None
        self.running = True
        self.background_service = None
        
        # Initialize Merkle integrity protection
        merkle_config = MerkleTreeConfig(
            enabled=getattr(self.config, 'merkle_integrity', True),
            merkle_tree_depth=getattr(self.config, 'merkle_tree_depth', 256),
            hash_algorithm=getattr(self.config, 'merkle_hash_algorithm', HashAlgorithm.SHA256),
            verify_on_read=getattr(self.config, 'merkle_verify_on_read', True),
            verify_on_write=getattr(self.config, 'merkle_verify_on_write', True),
            auto_recover=getattr(self.config, 'merkle_auto_recover', True)
        )
        self.integrity_manager = IntegrityManager(db_path, merkle_config)
        
        self._initialize_database()
        self._initialize_encryption()
        self._initialize_compression()
        self._initialize_serializer()
        
        # Background tasks including integrity checks
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
    
    def _start_background_tasks(self):
        """Start background tasks including integrity checks"""
        # Start the existing background service
        if hasattr(self, 'background_service') and self.background_service:
            self.background_service.start()
        
        # Start integrity check thread if enabled
        if self.integrity_manager.config.enabled:
            self._start_integrity_monitor()
    
    def _start_integrity_monitor(self):
        """Start background integrity monitoring"""
        def integrity_monitor():
            while self.running:
                try:
                    # Run integrity check at configured interval
                    time.sleep(self.integrity_manager.config.integrity_check_interval)
                    
                    if self.running:
                        logger.info("Running scheduled database integrity check...")
                        results = self.integrity_manager.run_integrity_check(self)
                        logger.info(f"Integrity check completed: {results}")
                        
                except Exception as e:
                    logger.error(f"Integrity monitor error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        monitor_thread = threading.Thread(target=integrity_monitor, daemon=True)
        monitor_thread.start()
        logger.info("Database integrity monitor started")
 
    def put(self, key: Union[str, bytes], value: Any, ttl: Optional[int] = None, 
            use_cache: bool = True, update_indexes: bool = True, 
            verify_integrity: bool = True) -> bool:
        """
        Store key-value pair with Merkle integrity protection
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
                
                # Verify integrity before write if enabled
                if verify_integrity and self.integrity_manager.config.verify_on_write:
                    valid, reason = self.integrity_manager.verify_data_integrity(key_bytes, value)
                    if not valid:
                        logger.warning(f"Integrity check failed before write for key {key_bytes}: {reason}")
                        if not self.integrity_manager.config.auto_recover:
                            raise IntegrityError(f"Data integrity violation: {reason}")
                
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
                        self.db[key_bytes] = prepared_value
                        
                except Exception as e:
                    raise DatabaseError(f"Database storage failed: {e}")
                
                # Register with Merkle integrity manager
                self.integrity_manager.register_put_operation(key_bytes, value)
                
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
            check_ttl: bool = True, verify_integrity: bool = True) -> Any:
        """
        Retrieve value by key with Merkle integrity verification
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
                            # Verify cache integrity if enabled
                            if verify_integrity and self.integrity_manager.config.verify_on_read:
                                valid, reason = self.integrity_manager.verify_data_integrity(key_bytes, value)
                                if not valid:
                                    logger.warning(f"Cache integrity violation for key {key_bytes}: {reason}")
                                    # Remove from cache and fall through to database read
                                    del self.cache[key_bytes]
                                else:
                                    with self.locks['stats']:
                                        self.stats.cache_hits += 1
                                    return value
                            else:
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
                
                # Verify data integrity using Merkle tree
                if verify_integrity and self.integrity_manager.config.verify_on_read:
                    valid, reason = self.integrity_manager.verify_data_integrity(key_bytes, value)
                    if not valid:
                        logger.error(f"DATA INTEGRITY VIOLATION for key {key_bytes}: {reason}")
                        
                        # Attempt automatic recovery
                        if self.integrity_manager.config.auto_recover:
                            if self.integrity_manager.attempt_recovery(key_bytes, value):
                                logger.info(f"Successfully recovered corrupted key: {key_bytes}")
                            else:
                                raise IntegrityError(f"Data corrupted and recovery failed: {reason}")
                        else:
                            raise IntegrityError(f"Data integrity violation: {reason}")
                
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
                
            except (KeyNotFoundError, IntegrityError):
                raise
            except Exception as e:
                with self.locks['stats']:
                    self.stats.get_errors += 1
                logger.error(f"Get operation failed for key {key_bytes}: {e}")
                raise DatabaseError(f"Get operation failed: {e}")
    
    def get_with_integrity_guarantee(self, key: Union[str, bytes]) -> Any:
        """
        Get value with guaranteed integrity verification and automatic recovery
        """
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                return self.get(key, verify_integrity=True)
            except IntegrityError as e:
                if attempt == max_attempts - 1:
                    raise
                logger.warning(f"Integrity error on attempt {attempt + 1}, retrying: {e}")
                time.sleep(0.1 * (attempt + 1))
        
        raise DatabaseError("Max integrity recovery attempts exceeded")
    
    def delete(self, key: Union[str, bytes], use_cache: bool = True, 
               update_indexes: bool = True) -> bool:
        """
        Delete key-value pair with Merkle integrity update
        """
        key_bytes = self._ensure_bytes(key)
        
        with self.locks['db']:
            try:
                # Get current value for index updates
                current_value = None
                if update_indexes:
                    try:
                        current_value = self.get(key_bytes, use_cache=False, check_ttl=False, verify_integrity=False)
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
                
                # Update Merkle integrity manager
                self.integrity_manager.register_delete_operation(key_bytes)
                
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
    
    # NEW INTEGRITY-RELATED METHODS
    
    def verify_data_integrity(self, key: Union[str, bytes]) -> Tuple[bool, Optional[str]]:
        """Verify integrity of a specific key-value pair"""
        key_bytes = self._ensure_bytes(key)
        
        try:
            value = self.get(key_bytes, use_cache=False, verify_integrity=False)
            return self.integrity_manager.verify_data_integrity(key_bytes, value)
        except KeyNotFoundError:
            return False, "Key not found"
        except Exception as e:
            return False, f"Verification error: {e}"
    
    def get_integrity_root(self) -> Optional[str]:
        """Get current Merkle integrity root hash"""
        return self.integrity_manager.get_integrity_root()
    
    def run_integrity_scan(self) -> Dict[str, Any]:
        """Run comprehensive integrity scan on entire database"""
        return self.integrity_manager.run_integrity_check(self)
    
    def get_corrupted_keys(self) -> List[bytes]:
        """Get list of corrupted keys"""
        return self.integrity_manager.get_corrupted_keys()
    
    def get_integrity_stats(self) -> Dict[str, Any]:
        """Get integrity protection statistics"""
        return self.integrity_manager.get_stats()
    
    def repair_corrupted_entries(self) -> Dict[str, Any]:
        """Attempt to repair all corrupted entries"""
        corrupted_keys = self.get_corrupted_keys()
        results = {
            'total_corrupted': len(corrupted_keys),
            'repaired': 0,
            'failed': 0,
            'details': []
        }
        
        for key in corrupted_keys:
            try:
                # Try to read the corrupted value
                value = self.get(key, use_cache=False, verify_integrity=False)
                
                # Attempt recovery through integrity manager
                if self.integrity_manager.attempt_recovery(key, value):
                    # Verify the repair worked
                    valid, reason = self.verify_data_integrity(key)
                    if valid:
                        results['repaired'] += 1
                        results['details'].append({
                            'key': key.hex(),
                            'status': 'repaired',
                            'reason': reason
                        })
                    else:
                        results['failed'] += 1
                        results['details'].append({
                            'key': key.hex(),
                            'status': 'failed',
                            'reason': f"Repair verification failed: {reason}"
                        })
                else:
                    results['failed'] += 1
                    results['details'].append({
                        'key': key.hex(),
                        'status': 'failed',
                        'reason': 'Recovery attempt failed'
                    })
                    
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'key': key.hex(),
                    'status': 'failed',
                    'reason': f"Error during repair: {e}"
                })
        
        return results
    
    # EXISTING DATABASE METHODS (unchanged but now protected by Merkle)
    
    def create_index(self, name: str, config: IndexConfig):
        """Create a new functional index"""
        with self.locks['indexes']:
            try:
                if config.index_type == IndexType.BTREE:
                    self.indexes[name] = FunctionalBTreeIndex(name, config, self)
                elif config.index_type == IndexType.HASH:
                    self.indexes[name] = FunctionalHashIndex(name, config, self)
                elif config.index_type == IndexType.BLOOM:
                    self.indexes[name] = BloomFilter(
                        config.bloom_filter_size, 
                        config.bloom_filter_error_rate
                    )
                elif config.index_type == IndexType.LSM:
                    self.indexes[name] = FunctionalLSMIndex(name, config, self)
                elif config.index_type == IndexType.COMPOUND:
                    self.indexes[name] = CompoundIndex(name, config, self)
                
                self.index_configs[name] = config
                logger.info(f"Created index: {name} with type {config.index_type}")
                
            except Exception as e:
                raise IndexError(f"Failed to create index {name}: {e}")
    
    def batch_write(self, operations: List[BatchOperation]) -> bool:
        """Execute batch operations atomically with integrity protection"""
        with self.locks['db']:
            try:
                if self.config.db_type == DatabaseType.MEMORY:
                    # For memory DB, execute operations sequentially
                    for op in operations:
                        if op.op_type == 'put':
                            self.db[op.key] = op.value
                            self.integrity_manager.register_put_operation(op.key, op.value)
                        elif op.op_type == 'delete':
                            if op.key in self.db:
                                del self.db[op.key]
                                self.integrity_manager.register_delete_operation(op.key)
                    return True
                
                # For plyvel, use write batch with integrity updates
                batch = self.db.write_batch()
                
                for op in operations:
                    try:
                        if op.op_type == 'put':
                            # Serialize and prepare value
                            serialized = self._serialize_value(op.value)
                            prepared = self._prepare_value_for_storage(serialized, op.ttl)
                            batch.put(op.key, prepared)
                            
                            # Register with integrity manager
                            self.integrity_manager.register_put_operation(op.key, op.value)
                            
                            # Update indexes if provided
                            if op.index_updates:
                                self._apply_index_updates(op.key, op.value, op.index_updates)
                                
                        elif op.op_type == 'delete':
                            batch.delete(op.key)
                            
                            # Update integrity manager
                            self.integrity_manager.register_delete_operation(op.key)
                            
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
               reverse: bool = False, include_metadata: bool = False,
               verify_integrity: bool = False) -> Iterator[Tuple[bytes, Any]]:
        """
        Iterate over key-value pairs with optional integrity verification
        """
        with self.locks['db']:
            try:
                if self.config.db_type == DatabaseType.MEMORY:
                    keys = sorted(self.db.keys(), reverse=reverse)
                    for key in keys:
                        if prefix is None or key.startswith(prefix):
                            value, metadata = self._extract_value_from_storage(self.db[key])
                            
                            # Verify integrity if requested
                            if verify_integrity:
                                valid, reason = self.integrity_manager.verify_data_integrity(key, value)
                                if not valid:
                                    logger.warning(f"Skipping corrupted key during iteration: {key.hex()}")
                                    continue
                            
                            yield (key, value) if not include_metadata else (key, value, metadata)
                else:
                    it = self.db.iterator(prefix=prefix, reverse=reverse)
                    for key, prepared_value in it:
                        value, metadata = self._extract_value_from_storage(prepared_value)
                        
                        # Verify integrity if requested
                        if verify_integrity:
                            valid, reason = self.integrity_manager.verify_data_integrity(key, value)
                            if not valid:
                                logger.warning(f"Skipping corrupted key during iteration: {key.hex()}")
                                continue
                        
                        yield (key, value) if not include_metadata else (key, value, metadata)
                
                with self.locks['stats']:
                    self.stats.iterate_operations += 1
                
            except Exception as e:
                with self.locks['stats']:
                    self.stats.iterate_errors += 1
                logger.error(f"Iterate operation failed: {e}")
                raise DatabaseError(f"Iterate operation failed: {e}")
    
    def close(self):
        """Close database with integrity state preservation"""
        try:
            self.running = False
            
            # Save Merkle integrity state
            if hasattr(self, 'integrity_manager'):
                self.integrity_manager._save_merkle_state()
            
            # Close underlying database
            if hasattr(self.db, 'close'):
                self.db.close()
            
            logger.info("Database closed with integrity state preserved")
            
        except Exception as e:
            logger.error(f"Error closing database: {e}")
    
    # EXISTING HELPER METHODS (unchanged)
    def _ensure_bytes(self, key: Union[str, bytes]) -> bytes:
        """Ensure key is bytes"""
        if isinstance(key, str):
            return key.encode('utf-8')
        return key
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value using configured serializer"""
        if self.serializer:
            return self.serializer.serialize(value)
        # Fallback to JSON
        return json.dumps(value, cls=AdvancedJSONEncoder).encode('utf-8')
    def _calculate_checksum(self, data: bytes) -> bytes:
        """Calculate checksum for data integrity"""
        return crc32c.crc32c(data).to_bytes(4, 'big')
    
    def _verify_checksum(self, data: bytes, checksum: bytes) -> bool:
        """Verify data checksum"""
        if checksum is None:
            return True
        return self._calculate_checksum(data) == checksum
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value using configured serializer"""
        if self.serializer:
            return self.serializer.deserialize(data)
        # Fallback to JSON
        return json.loads(data.decode('utf-8'))
    
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
            MAX_REASONABLE_METADATA_SIZE = 1 * 1024 * 1024  # 10MB max for metadata
            
            # Validate metadata length
            if (metadata_len > len(prepared_value) - 4 or metadata_len > MAX_REASONABLE_METADATA_SIZE or metadata_len == 0):
            	raise DatabaseError(f"Corrupted metadata length: {metadata_len}")
            	
            metadata_bytes = prepared_value[4:4 + metadata_len]
            encrypted_compressed_data = prepared_value[4 + metadata_len:]
            
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
            
            # Verify checksum on the STORED data first
            if not self._verify_checksum(encrypted_compressed_data, metadata.get('checksum')):
            	raise IntegrityError("Checksum verification failed")
            
            # Now process the data
            value_bytes = encrypted_compressed_data
            	
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
            	
            # Deserialize value
            try:
            	value = self._deserialize_value(value_bytes)
            except SerializationError as e:
            	raise DatabaseError(f"Deserialization failed: {e}")
            return value, metadata
            
        except Exception as e:
        	logger.error(f"Value extraction failed: {e}")
        	raise DatabaseError(f"Value extraction failed: {e}")
    
    def _is_expired(self, metadata: Dict) -> bool:
        """Check if value is expired based on metadata"""
        if metadata.get('ttl') is None:
            return False
        
        current_time = time.time()
        created_time = metadata.get('timestamp', current_time)
        return current_time > created_time + metadata['ttl']
    
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

# Enhanced DatabaseConfig with Merkle options
def enhance_database_config():
    """Add Merkle integrity options to DatabaseConfig"""
    original_fields = DatabaseConfig.__dataclass_fields__.copy()
    
    # Add Merkle-specific fields
    MerkleTreeConfig.__dataclass_fields__.update(original_fields)
    
    # Create enhanced config class
    class EnhancedDatabaseConfig(DatabaseConfig, MerkleTreeConfig):
        pass
    
    return EnhancedDatabaseConfig

# Usage example and quick test
if __name__ == "__main__":
    # Test the enhanced database
    db = AdvancedDatabase("./test_db")
    
    # Store some data
    db.put(b"key1", "value1")
    db.put(b"key2", {"data": "test", "number": 42})
    
    # Retrieve with integrity guarantee
    value = db.get_with_integrity_guarantee(b"key1")
    print(f"Retrieved: {value}")
    
    # Check integrity
    valid, reason = db.verify_data_integrity(b"key1")
    print(f"Integrity: {valid}, {reason}")
    
    # Get integrity root
    root_hash = db.get_integrity_root()
    print(f"Integrity root: {root_hash}")
    
    # Run integrity scan
    scan_results = db.run_integrity_scan()
    print(f"Integrity scan: {scan_results}")
    
    db.close()