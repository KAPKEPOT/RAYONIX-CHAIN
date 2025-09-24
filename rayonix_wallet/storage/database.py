import plyvel
import json
import threading
import time
import hashlib
import struct
from typing import List, Dict, Optional, Any, Iterator, Tuple
from contextlib import contextmanager
from enum import Enum
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

class DatabaseOperation(Enum):
    PUT = "put"
    DELETE = "delete"
    BATCH = "batch"

class DatabaseMetrics:
    """Database performance metrics"""
    def __init__(self):
        self.operations_total = 0
        self.operations_failed = 0
        self.read_latency_avg = 0.0
        self.write_latency_avg = 0.0
        self.batch_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_compaction = 0.0

class CompressionType(Enum):
    NONE = "none"
    SNAPPY = "snappy"
    ZLIB = "zlib"

class WalletDatabase:
    """Production-grade LevelDB wallet storage with advanced features"""
    
    def __init__(self, db_path: str, 
                 compression: CompressionType = CompressionType.SNAPPY,
                 cache_size: int = 512 * 1024 * 1024,  # 512MB
                 write_buffer_size: int = 64 * 1024 * 1024,  # 64MB
                 max_open_files: int = 1000,
                 create_if_missing: bool = True,
                 paranoid_checks: bool = False,
                 read_only: bool = False):
        
        self.db_path = db_path
        self.compression = compression
        self.cache_size = cache_size
        self.write_buffer_size = write_buffer_size
        self.max_open_files = max_open_files
        self.read_only = read_only
        
        # Performance tracking
        self.metrics = DatabaseMetrics()
        self.start_time = time.time()
        
        # Threading and synchronization
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        
        # Cache for frequent operations
        self._address_cache: Dict[str, bytes] = {}
        self._wallet_state_cache: Optional[bytes] = None
        self._cache_max_size = 10000
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Database instance
        self._db: Optional[plyvel.DB] = None
        self._is_open = False
        
        # Key prefixes for organized storage
        self._prefixes = {
            'metadata': b'meta_',
            'addresses': b'addr_',
            'transactions': b'tx_',
            'wallet_state': b'state_',
            'utxos': b'utxo_',
            'indexes': b'idx_',
            'counters': b'cnt_'
        }
        
        self._initialize_database(create_if_missing, paranoid_checks)
    
    def _initialize_database(self, create_if_missing: bool, paranoid_checks: bool):
        """Initialize LevelDB database with optimized settings"""
        try:
            # Create database directory
            import os
            os.makedirs(self.db_path, exist_ok=True)
            
            # Configure LevelDB options
            db_options = plyvel.Options(
                create_if_missing=create_if_missing,
                error_if_exists=False,
                paranoid_checks=paranoid_checks,
                write_buffer_size=self.write_buffer_size,
                max_open_files=self.max_open_files,
                lru_cache_size=self.cache_size,
                compression=self._get_compression_type()
            )
            
            self._db = plyvel.DB(self.db_path, options=db_options, read_only=self.read_only)
            self._is_open = True
            
            # Initialize schema if new database
            if create_if_missing and not self._check_schema_exists():
                self._create_schema()
            
            logger.info(f"LevelDB wallet database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _get_compression_type(self) -> plyvel.CompressionType:
        """Convert compression type to LevelDB format"""
        if self.compression == CompressionType.SNAPPY:
            return plyvel.CompressionType.snappy
        elif self.compression == CompressionType.ZLIB:
            return plyvel.CompressionType.zlib
        else:
            return plyvel.CompressionType.none
    
    def _check_schema_exists(self) -> bool:
        """Check if database schema exists"""
        try:
            schema_version = self._db.get(self._prefixes['metadata'] + b'schema_version')
            return schema_version is not None
        except Exception:
            return False
    
    def _create_schema(self):
        """Create initial database schema"""
        with self._write_batch() as batch:
            # Store schema version
            batch.put(self._prefixes['metadata'] + b'schema_version', b'2.0.0')
            
            # Store creation timestamp
            batch.put(self._prefixes['metadata'] + b'created_at', 
                     struct.pack('d', time.time()))
            
            # Initialize counters
            batch.put(self._prefixes['counters'] + b'address_count', struct.pack('Q', 0))
            batch.put(self._prefixes['counters'] + b'transaction_count', struct.pack('Q', 0))
            batch.put(self._prefixes['counters'] + b'utxo_count', struct.pack('Q', 0))
            
            logger.info("Database schema created successfully")
    
    @contextmanager
    def _write_batch(self, transaction: bool = True) -> Iterator[plyvel.WriteBatch]:
        """Context manager for batch writes with transaction support"""
        if not self._is_open:
            raise RuntimeError("Database is not open")
        
        batch = self._db.write_batch(transaction=transaction)
        
        try:
            yield batch
            batch.write()
            self.metrics.batch_operations += 1
        except Exception as e:
            logger.error(f"Batch write failed: {e}")
            if transaction:
                batch.clear()  # Clear batch on error for transactional safety
            raise
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize Python objects to bytes for storage"""
        if isinstance(value, (int, float)):
            return struct.pack('d', float(value)) if isinstance(value, float) else struct.pack('Q', value)
        elif isinstance(value, str):
            return value.encode('utf-8')
        elif isinstance(value, dict):
            return json.dumps(value, separators=(',', ':')).encode('utf-8')
        elif isinstance(value, list):
            return json.dumps(value, separators=(',', ':')).encode('utf-8')
        else:
            return str(value).encode('utf-8')
    
    def _deserialize_value(self, data: bytes, value_type: type) -> Any:
        """Deserialize bytes back to Python objects"""
        if not data:
            return None
        
        if value_type == int:
            return struct.unpack('Q', data)[0]
        elif value_type == float:
            return struct.unpack('d', data)[0]
        elif value_type == str:
            return data.decode('utf-8')
        elif value_type == dict:
            return json.loads(data.decode('utf-8'))
        elif value_type == list:
            return json.loads(data.decode('utf-8'))
        else:
            return data.decode('utf-8')
    
    def _make_key(self, prefix: bytes, key: str) -> bytes:
        """Create a namespaced key"""
        return prefix + key.encode('utf-8')
    
    def _make_composite_key(self, prefix: bytes, *parts) -> bytes:
        """Create a composite key from multiple parts"""
        key_parts = [str(part).encode('utf-8') for part in parts]
        return prefix + b':'.join(key_parts)
    
    # Address Management
    def save_address(self, address_info: Any):
        """Save address information with caching"""
        start_time = time.time()
        
        try:
            address_key = self._make_key(self._prefixes['addresses'], address_info.address)
            address_data = self._serialize_value(asdict(address_info))
            
            with self._write_batch() as batch:
                batch.put(address_key, address_data)
                
                # Update address counter
                current_count = self._get_counter('address_count')
                batch.put(self._prefixes['counters'] + b'address_count', 
                         struct.pack('Q', current_count + 1))
                
                # Create secondary indexes
                index_key = self._make_composite_key(self._prefixes['indexes'], 
                                                   'addr_by_index', 
                                                   address_info.index,
                                                   address_info.is_change)
                batch.put(index_key, address_info.address.encode('utf-8'))
            
            # Update cache
            self._address_cache[address_info.address] = address_data
            self._maintain_cache_size()
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            
        except Exception as e:
            logger.error(f"Failed to save address {address_info.address}: {e}")
            self.metrics.operations_failed += 1
            raise
    
    def get_address(self, address: str) -> Optional[Any]:
        """Get address information with cache support"""
        start_time = time.time()
        
        try:
            # Check cache first
            if address in self._address_cache:
                self._cache_hits += 1
                cached_data = self._address_cache[address]
                operation_time = (time.time() - start_time) * 1000
                self._update_metrics(DatabaseOperation.PUT, operation_time)
                
                from rayonix_wallet.core.types import AddressInfo
                return AddressInfo(**json.loads(cached_data.decode('utf-8')))
            
            self._cache_misses += 1
            
            address_key = self._make_key(self._prefixes['addresses'], address)
            address_data = self._db.get(address_key)
            
            if address_data:
                # Cache the result
                self._address_cache[address] = address_data
                self._maintain_cache_size()
                
                from rayonix_wallet.core.types import AddressInfo
                operation_time = (time.time() - start_time) * 1000
                self._update_metrics(DatabaseOperation.PUT, operation_time)
                return AddressInfo(**json.loads(address_data.decode('utf-8')))
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get address {address}: {e}")
            self.metrics.operations_failed += 1
            return None
    
    def get_all_addresses(self) -> List[Any]:
        """Get all addresses using efficient iteration"""
        addresses = []
        
        try:
            # Use prefix iterator for efficient scanning
            with self._db.iterator(prefix=self._prefixes['addresses']) as it:
                for key, value in it:
                    try:
                        from rayonix_wallet.core.types import AddressInfo
                        address_data = json.loads(value.decode('utf-8'))
                        addresses.append(AddressInfo(**address_data))
                    except Exception as e:
                        logger.warning(f"Failed to deserialize address data: {e}")
                        continue
            
            # Sort by index and change status
            addresses.sort(key=lambda x: (x.index, x.is_change))
            
        except Exception as e:
            logger.error(f"Failed to get all addresses: {e}")
        
        return addresses
    
    # Transaction Management
    def save_transaction(self, transaction: Any):
        """Save transaction with indexing"""
        start_time = time.time()
        
        try:
            tx_key = self._make_key(self._prefixes['transactions'], transaction.txid)
            tx_data = self._serialize_value(asdict(transaction))
            
            with self._write_batch() as batch:
                batch.put(tx_key, tx_data)
                
                # Update transaction counter
                current_count = self._get_counter('transaction_count')
                batch.put(self._prefixes['counters'] + b'transaction_count', 
                         struct.pack('Q', current_count + 1))
                
                # Create timestamp index for efficient querying
                timestamp_key = self._make_composite_key(self._prefixes['indexes'], 
                                                        'tx_by_time', 
                                                        transaction.timestamp,
                                                        transaction.txid)
                batch.put(timestamp_key, b'')
                
                # Create address indexes
                if hasattr(transaction, 'from_address'):
                    addr_from_key = self._make_composite_key(self._prefixes['indexes'],
                                                           'tx_by_addr',
                                                           transaction.from_address,
                                                           transaction.timestamp,
                                                           transaction.txid)
                    batch.put(addr_from_key, b'')
                
                if hasattr(transaction, 'to_address'):
                    addr_to_key = self._make_composite_key(self._prefixes['indexes'],
                                                         'tx_by_addr',
                                                         transaction.to_address,
                                                         transaction.timestamp,
                                                         transaction.txid)
                    batch.put(addr_to_key, b'')
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            
        except Exception as e:
            logger.error(f"Failed to save transaction {transaction.txid}: {e}")
            self.metrics.operations_failed += 1
            raise
    
    def get_transactions(self, limit: int = 50, offset: int = 0) -> List[Any]:
        """Get transactions with pagination using timestamp index"""
        transactions = []
        
        try:
            # Use reverse iterator for latest transactions first
            index_prefix = self._prefixes['indexes'] + b'tx_by_time'
            
            with self._db.iterator(prefix=index_prefix, reverse=True) as it:
                skipped = 0
                collected = 0
                
                for key, _ in it:
                    if skipped < offset:
                        skipped += 1
                        continue
                    
                    if collected >= limit:
                        break
                    
                    # Extract transaction ID from composite key
                    key_parts = key.split(b':')
                    if len(key_parts) >= 3:
                        txid = key_parts[2].decode('utf-8')
                        tx_key = self._make_key(self._prefixes['transactions'], txid)
                        tx_data = self._db.get(tx_key)
                        
                        if tx_data:
                            from rayonix_wallet.core.types import Transaction
                            tx_dict = json.loads(tx_data.decode('utf-8'))
                            transactions.append(Transaction(**tx_dict))
                            collected += 1
            
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
        
        return transactions
    
    # Wallet State Management
    def save_wallet_state(self, state: Any):
        """Save wallet state with caching"""
        start_time = time.time()
        
        try:
            state_key = self._prefixes['wallet_state'] + b'current'
            state_data = self._serialize_value(asdict(state))
            
            self._db.put(state_key, state_data)
            self._wallet_state_cache = state_data
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            
        except Exception as e:
            logger.error(f"Failed to save wallet state: {e}")
            self.metrics.operations_failed += 1
            raise
    
    def get_wallet_state(self) -> Optional[Any]:
        """Get wallet state with cache support"""
        start_time = time.time()
        
        try:
            if self._wallet_state_cache:
                self._cache_hits += 1
                from rayonix_wallet.core.types import WalletState
                operation_time = (time.time() - start_time) * 1000
                self._update_metrics(DatabaseOperation.PUT, operation_time)
                return WalletState(**json.loads(self._wallet_state_cache.decode('utf-8')))
            
            self._cache_misses += 1
            
            state_key = self._prefixes['wallet_state'] + b'current'
            state_data = self._db.get(state_key)
            
            if state_data:
                self._wallet_state_cache = state_data
                from rayonix_wallet.core.types import WalletState
                operation_time = (time.time() - start_time) * 1000
                self._update_metrics(DatabaseOperation.PUT, operation_time)
                return WalletState(**json.loads(state_data.decode('utf-8')))
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            return None
            
        except Exception as e:
            logger.error(f"Failed to get wallet state: {e}")
            self.metrics.operations_failed += 1
            return None
    
    # UTXO Management
    def save_utxo(self, txid: str, vout: int, address: str, amount: int, 
                 script_pubkey: str, confirmations: int = 0):
        """Save UTXO with efficient storage"""
        start_time = time.time()
        
        try:
            utxo_key = self._make_composite_key(self._prefixes['utxos'], txid, vout)
            utxo_data = {
                'txid': txid,
                'vout': vout,
                'address': address,
                'amount': amount,
                'script_pubkey': script_pubkey,
                'confirmations': confirmations,
                'is_spent': False,
                'spent_txid': None,
                'created_at': time.time()
            }
            
            with self._write_batch() as batch:
                batch.put(utxo_key, self._serialize_value(utxo_data))
                
                # Update UTXO counter
                current_count = self._get_counter('utxo_count')
                batch.put(self._prefixes['counters'] + b'utxo_count', 
                         struct.pack('Q', current_count + 1))
                
                # Create address index for UTXOs
                addr_index_key = self._make_composite_key(self._prefixes['indexes'],
                                                         'utxo_by_addr',
                                                         address,
                                                         txid,
                                                         vout)
                batch.put(addr_index_key, b'')
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            
        except Exception as e:
            logger.error(f"Failed to save UTXO {txid}:{vout}: {e}")
            self.metrics.operations_failed += 1
            raise
    
    def mark_utxo_spent(self, txid: str, vout: int, spent_txid: str):
        """Mark UTXO as spent"""
        start_time = time.time()
        
        try:
            utxo_key = self._make_composite_key(self._prefixes['utxos'], txid, vout)
            utxo_data = self._db.get(utxo_key)
            
            if utxo_data:
                utxo_dict = json.loads(utxo_data.decode('utf-8'))
                utxo_dict['is_spent'] = True
                utxo_dict['spent_txid'] = spent_txid
                
                self._db.put(utxo_key, self._serialize_value(utxo_dict))
                
                # Update UTXO counter
                current_count = self._get_counter('utxo_count')
                if current_count > 0:
                    self._db.put(self._prefixes['counters'] + b'utxo_count', 
                                struct.pack('Q', current_count - 1))
            
            operation_time = (time.time() - start_time) * 1000
            self._update_metrics(DatabaseOperation.PUT, operation_time)
            
        except Exception as e:
            logger.error(f"Failed to mark UTXO spent {txid}:{vout}: {e}")
            self.metrics.operations_failed += 1
            raise
    
    def get_utxos(self, address: Optional[str] = None, unspent_only: bool = True) -> List[Dict]:
        """Get UTXOs with optional filtering"""
        utxos = []
        
        try:
            if address:
                # Use address index for efficient lookup
                index_prefix = self._prefixes['indexes'] + b'utxo_by_addr:' + address.encode('utf-8')
                
                with self._db.iterator(prefix=index_prefix) as it:
                    for key, _ in it:
                        key_parts = key.split(b':')
                        if len(key_parts) >= 4:
                            txid = key_parts[2].decode('utf-8')
                            vout = int(key_parts[3])
                            
                            utxo_key = self._make_composite_key(self._prefixes['utxos'], txid, vout)
                            utxo_data = self._db.get(utxo_key)
                            
                            if utxo_data:
                                utxo_dict = json.loads(utxo_data.decode('utf-8'))
                                if not unspent_only or not utxo_dict.get('is_spent', False):
                                    utxos.append(utxo_dict)
            else:
                # Scan all UTXOs
                with self._db.iterator(prefix=self._prefixes['utxos']) as it:
                    for key, value in it:
                        utxo_dict = json.loads(value.decode('utf-8'))
                        if not unspent_only or not utxo_dict.get('is_spent', False):
                            utxos.append(utxo_dict)
        
        except Exception as e:
            logger.error(f"Failed to get UTXOs: {e}")
        
        return utxos
    
    # Counter Management
    def _get_counter(self, counter_name: str) -> int:
        """Get current counter value"""
        try:
            counter_key = self._prefixes['counters'] + counter_name.encode('utf-8')
            counter_data = self._db.get(counter_key)
            return struct.unpack('Q', counter_data)[0] if counter_data else 0
        except Exception:
            return 0
    
    # Cache Management
    def _maintain_cache_size(self):
        """Maintain cache size within limits"""
        if len(self._address_cache) > self._cache_max_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._address_cache.keys())[:len(self._address_cache) - self._cache_max_size]
            for key in keys_to_remove:
                del self._address_cache[key]
    
    # Performance Monitoring
    def _update_metrics(self, operation: DatabaseOperation, latency: float):
        """Update performance metrics"""
        self.metrics.operations_total += 1
        
        if operation == DatabaseOperation.PUT:
            self.metrics.write_latency_avg = (
                self.metrics.write_latency_avg * (self.metrics.operations_total - 1) + latency
            ) / self.metrics.operations_total
        else:
            self.metrics.read_latency_avg = (
                self.metrics.read_latency_avg * (self.metrics.operations_total - 1) + latency
            ) / self.metrics.operations_total
        
        self.metrics.cache_hits = self._cache_hits
        self.metrics.cache_misses = self._cache_misses
    
    # Database Maintenance
    def compact_database(self):
        """Compact database to reclaim space"""
        try:
            self._db.compact_range()
            self.metrics.last_compaction = time.time()
            logger.info("Database compaction completed")
        except Exception as e:
            logger.error(f"Database compaction failed: {e}")
    
    def backup(self, backup_path: str):
        """Create database backup using LevelDB's native backup"""
        try:
            import shutil
            import os
            
            # Create backup directory
            os.makedirs(backup_path, exist_ok=True)
            
            # Use LevelDB's backup functionality
            # Note: This is a simplified backup - for production, use proper LevelDB backup tools
            backup_db_path = os.path.join(backup_path, 'wallet_backup')
            
            # Close current database
            self.close()
            
            # Copy database files
            shutil.copytree(self.db_path, backup_db_path)
            
            # Reopen database
            self._initialize_database(create_if_missing=False, paranoid_checks=False)
            
            logger.info(f"Database backup created at {backup_db_path}")
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            # Try to reopen database on failure
            try:
                self._initialize_database(create_if_missing=False, paranoid_checks=False)
            except Exception:
                logger.error("Failed to reopen database after backup failure")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            'database_path': self.db_path,
            'is_open': self._is_open,
            'read_only': self.read_only,
            'uptime_seconds': time.time() - self.start_time,
            'address_count': self._get_counter('address_count'),
            'transaction_count': self._get_counter('transaction_count'),
            'utxo_count': self._get_counter('utxo_count'),
            'cache_size': len(self._address_cache),
            'cache_hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) 
                if (self._cache_hits + self._cache_misses) > 0 else 0,
            'metrics': asdict(self.metrics)
        }
        
        # Add LevelDB property statistics if available
        try:
            if self._is_open:
                stats['leveldb_stats'] = self._db.get_property('leveldb.stats')
        except Exception:
            pass
        
        return stats
    
    def close(self):
        """Close database connections and cleanup"""
        with self._lock:
            try:
                if self._db and self._is_open:
                    self._db.close()
                    self._is_open = False
                
                # Clear caches
                self._address_cache.clear()
                self._wallet_state_cache = None
                
                logger.info("Database closed successfully")
                
            except Exception as e:
                logger.error(f"Error closing database: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()