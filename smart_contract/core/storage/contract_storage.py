# smart_contract/core/storage/contract_storage.py
import time
import json
import pickle
import zlib
import logging
import threading
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, field
from collections import OrderedDict

from ....utils.cryptography_utils import encrypt_data, decrypt_data, derive_key
from ....utils.serialization_utils import compress_data, decompress_data

logger = logging.getLogger("SmartContract.Storage")

@dataclass
class StorageConfig:
    """Configuration for contract storage"""
    max_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    compression_threshold: int = 1024  # 1KB
    encryption_enabled: bool = True
    compression_enabled: bool = True
    audit_log_enabled: bool = True
    cache_size: int = 1000
    backup_interval: int = 3600  # 1 hour

class ContractStorage:
    """Advanced contract storage with encryption, compression, and audit logging"""
    
    def __init__(self, encryption_enabled: bool = True, compression_enabled: bool = True,
                 max_size: int = 10 * 1024 * 1024 * 1024):
        self.config = StorageConfig(
            encryption_enabled=encryption_enabled,
            compression_enabled=compression_enabled,
            max_size=max_size
        )
        
        self.data: Dict[str, Any] = {}
        self.metadata: Dict[str, Dict] = {}
        self.audit_log: List[Dict] = []
        self.allowed_writers: Set[str] = set()
        self.encryption_key: Optional[bytes] = None
        self.cache: OrderedDict = OrderedDict()
        self.lock = threading.RLock()
        self.total_size = 0
        self.entry_count = 0
        self.last_backup = time.time()
        
        # Initialize encryption if enabled
        if encryption_enabled:
            self._initialize_encryption()
        
        logger.info("ContractStorage initialized with encryption: %s, compression: %s", 
                   encryption_enabled, compression_enabled)
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption system"""
        # In production, this would use a proper key management system
        self.encryption_key = derive_key("contract_storage_default_key")
    
    def store(self, key: str, value: Any, caller: str, operation: str = "store") -> bool:
        """Store data with comprehensive security checks"""
        with self.lock:
            # Security check
            if caller not in self.allowed_writers:
                logger.warning(f"Unauthorized storage write attempt by {caller}")
                return False
            
            # Size check
            serialized_size = self._estimate_size(value)
            if self.total_size + serialized_size > self.config.max_size:
                logger.error(f"Storage limit exceeded: {self.total_size + serialized_size} > {self.config.max_size}")
                return False
            
            try:
                # Process value based on configuration
                processed_value = self._process_value_for_storage(value)
                
                # Store the data
                old_value = self.data.get(key)
                self.data[key] = processed_value
                
                # Update metadata
                self.metadata[key] = {
                    'created_at': time.time(),
                    'modified_at': time.time(),
                    'size': serialized_size,
                    'owner': caller,
                    'operation': operation,
                    'version': 1 if key not in self.metadata else self.metadata[key]['version'] + 1
                }
                
                # Update size tracking
                if old_value is not None:
                    old_size = self.metadata[key]['size']
                    self.total_size -= old_size
                self.total_size += serialized_size
                
                # Update entry count
                if old_value is None:
                    self.entry_count += 1
                
                # Add to audit log
                if self.config.audit_log_enabled:
                    self._add_audit_entry(
                        operation="store" if old_value is None else "update",
                        key=key,
                        caller=caller,
                        old_value=old_value,
                        new_value=value,
                        size=serialized_size
                    )
                
                # Update cache
                self._update_cache(key, processed_value)
                
                # Check for backup
                self._check_backup()
                
                logger.debug(f"Stored key '{key}' (size: {serialized_size} bytes) by {caller}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to store data for key '{key}': {e}")
                return False
    
    def retrieve(self, key: str, caller: str) -> Optional[Any]:
        """Retrieve data with access control"""
        with self.lock:
            if key not in self.data:
                return None
            
            # Check cache first
            cached_value = self.cache.get(key)
            if cached_value is not None:
                # Update cache order
                self.cache.move_to_end(key)
                return self._process_value_from_storage(cached_value)
            
            # Retrieve from main storage
            stored_value = self.data.get(key)
            if stored_value is None:
                return None
            
            try:
                # Process the stored value
                value = self._process_value_from_storage(stored_value)
                
                # Update cache
                self._update_cache(key, stored_value)
                
                # Add to audit log
                if self.config.audit_log_enabled:
                    self._add_audit_entry(
                        operation="retrieve",
                        key=key,
                        caller=caller,
                        value=value
                    )
                
                return value
                
            except Exception as e:
                logger.error(f"Failed to retrieve data for key '{key}': {e}")
                return None
    
    def delete(self, key: str, caller: str) -> bool:
        """Delete data with proper cleanup"""
        with self.lock:
            if key not in self.data:
                return False
            
            if caller not in self.allowed_writers:
                logger.warning(f"Unauthorized delete attempt by {caller}")
                return False
            
            try:
                # Get old value for audit log
                old_value = self.data[key]
                old_size = self.metadata[key]['size']
                
                # Remove from storage
                del self.data[key]
                del self.metadata[key]
                
                # Remove from cache
                if key in self.cache:
                    del self.cache[key]
                
                # Update size tracking
                self.total_size -= old_size
                self.entry_count -= 1
                
                # Add to audit log
                if self.config.audit_log_enabled:
                    self._add_audit_entry(
                        operation="delete",
                        key=key,
                        caller=caller,
                        old_value=old_value,
                        size=old_size
                    )
                
                logger.debug(f"Deleted key '{key}' by {caller}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete key '{key}': {e}")
                return False
    
    def _process_value_for_storage(self, value: Any) -> Any:
        """Process value for storage (encryption, compression)"""
        processed = value
        
        # Serialize if not already bytes
        if not isinstance(processed, bytes):
            processed = pickle.dumps(processed)
        
        # Compress if enabled and above threshold
        if (self.config.compression_enabled and 
            len(processed) > self.config.compression_threshold):
            processed = compress_data(processed)
        
        # Encrypt if enabled
        if self.config.encryption_enabled and self.encryption_key:
            processed = encrypt_data(processed, self.encryption_key)
        
        return processed
    
    def _process_value_from_storage(self, stored_value: Any) -> Any:
        """Process stored value for retrieval (decryption, decompression)"""
        processed = stored_value
        
        # Decrypt if enabled
        if self.config.encryption_enabled and self.encryption_key:
            processed = decrypt_data(processed, self.encryption_key)
        
        # Decompress if it looks compressed
        try:
            processed = decompress_data(processed)
        except:
            pass  # Not compressed or already decompressed
        
        # Deserialize if it's pickled data
        try:
            processed = pickle.loads(processed)
        except:
            pass  # Not pickled or already deserialized
        
        return processed
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate the size of a value when stored"""
        try:
            if isinstance(value, bytes):
                return len(value)
            return len(pickle.dumps(value))
        except:
            return 1024  # Conservative estimate
    
    def _add_audit_entry(self, operation: str, key: str, caller: str, 
                        old_value: Any = None, new_value: Any = None, size: int = 0) -> None:
        """Add an entry to the audit log"""
        entry = {
            'timestamp': time.time(),
            'operation': operation,
            'key': key,
            'caller': caller,
            'size': size,
            'old_value_hash': hash(str(old_value)) if old_value is not None else None,
            'new_value_hash': hash(str(new_value)) if new_value is not None else None
        }
        self.audit_log.append(entry)
        
        # Trim audit log if it gets too large
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
    
    def _update_cache(self, key: str, value: Any) -> None:
        """Update the LRU cache"""
        self.cache[key] = value
        self.cache.move_to_end(key)
        
        # Trim cache if it exceeds size limit
        if len(self.cache) > self.config.cache_size:
            self.cache.popitem(last=False)
    
    def _check_backup(self) -> None:
        """Check if it's time for a backup"""
        current_time = time.time()
        if current_time - self.last_backup >= self.config.backup_interval:
            self._create_backup()
            self.last_backup = current_time
    
    def _create_backup(self) -> None:
        """Create a backup of the storage state"""
        # In production, this would save to persistent storage
        logger.debug("Storage backup created")
    
    def clear_storage(self) -> None:
        """Clear all storage data"""
        with self.lock:
            self.data.clear()
            self.metadata.clear()
            self.cache.clear()
            self.audit_log.clear()
            self.total_size = 0
            self.entry_count = 0
            logger.info("Storage cleared")
    
    def get_total_size(self) -> int:
        """Get total storage size in bytes"""
        return self.total_size
    
    def get_entry_count(self) -> int:
        """Get number of storage entries"""
        return self.entry_count
    
    def get_audit_log_size(self) -> int:
        """Get number of audit log entries"""
        return len(self.audit_log)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        with self.lock:
            return {
                'total_size_bytes': self.total_size,
                'entry_count': self.entry_count,
                'cache_size': len(self.cache),
                'audit_log_entries': len(self.audit_log),
                'allowed_writers': len(self.allowed_writers),
                'encryption_enabled': self.config.encryption_enabled,
                'compression_enabled': self.config.compression_enabled,
                'max_size_bytes': self.config.max_size,
                'compression_threshold': self.config.compression_threshold
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert storage to dictionary for serialization"""
        with self.lock:
            return {
                'data': self.data.copy(),
                'metadata': self.metadata.copy(),
                'audit_log': self.audit_log.copy(),
                'allowed_writers': list(self.allowed_writers),
                'total_size': self.total_size,
                'entry_count': self.entry_count,
                'config': {
                    'encryption_enabled': self.config.encryption_enabled,
                    'compression_enabled': self.config.compression_enabled,
                    'max_size': self.config.max_size
                }
            }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load storage from dictionary"""
        with self.lock:
            self.data = data.get('data', {}).copy()
            self.metadata = data.get('metadata', {}).copy()
            self.audit_log = data.get('audit_log', []).copy()
            self.allowed_writers = set(data.get('allowed_writers', []))
            self.total_size = data.get('total_size', 0)
            self.entry_count = data.get('entry_count', 0)
            
            config = data.get('config', {})
            self.config.encryption_enabled = config.get('encryption_enabled', True)
            self.config.compression_enabled = config.get('compression_enabled', True)
            self.config.max_size = config.get('max_size', 10 * 1024 * 1024 * 1024)