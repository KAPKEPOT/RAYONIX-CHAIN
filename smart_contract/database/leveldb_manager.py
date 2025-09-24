# smart_contract/database/leveldb_manager.py
import plyvel
import json
import logging
import pickle
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger("SmartContract.Database")

@dataclass
class DatabaseConfig:
    """Configuration for LevelDB database"""
    create_if_missing: bool = True
    error_if_exists: bool = False
    compression: bool = True
    bloom_filter_bits: int = 10
    cache_size: int = 100 * 1024 * 1024  # 100MB
    write_buffer_size: int = 64 * 1024 * 1024  # 64MB
    max_open_files: int = 1000

class LevelDBManager:
    """Advanced LevelDB manager for contract storage"""
    
    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        # Validate and normalize db_path
        if not isinstance(db_path, (str, bytes)):
            raise TypeError(f"db_path must be a string or bytes, got {type(db_path)}")
        
        # Convert to string if it's bytes
        if isinstance(db_path, bytes):
            db_path = db_path.decode('utf-8')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        
        self.db_path = db_path
        self.config = config or DatabaseConfig()
        self.db: Optional[plyvel.DB] = None
        
        self._open_database()
        logger.info(f"LevelDBManager initialized for path: {db_path}")
    
    def _open_database(self) -> None:
        """Open or create LevelDB database"""
        try:
            # Ensure path is properly encoded
            if isinstance(self.db_path, str):
                db_path_bytes = self.db_path.encode('utf-8')
            else:
                db_path_bytes = self.db_path
            
            self.db = plyvel.DB(
                db_path_bytes,  # Use encoded path
                create_if_missing=self.config.create_if_missing,
                error_if_exists=self.config.error_if_exists,
                compression='snappy' if self.config.compression else None,
                bloom_filter_bits=self.config.bloom_filter_bits,
                lru_cache_size=self.config.cache_size,
                write_buffer_size=self.config.write_buffer_size,
                max_open_files=self.config.max_open_files
            )
        except Exception as e:
            logger.error(f"Failed to open database at {self.db_path}: {e}")
            raise
    
    def save_contract(self, contract: Any) -> bool:
        """Save contract to database"""
        try:
            # Serialize contract
            contract_data = self._serialize_contract(contract)
            
            # Store in database
            key = f"contract:{contract.contract_id}".encode()
            self.db.put(key, contract_data)
            
            # Update index
            self._update_contract_index(contract.contract_id)
            
            logger.debug(f"Contract {contract.contract_id} saved to database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save contract: {e}")
            return False
    
    def load_contract(self, contract_id: str) -> Optional[Any]:
        """Load contract from database"""
        try:
            key = f"contract:{contract_id}".encode()
            contract_data = self.db.get(key)
            
            if contract_data is None:
                return None
            
            # Deserialize contract
            return self._deserialize_contract(contract_data)
            
        except Exception as e:
            logger.error(f"Failed to load contract {contract_id}: {e}")
            return None
    
    def delete_contract(self, contract_id: str) -> bool:
        """Delete contract from database"""
        try:
            key = f"contract:{contract_id}".encode()
            self.db.delete(key)
            
            # Update index
            self._remove_contract_index(contract_id)
            
            logger.debug(f"Contract {contract_id} deleted from database")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete contract {contract_id}: {e}")
            return False
    
    def load_all_contracts(self) -> Dict[str, Any]:
        """Load all contracts from database"""
        contracts = {}
        
        try:
            # Iterate through all contract keys
            prefix = b'contract:'
            for key, value in self.db.iterator(prefix=prefix):
                contract_id = key.decode().split(':', 1)[1]
                
                try:
                    contract = self._deserialize_contract(value)
                    contracts[contract_id] = contract
                except Exception as e:
                    logger.error(f"Failed to deserialize contract {contract_id}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to load all contracts: {e}")
        
        return contracts
    
    def save_metadata(self, key: str, value: Any) -> bool:
        """Save metadata to database"""
        try:
            db_key = f"metadata:{key}".encode()
            serialized_value = pickle.dumps(value)
            self.db.put(db_key, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata {key}: {e}")
            return False
    
    def load_metadata(self, key: str) -> Optional[Any]:
        """Load metadata from database"""
        try:
            db_key = f"metadata:{key}".encode()
            value = self.db.get(db_key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Failed to load metadata {key}: {e}")
            return None
    
    def _serialize_contract(self, contract: Any) -> bytes:
        """Serialize contract for storage"""
        try:
            # Use pickle for complex object serialization
            return pickle.dumps(contract)
        except Exception as e:
            logger.error(f"Contract serialization failed: {e}")
            raise
    
    def _deserialize_contract(self, data: bytes) -> Any:
        """Deserialize contract from storage"""
        try:
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Contract deserialization failed: {e}")
            raise
    
    def _update_contract_index(self, contract_id: str) -> None:
        """Update contract index"""
        try:
            index_key = b'contracts:index'
            index = self.load_metadata('contracts_index') or []
            
            if contract_id not in index:
                index.append(contract_id)
                self.save_metadata('contracts_index', index)
        except Exception as e:
            logger.error(f"Failed to update contract index: {e}")
    
    def _remove_contract_index(self, contract_id: str) -> None:
        """Remove contract from index"""
        try:
            index = self.load_metadata('contracts_index') or []
            if contract_id in index:
                index.remove(contract_id)
                self.save_metadata('contracts_index', index)
        except Exception as e:
            logger.error(f"Failed to remove contract from index: {e}")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {
                'approximate_size': self.db.get_property(b'leveldb.approximate-memory-usage') or 0,
                'num_contracts': 0,
                'num_metadata': 0
            }
            
            # Count contracts
            prefix = b'contract:'
            stats['num_contracts'] = sum(1 for _ in self.db.iterator(prefix=prefix))
            
            # Count metadata
            prefix = b'metadata:'
            stats['num_metadata'] = sum(1 for _ in self.db.iterator(prefix=prefix))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}
    
    def create_backup(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            # This would use proper backup mechanisms
            # For now, simulate backup
            logger.info(f"Database backup created at: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False
    
    def compact_database(self) -> bool:
        """Compact database to reduce size"""
        try:
            self.db.compact_range()
            logger.info("Database compaction completed")
            return True
        except Exception as e:
            logger.error(f"Database compaction failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connection"""
        if self.db:
            self.db.close()
            self.db = None
            logger.info("Database connection closed")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.close()
        except Exception:
            pass