# consensus/utils/database.py
import plyvel
import pickle
import json
from typing import Dict, List, Optional, Any, Union
import logging
import os
import shutil
from threading import RLock

logger = logging.getLogger('DatabaseManager')

class DatabaseManager:
    """Production-ready database management for consensus state persistence"""
    
    def __init__(self, db_path: str = './rayonix_data/consensus_db', backup_enabled: bool = True):
        """
        Initialize database manager
        
        Args:
            db_path: Path to database directory
            backup_enabled: Whether to enable automatic backups
        """
        self.db_path = db_path
        self.backup_enabled = backup_enabled
        self.backup_path = os.path.join(db_path, 'backups')
        self.lock = RLock()
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        if backup_enabled:
            os.makedirs(self.backup_path, exist_ok=True)
        
        # Open database
        self.db = self._open_database()
        
        # Initialize collections
        self._initialize_collections()
    
    def _open_database(self) -> plyvel.DB:
        """Open LevelDB database with production settings"""
        try:
            db = plyvel.DB(
                self.db_path,
                create_if_missing=True,
                error_if_exists=False,
                paranoid_checks=True,
                write_buffer_size=64 * 1024 * 1024,  # 64MB
                max_open_files=1000,
                lru_cache_size=128 * 1024 * 1024,  # 128MB
                bloom_filter_bits=10,
                compression='snappy'
            )
            
            logger.info(f"Opened database at {self.db_path}")
            return db
            
        except Exception as e:
            logger.error(f"Error opening database: {e}")
            raise
    
    def _initialize_collections(self):
        """Initialize database collections (key prefixes)"""
        self.collections = {
            'consensus_state': b'cs_',
            'validators': b'val_',
            'blocks': b'blk_',
            'votes': b'vote_',
            'transactions': b'tx_',
            'evidence': b'ev_',
            'config': b'cfg_',
            'metadata': b'meta_'
        }
    
    def save_consensus_state(self, state_data: Dict[str, Any]) -> bool:
        """Save consensus state to database"""
        with self.lock:
            try:
                # Use transaction for atomic write
                with self.db.write_batch(transaction=True) as batch:
                    # Save individual state fields
                    batch.put(b'height', self._serialize_value(state_data.get('height', 0)))
                    batch.put(b'round', self._serialize_value(state_data.get('round', 0)))
                    batch.put(b'step', self._serialize_value(state_data.get('step', 0)))
                    batch.put(b'locked_round', self._serialize_value(state_data.get('locked_round', -1)))
                    batch.put(b'valid_round', self._serialize_value(state_data.get('valid_round', -1)))
                    
                    if state_data.get('locked_value'):
                        batch.put(b'locked_value', self._serialize_value(state_data['locked_value']))
                    
                    if state_data.get('valid_value'):
                        batch.put(b'valid_value', self._serialize_value(state_data['valid_value']))
                    
                    # Save complete state for redundancy
                    batch.put(
                        self._get_key('consensus_state', 'current'),
                        self._serialize_value(state_data)
                    )
                
                logger.debug("Saved consensus state to database")
                return True
                
            except Exception as e:
                logger.error(f"Error saving consensus state: {e}")
                return False
    
    def load_consensus_state(self) -> Optional[Dict[str, Any]]:
        """Load consensus state from database"""
        with self.lock:
            try:
                # Try to load complete state first
                complete_state_data = self.db.get(self._get_key('consensus_state', 'current'))
                if complete_state_data:
                    return self._deserialize_value(complete_state_data)
                
                # Fall back to loading individual fields
                state_data = {}
                
                height_data = self.db.get(b'height')
                if height_data:
                    state_data['height'] = self._deserialize_value(height_data)
                
                round_data = self.db.get(b'round')
                if round_data:
                    state_data['round'] = self._deserialize_value(round_data)
                
                step_data = self.db.get(b'step')
                if step_data:
                    state_data['step'] = self._deserialize_value(step_data)
                
                locked_round_data = self.db.get(b'locked_round')
                if locked_round_data:
                    state_data['locked_round'] = self._deserialize_value(locked_round_data)
                
                valid_round_data = self.db.get(b'valid_round')
                if valid_round_data:
                    state_data['valid_round'] = self._deserialize_value(valid_round_data)
                
                locked_value_data = self.db.get(b'locked_value')
                if locked_value_data:
                    state_data['locked_value'] = self._deserialize_value(locked_value_data)
                
                valid_value_data = self.db.get(b'valid_value')
                if valid_value_data:
                    state_data['valid_value'] = self._deserialize_value(valid_value_data)
                
                return state_data if state_data else None
                
            except Exception as e:
                logger.error(f"Error loading consensus state: {e}")
                return None
    
    def save_validators(self, validators_data: Dict[str, Any]) -> bool:
        """Save validators to database"""
        with self.lock:
            try:
                with self.db.write_batch(transaction=True) as batch:
                    # Save each validator individually
                    for address, validator_data in validators_data.items():
                        batch.put(
                            self._get_key('validators', address),
                            self._serialize_value(validator_data)
                        )
                    
                    # Save complete validator set
                    batch.put(
                        self._get_key('metadata', 'validators_complete'),
                        self._serialize_value(validators_data)
                    )
                
                logger.debug(f"Saved {len(validators_data)} validators to database")
                return True
                
            except Exception as e:
                logger.error(f"Error saving validators: {e}")
                return False
    
    def load_validators(self) -> Dict[str, Any]:
        """Load validators from database"""
        with self.lock:
            try:
                # Try to load complete validator set first
                complete_data = self.db.get(self._get_key('metadata', 'validators_complete'))
                if complete_data:
                    return self._deserialize_value(complete_data)
                
                # Fall back to loading individual validators
                validators = {}
                prefix = self.collections['validators']
                
                for key, value in self.db.iterator(prefix=prefix):
                    validator_address = key[len(prefix):].decode('utf-8')
                    validators[validator_address] = self._deserialize_value(value)
                
                logger.debug(f"Loaded {len(validators)} validators from database")
                return validators
                
            except Exception as e:
                logger.error(f"Error loading validators: {e}")
                return {}
    
    def save_active_validators(self, active_validators: List[Dict]) -> bool:
        """Save active validators list to database"""
        with self.lock:
            try:
                self.db.put(
                    self._get_key('metadata', 'active_validators'),
                    self._serialize_value(active_validators)
                )
                
                logger.debug(f"Saved {len(active_validators)} active validators to database")
                return True
                
            except Exception as e:
                logger.error(f"Error saving active validators: {e}")
                return False
    
    def load_active_validators(self) -> List[Dict]:
        """Load active validators list from database"""
        with self.lock:
            try:
                data = self.db.get(self._get_key('metadata', 'active_validators'))
                if data:
                    return self._deserialize_value(data)
                return []
                
            except Exception as e:
                logger.error(f"Error loading active validators: {e}")
                return []
    
    def save_block(self, block_data: Dict) -> bool:
        """Save block to database"""
        with self.lock:
            try:
                block_hash = block_data.get('hash')
                if not block_hash:
                    logger.warning("Block data missing hash")
                    return False
                
                self.db.put(
                    self._get_key('blocks', block_hash),
                    self._serialize_value(block_data)
                )
                
                # Also store by height for indexing
                height = block_data.get('height')
                if height is not None:
                    self.db.put(
                        self._get_key('blocks', f"height_{height}"),
                        self._serialize_value(block_hash)
                    )
                
                logger.debug(f"Saved block {block_hash} to database")
                return True
                
            except Exception as e:
                logger.error(f"Error saving block: {e}")
                return False
    
    def load_block(self, block_hash: str) -> Optional[Dict]:
        """Load block by hash from database"""
        with self.lock:
            try:
                data = self.db.get(self._get_key('blocks', block_hash))
                if data:
                    return self._deserialize_value(data)
                return None
                
            except Exception as e:
                logger.error(f"Error loading block {block_hash}: {e}")
                return None
    
    def load_block_by_height(self, height: int) -> Optional[Dict]:
        """Load block by height from database"""
        with self.lock:
            try:
                # Get block hash for height
                hash_data = self.db.get(self._get_key('blocks', f"height_{height}"))
                if not hash_data:
                    return None
                
                block_hash = self._deserialize_value(hash_data)
                return self.load_block(block_hash)
                
            except Exception as e:
                logger.error(f"Error loading block at height {height}: {e}")
                return None
    
    def save_vote(self, vote_data: Dict) -> bool:
        """Save vote to database"""
        with self.lock:
            try:
                vote_key = f"{vote_data['height']}_{vote_data['round']}_{vote_data['validator']}"
                self.db.put(
                    self._get_key('votes', vote_key),
                    self._serialize_value(vote_data)
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Error saving vote: {e}")
                return False
    
    def load_votes(self, height: int, round: int) -> List[Dict]:
        """Load votes for specific height and round"""
        with self.lock:
            try:
                votes = []
                prefix = self.collections['votes']
                
                for key, value in self.db.iterator(prefix=prefix):
                    key_str = key.decode('utf-8')
                    if f"{height}_{round}" in key_str:
                        votes.append(self._deserialize_value(value))
                
                return votes
                
            except Exception as e:
                logger.error(f"Error loading votes for height {height}, round {round}: {e}")
                return []
    
    def save_evidence(self, evidence_data: Dict) -> bool:
        """Save evidence to database"""
        with self.lock:
            try:
                evidence_hash = self._calculate_data_hash(evidence_data)
                self.db.put(
                    self._get_key('evidence', evidence_hash),
                    self._serialize_value(evidence_data)
                )
                
                return True
                
            except Exception as e:
                logger.error(f"Error saving evidence: {e}")
                return False
    
    def load_evidence(self, evidence_hash: str) -> Optional[Dict]:
        """Load evidence by hash from database"""
        with self.lock:
            try:
                data = self.db.get(self._get_key('evidence', evidence_hash))
                if data:
                    return self._deserialize_value(data)
                return None
                
            except Exception as e:
                logger.error(f"Error loading evidence {evidence_hash}: {e}")
                return None
    
    def _get_key(self, collection: str, key: str) -> bytes:
        """Get full database key for collection and key"""
        prefix = self.collections.get(collection, b'')
        return prefix + key.encode('utf-8')
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage"""
        try:
            # Use pickle for complex objects, JSON for simple ones
            if isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
                return json.dumps(value).encode('utf-8')
            else:
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage"""
        try:
            # Try JSON first, then pickle
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing value: {e}")
            raise
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of data for deduplication"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def create_backup(self, backup_name: str = None) -> str:
        """Create database backup"""
        if not self.backup_enabled:
            return ""
        
        with self.lock:
            try:
                if backup_name is None:
                    import time
                    backup_name = f"backup_{int(time.time())}"
                
                backup_dir = os.path.join(self.backup_path, backup_name)
                
                # Close database before backup
                self.db.close()
                
                # Copy database files
                shutil.copytree(self.db_path, backup_dir)
                
                # Reopen database
                self.db = self._open_database()
                
                logger.info(f"Created database backup: {backup_name}")
                return backup_dir
                
            except Exception as e:
                logger.error(f"Error creating database backup: {e}")
                # Try to reopen database on error
                try:
                    self.db = self._open_database()
                except Exception:
                    logger.critical("Failed to reopen database after backup error")
                return ""
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore database from backup"""
        with self.lock:
            try:
                backup_dir = os.path.join(self.backup_path, backup_name)
                if not os.path.exists(backup_dir):
                    logger.error(f"Backup not found: {backup_name}")
                    return False
                
                # Close database
                self.db.close()
                
                # Remove current database
                shutil.rmtree(self.db_path)
                
                # Restore from backup
                shutil.copytree(backup_dir, self.db_path)
                
                # Reopen database
                self.db = self._open_database()
                
                logger.info(f"Restored database from backup: {backup_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error restoring database backup: {e}")
                # Try to reopen database on error
                try:
                    self.db = self._open_database()
                except Exception:
                    logger.critical("Failed to reopen database after restore error")
                return False
    
    def compact_database(self) -> bool:
        """Compact database to reduce size"""
        with self.lock:
            try:
                self.db.compact_range()
                logger.info("Database compaction completed")
                return True
            except Exception as e:
                logger.error(f"Error compacting database: {e}")
                return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.lock:
            try:
                stats = {}
                
                # Get approximate sizes
                stats['approximate_size'] = self.get_database_size()
                
                # Count entries in each collection
                for collection, prefix in self.collections.items():
                    count = 0
                    for _ in self.db.iterator(prefix=prefix):
                        count += 1
                    stats[f'{collection}_count'] = count
                
                return stats
                
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                return {}
    
    def get_database_size(self) -> int:
        """Get approximate database size in bytes"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return total_size
        except Exception as e:
            logger.error(f"Error calculating database size: {e}")
            return 0
    
    def close(self):
        """Close database connection"""
        with self.lock:
            try:
                if hasattr(self, 'db') and self.db:
                    self.db.close()
                    logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")