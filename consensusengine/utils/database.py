# consensus/utils/database.py
import logging
from typing import Dict, List, Optional, Any, Union
from database.core.database import AdvancedDatabase
from database.utils.types import DatabaseConfig, CompressionType, EncryptionType
from database.utils.exceptions import DatabaseError, KeyNotFoundError

logger = logging.getLogger('ConsensusDatabase')

class ConsensusDatabase:
    """
    Thin wrapper around AdvancedDatabase for consensus-specific data.
    Handles ONLY consensus data modeling and business logic.
    """
    
    def __init__(self, db_path: str = './consensus_db', config: DatabaseConfig = None):
        """
        Initialize consensus database wrapper
        
        Args:
            db_path: Path to database directory
            config: Advanced database configuration (optional)
        """
        # Use optimized config for consensus data if not provided
        if config is None:
            config = DatabaseConfig(
                compression=CompressionType.SNAPPY,  # Good for structured data
                encryption=EncryptionType.AES256,    # Secure by default
                merkle_integrity=True,               # Critical for consensus
                cache_size=256 * 1024 * 1024,        # 256MB cache
                max_cache_size=10000,                # 10K entries
                create_if_missing=True
            )
        
        # Initialize the advanced database engine
        self.storage = AdvancedDatabase(db_path, config)
        
        # Consensus-specific key prefixes
        self.key_prefixes = {
            'consensus_state': b'cs_',
            'validators': b'val_',
            'blocks': b'blk_',
            'votes': b'vote_',
            'transactions': b'tx_',
            'evidence': b'ev_',
            'config': b'cfg_',
            'metadata': b'meta_'
        }
        
        logger.info(f"Consensus database initialized at {db_path}")
    
    # === CONSENSUS STATE METHODS ===
    
    def save_consensus_state(self, state_data: Dict[str, Any]) -> bool:
        """Save consensus state to database"""
        try:
            # Use integrity-guaranteed storage
            return self.storage.put(
                self._get_key('consensus_state', 'current'),
                state_data,
                verify_integrity=True
            )
        except DatabaseError as e:
            logger.error(f"Error saving consensus state: {e}")
            return False
    
    def load_consensus_state(self) -> Optional[Dict[str, Any]]:
        """Load consensus state from database with integrity guarantee"""
        try:
            return self.storage.get_with_integrity_guarantee(
                self._get_key('consensus_state', 'current')
            )
        except (KeyNotFoundError, DatabaseError):
            return None
    
    # === VALIDATOR MANAGEMENT ===
    
    def save_validators(self, validators_data: Dict[str, Any]) -> bool:
        """Save validators to database"""
        try:
            # Store complete validator set
            success = self.storage.put(
                self._get_key('metadata', 'validators_complete'),
                validators_data,
                verify_integrity=True
            )
            
            if success:
                logger.debug(f"Saved {len(validators_data)} validators to database")
            
            return success
            
        except DatabaseError as e:
            logger.error(f"Error saving validators: {e}")
            return False
    
    def load_validators(self) -> Dict[str, Any]:
        """Load validators from database"""
        try:
            return self.storage.get_with_integrity_guarantee(
                self._get_key('metadata', 'validators_complete')
            ) or {}
        except (KeyNotFoundError, DatabaseError):
            return {}
    
    def save_active_validators(self, active_validators: List[Dict]) -> bool:
        """Save active validators list"""
        try:
            return self.storage.put(
                self._get_key('metadata', 'active_validators'),
                active_validators,
                verify_integrity=True
            )
        except DatabaseError as e:
            logger.error(f"Error saving active validators: {e}")
            return False
    
    def load_active_validators(self) -> List[Dict]:
        """Load active validators list"""
        try:
            return self.storage.get_with_integrity_guarantee(
                self._get_key('metadata', 'active_validators')
            ) or []
        except (KeyNotFoundError, DatabaseError):
            return []
    
    # === BLOCK MANAGEMENT ===
    
    def save_block(self, block_data: Dict) -> bool:
        """Save block to database"""
        try:
            block_hash = block_data.get('hash')
            if not block_hash:
                logger.warning("Block data missing hash")
                return False
            
            # Store block by hash
            success = self.storage.put(
                self._get_key('blocks', block_hash),
                block_data,
                verify_integrity=True
            )
            
            if success:
                # Also index by height for fast lookup
                height = block_data.get('height')
                if height is not None:
                    self.storage.put(
                        self._get_key('blocks', f"height_{height}"),
                        block_hash,
                        verify_integrity=False  # Hash doesn't need integrity
                    )
                
                logger.debug(f"Saved block {block_hash} to database")
            
            return success
            
        except DatabaseError as e:
            logger.error(f"Error saving block: {e}")
            return False
    
    def load_block(self, block_hash: str) -> Optional[Dict]:
        """Load block by hash"""
        try:
            return self.storage.get_with_integrity_guarantee(
                self._get_key('blocks', block_hash)
            )
        except (KeyNotFoundError, DatabaseError):
            return None
    
    def load_block_by_height(self, height: int) -> Optional[Dict]:
        """Load block by height"""
        try:
            # Get block hash for height
            block_hash = self.storage.get(
                self._get_key('blocks', f"height_{height}"),
                verify_integrity=False
            )
            
            if block_hash:
                return self.load_block(block_hash)
            return None
            
        except (KeyNotFoundError, DatabaseError):
            return None
    
    # === VOTE MANAGEMENT ===
    
    def save_vote(self, vote_data: Dict) -> bool:
        """Save vote to database"""
        try:
            vote_key = f"{vote_data['height']}_{vote_data['round']}_{vote_data['validator']}"
            return self.storage.put(
                self._get_key('votes', vote_key),
                vote_data,
                verify_integrity=True
            )
        except DatabaseError as e:
            logger.error(f"Error saving vote: {e}")
            return False
    
    def load_votes(self, height: int, round: int) -> List[Dict]:
        """Load votes for specific height and round"""
        try:
            votes = []
            prefix = self.key_prefixes['votes']
            
            # Use advanced iteration with filtering
            for key, value in self.storage.iterate(prefix=prefix, verify_integrity=True):
                key_str = key.decode('utf-8')
                if f"{height}_{round}" in key_str:
                    votes.append(value)
            
            return votes
            
        except DatabaseError as e:
            logger.error(f"Error loading votes for height {height}, round {round}: {e}")
            return []
    
    # === EVIDENCE MANAGEMENT ===
    
    def save_evidence(self, evidence_data: Dict) -> bool:
        """Save evidence to database"""
        try:
            import hashlib
            evidence_hash = hashlib.sha256(
                str(evidence_data).encode()
            ).hexdigest()
            
            return self.storage.put(
                self._get_key('evidence', evidence_hash),
                evidence_data,
                verify_integrity=True
            )
        except DatabaseError as e:
            logger.error(f"Error saving evidence: {e}")
            return False
    
    def load_evidence(self, evidence_hash: str) -> Optional[Dict]:
        """Load evidence by hash"""
        try:
            return self.storage.get_with_integrity_guarantee(
                self._get_key('evidence', evidence_hash)
            )
        except (KeyNotFoundError, DatabaseError):
            return None
    
    # === BATCH OPERATIONS ===
    
    def save_consensus_batch(self, state_data: Dict, validators_data: Dict, 
                           blocks: List[Dict] = None, votes: List[Dict] = None) -> bool:
        """Save multiple consensus entities atomically"""
        from database.utils.types import BatchOperation
        
        try:
            operations = []
            
            # Consensus state
            if state_data:
                operations.append(BatchOperation(
                    op_type='put',
                    key=self._get_key('consensus_state', 'current'),
                    value=state_data
                ))
            
            # Validators
            if validators_data:
                operations.append(BatchOperation(
                    op_type='put',
                    key=self._get_key('metadata', 'validators_complete'),
                    value=validators_data
                ))
            
            # Blocks
            if blocks:
                for block in blocks:
                    block_hash = block.get('hash')
                    if block_hash:
                        operations.append(BatchOperation(
                            op_type='put',
                            key=self._get_key('blocks', block_hash),
                            value=block
                        ))
            
            # Votes
            if votes:
                for vote in votes:
                    vote_key = f"{vote['height']}_{vote['round']}_{vote['validator']}"
                    operations.append(BatchOperation(
                        op_type='put',
                        key=self._get_key('votes', vote_key),
                        value=vote
                    ))
            
            # Execute batch atomically
            return self.storage.batch_write(operations)
            
        except DatabaseError as e:
            logger.error(f"Error in consensus batch save: {e}")
            return False
    
    # === INTEGRITY AND MAINTENANCE ===
    
    def verify_data_integrity(self) -> Dict[str, Any]:
        """Run comprehensive integrity check on consensus data"""
        try:
            return self.storage.run_integrity_scan()
        except DatabaseError as e:
            logger.error(f"Integrity check failed: {e}")
            return {'error': str(e)}
    
    def get_corrupted_entries(self) -> List[str]:
        """Get list of corrupted consensus entries"""
        try:
            corrupted_keys = self.storage.get_corrupted_keys()
            # Filter to only consensus-related keys
            return [
                key.decode('utf-8') for key in corrupted_keys 
                if any(prefix in key for prefix in self.key_prefixes.values())
            ]
        except DatabaseError as e:
            logger.error(f"Error getting corrupted entries: {e}")
            return []
    
    def repair_corrupted_data(self) -> Dict[str, Any]:
        """Attempt to repair corrupted consensus data"""
        try:
            return self.storage.repair_corrupted_entries()
        except DatabaseError as e:
            logger.error(f"Data repair failed: {e}")
            return {'error': str(e)}
    
    def compact_database(self) -> bool:
        """Compact database to reduce size"""
        try:
            # Advanced DB handles compaction automatically
            # This is just for manual triggering if needed
            return self.storage.compact_database()
        except DatabaseError as e:
            logger.error(f"Compaction failed: {e}")
            return False
    
    # === STATISTICS AND MONITORING ===
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get consensus database statistics"""
        try:
            # Get base storage stats
            storage_stats = self.storage.get_stats()
            
            # Add consensus-specific stats
            consensus_stats = {
                'consensus_state_exists': self._key_exists('consensus_state', 'current'),
                'validators_count': len(self.load_validators()),
                'active_validators_count': len(self.load_active_validators()),
            }
            
            return {**storage_stats, **consensus_stats}
            
        except DatabaseError as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def get_integrity_stats(self) -> Dict[str, Any]:
        """Get integrity protection statistics"""
        try:
            return self.storage.get_integrity_stats()
        except DatabaseError as e:
            logger.error(f"Error getting integrity stats: {e}")
            return {}
    
    # === HELPER METHODS ===
    
    def _get_key(self, collection: str, key: str) -> bytes:
        """Get full database key for collection and key"""
        prefix = self.key_prefixes.get(collection, b'')
        return prefix + key.encode('utf-8')
    
    def _key_exists(self, collection: str, key: str) -> bool:
        """Check if a key exists without loading the value"""
        try:
            self.storage.get(
                self._get_key(collection, key),
                use_cache=False,
                verify_integrity=False
            )
            return True
        except KeyNotFoundError:
            return False
        except DatabaseError:
            return False
    
    # === LIFECYCLE MANAGEMENT ===
    
    def close(self):
        """Close database connection"""
        try:
            self.storage.close()
            logger.info("Consensus database closed")
        except DatabaseError as e:
            logger.error(f"Error closing database: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

# Convenience function for quick access
def get_consensus_database(db_path: str = './consensus_db') -> ConsensusDatabase:
    """Get a consensus database instance with default settings"""
    return ConsensusDatabase(db_path)