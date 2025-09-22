"""
Block storage implementation using LevelDB
"""

import plyvel
import pickle
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from ..exceptions import StorageError

logger = logging.getLogger('consensus.storage')

class BlockStore:
    """Block storage using LevelDB"""
    
    def __init__(self, db_path: str, max_open_files: int = 1000, 
                 write_buffer_size: int = 4 * 1024 * 1024,
                 read_buffer_size: int = 1 * 1024 * 1024):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.db = plyvel.DB(
            str(self.db_path),
            create_if_missing=True,
            max_open_files=max_open_files,
            write_buffer_size=write_buffer_size,
            read_buffer_size=read_buffer_size
        )
        
        logger.info(f"Block store initialized at {db_path}")
    
    def store_block(self, block_data: Dict) -> None:
        """Store block data"""
        try:
            block_hash = block_data['hash']
            key = f"block:{block_hash}".encode()
            value = pickle.dumps(block_data)
            self.db.put(key, value)
        except Exception as e:
            raise StorageError(f"Failed to store block: {e}")
    
    def get_block(self, block_hash: str) -> Optional[Dict]:
        """Get block by hash"""
        try:
            key = f"block:{block_hash}".encode()
            value = self.db.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            raise StorageError(f"Failed to get block: {e}")
    
    def get_block_by_height(self, height: int) -> Optional[Dict]:
        """Get block by height"""
        try:
            key = f"height:{height}".encode()
            block_hash = self.db.get(key)
            if block_hash:
                return self.get_block(block_hash.decode())
            return None
        except Exception as e:
            raise StorageError(f"Failed to get block by height: {e}")
    
    def store_proposal(self, proposal: Dict) -> None:
        """Store block proposal"""
        try:
            key = f"proposal:{proposal['height']}:{proposal['round']}".encode()
            value = pickle.dumps(proposal)
            self.db.put(key, value)
        except Exception as e:
            raise StorageError(f"Failed to store proposal: {e}")
    
    def get_proposal(self, height: int, round_num: int) -> Optional[Dict]:
        """Get proposal for specific height and round"""
        try:
            key = f"proposal:{height}:{round_num}".encode()
            value = self.db.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            raise StorageError(f"Failed to get proposal: {e}")
    
    def save_consensus_state(self, state: Dict) -> None:
        """Save consensus state"""
        try:
            key = b"consensus:state"
            value = pickle.dumps(state)
            self.db.put(key, value)
        except Exception as e:
            raise StorageError(f"Failed to save consensus state: {e}")
    
    def load_consensus_state(self) -> Optional[Dict]:
        """Load consensus state"""
        try:
            key = b"consensus:state"
            value = self.db.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            raise StorageError(f"Failed to load consensus state: {e}")
    
    def save_validators(self, validators: Dict) -> None:
        """Save validator data"""
        try:
            key = b"validators:data"
            value = pickle.dumps(validators)
            self.db.put(key, value)
        except Exception as e:
            raise StorageError(f"Failed to save validators: {e}")
    
    def load_validators(self) -> Dict:
        """Load validator data"""
        try:
            key = b"validators:data"
            value = self.db.get(key)
            if value:
                return pickle.loads(value)
            return {}
        except Exception as e:
            raise StorageError(f"Failed to load validators: {e}")
    
    def close(self) -> None:
        """Close database connection"""
        self.db.close()
        logger.info("Block store closed")