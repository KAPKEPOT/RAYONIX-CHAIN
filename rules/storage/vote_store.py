"""
Vote storage implementation
"""

import plyvel
import pickle
from typing import Dict, List, Optional
import logging
from collections import defaultdict

from ..exceptions import StorageError

logger = logging.getLogger('consensus.storage')

class VoteStore:
    """Vote storage using LevelDB"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = plyvel.DB(db_path, create_if_missing=True)
        
        # In-memory cache for recent votes
        self.vote_cache = defaultdict(dict)
        self.cache_size = 1000
        
        logger.info(f"Vote store initialized at {db_path}")
    
    def store_vote(self, vote: Dict) -> None:
        """Store vote"""
        try:
            vote_key = self._get_vote_key(vote)
            value = pickle.dumps(vote)
            self.db.put(vote_key, value)
            
            # Update cache
            cache_key = (vote['height'], vote['round'], vote['vote_type'])
            if len(self.vote_cache[cache_key]) < self.cache_size:
                self.vote_cache[cache_key][vote['voter']] = vote
            
        except Exception as e:
            raise StorageError(f"Failed to store vote: {e}")
    
    def get_votes(self, height: int, round_num: int, vote_type: str) -> List[Dict]:
        """Get all votes for specific height, round and type"""
        try:
            cache_key = (height, round_num, vote_type)
            if cache_key in self.vote_cache:
                return list(self.vote_cache[cache_key].values())
            
            # If not in cache, read from DB
            votes = []
            prefix = f"vote:{height}:{round_num}:{vote_type}:".encode()
            
            for key, value in self.db.iterator(prefix=prefix):
                vote = pickle.loads(value)
                votes.append(vote)
            
            return votes
            
        except Exception as e:
            raise StorageError(f"Failed to get votes: {e}")
    
    def get_votes_for_block(self, height: int, round_num: int, vote_type: str, block_hash: str) -> List[Dict]:
        """Get votes for specific block"""
        try:
            votes = self.get_votes(height, round_num, vote_type)
            return [vote for vote in votes if vote.get('block_hash') == block_hash]
        except Exception as e:
            raise StorageError(f"Failed to get votes for block: {e}")
    
    def has_voted(self, height: int, round_num: int, vote_type: str, voter: str) -> bool:
        """Check if validator has already voted"""
        try:
            cache_key = (height, round_num, vote_type)
            if cache_key in self.vote_cache and voter in self.vote_cache[cache_key]:
                return True
            
            key = f"vote:{height}:{round_num}:{vote_type}:{voter}".encode()
            return self.db.get(key) is not None
            
        except Exception as e:
            raise StorageError(f"Failed to check vote existence: {e}")
    
    def cleanup_old_votes(self, current_height: int, keep_previous: int = 100) -> None:
        """Clean up old votes to save space"""
        try:
            # Keep votes from last N heights
            max_old_height = current_height - keep_previous
            
            for key, value in self.db.iterator():
                if key.startswith(b"vote:"):
                    try:
                        parts = key.decode().split(':')
                        if len(parts) >= 2:
                            height = int(parts[1])
                            if height < max_old_height:
                                self.db.delete(key)
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            raise StorageError(f"Failed to cleanup old votes: {e}")
    
    def _get_vote_key(self, vote: Dict) -> bytes:
        """Get database key for vote"""
        return f"vote:{vote['height']}:{vote['round']}:{vote['vote_type']}:{vote['voter']}".encode()
    
    def close(self) -> None:
        """Close database connection"""
        self.db.close()
        logger.info("Vote store closed")