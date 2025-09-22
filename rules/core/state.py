"""
Consensus state management
"""

import time
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger('consensus.state')

@dataclass
class ConsensusState:
    """Current consensus state"""
    height: int = 0
    round: int = 0
    step: str = "NEW_HEIGHT"
    locked_round: int = -1
    valid_round: int = -1
    locked_value: Optional[str] = None
    valid_value: Optional[str] = None
    current_proposer: Optional[str] = None
    start_time: float = field(default_factory=time.time)

@dataclass
class RoundState:
    """State for specific consensus round"""
    height: int
    round: int
    step: str
    proposer: Optional[str] = None
    proposal: Optional[Dict] = None
    prevotes: Dict[str, List[Dict]] = field(default_factory=dict)  # block_hash -> votes
    precommits: Dict[str, List[Dict]] = field(default_factory=dict)  # block_hash -> votes
    prevote_polka: Optional[Set[str]] = None
    precommit_polka: Optional[Set[str]] = None
    start_time: float = field(default_factory=time.time)

class StateManager:
    """Manager for consensus state"""
    
    def __init__(self):
        self.consensus_state = ConsensusState()
        self.round_states: Dict[Tuple[int, int], RoundState] = {}
        self.lock = threading.RLock()
    
    def get_consensus_state(self) -> ConsensusState:
        """Get current consensus state"""
        with self.lock:
            return self.consensus_state
    
    def set_consensus_state(self, state: ConsensusState) -> None:
        """Set consensus state"""
        with self.lock:
            self.consensus_state = state
    
    def update_consensus_state(self, **kwargs) -> None:
        """Update consensus state with given fields"""
        with self.lock:
            for key, value in kwargs.items():
                if hasattr(self.consensus_state, key):
                    setattr(self.consensus_state, key, value)
    
    def get_round_state(self, height: int, round_num: int) -> Optional[RoundState]:
        """Get round state for specific height and round"""
        with self.lock:
            return self.round_states.get((height, round_num))
    
    def create_round_state(self, height: int, round_num: int, step: str, proposer: Optional[str] = None) -> RoundState:
        """Create new round state"""
        with self.lock:
            round_state = RoundState(height=height, round=round_num, step=step, proposer=proposer)
            self.round_states[(height, round_num)] = round_state
            return round_state
    
    def update_round_state(self, height: int, round_num: int, **kwargs) -> None:
        """Update round state with given fields"""
        with self.lock:
            round_state = self.round_states.get((height, round_num))
            if round_state:
                for key, value in kwargs.items():
                    if hasattr(round_state, key):
                        setattr(round_state, key, value)
    
    def add_vote_to_round(self, height: int, round_num: int, vote: Dict, vote_type: str) -> None:
        """Add vote to round state"""
        with self.lock:
            round_state = self.round_states.get((height, round_num))
            if not round_state:
                return
            
            block_hash = vote.get('block_hash', 'nil')
            votes_dict = round_state.prevotes if vote_type == "PREVOTE" else round_state.precommits
            
            if block_hash not in votes_dict:
                votes_dict[block_hash] = []
            
            # Check if this validator already voted for this block
            existing_votes = [v for v in votes_dict[block_hash] if v.get('voter') == vote.get('voter')]
            if not existing_votes:
                votes_dict[block_hash].append(vote)
    
    def get_round_votes(self, height: int, round_num: int, vote_type: str, block_hash: Optional[str] = None) -> List[Dict]:
        """Get votes for specific round and optional block hash"""
        with self.lock:
            round_state = self.round_states.get((height, round_num))
            if not round_state:
                return []
            
            votes_dict = round_state.prevotes if vote_type == "PREVOTE" else round_state.precommits
            
            if block_hash:
                return votes_dict.get(block_hash, [])
            else:
                # Return all votes
                all_votes = []
                for votes in votes_dict.values():
                    all_votes.extend(votes)
                return all_votes
    
    def check_polka(self, height: int, round_num: int, vote_type: str, block_hash: str, total_power: int, get_voting_power: callable) -> bool:
        """Check if +2/3 voting power has been reached for a block"""
        votes = self.get_round_votes(height, round_num, vote_type, block_hash)
        voting_power = sum(get_voting_power(vote.get('voter')) for vote in votes)
        
        return voting_power > (2 * total_power) / 3
    
    def cleanup_old_rounds(self, current_height: int, keep_previous: int = 10) -> None:
        """Clean up old round states to save memory"""
        with self.lock:
            keys_to_remove = []
            for (height, round_num) in self.round_states.keys():
                if height < current_height - keep_previous:
                    keys_to_remove.append((height, round_num))
            
            for key in keys_to_remove:
                del self.round_states[key]
            
            logger.debug(f"Cleaned up {len(keys_to_remove)} old round states")