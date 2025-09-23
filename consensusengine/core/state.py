# consensus/core/state.py
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import time

class ConsensusState(Enum):
    """Consensus process states"""
    IDLE = auto()
    PROPOSE = auto()
    PREVOTE = auto()
    PRECOMMIT = auto()
    COMMITTED = auto()
    VIEW_CHANGE = auto()
    RECOVERY = auto()
    NEW_HEIGHT = auto()

class ValidatorStatus(Enum):
    """Validator status levels"""
    ACTIVE = auto()
    JAILED = auto()
    INACTIVE = auto()
    SLASHED = auto()
    PENDING = auto()
    UNBONDING = auto()

class VoteType(Enum):
    """Vote types for BFT consensus"""
    PREVOTE = auto()
    PRECOMMIT = auto()

@dataclass
class RoundState:
    """State for a specific height and round"""
    height: int
    round: int
    step: ConsensusState
    proposer: Optional[Any] = None  # Validator object
    proposal: Optional[Any] = None  # BlockProposal object
    prevotes: Dict[str, Any] = field(default_factory=dict)  # validator_address -> vote
    precommits: Dict[str, Any] = field(default_factory=dict)  # validator_address -> vote
    prevote_polka: Optional[str] = None  # Block hash with 2/3+ prevotes
    precommit_polka: Optional[str] = None  # Block hash with 2/3+ precommits
    locked_value: Optional[str] = None  # Locked block hash for this round
    valid_value: Optional[str] = None  # Valid block hash for this round
    start_time: float = field(default_factory=time.time)

class EpochState:
    """State for managing epoch transitions"""
    
    def __init__(self, epoch_blocks: int = 100):
        self.epoch_blocks = epoch_blocks
        self.current_epoch = 0
        self.next_epoch_validators: Dict[str, Any] = {}
        self.pending_stakes: List[Tuple[str, int]] = []  # (address, amount)
        self.pending_unstakes: List[Tuple[str, int]] = []  # (address, amount)
        self.pending_delegations: List[Tuple[str, str, int]] = []  # (delegator, validator, amount)
        self.pending_undelegations: List[Tuple[str, str, int]] = []  # (delegator, validator, amount)
        self.reward_pool: int = 0  # Accumulated rewards for distribution
    
    def to_dict(self) -> Dict:
        """Serialize epoch state to dictionary"""
        return {
            'epoch_blocks': self.epoch_blocks,
            'current_epoch': self.current_epoch,
            'next_epoch_validators': {k: v.to_dict() for k, v in self.next_epoch_validators.items()},
            'pending_stakes': self.pending_stakes,
            'pending_unstakes': self.pending_unstakes,
            'pending_delegations': self.pending_delegations,
            'pending_undelegations': self.pending_undelegations,
            'reward_pool': self.reward_pool
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EpochState':
        """Deserialize epoch state from dictionary"""
        from consensusengine.models.validators import Validator
        
        epoch_state = cls(data['epoch_blocks'])
        epoch_state.current_epoch = data['current_epoch']
        epoch_state.next_epoch_validators = {
            k: Validator.from_dict(v) for k, v in data['next_epoch_validators'].items()
        }
        epoch_state.pending_stakes = data['pending_stakes']
        epoch_state.pending_unstakes = data['pending_unstakes']
        epoch_state.pending_delegations = data['pending_delegations']
        epoch_state.pending_undelegations = data['pending_undelegations']
        epoch_state.reward_pool = data['reward_pool']
        
        return epoch_state