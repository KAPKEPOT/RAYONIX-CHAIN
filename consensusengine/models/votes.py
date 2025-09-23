# consensus/models/votes.py
import time
import hashlib
from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend

class VoteType(Enum):
    """Vote types for BFT consensus"""
    PREVOTE = auto()
    PRECOMMIT = auto()

@dataclass
class Vote:
    """Vote for block proposal with production-ready validation"""
    height: int
    block_hash: str
    validator_address: str
    timestamp: float
    signature: str
    round_number: int
    vote_type: VoteType
    voting_power: int = 0  # Voting power of the validator at time of vote
    
    def to_dict(self) -> Dict:
        """Serialize vote to dictionary"""
        return {
            'height': self.height,
            'block_hash': self.block_hash,
            'validator_address': self.validator_address,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'round_number': self.round_number,
            'vote_type': self.vote_type.name,
            'voting_power': self.voting_power
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Vote':
        """Deserialize vote from dictionary"""
        return cls(
            height=data['height'],
            block_hash=data['block_hash'],
            validator_address=data['validator_address'],
            timestamp=data['timestamp'],
            signature=data['signature'],
            round_number=data['round_number'],
            vote_type=VoteType[data['vote_type']],
            voting_power=data.get('voting_power', 0)
        )
    
    def get_signing_data(self) -> bytes:
        """Get data that should be signed for the vote"""
        data = f"{self.height}|{self.round_number}|{self.vote_type.name}|{self.block_hash}"
        return data.encode('utf-8')
    
    def calculate_hash(self) -> str:
        """Calculate vote hash for deduplication"""
        vote_data = f"{self.height}{self.round_number}{self.vote_type.name}{self.block_hash}{self.validator_address}{self.timestamp}"
        return hashlib.sha256(vote_data.encode()).hexdigest()
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify vote signature using validator's public key"""
        try:
            if isinstance(self.signature, str):
                signature_bytes = bytes.fromhex(self.signature)
            else:
                signature_bytes = self.signature
            
            # Load public key
            verifying_key = serialization.load_der_public_key(
                bytes.fromhex(public_key),
                backend=default_backend()
            )
            
            # Verify signature
            signing_data = self.get_signing_data()
            verifying_key.verify(
                signature_bytes,
                signing_data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
            
        except (InvalidSignature, ValueError, Exception) as e:
            return False
    
    def validate(self, current_height: int, current_round: int, max_future_time: float = 10.0) -> bool:
        """
        Validate vote structure and timing
        
        Args:
            current_height: Current consensus height
            current_round: Current consensus round
            max_future_time: Maximum allowed future time in seconds
            
        Returns:
            True if vote is valid, False otherwise
        """
        # Check height and round
        if self.height != current_height:
            return False
        
        if self.round_number != current_round:
            return False
        
        # Check block hash format
        if not self.block_hash or (self.block_hash != "nil" and len(self.block_hash) != 64):
            return False
        
        # Check validator address format (assuming 42 char address)
        if not self.validator_address or len(self.validator_address) != 42:
            return False
        
        # Check timestamp validity
        current_time = time.time()
        if self.timestamp > current_time + max_future_time:
            return False
        
        # Allow some tolerance for clock skew (5 minutes)
        if self.timestamp < current_time - 300:
            return False
        
        return True
    
    def is_nil_vote(self) -> bool:
        """Check if this is a nil vote"""
        return self.block_hash == "nil"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Vote):
            return False
        
        return (self.height == other.height and
                self.round_number == other.round_number and
                self.vote_type == other.vote_type and
                self.validator_address == other.validator_address and
                self.block_hash == other.block_hash)
    
    def __hash__(self) -> int:
        return hash((self.height, self.round_number, self.vote_type, 
                    self.validator_address, self.block_hash))

@dataclass
class VoteSet:
    """Collection of votes for a specific height, round, and type"""
    height: int
    round: int
    vote_type: VoteType
    votes: Dict[str, Vote] = None  # validator_address -> Vote
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
    
    def add_vote(self, vote: Vote) -> bool:
        """Add a vote to the set if it's valid for this set"""
        if (vote.height != self.height or 
            vote.round_number != self.round or 
            vote.vote_type != self.vote_type):
            return False
        
        self.votes[vote.validator_address] = vote
        return True
    
    def get_votes_for_block(self, block_hash: str) -> list[Vote]:
        """Get all votes for a specific block hash"""
        return [vote for vote in self.votes.values() if vote.block_hash == block_hash]
    
    def has_vote_from(self, validator_address: str) -> bool:
        """Check if we have a vote from a specific validator"""
        return validator_address in self.votes
    
    def get_vote_power(self, validators: Dict[str, Any]) -> int:
        """Calculate total voting power in this vote set"""
        total_power = 0
        for vote in self.votes.values():
            if vote.validator_address in validators:
                total_power += validators[vote.validator_address].voting_power
        return total_power
    
    def get_vote_power_for_block(self, block_hash: str, validators: Dict[str, Any]) -> int:
        """Calculate voting power for a specific block hash"""
        total_power = 0
        for vote in self.get_votes_for_block(block_hash):
            if vote.validator_address in validators:
                total_power += validators[vote.validator_address].voting_power
        return total_power
    
    def has_supermajority(self, block_hash: str, validators: Dict[str, Any], total_voting_power: int) -> bool:
        """Check if a block hash has +2/3 voting power"""
        if total_voting_power == 0:
            return False
        
        block_power = self.get_vote_power_for_block(block_hash, validators)
        return block_power > (2 * total_voting_power) / 3
    
    def has_quorum(self, validators: Dict[str, Any], total_voting_power: int) -> bool:
        """Check if we have at least +2/3 voting power in the set"""
        if total_voting_power == 0:
            return False
        
        total_power = self.get_vote_power(validators)
        return total_power > (2 * total_voting_power) / 3
    
    def get_majority_block(self, validators: Dict[str, Any]) -> Optional[str]:
        """Get the block hash with the majority of votes, if any"""
        block_powers = {}
        
        for vote in self.votes.values():
            if vote.validator_address in validators:
                power = validators[vote.validator_address].voting_power
                block_powers[vote.block_hash] = block_powers.get(vote.block_hash, 0) + power
        
        if not block_powers:
            return None
        
        # Return block with highest voting power
        return max(block_powers.items(), key=lambda x: x[1])[0]