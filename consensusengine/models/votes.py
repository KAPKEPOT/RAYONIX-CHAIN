# consensus/models/votes.py
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, ClassVar
from enum import Enum, auto
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
import json
import secrets

# Configure logging
logger = logging.getLogger(__name__)

class VoteType(Enum):
    """Vote types for BFT consensus with proper string representations"""
    PREVOTE = auto()
    PRECOMMIT = auto()
    
    def __str__(self):
        return self.name

class VoteValidationError(Exception):
    """Custom exception for vote validation failures"""
    pass

class VoteSignatureError(Exception):
    """Custom exception for vote signature verification failures"""
    pass

@dataclass
class Vote:
    """Production-ready vote implementation for BFT consensus"""
    
    # Core vote fields
    height: int
    round_number: int
    vote_type: VoteType
    block_hash: str
    validator_address: str
    signature: str
    timestamp: float = field(default_factory=time.time)
    voting_power: int = 0
    
    # Constants
    NIL_VOTE_HASH: ClassVar[str] = "nil"
    VALIDATOR_ADDRESS_LENGTH: ClassVar[int] = 42
    BLOCK_HASH_LENGTH: ClassVar[int] = 64
    MAX_FUTURE_TIME_TOLERANCE: ClassVar[float] = 10.0  # seconds
    MAX_PAST_TIME_TOLERANCE: ClassVar[float] = 300.0   # seconds
    
    def __post_init__(self):
        """Validate vote upon initialization"""
        self._validate_basic()
    
    def _validate_basic(self) -> None:
        """Perform basic structural validation"""
        if self.height < 0:
            raise VoteValidationError(f"Invalid height: {self.height}")
        
        if self.round_number < 0:
            raise VoteValidationError(f"Invalid round number: {self.round_number}")
        
        if not self.block_hash or (self.block_hash != self.NIL_VOTE_HASH and 
                                 len(self.block_hash) != self.BLOCK_HASH_LENGTH):
            raise VoteValidationError(f"Invalid block hash: {self.block_hash}")
        
        if not self.validator_address or len(self.validator_address) != self.VALIDATOR_ADDRESS_LENGTH:
            raise VoteValidationError(f"Invalid validator address: {self.validator_address}")
        
        if not self.signature:
            raise VoteValidationError("Vote signature cannot be empty")
        
        if self.voting_power < 0:
            raise VoteValidationError(f"Invalid voting power: {self.voting_power}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize vote to dictionary with proper error handling"""
        try:
            return {
                'height': self.height,
                'round_number': self.round_number,
                'vote_type': str(self.vote_type),
                'block_hash': self.block_hash,
                'validator_address': self.validator_address,
                'signature': self.signature,
                'timestamp': self.timestamp,
                'voting_power': self.voting_power,
                'vote_hash': self.calculate_hash()
            }
        except Exception as e:
            logger.error(f"Failed to serialize vote to dict: {e}")
            raise
    
    def to_json(self) -> str:
        """Serialize vote to JSON string"""
        try:
            return json.dumps(self.to_dict(), sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to serialize vote to JSON: {e}")
            raise
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vote':
        """Deserialize vote from dictionary with robust error handling"""
        try:
            required_fields = ['height', 'round_number', 'vote_type', 'block_hash', 
                             'validator_address', 'signature']
            
            for field in required_fields:
                if field not in data:
                    raise VoteValidationError(f"Missing required field: {field}")
            
            return cls(
                height=int(data['height']),
                round_number=int(data['round_number']),
                vote_type=VoteType[data['vote_type']],
                block_hash=str(data['block_hash']),
                validator_address=str(data['validator_address']),
                signature=str(data['signature']),
                timestamp=float(data.get('timestamp', time.time())),
                voting_power=int(data.get('voting_power', 0))
            )
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to deserialize vote from dict: {e}")
            raise VoteValidationError(f"Invalid vote data: {e}")
    
    @classmethod
    def from_json(cls, json_data: str) -> 'Vote':
        """Deserialize vote from JSON string"""
        try:
            data = json.loads(json_data)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON data: {e}")
            raise VoteValidationError(f"Invalid JSON format: {e}")
    
    def get_signing_data(self) -> bytes:
        """Get canonical data that should be signed for the vote"""
        # Use deterministic JSON serialization for signing
        signing_dict = {
            'height': self.height,
            'round': self.round_number,
            'type': str(self.vote_type),
            'block_hash': self.block_hash,
            'validator': self.validator_address,
            'timestamp': int(self.timestamp)  # Use integer timestamp for determinism
        }
        return json.dumps(signing_dict, sort_keys=True, separators=(',', ':')).encode('utf-8')
    
    def calculate_hash(self) -> str:
        """Calculate unique vote hash for deduplication and identification"""
        hash_data = f"{self.height}:{self.round_number}:{self.vote_type}:{self.block_hash}:{self.validator_address}:{int(self.timestamp)}"
        return hashlib.sha3_256(hash_data.encode()).hexdigest()
    
    def sign(self, private_key: ec.EllipticCurvePrivateKey) -> None:
        """Sign the vote with validator's private key"""
        try:
            signing_data = self.get_signing_data()
            signature = private_key.sign(signing_data, ec.ECDSA(hashes.SHA3_256()))
            self.signature = signature.hex()
        except Exception as e:
            logger.error(f"Failed to sign vote: {e}")
            raise VoteSignatureError(f"Vote signing failed: {e}")
    
    def verify_signature(self, public_key_pem: str) -> bool:
        """Verify vote signature using validator's public key with comprehensive error handling"""
        try:
            # Parse public key
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=default_backend()
            )
            
            if not isinstance(public_key, ec.EllipticCurvePublicKey):
                logger.error("Invalid public key type for ECDSA verification")
                return False
            
            # Convert signature from hex
            signature_bytes = bytes.fromhex(self.signature)
            
            # Verify signature
            signing_data = self.get_signing_data()
            public_key.verify(
                signature_bytes,
                signing_data,
                ec.ECDSA(hashes.SHA3_256())
            )
            
            return True
            
        except InvalidSignature:
            logger.warning(f"Invalid signature for vote from {self.validator_address}")
            return False
        except ValueError as e:
            logger.error(f"Signature verification value error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during signature verification: {e}")
            return False
    
    def validate(self, current_height: int, current_round: int, 
                max_future_time: float = MAX_FUTURE_TIME_TOLERANCE,
                max_past_time: float = MAX_PAST_TIME_TOLERANCE) -> bool:
        """
        Comprehensive vote validation including timing and consensus rules
        """
        try:
            # Re-validate basic structure
            self._validate_basic()
            
            # Check consensus context
            if self.height != current_height:
                logger.warning(f"Vote height {self.height} doesn't match current height {current_height}")
                return False
            
            if self.round_number != current_round:
                logger.warning(f"Vote round {self.round_number} doesn't match current round {current_round}")
                return False
            
            # Validate timestamp
            current_time = time.time()
            if self.timestamp > current_time + max_future_time:
                logger.warning(f"Vote timestamp {self.timestamp} is too far in the future")
                return False
            
            if self.timestamp < current_time - max_past_time:
                logger.warning(f"Vote timestamp {self.timestamp} is too far in the past")
                return False
            
            return True
            
        except VoteValidationError as e:
            logger.warning(f"Vote validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during vote validation: {e}")
            return False
    
    def is_nil_vote(self) -> bool:
        """Check if this is a nil vote (vote for no block)"""
        return self.block_hash == self.NIL_VOTE_HASH
    
    def get_vote_id(self) -> str:
        """Get unique vote identifier"""
        return f"{self.validator_address}:{self.height}:{self.round_number}:{self.vote_type}"
    
    def __eq__(self, other: object) -> bool:
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
    
    def __str__(self) -> str:
        return (f"Vote(height={self.height}, round={self.round_number}, "
                f"type={self.vote_type}, block_hash={self.block_hash[:16]}..., "
                f"validator={self.validator_address[:16]}...)")

@dataclass
class VoteSet:
    """Production-ready vote set implementation with thread-safe operations"""
    
    height: int
    round: int
    vote_type: VoteType
    votes: Dict[str, Vote] = field(default_factory=dict)  # validator_address -> Vote
    _lock: Any = field(default=None)  # Will be initialized in __post_init__
    
    def __post_init__(self):
        """Initialize vote set with thread safety"""
        import threading
        self._lock = threading.RLock()
        
        if self.height < 0 or self.round < 0:
            raise VoteValidationError("Invalid height or round in VoteSet")
    
    def add_vote(self, vote: Vote, validators: Dict[str, Any] = None) -> bool:
        """
        Add a vote to the set with optional validator verification
        
        Returns:
            bool: True if vote was added successfully, False otherwise
        """
        with self._lock:
            # Validate vote matches set context
            if (vote.height != self.height or 
                vote.round_number != self.round or 
                vote.vote_type != self.vote_type):
                logger.warning(f"Vote context mismatch for vote set {self.height}/{self.round}/{self.vote_type}")
                return False
            
            # Check for duplicate vote
            if vote.validator_address in self.votes:
                existing_vote = self.votes[vote.validator_address]
                if existing_vote == vote:
                    logger.debug(f"Duplicate vote from {vote.validator_address}")
                    return True
                else:
                    logger.warning(f"Conflicting vote from {vote.validator_address}")
                    return False
            
            # Optional validator verification
            if validators is not None:
                if vote.validator_address not in validators:
                    logger.warning(f"Vote from unknown validator: {vote.validator_address}")
                    return False
            
            self.votes[vote.validator_address] = vote
            logger.debug(f"Added vote from {vote.validator_address} to vote set")
            return True
    
    def remove_vote(self, validator_address: str) -> bool:
        """Remove a vote from the set"""
        with self._lock:
            if validator_address in self.votes:
                del self.votes[validator_address]
                return True
            return False
    
    def get_votes_for_block(self, block_hash: str) -> List[Vote]:
        """Get all votes for a specific block hash"""
        with self._lock:
            return [vote for vote in self.votes.values() if vote.block_hash == block_hash]
    
    def get_validator_votes(self) -> Set[str]:
        """Get set of validator addresses that have voted"""
        with self._lock:
            return set(self.votes.keys())
    
    def has_vote_from(self, validator_address: str) -> bool:
        """Check if we have a vote from a specific validator"""
        with self._lock:
            return validator_address in self.votes
    
    def get_vote_count(self) -> int:
        """Get total number of votes in the set"""
        with self._lock:
            return len(self.votes)
    
    def get_vote_power(self, validators: Dict[str, Any]) -> int:
        """Calculate total voting power in this vote set"""
        with self._lock:
            total_power = 0
            for vote in self.votes.values():
                validator = validators.get(vote.validator_address)
                if validator and hasattr(validator, 'voting_power'):
                    total_power += validator.voting_power
            return total_power
    
    def get_vote_power_for_block(self, block_hash: str, validators: Dict[str, Any]) -> int:
        """Calculate voting power for a specific block hash"""
        with self._lock:
            total_power = 0
            for vote in self.get_votes_for_block(block_hash):
                validator = validators.get(vote.validator_address)
                if validator and hasattr(validator, 'voting_power'):
                    total_power += validator.voting_power
            return total_power
    
    def has_supermajority(self, block_hash: str, validators: Dict[str, Any], 
                         total_voting_power: int) -> bool:
        """Check if a block hash has +2/3 voting power (strict inequality)"""
        if total_voting_power <= 0:
            return False
        
        block_power = self.get_vote_power_for_block(block_hash, validators)
        return block_power > (2 * total_voting_power) / 3
    
    def has_quorum(self, validators: Dict[str, Any], total_voting_power: int) -> bool:
        """Check if we have at least +2/3 voting power in the set"""
        if total_voting_power <= 0:
            return False
        
        total_power = self.get_vote_power(validators)
        return total_power > (2 * total_voting_power) / 3
    
    def get_majority_block(self, validators: Dict[str, Any]) -> Optional[str]:
        """Get the block hash with the majority of voting power, if any"""
        with self._lock:
            block_powers = {}
            
            for vote in self.votes.values():
                validator = validators.get(vote.validator_address)
                if validator and hasattr(validator, 'voting_power'):
                    power = validator.voting_power
                    block_powers[vote.block_hash] = block_powers.get(vote.block_hash, 0) + power
            
            if not block_powers:
                return None
            
            # Return block with highest voting power
            max_block, max_power = max(block_powers.items(), key=lambda x: x[1])
            
            # Only return if we have at least one vote worth of power
            return max_block if max_power > 0 else None
    
    def get_all_votes(self) -> List[Vote]:
        """Get all votes in the set as a list"""
        with self._lock:
            return list(self.votes.values())
    
    def clear(self) -> None:
        """Clear all votes from the set"""
        with self._lock:
            self.votes.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize vote set to dictionary"""
        with self._lock:
            return {
                'height': self.height,
                'round': self.round,
                'vote_type': str(self.vote_type),
                'vote_count': len(self.votes),
                'votes': [vote.to_dict() for vote in self.votes.values()]
            }
    
    def __len__(self) -> int:
        return self.get_vote_count()
    
    def __contains__(self, validator_address: str) -> bool:
        return self.has_vote_from(validator_address)

class VotePool:
    """Thread-safe pool for managing multiple vote sets across heights and rounds"""
    
    def __init__(self):
        import threading
        self._lock = threading.RLock()
        self._vote_sets: Dict[tuple[int, int, VoteType], VoteSet] = {}
    
    def get_vote_set(self, height: int, round: int, vote_type: VoteType) -> VoteSet:
        """Get or create vote set for specific height, round, and type"""
        key = (height, round, vote_type)
        
        with self._lock:
            if key not in self._vote_sets:
                self._vote_sets[key] = VoteSet(height, round, vote_type)
            
            return self._vote_sets[key]
    
    def add_vote(self, vote: Vote, validators: Dict[str, Any] = None) -> bool:
        """Add vote to appropriate vote set"""
        vote_set = self.get_vote_set(vote.height, vote.round_number, vote.vote_type)
        return vote_set.add_vote(vote, validators)
    
    def prune_old_votes(self, current_height: int, keep_previous_n: int = 10):
        """Prune vote sets older than current_height - keep_previous_n"""
        with self._lock:
            keys_to_remove = [
                key for key in self._vote_sets.keys() 
                if key[0] < current_height - keep_previous_n
            ]
            
            for key in keys_to_remove:
                del self._vote_sets[key]
            
            if keys_to_remove:
                logger.info(f"Pruned {len(keys_to_remove)} old vote sets")