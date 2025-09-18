# consensus.py
import os
import hashlib
import json
import time
import random
import threading
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
import plyvel
import pickle
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ConsensusEngine')

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
class Validator:
    """Complete validator information"""
    address: str
    public_key: str
    staked_amount: int
    commission_rate: float  # 0.0 to 1.0
    total_delegated: int = 0
    status: ValidatorStatus = ValidatorStatus.PENDING
    uptime: float = 100.0  # Percentage
    last_active: float = field(default_factory=time.time)
    created_block_height: int = 0
    total_rewards: int = 0
    slashing_count: int = 0
    voting_power: int = 0
    jail_until: Optional[float] = None
    delegators: Dict[str, int] = field(default_factory=dict)  # address -> amount
    missed_blocks: int = 0  # Track blocks missed for unavailability slashing
    signed_blocks: int = 0  # Track blocks signed
    
    @property
    def total_stake(self) -> int:
        return self.staked_amount + self.total_delegated
    
    @property
    def effective_stake(self) -> int:
        """Calculate effective stake considering status and uptime"""
        if self.status in [ValidatorStatus.JAILED, ValidatorStatus.SLASHED]:
            return 0
        return int(self.total_stake * (self.uptime / 100.0))
    
    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'public_key': self.public_key,
            'staked_amount': self.staked_amount,
            'commission_rate': self.commission_rate,
            'total_delegated': self.total_delegated,
            'status': self.status.name,
            'uptime': self.uptime,
            'last_active': self.last_active,
            'created_block_height': self.created_block_height,
            'total_rewards': self.total_rewards,
            'slashing_count': self.slashing_count,
            'voting_power': self.voting_power,
            'jail_until': self.jail_until,
            'delegators': self.delegators,
            'missed_blocks': self.missed_blocks,
            'signed_blocks': self.signed_blocks
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Validator':
        return cls(
            address=data['address'],
            public_key=data['public_key'],
            staked_amount=data['staked_amount'],
            commission_rate=data['commission_rate'],
            total_delegated=data['total_delegated'],
            status=ValidatorStatus[data['status']],
            uptime=data['uptime'],
            last_active=data['last_active'],
            created_block_height=data['created_block_height'],
            total_rewards=data['total_rewards'],
            slashing_count=data['slashing_count'],
            voting_power=data['voting_power'],
            jail_until=data.get('jail_until'),
            delegators=data.get('delegators', {}),
            missed_blocks=data.get('missed_blocks', 0),
            signed_blocks=data.get('signed_blocks', 0)
        )

@dataclass
class BlockProposal:
    """Block proposal structure"""
    height: int
    block_hash: str
    validator_address: str
    timestamp: float
    signature: str
    view_number: int
    round_number: int
    parent_hash: str
    tx_hashes: List[str] = field(default_factory=list)
    justification: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'height': self.height,
            'block_hash': self.block_hash,
            'validator_address': self.validator_address,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'view_number': self.view_number,
            'round_number': self.round_number,
            'parent_hash': self.parent_hash,
            'tx_hashes': self.tx_hashes,
            'justification': self.justification
        }

@dataclass
class Vote:
    """Vote for block proposal"""
    height: int
    block_hash: str
    validator_address: str
    timestamp: float
    signature: str
    round_number: int
    vote_type: VoteType
    
    def to_dict(self) -> Dict:
        return {
            'height': self.height,
            'block_hash': self.block_hash,
            'validator_address': self.validator_address,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'round_number': self.round_number,
            'vote_type': self.vote_type.name
        }

@dataclass
class RoundState:
    """State for a specific height and round"""
    height: int
    round: int
    step: ConsensusState
    proposer: Optional[Validator] = None
    proposal: Optional[BlockProposal] = None
    prevotes: Dict[str, Vote] = field(default_factory=dict)  # validator_address -> vote
    precommits: Dict[str, Vote] = field(default_factory=dict)  # validator_address -> vote
    prevote_polka: Optional[Set[str]] = None  # Set of block hashes with 2/3+ prevotes
    precommit_polka: Optional[Set[str]] = None  # Set of block hashes with 2/3+ precommits
    locked_value: Optional[str] = None  # Locked block hash for this round
    valid_value: Optional[str] = None  # Valid block hash for this round
    start_time: float = field(default_factory=time.time)

class EpochState:
    """State for managing epoch transitions"""
    def __init__(self, epoch_blocks: int = 100):
        self.epoch_blocks = epoch_blocks
        self.current_epoch = 0
        self.next_epoch_validators: Dict[str, Validator] = {}
        self.pending_stakes: List[Tuple[str, int]] = []  # (address, amount)
        self.pending_unstakes: List[Tuple[str, int]] = []  # (address, amount)
        self.pending_delegations: List[Tuple[str, str, int]] = []  # (delegator, validator, amount)
        self.pending_undelegations: List[Tuple[str, str, int]] = []  # (delegator, validator, amount)
        self.reward_pool: int = 0  # Accumulated rewards for distribution

class ABCIApplication:
    """Application Blockchain Interface for decoupling consensus from execution"""
    def __init__(self):
        self.check_tx_fn: Optional[Callable[[str], bool]] = None
        self.deliver_tx_fn: Optional[Callable[[str], bool]] = None
        self.commit_fn: Optional[Callable[[], str]] = None
        self.begin_block_fn: Optional[Callable[[int, str], None]] = None
        self.end_block_fn: Optional[Callable[[int], None]] = None
    
    def set_check_tx(self, fn: Callable[[str], bool]):
        self.check_tx_fn = fn
    
    def set_deliver_tx(self, fn: Callable[[str], bool]):
        self.deliver_tx_fn = fn
    
    def set_commit(self, fn: Callable[[], str]):
        self.commit_fn = fn
    
    def set_begin_block(self, fn: Callable[[int, str], None]):
        self.begin_block_fn = fn
    
    def set_end_block(self, fn: Callable[[int], None]):
        self.end_block_fn = fn

class ProofOfStake:
    """Complete Proof-of-Stake consensus implementation with BFT features"""
    
    def __init__(self, min_stake: int = 1000, jail_duration: int = 3600,
                 slash_percentage: float = 0.01, epoch_blocks: int = 100,
                 max_validators: int = 100, db_path: str = './consensus_db',
                 timeout_propose: int = 3000, timeout_prevote: int = 1000,
                 timeout_precommit: int = 1000):
        """
        Initialize Proof-of-Stake consensus
        
        Args:
            min_stake: Minimum stake required to become validator
            jail_duration: Jail duration in seconds for misbehavior
            slash_percentage: Percentage of stake to slash for violations
            epoch_blocks: Number of blocks per epoch
            max_validators: Maximum number of active validators
            timeout_propose: Propose timeout in milliseconds
            timeout_prevote: Prevote timeout in milliseconds
            timeout_precommit: Precommit timeout in milliseconds
        """        
        self.min_stake = min_stake
        self.jail_duration = jail_duration
        self.slash_percentage = slash_percentage
        self.epoch_blocks = epoch_blocks
        self.max_validators = max_validators
        
        # Timeout values in seconds
        self.timeout_propose = timeout_propose / 1000.0
        self.timeout_prevote = timeout_prevote / 1000.0
        self.timeout_precommit = timeout_precommit / 1000.0
        
        self.validators: Dict[str, Validator] = {}
        self.active_validators: List[Validator] = []
        self.pending_validators: List[Validator] = []
        
        # BFT consensus state
        self.height = 0
        self.round = 0
        self.step = ConsensusState.NEW_HEIGHT
        self.locked_round = -1
        self.valid_round = -1
        self.locked_value: Optional[str] = None
        self.valid_value: Optional[str] = None
        
        # Round states for each height and round
        self.round_states: Dict[Tuple[int, int], RoundState] = {}
        
        # Epoch management
        self.epoch_state = EpochState(epoch_blocks)
        
        # Block and vote storage
        self.block_proposals: Dict[str, BlockProposal] = {}
        self.votes: Dict[Tuple[int, int, VoteType], Dict[str, Vote]] = defaultdict(dict)
        self.executed_blocks: Set[str] = set()
        
        # ABCI interface
        self.abci = ABCIApplication()
        
        # Database for persistence
        self.db = plyvel.DB(db_path, create_if_missing=True)
        
        # Total stake for compatibility
        self.total_stake = 0
        
        # Locks for thread safety
        self.lock = threading.RLock()
        self.validator_lock = threading.RLock()
        self.consensus_lock = threading.RLock()
        
        # Background task management
        self._running = True
        self._timeout_handlers: Dict[Tuple[int, int, ConsensusState], threading.Timer] = {}
        
        self._load_state()
        self._start_background_tasks()
    
    def _load_state(self):
        """Load consensus state from database"""
        try:
            # Load validators
            validators_data_bytes = self.db.get(b'validators')
            if validators_data_bytes:
                validators_data = pickle.loads(validators_data_bytes)
                self.validators = {k: Validator.from_dict(v) for k, v in validators_data.items()}
            
            # Load active validators
            active_data_bytes = self.db.get(b'active_validators')
            if active_data_bytes:
                active_data = pickle.loads(active_data_bytes)
                self.active_validators = [Validator.from_dict(v) for v in active_data]
            
            # Load height and round
            height_bytes = self.db.get(b'height')
            if height_bytes:
                self.height = int.from_bytes(height_bytes, 'big')
            
            round_bytes = self.db.get(b'round')
            if round_bytes:
                self.round = int.from_bytes(round_bytes, 'big')
            
            # Load step
            step_bytes = self.db.get(b'step')
            if step_bytes:
                self.step = ConsensusState(int.from_bytes(step_bytes, 'big'))
            
            # Load locked and valid values
            locked_value_bytes = self.db.get(b'locked_value')
            if locked_value_bytes:
                self.locked_value = locked_value_bytes.decode('utf-8')
            
            valid_value_bytes = self.db.get(b'valid_value')
            if valid_value_bytes:
                self.valid_value = valid_value_bytes.decode('utf-8')
            
            # Load locked and valid rounds
            locked_round_bytes = self.db.get(b'locked_round')
            if locked_round_bytes:
                self.locked_round = int.from_bytes(locked_round_bytes, 'big')
            
            valid_round_bytes = self.db.get(b'valid_round')
            if valid_round_bytes:
                self.valid_round = int.from_bytes(valid_round_bytes, 'big')
            
            # Update total stake
            self.update_total_stake()
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Initialize fresh state
            self.update_total_stake()
            self._save_state()
    
    def _save_state(self):
        """Save consensus state to database"""
        with self.lock:
            # Update total stake before saving
            self.update_total_stake()
            
            # Save validators
            validators_data = {k: v.to_dict() for k, v in self.validators.items()}
            self.db.put(b'validators', pickle.dumps(validators_data))
            
            # Save active validators
            active_data = [v.to_dict() for v in self.active_validators]
            self.db.put(b'active_validators', pickle.dumps(active_data))
            
            # Save consensus state
            self.db.put(b'height', self.height.to_bytes(8, 'big'))
            self.db.put(b'round', self.round.to_bytes(8, 'big'))
            self.db.put(b'step', self.step.value.to_bytes(4, 'big'))
            
            if self.locked_value:
                self.db.put(b'locked_value', self.locked_value.encode('utf-8'))
            
            if self.valid_value:
                self.db.put(b'valid_value', self.valid_value.encode('utf-8'))
            
            self.db.put(b'locked_round', self.locked_round.to_bytes(8, 'big'))
            self.db.put(b'valid_round', self.valid_round.to_bytes(8, 'big'))
            
            # Save total stake
            self.db.put(b'total_stake', self.total_stake.to_bytes(8, 'big'))
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def epoch_processor():
            while self._running:
                time.sleep(30)
                self._process_epoch_transition()
        
        def validator_updater():
            while self._running:
                time.sleep(60)
                self._update_validator_set()
        
        def jail_checker():
            while self._running:
                time.sleep(300)
                self._check_jailed_validators()
        
        def unavailability_checker():
            while self._running:
                time.sleep(self.epoch_blocks * 5)  # Check every ~5 blocks worth of time
                self._check_unavailability()
        
        # Start background threads
        threading.Thread(target=epoch_processor, daemon=True).start()
        threading.Thread(target=validator_updater, daemon=True).start()
        threading.Thread(target=jail_checker, daemon=True).start()
        threading.Thread(target=unavailability_checker, daemon=True).start()
    
    def _check_unavailability(self):
        """Check for validator unavailability and apply slashing"""
        with self.validator_lock:
            for validator in self.validators.values():
                if validator.status != ValidatorStatus.ACTIVE:
                    continue
                
                # Calculate missed block percentage
                total_blocks = validator.missed_blocks + validator.signed_blocks
                if total_blocks == 0:
                    continue
                
                missed_percentage = validator.missed_blocks / total_blocks
                
                # Slash if missed more than 50% of blocks
                if missed_percentage > 0.5:
                    self.slash_validator(
                        validator.address,
                        {'type': 'unavailability', 'missed_percentage': missed_percentage},
                        'system'
                    )
                    logger.warning(f"Validator {validator.address} slashed for unavailability")
    
    def _process_epoch_transition(self):
        """Process epoch transition and distribute rewards"""
        if self.height % self.epoch_blocks != 0:
            return
        
        with self.lock:
            # Process all pending staking operations
            self._process_pending_stakes()
            
            # Update validator set for next epoch
            self._update_validator_set()
            
            # Distribute rewards if we have a reward pool
            if self.epoch_state.reward_pool > 0:
                self._distribute_epoch_rewards()
            
            # Reset reward pool for next epoch
            self.epoch_state.reward_pool = 0
            self.epoch_state.current_epoch += 1
            
            self._save_state()
    
    def _process_pending_stakes(self):
        """Process all pending stake operations"""
        with self.validator_lock:
            # Process new stakes
            for address, amount in self.epoch_state.pending_stakes:
                if address in self.validators:
                    self.validators[address].staked_amount += amount
                else:
                    # This should not happen as validators must be registered first
                    logger.warning(f"Pending stake for unknown validator: {address}")
            
            # Process unstakes
            for address, amount in self.epoch_state.pending_unstakes:
                if address in self.validators:
                    validator = self.validators[address]
                    if validator.staked_amount >= amount:
                        validator.staked_amount -= amount
                    else:
                        # Slash the validator for trying to unstake more than they have
                        self.slash_validator(
                            address,
                            {'type': 'over_unstake', 'attempted': amount, 'actual': validator.staked_amount},
                            'system'
                        )
            
            # Process delegations
            for delegator, validator_addr, amount in self.epoch_state.pending_delegations:
                if validator_addr in self.validators:
                    validator = self.validators[validator_addr]
                    current_delegation = validator.delegators.get(delegator, 0)
                    validator.delegators[delegator] = current_delegation + amount
                    validator.total_delegated += amount
            
            # Process undelegations
            for delegator, validator_addr, amount in self.epoch_state.pending_undelegations:
                if validator_addr in self.validators:
                    validator = self.validators[validator_addr]
                    if delegator in validator.delegators:
                        current_delegation = validator.delegators[delegator]
                        if current_delegation >= amount:
                            validator.delegators[delegator] = current_delegation - amount
                            validator.total_delegated -= amount
                            
                            if validator.delegators[delegator] == 0:
                                del validator.delegators[delegator]
            
            # Clear pending operations
            self.epoch_state.pending_stakes.clear()
            self.epoch_state.pending_unstakes.clear()
            self.epoch_state.pending_delegations.clear()
            self.epoch_state.pending_undelegations.clear()
    
    def _distribute_epoch_rewards(self):
        """Distribute epoch rewards to validators and delegators"""
        total_effective_stake = sum(v.effective_stake for v in self.active_validators)
        if total_effective_stake == 0:
            return
        
        total_reward = self.epoch_state.reward_pool
        
        for validator in self.active_validators:
            if validator.effective_stake == 0:
                continue
            
            # Validator's share of rewards
            validator_share = (validator.effective_stake / total_effective_stake) * total_reward
            
            # Commission goes to validator
            commission = validator_share * validator.commission_rate
            validator.total_rewards += commission
            
            # Remainder goes to delegators proportionally
            delegator_rewards = validator_share - commission
            
            if validator.total_delegated > 0:
                self._distribute_delegator_rewards(validator, delegator_rewards)
    
    def _distribute_delegator_rewards(self, validator: Validator, total_rewards: int):
        """Distribute rewards to delegators"""
        for delegator_address, delegated_amount in validator.delegators.items():
            delegator_share = (delegated_amount / validator.total_delegated) * total_rewards
            # In real implementation, this would transfer tokens to delegators
            # For now, just track in validator object
            if 'delegator_rewards' not in validator.__dict__:
                validator.delegator_rewards = {}
            validator.delegator_rewards[delegator_address] = \
                validator.delegator_rewards.get(delegator_address, 0) + delegator_share
    
    def _update_validator_set(self):
        """Update the active validator set based on stake at epoch boundaries"""
        if self.height % self.epoch_blocks != 0:
            return
        
        with self.validator_lock:
            # Sort validators by effective stake
            sorted_validators = sorted(
                [v for v in self.validators.values() 
                 if v.status in [ValidatorStatus.ACTIVE, ValidatorStatus.PENDING]],
                key=lambda x: x.effective_stake,
                reverse=True
            )
            
            # Select top validators up to max_validators
            new_active = sorted_validators[:self.max_validators]
            
            # Update statuses
            for validator in new_active:
                if validator.status == ValidatorStatus.PENDING:
                    validator.status = ValidatorStatus.ACTIVE
            
            for validator in sorted_validators[self.max_validators:]:
                if validator.status == ValidatorStatus.ACTIVE:
                    validator.status = ValidatorStatus.INACTIVE
            
            self.active_validators = new_active
            self._save_state()
    
    def _check_jailed_validators(self):
        """Check and release jailed validators"""
        current_time = time.time()
        
        with self.validator_lock:
            for validator in self.validators.values():
                if (validator.status == ValidatorStatus.JAILED and 
                    validator.jail_until and 
                    validator.jail_until <= current_time):
                    
                    validator.status = ValidatorStatus.ACTIVE
                    validator.jail_until = None
                    validator.missed_blocks = 0
                    validator.signed_blocks = 0
            
            self._save_state()
    
    def select_proposer(self, height: int, round: int) -> Optional[Validator]:
        """
        Select block proposer for current height and round
        
        Uses weighted random selection based on stake
        """
        with self.validator_lock:
            if not self.active_validators:
                return None
            
            total_stake = sum(v.effective_stake for v in self.active_validators)
            if total_stake == 0:
                return None
            
            # Deterministic selection based on height and round
            random_seed = hashlib.sha256(
                f"{height}_{round}_{self.epoch_state.current_epoch}".encode()
            ).digest()
            
            random_number = int.from_bytes(random_seed, 'big') % total_stake
            
            current_sum = 0
            for validator in self.active_validators:
                current_sum += validator.effective_stake
                if random_number < current_sum:
                    return validator
            
            return self.active_validators[-1]
    
    def start_new_height(self, height: int):
        """Start consensus for a new block height"""
        with self.consensus_lock:
            self.height = height
            self.round = 0
            self.step = ConsensusState.PROPOSE
            self.locked_round = -1
            self.valid_round = -1
            self.locked_value = None
            self.valid_value = None
            
            # Clear old round states
            self.round_states = {}
            
            # Start the propose step
            self._start_round(0)
    
    def _start_round(self, round: int):
        """Start a new round"""
        with self.consensus_lock:
            self.round = round
            self.step = ConsensusState.PROPOSE
            
            # Get the proposer for this round
            proposer = self.select_proposer(self.height, round)
            
            # Create round state
            round_state = RoundState(
                height=self.height,
                round=round,
                step=ConsensusState.PROPOSE,
                proposer=proposer,
                start_time=time.time()
            )
            
            self.round_states[(self.height, round)] = round_state
            
            # Set timeout for propose step
            self._set_timeout(self.height, round, ConsensusState.PROPOSE, self.timeout_propose)
            
            # If we're the proposer, create a proposal
            # In real implementation, this would be triggered by the proposer's logic
            logger.info(f"Starting round {round} at height {self.height}, proposer: {proposer.address if proposer else 'None'}")
    
    def _set_timeout(self, height: int, round: int, step: ConsensusState, timeout: float):
        """Set a timeout for a specific step"""
        key = (height, round, step)
        
        # Cancel any existing timeout
        if key in self._timeout_handlers:
            self._timeout_handlers[key].cancel()
        
        # Create new timeout
        def timeout_handler():
            with self.consensus_lock:
                if (self.height == height and self.round == round and self.step == step):
                    self._on_timeout(height, round, step)
        
        timer = threading.Timer(timeout, timeout_handler)
        timer.daemon = True
        timer.start()
        
        self._timeout_handlers[key] = timer
    
    def _on_timeout(self, height: int, round: int, step: ConsensusState):
        """Handle timeout for a step"""
        logger.warning(f"Timeout at height {height}, round {round}, step {step}")
        
        if step == ConsensusState.PROPOSE:
            # Move to prevote step with nil value
            self._prevote(height, round, None)
        elif step == ConsensusState.PREVOTE:
            # Move to precommit step with nil value
            self._precommit(height, round, None)
        elif step == ConsensusState.PRECOMMIT:
            # Move to next round
            self._start_round(round + 1)
    
    def receive_proposal(self, proposal: BlockProposal) -> bool:
        """
        Receive a block proposal from the network
        
        Returns:
            True if proposal is valid and accepted, False otherwise
        """
        with self.consensus_lock:
            # Check if this proposal is for the current height and round
            if proposal.height != self.height or proposal.round_number != self.round:
                logger.warning(f"Proposal for wrong height/round: {proposal.height}/{proposal.round_number}, current: {self.height}/{self.round}")
                return False
            
            # Check if we're in the propose step
            if self.step != ConsensusState.PROPOSE:
                logger.warning(f"Not in propose step, current step: {self.step}")
                return False
            
            # Verify the proposer is correct
            expected_proposer = self.select_proposer(self.height, self.round)
            if not expected_proposer or expected_proposer.address != proposal.validator_address:
                logger.warning(f"Invalid proposer: {proposal.validator_address}, expected: {expected_proposer.address if expected_proposer else 'None'}")
                return False
            
            # Verify the signature
            if not self.verify_validator_signature(proposal.validator_address, self._get_proposal_signing_data(proposal), proposal.signature):
                logger.warning(f"Invalid proposal signature from {proposal.validator_address}")
                return False
            
            # Verify the block is valid using ABCI
            if self.abci.check_tx_fn:
                for tx_hash in proposal.tx_hashes:
                    if not self.abci.check_tx_fn(tx_hash):
                        logger.warning(f"Invalid transaction in proposal: {tx_hash}")
                        return False
            
            # Store the proposal
            self.block_proposals[proposal.block_hash] = proposal
            
            # Update round state
            round_state = self.round_states.get((self.height, self.round))
            if round_state:
                round_state.proposal = proposal
                round_state.step = ConsensusState.PREVOTE
            
            # Move to prevote step
            self.step = ConsensusState.PREVOTE
            self._set_timeout(self.height, self.round, ConsensusState.PREVOTE, self.timeout_prevote)
            
            # Send prevote for this block
            self._prevote(self.height, self.round, proposal.block_hash)
            
            return True
    
    def _prevote(self, height: int, round: int, block_hash: Optional[str]):
        """Send a prevote for a block"""
        with self.consensus_lock:
            if height != self.height or round != self.round:
                return
            
            # Create vote
            vote = self._create_vote(height, round, block_hash, VoteType.PREVOTE)
            if not vote:
                return
            
            # Store vote locally
            vote_key = (height, round, VoteType.PREVOTE)
            self.votes[vote_key][vote.validator_address] = vote
            
            # Check if we have +2/3 prevotes for this block
            self._check_prevote_polka(height, round, block_hash)
            
            # In real implementation, broadcast the vote to the network
            logger.info(f"Prevote for block {block_hash} at height {height}, round {round}")
    
    def _precommit(self, height: int, round: int, block_hash: Optional[str]):
        """Send a precommit for a block"""
        with self.consensus_lock:
            if height != self.height or round != self.round:
                return
            
            # Create vote
            vote = self._create_vote(height, round, block_hash, VoteType.PRECOMMIT)
            if not vote:
                return
            
            # Store vote locally
            vote_key = (height, round, VoteType.PRECOMMIT)
            self.votes[vote_key][vote.validator_address] = vote
            
            # Check if we have +2/3 precommits for this block
            self._check_precommit_polka(height, round, block_hash)
            
            # In real implementation, broadcast the vote to the network
            logger.info(f"Precommit for block {block_hash} at height {height}, round {round}")
    
    def _create_vote(self, height: int, round: int, block_hash: Optional[str], vote_type: VoteType) -> Optional[Vote]:
        """Create a vote object"""
        # This would use the validator's private key to sign in real implementation
        # For now, we'll just create the vote object without a real signature
        
        validator_address = "current_validator"  # This would be the address of the current validator
        
        vote = Vote(
            height=height,
            block_hash=block_hash or "nil",
            validator_address=validator_address,
            timestamp=time.time(),
            signature="dummy_signature",  # This would be a real signature
            round_number=round,
            vote_type=vote_type
        )
        
        return vote
    
    def _check_prevote_polka(self, height: int, round: int, block_hash: str):
        """Check if we have +2/3 prevotes for a block (polka)"""
        vote_key = (height, round, VoteType.PREVOTE)
        votes = self.votes.get(vote_key, {})
        
        # Count votes for this block
        block_votes = [v for v in votes.values() if v.block_hash == block_hash]
        
        # Calculate voting power
        total_voting_power = sum(v.voting_power for v in self.active_validators)
        voted_power = sum(self.validators[v.validator_address].voting_power 
                         for v in block_votes if v.validator_address in self.validators)
        
        if voted_power > (2 * total_voting_power) / 3:
            # We have a polka!
            round_state = self.round_states.get((height, round))
            if round_state:
                round_state.prevote_polka = block_hash
            
            # Update locked and valid values
            if self.locked_round < round:
                self.locked_value = block_hash
                self.locked_round = round
            
            self.valid_value = block_hash
            self.valid_round = round
            
            # Move to precommit step
            self.step = ConsensusState.PRECOMMIT
            self._set_timeout(height, round, ConsensusState.PRECOMMIT, self.timeout_precommit)
            
            # Send precommit for this block
            self._precommit(height, round, block_hash)
    
    def _check_precommit_polka(self, height: int, round: int, block_hash: str):
        """Check if we have +2/3 precommits for a block (commit)"""
        vote_key = (height, round, VoteType.PRECOMMIT)
        votes = self.votes.get(vote_key, {})
        
        # Count votes for this block
        block_votes = [v for v in votes.values() if v.block_hash == block_hash]
        
        # Calculate voting power
        total_voting_power = sum(v.voting_power for v in self.active_validators)
        voted_power = sum(self.validators[v.validator_address].voting_power 
                         for v in block_votes if v.validator_address in self.validators)
        
        if voted_power > (2 * total_voting_power) / 3:
            # We have a commit!
            round_state = self.round_states.get((height, round))
            if round_state:
                round_state.precommit_polka = block_hash
            
            # Commit the block
            self._commit_block(height, block_hash)
    
    def _commit_block(self, height: int, block_hash: str):
        """Commit a block and move to the next height"""
        with self.consensus_lock:
            # Get the block proposal
            proposal = self.block_proposals.get(block_hash)
            if not proposal:
                logger.error(f"Block proposal not found for hash: {block_hash}")
                return
            
            # Use ABCI to deliver transactions
            if self.abci.begin_block_fn:
                self.abci.begin_block_fn(height, block_hash)
            
            if self.abci.deliver_tx_fn:
                for tx_hash in proposal.tx_hashes:
                    if not self.abci.deliver_tx_fn(tx_hash):
                        logger.error(f"Failed to deliver transaction: {tx_hash}")
                        # In real implementation, this would be a serious error
            
            if self.abci.end_block_fn:
                self.abci.end_block_fn(height)
            
            # Get app hash from ABCI
            app_hash = self.abci.commit_fn() if self.abci.commit_fn else ""
            
            # Mark block as executed
            self.executed_blocks.add(block_hash)
            
            # Update validator stats
            with self.validator_lock:
                if proposal.validator_address in self.validators:
                    validator = self.validators[proposal.validator_address]
                    validator.last_active = time.time()
                    validator.signed_blocks += 1
                    
                    # Update uptime
                    total_blocks = validator.missed_blocks + validator.signed_blocks
                    if total_blocks > 0:
                        validator.uptime = (validator.signed_blocks / total_blocks) * 100
            
            # Move to next height
            self.step = ConsensusState.COMMITTED
            logger.info(f"Committed block {block_hash} at height {height}")
            
            # Start next height
            self.start_new_height(height + 1)
            
            self._save_state()
    
    def receive_vote(self, vote: Vote) -> bool:
        """
        Receive a vote from the network
        
        Returns:
            True if vote is valid and accepted, False otherwise
        """
        with self.consensus_lock:
            # Check if this vote is for the current height and round
            if vote.height != self.height or vote.round_number != self.round:
                logger.warning(f"Vote for wrong height/round: {vote.height}/{vote.round_number}, current: {self.height}/{self.round}")
                return False
            
            # Verify the validator is active
            if vote.validator_address not in self.validators:
                logger.warning(f"Vote from unknown validator: {vote.validator_address}")
                return False
            
            validator = self.validators[vote.validator_address]
            if validator.status != ValidatorStatus.ACTIVE:
                logger.warning(f"Vote from inactive validator: {vote.validator_address}")
                return False
            
            # Verify the signature
            if not self.verify_validator_signature(vote.validator_address, self._get_vote_signing_data(vote), vote.signature):
                logger.warning(f"Invalid vote signature from {vote.validator_address}")
                return False
            
            # Store the vote
            vote_key = (vote.height, vote.round_number, vote.vote_type)
            self.votes[vote_key][vote.validator_address] = vote
            
            # Check for polka if this is a prevote
            if vote.vote_type == VoteType.PREVOTE and vote.block_hash != "nil":
                self._check_prevote_polka(vote.height, vote.round_number, vote.block_hash)
            
            # Check for commit if this is a precommit
            if vote.vote_type == VoteType.PRECOMMIT and vote.block_hash != "nil":
                self._check_precommit_polka(vote.height, vote.round_number, vote.block_hash)
            
            return True
    
    def _proposal_signing_data(self, proposal: BlockProposal) -> bytes:
        """Get the data that should be signed for a proposal"""
        data = f"{proposal.height}|{proposal.round_number}|{proposal.block_hash}|{proposal.parent_hash}|{','.join(proposal.tx_hashes)}"
        return data.encode('utf-8')
    
    def _get_vote_signing_data(self, vote: Vote) -> bytes:
        """Get the data that should be signed for a vote"""
        data = f"{vote.height}|{vote.round_number}|{vote.vote_type.name}|{vote.block_hash}"
        return data.encode('utf-8')
    
    def verify_validator_signature(self, validator_address: str, data: bytes, signature: str) -> bool:
        """
        Verify a signature from a validator
        
        In a real implementation, this would use the validator's public key
        to verify the signature against the data.
        """
        # This is a placeholder implementation
        # In production, you would use proper cryptographic verification
        
        if validator_address not in self.validators:
            return False
        
        # For demonstration, we'll just check if the signature is a non-empty string
        return bool(signature and isinstance(signature, str))
    
    def add_reward(self, amount: int):
        """Add rewards to the epoch reward pool"""
        with self.lock:
            self.epoch_state.reward_pool += amount
    
    def slash_validator(self, validator_address: str, evidence: Dict, reporter: str) -> bool:
        """
        Slash a validator for misbehavior
        
        Args:
            validator_address: Address of validator to slash
            evidence: Dictionary containing evidence of misbehavior
            reporter: Address of the reporter (for potential rewards)
        
        Returns:
            True if slashing was successful, False otherwise
        """
        with self.validator_lock:
            if validator_address not in self.validators:
                logger.warning(f"Attempt to slash unknown validator: {validator_address}")
                return False
            
            validator = self.validators[validator_address]
            
            # Check evidence type
            evidence_type = evidence.get('type')
            
            if evidence_type == 'double_sign':
                # Verify double signing evidence
                if not self._verify_double_sign_evidence(validator, evidence):
                    logger.warning(f"Invalid double sign evidence for validator {validator_address}")
                    return False
                
                # Apply severe slashing for double signing
                slash_amount = int(validator.total_stake * self.slash_percentage)
                validator.staked_amount = max(0, validator.staked_amount - slash_amount)
                validator.status = ValidatorStatus.SLASHED
                validator.slashing_count += 1
                
                logger.warning(f"Validator {validator_address} slashed for double signing: {slash_amount}")
                
            elif evidence_type == 'unavailability':
                # Verify unavailability evidence
                missed_percentage = evidence.get('missed_percentage', 0)
                if missed_percentage < 0.5:  # Only slash if missed more than 50%
                    logger.warning(f"Insufficient unavailability evidence for validator {validator_address}: {missed_percentage}")
                    return False
                
                # Apply moderate slashing for unavailability
                slash_amount = int(validator.staked_amount * (self.slash_percentage / 2))
                validator.staked_amount = max(0, validator.staked_amount - slash_amount)
                validator.status = ValidatorStatus.JAILED
                validator.jail_until = time.time() + self.jail_duration
                validator.slashing_count += 1
                
                logger.warning(f"Validator {validator_address} jailed for unavailability: {slash_amount}")
            
            else:
                logger.warning(f"Unknown evidence type: {evidence_type}")
                return False
            
            # Update validator set if needed
            if validator.status != ValidatorStatus.ACTIVE:
                self._update_validator_set()
            
            self._save_state()
            return True
    
    def _verify_double_sign_evidence(self, validator: Validator, evidence: Dict) -> bool:
        """
        Verify double signing evidence
        
        In a real implementation, this would cryptographically verify
        that the validator signed two different blocks at the same height.
        """
        # Extract evidence
        block1 = evidence.get('block1')
        block2 = evidence.get('block2')
        signature1 = evidence.get('signature1')
        signature2 = evidence.get('signature2')
        
        if not all([block1, block2, signature1, signature2]):
            return False
        
        # Check if blocks are at the same height
        if block1.get('height') != block2.get('height'):
            return False
        
        # Check if blocks are different
        if block1.get('hash') == block2.get('hash'):
            return False
        
        # In real implementation, verify both signatures against validator's public key
        # For now, we'll assume the evidence is valid if all required fields are present
        
        return True
    
    def register_validator(self, address: str, public_key: str, stake_amount: int, commission_rate: float = 0.1) -> bool:
        """
        Register a new validator
        
        Args:
            address: Validator address
            public_key: Validator public key
            stake_amount: Amount of stake to lock
            commission_rate: Commission rate (0.0 to 1.0)
        
        Returns:
            True if registration successful, False otherwise
        """
        with self.validator_lock:
            if address in self.validators:
                logger.warning(f"Validator already registered: {address}")
                return False
            
            if stake_amount < self.min_stake:
                logger.warning(f"Insufficient stake: {stake_amount}, minimum: {self.min_stake}")
                return False
            
            if commission_rate < 0 or commission_rate > 1:
                logger.warning(f"Invalid commission rate: {commission_rate}")
                return False
            
            # Create new validator
            validator = Validator(
                address=address,
                public_key=public_key,
                staked_amount=stake_amount,
                commission_rate=commission_rate,
                status=ValidatorStatus.PENDING,
                created_block_height=self.height
            )
            
            self.validators[address] = validator
            self.pending_validators.append(validator)
            
            # Add to pending stakes for next epoch
            self.epoch_state.pending_stakes.append((address, stake_amount))
            
            self.update_total_stake()
            self._save_state()
            
            logger.info(f"Registered new validator: {address} with stake {stake_amount}")
            return True
    
    def stake(self, validator_address: str, amount: int) -> bool:
        """
        Add stake to an existing validator
        
        Args:
            validator_address: Validator address
            amount: Amount to stake
        
        Returns:
            True if staking successful, False otherwise
        """
        with self.validator_lock:
            if validator_address not in self.validators:
                logger.warning(f"Unknown validator: {validator_address}")
                return False
            
            # Add to pending stakes for next epoch
            self.epoch_state.pending_stakes.append((validator_address, amount))
            
            logger.info(f"Added stake {amount} to validator {validator_address}")
            return True
    
    def unstake(self, validator_address: str, amount: int) -> bool:
        """
        Remove stake from a validator
        
        Args:
            validator_address: Validator address
            amount: Amount to unstake
        
        Returns:
            True if unstaking successful, False otherwise
        """
        with self.validator_lock:
            if validator_address not in self.validators:
                logger.warning(f"Unknown validator: {validator_address}")
                return False
            
            validator = self.validators[validator_address]
            
            if amount > validator.staked_amount:
                logger.warning(f"Attempt to unstake more than available: {amount} > {validator.staked_amount}")
                return False
            
            # Add to pending unstakes for next epoch
            self.epoch_state.pending_unstakes.append((validator_address, amount))
            
            logger.info(f"Unstaked {amount} from validator {validator_address}")
            return True
    
    def delegate(self, delegator_address: str, validator_address: str, amount: int) -> bool:
        """
        Delegate stake to a validator
        
        Args:
            delegator_address: Delegator address
            validator_address: Validator address
            amount: Amount to delegate
        
        Returns:
            True if delegation successful, False otherwise
        """
        with self.validator_lock:
            if validator_address not in self.validators:
                logger.warning(f"Unknown validator: {validator_address}")
                return False
            
            # Add to pending delegations for next epoch
            self.epoch_state.pending_delegations.append((delegator_address, validator_address, amount))
            
            logger.info(f"Delegated {amount} to validator {validator_address} from {delegator_address}")
            return True
    
    def undelegate(self, delegator_address: str, validator_address: str, amount: int) -> bool:
        """
        Remove delegation from a validator
        
        Args:
            delegator_address: Delegator address
            validator_address: Validator address
            amount: Amount to undelegate
        
        Returns:
            True if undelegation successful, False otherwise
        """
        with self.validator_lock:
            if validator_address not in self.validators:
                logger.warning(f"Unknown validator: {validator_address}")
                return False
            
            validator = self.validators[validator_address]
            
            if delegator_address not in validator.delegators:
                logger.warning(f"No delegation found for {delegator_address} with validator {validator_address}")
                return False
            
            if amount > validator.delegators[delegator_address]:
                logger.warning(f"Attempt to undelegate more than delegated: {amount} > {validator.delegators[delegator_address]}")
                return False
            
            # Add to pending undelegations for next epoch
            self.epoch_state.pending_undelegations.append((delegator_address, validator_address, amount))
            
            logger.info(f"Undelegated {amount} from validator {validator_address} by {delegator_address}")
            return True
    
    def update_total_stake(self):
        """Update the total stake in the system"""
        with self.validator_lock:
            self.total_stake = sum(v.total_stake for v in self.validators.values())
    
    def get_validator(self, address: str) -> Optional[Validator]:
        """Get validator by address"""
        return self.validators.get(address)
    
    def get_active_validators(self) -> List[Validator]:
        """Get list of active validators"""
        return self.active_validators.copy()
    
    def get_consensus_state(self) -> Dict:
        """Get current consensus state"""
        return {
            'height': self.height,
            'round': self.round,
            'step': self.step.name,
            'locked_round': self.locked_round,
            'valid_round': self.valid_round,
            'locked_value': self.locked_value,
            'valid_value': self.valid_value,
            'total_stake': self.total_stake,
            'active_validators_count': len(self.active_validators),
            'total_validators_count': len(self.validators)
        }
    
    def shutdown(self):
        """Shutdown the consensus engine"""
        self._running = False
        
        # Cancel all timeouts
        for timer in self._timeout_handlers.values():
            timer.cancel()
        
        # Save final state
        self._save_state()
        
        # Close database
        self.db.close()
        
        logger.info("Consensus engine shutdown complete")