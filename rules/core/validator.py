"""
Validator management for Proof-of-Stake consensus
"""

import time
import threading
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging
import hashlib

from ..storage import BlockStore
from ..crypto import CryptoService
from ..config import ConsensusConfig
from ..utils import synchronized
from ..exceptions import ValidatorError

logger = logging.getLogger('consensus.validator')

@dataclass
class Validator:
    """Validator information"""
    address: str
    public_key: str
    staked_amount: int
    commission_rate: float
    status: str = "PENDING"
    voting_power: int = 0
    uptime: float = 100.0
    last_active: float = field(default_factory=time.time)
    missed_blocks: int = 0
    signed_blocks: int = 0
    jailed_until: Optional[float] = None

class ValidatorManager:
    """Manager for validator set and staking operations"""
    
    def __init__(self, config: ConsensusConfig, crypto_service: CryptoService, block_store: BlockStore):
        self.config = config
        self.crypto = crypto_service
        self.blocks = block_store
        
        self.validators: Dict[str, Validator] = {}
        self.active_validators: List[Validator] = []
        self.total_voting_power: int = 0
        
        self.lock = threading.RLock()
        
        self._load_validators()
        self._update_validator_set()
    
    @synchronized(lock)
    def _load_validators(self) -> None:
        """Load validators from storage"""
        try:
            validators_data = self.blocks.load_validators()
            for addr, data in validators_data.items():
                self.validators[addr] = Validator(**data)
        except Exception as e:
            logger.error(f"Failed to load validators: {e}")
    
    @synchronized(lock)
    def _save_validators(self) -> None:
        """Save validators to storage"""
        try:
            validators_data = {
                addr: {
                    'address': v.address,
                    'public_key': v.public_key,
                    'staked_amount': v.staked_amount,
                    'commission_rate': v.commission_rate,
                    'status': v.status,
                    'voting_power': v.voting_power,
                    'uptime': v.uptime,
                    'last_active': v.last_active,
                    'missed_blocks': v.missed_blocks,
                    'signed_blocks': v.signed_blocks,
                    'jailed_until': v.jailed_until
                }
                for addr, v in self.validators.items()
            }
            self.blocks.save_validators(validators_data)
        except Exception as e:
            logger.error(f"Failed to save validators: {e}")
    
    @synchronized(lock)
    def _update_validator_set(self) -> None:
        """Update active validator set based on stake"""
        # Sort by stake descending
        sorted_validators = sorted(
            [v for v in self.validators.values() if v.status == "ACTIVE"],
            key=lambda x: x.staked_amount,
            reverse=True
        )
        
        # Take top N validators
        self.active_validators = sorted_validators[:self.config.max_validators]
        
        # Update voting power (proportional to stake)
        total_stake = sum(v.staked_amount for v in self.active_validators)
        for validator in self.active_validators:
            if total_stake > 0:
                validator.voting_power = int((validator.staked_amount / total_stake) * 10000)
            else:
                validator.voting_power = 0
        
        self.total_voting_power = sum(v.voting_power for v in self.active_validators)
        
        logger.info(f"Updated validator set: {len(self.active_validators)} active validators")
    
    @synchronized(lock)
    def select_proposer(self, height: int, round_num: int) -> Optional[Validator]:
        """Select proposer for given height and round"""
        if not self.active_validators:
            return None
        
        # Deterministic selection based on height and round
        seed = hashlib.sha256(f"{height}_{round_num}".encode()).digest()
        seed_int = int.from_bytes(seed, 'big')
        
        # Weighted random selection by voting power
        total_power = self.total_voting_power
        if total_power == 0:
            return self.active_validators[0]
        
        selected = seed_int % total_power
        current = 0
        
        for validator in self.active_validators:
            current += validator.voting_power
            if selected < current:
                return validator
        
        return self.active_validators[-1]
    
    @synchronized(lock)
    def register_validator(self, address: str, public_key: str, stake_amount: int, commission_rate: float = 0.1) -> bool:
        """Register new validator"""
        if address in self.validators:
            raise ValidatorError(f"Validator already registered: {address}")
        
        if stake_amount < self.config.min_stake:
            raise ValidatorError(f"Insufficient stake: {stake_amount}")
        
        if not 0 <= commission_rate <= 1:
            raise ValidatorError("Commission rate must be between 0 and 1")
        
        validator = Validator(
            address=address,
            public_key=public_key,
            staked_amount=stake_amount,
            commission_rate=commission_rate,
            status="PENDING"
        )
        
        self.validators[address] = validator
        self._save_validators()
        
        logger.info(f"Registered new validator: {address}")
        return True
    
    @synchronized(lock)
    def stake(self, validator_address: str, amount: int) -> bool:
        """Add stake to validator"""
        if validator_address not in self.validators:
            raise ValidatorError(f"Unknown validator: {validator_address}")
        
        validator = self.validators[validator_address]
        validator.staked_amount += amount
        
        # If validator was pending and now meets minimum stake, activate
        if validator.status == "PENDING" and validator.staked_amount >= self.config.min_stake:
            validator.status = "ACTIVE"
        
        self._update_validator_set()
        self._save_validators()
        
        logger.info(f"Added {amount} stake to validator {validator_address}")
        return True
    
    @synchronized(lock)
    def unstake(self, validator_address: str, amount: int) -> bool:
        """Remove stake from validator"""
        if validator_address not in self.validators:
            raise ValidatorError(f"Unknown validator: {validator_address}")
        
        validator = self.validators[validator_address]
        
        if amount > validator.staked_amount:
            raise ValidatorError("Cannot unstake more than current stake")
        
        validator.staked_amount -= amount
        
        # If stake falls below minimum, deactivate
        if validator.staked_amount < self.config.min_stake and validator.status == "ACTIVE":
            validator.status = "PENDING"
        
        self._update_validator_set()
        self._save_validators()
        
        logger.info(f"Unstaked {amount} from validator {validator_address}")
        return True
    
    @synchronized(lock)
    def slash_validator(self, validator_address: str, slash_percentage: float, reason: str) -> bool:
        """Slash validator for misbehavior"""
        if validator_address not in self.validators:
            raise ValidatorError(f"Unknown validator: {validator_address}")
        
        validator = self.validators[validator_address]
        slash_amount = int(validator.staked_amount * slash_percentage)
        
        validator.staked_amount -= slash_amount
        validator.status = "JAILED"
        validator.jailed_until = time.time() + self.config.jail_duration
        
        self._update_validator_set()
        self._save_validators()
        
        logger.warning(f"Slashed validator {validator_address}: {slash_amount} tokens ({reason})")
        return True
    
    @synchronized(lock)
    def record_block_creation(self, validator_address: str) -> None:
        """Record that validator created a block"""
        if validator_address in self.validators:
            validator = self.validators[validator_address]
            validator.signed_blocks += 1
            validator.last_active = time.time()
            
            # Update uptime
            total_blocks = validator.missed_blocks + validator.signed_blocks
            if total_blocks > 0:
                validator.uptime = (validator.signed_blocks / total_blocks) * 100
            
            self._save_validators()
    
    @synchronized(lock)
    def record_block_miss(self, validator_address: str) -> None:
        """Record that validator missed a block"""
        if validator_address in self.validators:
            validator = self.validators[validator_address]
            validator.missed_blocks += 1
            
            # Update uptime
            total_blocks = validator.missed_blocks + validator.signed_blocks
            if total_blocks > 0:
                validator.uptime = (validator.signed_blocks / total_blocks) * 100
            
            self._save_validators()
    
    @synchronized(lock)
    def is_active_validator(self, address: str) -> bool:
        """Check if address is an active validator"""
        return address in self.validators and self.validators[address].status == "ACTIVE"
    
    @synchronized(lock)
    def is_current_validator(self, address: str) -> bool:
        """Check if address is the current validator (this node)"""
        # This would check if the address matches this node's validator address
        # For now, return True for demonstration
        return self.is_active_validator(address)
    
    @synchronized(lock)
    def get_voting_power(self, address: str) -> int:
        """Get voting power for validator"""
        if address in self.validators:
            return self.validators[address].voting_power
        return 0
    
    @synchronized(lock)
    def get_total_voting_power(self) -> int:
        """Get total voting power of all active validators"""
        return self.total_voting_power
    
    @synchronized(lock)
    def get_validator(self, address: str) -> Optional[Validator]:
        """Get validator by address"""
        return self.validators.get(address)
    
    @synchronized(lock)
    def get_active_validators(self) -> List[Validator]:
        """Get list of active validators"""
        return self.active_validators.copy()
    
    @synchronized(lock)
    def check_jailed_validators(self) -> None:
        """Check and release jailed validators whose jail time has expired"""
        current_time = time.time()
        
        for validator in self.validators.values():
            if validator.status == "JAILED" and validator.jailed_until and validator.jailed_until <= current_time:
                validator.status = "ACTIVE"
                validator.jailed_until = None
                logger.info(f"Released validator {validator.address} from jail")
        
        self._update_validator_set()
        self._save_validators()