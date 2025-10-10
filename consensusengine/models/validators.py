# consensus/models/validators.py
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum, auto
import logging

logger = logging.getLogger('ValidatorModel')

class ValidatorStatus(Enum):
    """Validator status levels"""
    ACTIVE = auto()
    JAILED = auto()
    INACTIVE = auto()
    SLASHED = auto()
    PENDING = auto()
    UNBONDING = auto()

@dataclass
class Validator:
    """Complete validator information with production-ready logic"""
    
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
    delegator_rewards: Dict[str, int] = field(default_factory=dict)  # Track rewards per delegator
    metadata: Dict[str, str] = field(default_factory=dict)  # Additional metadata
    # New fields for stake management
    last_stake_change: float = field(default_factory=time.time)  # Track last stake modification
    unbonding_end_time: Optional[float] = None  # When unbonding period ends
    min_self_delegation: int = 1000  # Minimum self-delegation requirement
    
    @property
    def total_stake(self) -> int:
        """Calculate total stake including delegations"""
        return self.staked_amount + self.total_delegated
    
    @property
    def effective_stake(self) -> int:
        """Calculate effective stake considering status and uptime"""
        if self.status in [ValidatorStatus.JAILED, ValidatorStatus.SLASHED]:
            return 0
        
        # Apply uptime penalty
        stake_penalty = 1.0 - ((100.0 - min(self.uptime, 100.0)) / 200.0)  # Max 50% penalty
        return int(self.total_stake * stake_penalty)
    
    @property
    def is_active(self) -> bool:
        """Check if validator is currently active"""
        if self.status != ValidatorStatus.ACTIVE:
            return False
        
        if self.jail_until and time.time() < self.jail_until:
            return False
            
        return self.effective_stake > 0
    
    @property
    def is_unbonding(self) -> bool:
        """Check if validator is in unbonding period"""
        if self.unbonding_end_time and time.time() < self.unbonding_end_time:
            return True
        return self.status == ValidatorStatus.UNBONDING

    def can_accept_stakes(self) -> bool:
        """
        Check if validator can accept new stakes
        
        Returns:
            bool: True if validator can accept new stakes, False otherwise
        """
        try:
            # Validators can only accept stakes when active or pending
            if self.status not in [ValidatorStatus.ACTIVE, ValidatorStatus.PENDING]:
                logger.debug(f"Validator {self.address} cannot accept stakes: status={self.status.name}")
                return False
            
            # Check if validator is jailed
            if self.jail_until and time.time() < self.jail_until:
                logger.debug(f"Validator {self.address} cannot accept stakes: currently jailed")
                return False
            
            # Check if validator is in unbonding period
            if self.is_unbonding:
                logger.debug(f"Validator {self.address} cannot accept stakes: in unbonding period")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if validator {self.address} can accept stakes: {e}")
            return False

    def can_unstake(self, amount: int) -> bool:
        """
        Check if validator can unstake the given amount
        
        Args:
            amount: Amount to unstake
            
        Returns:
            bool: True if validator can unstake, False otherwise
        """
        try:
            # Basic validation
            if amount <= 0:
                logger.warning(f"Invalid unstake amount: {amount}")
                return False
            
            # Check if validator is jailed
            if self.status == ValidatorStatus.JAILED:
                logger.debug(f"Validator {self.address} cannot unstake: currently jailed")
                return False
            
            # Check if validator has sufficient stake
            if amount > self.staked_amount:
                logger.warning(f"Validator {self.address} cannot unstake {amount}: only {self.staked_amount} available")
                return False
            
            # Check cooldown period (24 hours between stake changes)
            current_time = time.time()
            if current_time - self.last_stake_change < 86400:  # 24 hours
                logger.debug(f"Validator {self.address} cannot unstake: in cooldown period")
                return False
            
            # Check minimum self-delegation requirement after unstake
            remaining_stake = self.staked_amount - amount
            if remaining_stake < self.min_self_delegation and remaining_stake > 0:
                logger.warning(f"Validator {self.address} cannot unstake: would violate minimum self-delegation")
                return False
            
            # Check if validator is in unbonding period
            if self.is_unbonding:
                logger.debug(f"Validator {self.address} cannot unstake: already in unbonding period")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if validator {self.address} can unstake: {e}")
            return False

    def can_accept_delegations(self) -> bool:
        """
        Check if validator can accept new delegations
        
        Returns:
            bool: True if validator can accept new delegations, False otherwise
        """
        try:
            # Only active validators can accept delegations
            if self.status != ValidatorStatus.ACTIVE:
                logger.debug(f"Validator {self.address} cannot accept delegations: status={self.status.name}")
                return False
            
            # Check if validator is jailed
            if self.jail_until and time.time() < self.jail_until:
                logger.debug(f"Validator {self.address} cannot accept delegations: currently jailed")
                return False
            
            # Check if validator is in unbonding period
            if self.is_unbonding:
                logger.debug(f"Validator {self.address} cannot accept delegations: in unbonding period")
                return False
            
            # Check if validator meets minimum self-delegation
            if self.staked_amount < self.min_self_delegation:
                logger.debug(f"Validator {self.address} cannot accept delegations: insufficient self-delegation")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if validator {self.address} can accept delegations: {e}")
            return False

    def can_undelegate(self) -> bool:
        """
        Check if delegations can be removed from this validator
        
        Returns:
            bool: True if delegations can be removed, False otherwise
        """
        try:
            # Cannot remove delegations from jailed validators
            if self.status == ValidatorStatus.JAILED:
                logger.debug(f"Cannot undelegate from validator {self.address}: currently jailed")
                return False
            
            # Check if validator is in unbonding period
            if self.is_unbonding:
                logger.debug(f"Cannot undelegate from validator {self.address}: in unbonding period")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking if delegations can be removed from validator {self.address}: {e}")
            return False

    def update_uptime(self, signed: bool, window_size: int = 100) -> None:
        """
        Update validator uptime statistics
        
        Args:
            signed: Whether the validator signed the current block
            window_size: Rolling window size for uptime calculation
        """
        if signed:
            self.signed_blocks += 1
        else:
            self.missed_blocks += 1
        
        # Apply rolling window if we have enough blocks
        total_blocks = self.signed_blocks + self.missed_blocks
        if total_blocks > window_size:
            # Keep only the most recent window_size blocks
            excess_blocks = total_blocks - window_size
            if self.missed_blocks > excess_blocks:
                self.missed_blocks -= excess_blocks
            else:
                self.signed_blocks = window_size - (excess_blocks - self.missed_blocks)
                self.missed_blocks = 0
        
        # Recalculate uptime
        total_blocks = self.signed_blocks + self.missed_blocks
        if total_blocks > 0:
            self.uptime = (self.signed_blocks / total_blocks) * 100.0
    
    def add_delegation(self, delegator_address: str, amount: int) -> bool:
        """
        Add delegation to validator
        
        Args:
            delegator_address: Address of delegator
            amount: Amount to delegate
            
        Returns:
            True if successful, False otherwise
        """
        if amount <= 0:
            logger.warning(f"Invalid delegation amount: {amount}")
            return False
        
        if not self.can_accept_delegations():
            logger.warning(f"Validator {self.address} cannot accept new delegations")
            return False
        
        current_delegation = self.delegators.get(delegator_address, 0)
        self.delegators[delegator_address] = current_delegation + amount
        self.total_delegated += amount
        
        logger.debug(f"Added delegation: {delegator_address} -> {self.address}, amount: {amount}")
        return True
    
    def remove_delegation(self, delegator_address: str, amount: int) -> bool:
        """
        Remove delegation from validator
        
        Args:
            delegator_address: Address of delegator
            amount: Amount to undelegate
            
        Returns:
            True if successful, False otherwise
        """
        if delegator_address not in self.delegators:
            logger.warning(f"No delegation found for {delegator_address}")
            return False
        
        if not self.can_undelegate():
            logger.warning(f"Cannot remove delegation from validator {self.address}")
            return False
        
        current_delegation = self.delegators[delegator_address]
        if amount > current_delegation:
            logger.warning(f"Attempt to remove more than delegated: {amount} > {current_delegation}")
            return False
        
        self.delegators[delegator_address] = current_delegation - amount
        self.total_delegated -= amount
        
        # Remove delegator if delegation is zero
        if self.delegators[delegator_address] == 0:
            del self.delegators[delegator_address]
        
        logger.debug(f"Removed delegation: {delegator_address} -> {self.address}, amount: {amount}")
        return True
    
    def add_stake(self, amount: int) -> bool:
        """
        Add stake to validator's own stake
        
        Args:
            amount: Amount to stake
            
        Returns:
            True if successful, False otherwise
        """
        if amount <= 0:
            logger.warning(f"Invalid stake amount: {amount}")
            return False
        
        if not self.can_accept_stakes():
            logger.warning(f"Validator {self.address} cannot accept new stakes")
            return False
        
        self.staked_amount += amount
        self.last_stake_change = time.time()
        logger.debug(f"Added stake to validator {self.address}: {amount}")
        return True
    
    def remove_stake(self, amount: int) -> bool:
        """
        Remove stake from validator's own stake
        
        Args:
            amount: Amount to unstake
            
        Returns:
            True if successful, False otherwise
        """
        if amount <= 0:
            logger.warning(f"Invalid unstake amount: {amount}")
            return False
        
        if not self.can_unstake(amount):
            logger.warning(f"Validator {self.address} cannot unstake {amount}")
            return False
        
        self.staked_amount -= amount
        self.last_stake_change = time.time()
        
        # If stake drops below minimum, mark for unbonding
        if self.staked_amount < self.min_self_delegation and self.staked_amount > 0:
            self.status = ValidatorStatus.UNBONDING
            self.unbonding_end_time = time.time() + 86400 * 21  # 21-day unbonding period
            logger.warning(f"Validator {self.address} entered unbonding period due to low self-delegation")
        
        logger.debug(f"Removed stake from validator {self.address}: {amount}")
        return True
    
    def jail(self, duration: float) -> None:
        """
        Jail the validator for specified duration
        
        Args:
            duration: Jail duration in seconds
        """
        self.status = ValidatorStatus.JAILED
        self.jail_until = time.time() + duration
        logger.warning(f"Validator {self.address} jailed for {duration} seconds")
    
    def unjail(self) -> None:
        """Release validator from jail"""
        if self.status == ValidatorStatus.JAILED:
            self.status = ValidatorStatus.ACTIVE
            self.jail_until = None
            # Reset missed blocks counter when unjailed
            self.missed_blocks = 0
            logger.info(f"Validator {self.address} unjailed")
    
    def slash(self, amount: int) -> bool:
        """
        Slash validator's stake
        
        Args:
            amount: Amount to slash
            
        Returns:
            True if successful, False otherwise
        """
        if amount <= 0:
            logger.warning(f"Invalid slash amount: {amount}")
            return False
        
        # Slash from own stake first, then from delegators proportionally
        remaining_slash = amount
        
        # Slash from validator's own stake
        if self.staked_amount >= remaining_slash:
            self.staked_amount -= remaining_slash
            remaining_slash = 0
        else:
            remaining_slash -= self.staked_amount
            self.staked_amount = 0
        
        # Slash from delegators proportionally if needed
        if remaining_slash > 0 and self.total_delegated > 0:
            slash_ratio = remaining_slash / self.total_delegated
            
            for delegator_address in list(self.delegators.keys()):
                delegation_amount = self.delegators[delegator_address]
                delegator_slash = int(delegation_amount * slash_ratio)
                
                if delegator_slash > 0:
                    self.delegators[delegator_address] = delegation_amount - delegator_slash
                    remaining_slash -= delegator_slash
                    
                    # Remove if delegation becomes zero
                    if self.delegators[delegator_address] == 0:
                        del self.delegators[delegator_address]
            
            self.total_delegated -= (amount - remaining_slash)
        
        self.slashing_count += 1
        
        # Update last stake change time
        self.last_stake_change = time.time()
        
        logger.warning(f"Validator {self.address} slashed by {amount}")
        return True
    
    def calculate_voting_power(self, total_stake: int) -> int:
        """
        Calculate voting power based on effective stake
        
        Args:
            total_stake: Total stake in the system
            
        Returns:
            Voting power as integer
        """
        if total_stake == 0:
            return 0
        
        # Voting power is proportional to effective stake
        power = (self.effective_stake * 10000) // total_stake  # Scale to avoid floating point
        self.voting_power = power
        return power
    
    def to_dict(self) -> Dict:
        """Serialize validator to dictionary"""
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
            'delegators': self.delegators.copy(),
            'missed_blocks': self.missed_blocks,
            'signed_blocks': self.signed_blocks,
            'delegator_rewards': self.delegator_rewards.copy(),
            'metadata': self.metadata.copy(),
            'last_stake_change': self.last_stake_change,
            'unbonding_end_time': self.unbonding_end_time,
            'min_self_delegation': self.min_self_delegation
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Validator':
        """Deserialize validator from dictionary"""
        validator = cls(
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
            signed_blocks=data.get('signed_blocks', 0),
            delegator_rewards=data.get('delegator_rewards', {}),
            metadata=data.get('metadata', {})
        )
        
        # Set new fields with defaults if not present
        validator.last_stake_change = data.get('last_stake_change', time.time())
        validator.unbonding_end_time = data.get('unbonding_end_time')
        validator.min_self_delegation = data.get('min_self_delegation', 1000)
        
        return validator
    
    def validate(self) -> bool:
        """Validate validator data integrity"""
        if not self.address or len(self.address) != 42:  # Assuming 42 char address
            return False
        
        if not self.public_key:
            return False
        
        if self.staked_amount < 0:
            return False
        
        if self.total_delegated < 0:
            return False
        
        if self.commission_rate < 0 or self.commission_rate > 1:
            return False
        
        if self.uptime < 0 or self.uptime > 100:
            return False
        
        if self.last_stake_change < 0:
            return False
        
        if self.min_self_delegation < 0:
            return False
        
        return True
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Validator):
            return False
        return self.address == other.address
    
    def __hash__(self) -> int:
        return hash(self.address)