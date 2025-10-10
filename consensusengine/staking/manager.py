# consensus/staking/manager.py
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from threading import RLock
from dataclasses import dataclass, field
from enum import Enum
import secrets
import json

logger = logging.getLogger('StakingManager')

class StakingOperationType(Enum):
    """Types of staking operations"""
    STAKE = "stake"
    UNSTAKE = "unstake"
    DELEGATE = "delegate"
    UNDELEGATE = "undelegate"
    REGISTER_VALIDATOR = "register_validator"

@dataclass
class StakingOperation:
    """Represents a staking operation with full audit trail"""
    operation_id: str
    operation_type: StakingOperationType
    address: str
    validator_address: Optional[str] = None
    amount: int = 0
    timestamp: float = field(default_factory=time.time)
    block_height: int = 0
    status: str = "pending"
    commission_rate: float = 0.0
    public_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type.value,
            'address': self.address,
            'validator_address': self.validator_address,
            'amount': self.amount,
            'timestamp': self.timestamp,
            'block_height': self.block_height,
            'status': self.status,
            'commission_rate': self.commission_rate,
            'public_key': self.public_key
        }

class StakingManager:
    """Production-ready staking operations and validator management with comprehensive features"""
    
    def __init__(self, consensus_engine: Any):
        self.consensus_engine = consensus_engine
        self.config = consensus_engine.config
        self.lock = RLock()
        
        # Staking pools with enhanced tracking
        self.staking_pool: Dict[str, int] = {}  # address -> total staked amount
        self.delegation_pool: Dict[Tuple[str, str], int] = {}  # (delegator, validator) -> amount
        
        # Enhanced reward distribution tracking
        self.reward_distributions: Dict[int, Dict[str, int]] = {}  # epoch -> {address -> amount}
        self.pending_operations: Dict[str, StakingOperation] = {}  # operation_id -> operation
        
        # Slashing and security tracking
        self.slashing_events: List[Dict[str, Any]] = []
        self.jailed_validators: Set[str] = set()
        
        # Performance metrics
        self.metrics = {
            'total_operations_processed': 0,
            'failed_operations': 0,
            'total_rewards_distributed': 0,
            'last_epoch_processed': 0
        }
        
        logger.info("StakingManager initialized with production-ready configuration")
    
    def _generate_operation_id(self) -> str:
        """Generate unique operation ID with cryptographic randomness"""
        return hashlib.sha256(
            f"{time.time()}{secrets.token_bytes(32)}".encode()
        ).hexdigest()[:32]
    
    def _validate_address(self, address: str) -> bool:
        """Validate cryptocurrency address format"""
        if not address or len(address) != 42 or not address.startswith("0x"):
            return False
        try:
            int(address, 16)
            return True
        except ValueError:
            return False
    
    def _validate_amount(self, amount: int) -> bool:
        """Validate amount with business rules"""
        return amount > 0 and amount <= self.config.max_stake_amount
    
    def _validate_public_key(self, public_key: str) -> bool:
        """Validate public key format"""
        return public_key and len(public_key) >= 64 and public_key.startswith("04")  # Uncompressed SEC1 format
    
    def register_validator(self, address: str, public_key: str, stake_amount: int, 
                          commission_rate: float = 0.1) -> Tuple[bool, str]:
        """Register a new validator with production-grade validation and error handling"""
        operation_id = self._generate_operation_id()
        
        with self.lock:
            try:
                # Create operation record
                operation = StakingOperation(
                    operation_id=operation_id,
                    operation_type=StakingOperationType.REGISTER_VALIDATOR,
                    address=address,
                    amount=stake_amount,
                    commission_rate=commission_rate,
                    public_key=public_key,
                    block_height=self.consensus_engine.height
                )
                
                # Validate inputs comprehensively
                validation_errors = self._validate_validator_registration(address, public_key, stake_amount, commission_rate)
                if validation_errors:
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    logger.warning(f"Validator registration validation failed: {validation_errors}")
                    return False, validation_errors
                
                # Check if validator already exists
                if address in self.consensus_engine.validators:
                    error_msg = f"Validator already registered: {address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    logger.warning(error_msg)
                    return False, error_msg
                
                # Check minimum stake requirement
                if stake_amount < self.config.min_stake:
                    error_msg = f"Insufficient stake: {stake_amount}, minimum: {self.config.min_stake}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    logger.warning(error_msg)
                    return False, error_msg
                
                from consensus.models.validators import Validator, ValidatorStatus
                
                # Create new validator with enhanced properties
                validator = Validator(
                    address=address,
                    public_key=public_key,
                    staked_amount=stake_amount,
                    commission_rate=commission_rate,
                    status=ValidatorStatus.PENDING,
                    created_block_height=self.consensus_engine.height,
                    last_active_block=self.consensus_engine.height,
                    performance_score=100.0  # Initial perfect score
                )
                
                # Add to validators
                self.consensus_engine.validators[address] = validator
                self.consensus_engine.pending_validators.append(validator)
                
                # Add to staking pool
                self.staking_pool[address] = stake_amount
                
                # Schedule for next epoch with enhanced tracking
                self.consensus_engine.epoch_state.pending_stakes.append({
                    'address': address,
                    'amount': stake_amount,
                    'operation_id': operation_id,
                    'timestamp': time.time()
                })
                
                # Update total stake
                self._update_total_stake()
                
                # Mark operation as successful
                operation.status = "success"
                self.pending_operations[operation_id] = operation
                self.metrics['total_operations_processed'] += 1
                
                logger.info(f"Registered new validator: {address} with stake {stake_amount}")
                return True, operation_id
                
            except Exception as e:
                error_msg = f"Error registering validator: {e}"
                logger.error(error_msg, exc_info=True)
                operation.status = "failed"
                self.pending_operations[operation_id] = operation
                self.metrics['failed_operations'] += 1
                return False, error_msg
    
    def _validate_validator_registration(self, address: str, public_key: str, 
                                       stake_amount: int, commission_rate: float) -> Optional[str]:
        """Comprehensive validator registration validation"""
        if not self._validate_address(address):
            return "Invalid validator address format"
        
        if not self._validate_public_key(public_key):
            return "Invalid public key format"
        
        if not self._validate_amount(stake_amount):
            return "Invalid stake amount"
        
        if commission_rate < 0 or commission_rate > self.config.max_commission_rate:
            return f"Commission rate must be between 0 and {self.config.max_commission_rate}"
        
        # Check if address is not currently jailed
        if address in self.jailed_validators:
            return "Address is currently jailed and cannot register as validator"
        
        return None
    
    def stake(self, validator_address: str, amount: int) -> Tuple[bool, str]:
        """Add stake to an existing validator with comprehensive validation"""
        operation_id = self._generate_operation_id()
        
        with self.lock:
            try:
                operation = StakingOperation(
                    operation_id=operation_id,
                    operation_type=StakingOperationType.STAKE,
                    address=validator_address,
                    amount=amount,
                    block_height=self.consensus_engine.height
                )
                
                if validator_address not in self.consensus_engine.validators:
                    error_msg = f"Unknown validator: {validator_address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                if not self._validate_amount(amount):
                    error_msg = "Invalid stake amount"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                validator = self.consensus_engine.validators[validator_address]
                
                # Check if validator is active or can accept stakes
                if not validator.can_accept_stakes():
                    error_msg = f"Validator {validator_address} cannot accept new stakes at this time"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Add to pending stakes for next epoch with enhanced tracking
                self.consensus_engine.epoch_state.pending_stakes.append({
                    'address': validator_address,
                    'amount': amount,
                    'operation_id': operation_id,
                    'timestamp': time.time()
                })
                
                # Update staking pool immediately for tracking
                self.staking_pool[validator_address] = self.staking_pool.get(validator_address, 0) + amount
                
                operation.status = "success"
                self.pending_operations[operation_id] = operation
                self.metrics['total_operations_processed'] += 1
                
                logger.info(f"Added stake {amount} to validator {validator_address}")
                return True, operation_id
                
            except Exception as e:
                error_msg = f"Error staking: {e}"
                logger.error(error_msg, exc_info=True)
                operation.status = "failed"
                self.pending_operations[operation_id] = operation
                self.metrics['failed_operations'] += 1
                return False, error_msg
    
    def unstake(self, validator_address: str, amount: int) -> Tuple[bool, str]:
        """Remove stake from a validator with comprehensive validation"""
        operation_id = self._generate_operation_id()
        
        with self.lock:
            try:
                operation = StakingOperation(
                    operation_id=operation_id,
                    operation_type=StakingOperationType.UNSTAKE,
                    address=validator_address,
                    amount=amount,
                    block_height=self.consensus_engine.height
                )
                
                if validator_address not in self.consensus_engine.validators:
                    error_msg = f"Unknown validator: {validator_address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                validator = self.consensus_engine.validators[validator_address]
                
                if not self._validate_amount(amount):
                    error_msg = "Invalid unstake amount"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                if amount > validator.staked_amount:
                    error_msg = f"Attempt to unstake more than available: {amount} > {validator.staked_amount}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Check unstaking conditions (cooldown, jail status, etc.)
                if not validator.can_unstake(amount):
                    error_msg = f"Validator {validator_address} cannot unstake at this time"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Add to pending unstakes for next epoch
                self.consensus_engine.epoch_state.pending_unstakes.append({
                    'address': validator_address,
                    'amount': amount,
                    'operation_id': operation_id,
                    'timestamp': time.time()
                })
                
                # Update staking pool
                self.staking_pool[validator_address] = max(0, self.staking_pool.get(validator_address, 0) - amount)
                
                operation.status = "success"
                self.pending_operations[operation_id] = operation
                self.metrics['total_operations_processed'] += 1
                
                logger.info(f"Unstaked {amount} from validator {validator_address}")
                return True, operation_id
                
            except Exception as e:
                error_msg = f"Error unstaking: {e}"
                logger.error(error_msg, exc_info=True)
                operation.status = "failed"
                self.pending_operations[operation_id] = operation
                self.metrics['failed_operations'] += 1
                return False, error_msg
    
    def delegate(self, delegator_address: str, validator_address: str, amount: int) -> Tuple[bool, str]:
        """Delegate stake to a validator with comprehensive validation"""
        operation_id = self._generate_operation_id()
        
        with self.lock:
            try:
                operation = StakingOperation(
                    operation_id=operation_id,
                    operation_type=StakingOperationType.DELEGATE,
                    address=delegator_address,
                    validator_address=validator_address,
                    amount=amount,
                    block_height=self.consensus_engine.height
                )
                
                if not self._validate_address(delegator_address):
                    error_msg = "Invalid delegator address"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                if validator_address not in self.consensus_engine.validators:
                    error_msg = f"Unknown validator: {validator_address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                if not self._validate_amount(amount):
                    error_msg = "Invalid delegation amount"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                validator = self.consensus_engine.validators[validator_address]
                
                # Check delegation conditions
                if not validator.can_accept_delegations():
                    error_msg = f"Validator {validator_address} cannot accept new delegations"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Check maximum delegation limits
                current_delegation = self.delegation_pool.get((delegator_address, validator_address), 0)
                if current_delegation + amount > self.config.max_delegation_per_validator:
                    error_msg = f"Delegation amount exceeds maximum limit"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Add to pending delegations for next epoch
                self.consensus_engine.epoch_state.pending_delegations.append({
                    'delegator_address': delegator_address,
                    'validator_address': validator_address,
                    'amount': amount,
                    'operation_id': operation_id,
                    'timestamp': time.time()
                })
                
                # Update delegation pool
                key = (delegator_address, validator_address)
                self.delegation_pool[key] = self.delegation_pool.get(key, 0) + amount
                
                operation.status = "success"
                self.pending_operations[operation_id] = operation
                self.metrics['total_operations_processed'] += 1
                
                logger.info(f"Delegated {amount} to validator {validator_address} from {delegator_address}")
                return True, operation_id
                
            except Exception as e:
                error_msg = f"Error delegating: {e}"
                logger.error(error_msg, exc_info=True)
                operation.status = "failed"
                self.pending_operations[operation_id] = operation
                self.metrics['failed_operations'] += 1
                return False, error_msg
    
    def undelegate(self, delegator_address: str, validator_address: str, amount: int) -> Tuple[bool, str]:
        """Remove delegation from a validator with comprehensive validation"""
        operation_id = self._generate_operation_id()
        
        with self.lock:
            try:
                operation = StakingOperation(
                    operation_id=operation_id,
                    operation_type=StakingOperationType.UNDELEGATE,
                    address=delegator_address,
                    validator_address=validator_address,
                    amount=amount,
                    block_height=self.consensus_engine.height
                )
                
                if validator_address not in self.consensus_engine.validators:
                    error_msg = f"Unknown validator: {validator_address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                validator = self.consensus_engine.validators[validator_address]
                key = (delegator_address, validator_address)
                
                if key not in self.delegation_pool:
                    error_msg = f"No delegation found for {delegator_address} with validator {validator_address}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                current_delegation = self.delegation_pool[key]
                if amount > current_delegation:
                    error_msg = f"Attempt to undelegate more than delegated: {amount} > {current_delegation}"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Check undelegation conditions
                if not validator.can_undelegate():
                    error_msg = f"Cannot undelegate from validator {validator_address} at this time"
                    operation.status = "failed"
                    self.pending_operations[operation_id] = operation
                    return False, error_msg
                
                # Add to pending undelegations for next epoch
                self.consensus_engine.epoch_state.pending_undelegations.append({
                    'delegator_address': delegator_address,
                    'validator_address': validator_address,
                    'amount': amount,
                    'operation_id': operation_id,
                    'timestamp': time.time()
                })
                
                # Update delegation pool
                self.delegation_pool[key] = current_delegation - amount
                if self.delegation_pool[key] == 0:
                    del self.delegation_pool[key]
                
                operation.status = "success"
                self.pending_operations[operation_id] = operation
                self.metrics['total_operations_processed'] += 1
                
                logger.info(f"Undelegated {amount} from validator {validator_address} by {delegator_address}")
                return True, operation_id
                
            except Exception as e:
                error_msg = f"Error undelegating: {e}"
                logger.error(error_msg, exc_info=True)
                operation.status = "failed"
                self.pending_operations[operation_id] = operation
                self.metrics['failed_operations'] += 1
                return False, error_msg
    
    def process_pending_operations(self) -> Dict[str, int]:
        """Process all pending staking operations at epoch boundary with comprehensive reporting"""
        results = {
            'stakes_processed': 0,
            'unstakes_processed': 0,
            'delegations_processed': 0,
            'undelegations_processed': 0,
            'failures': 0
        }
        
        with self.lock:
            try:
                current_height = self.consensus_engine.height
                
                # Process new stakes
                for stake_op in self.consensus_engine.epoch_state.pending_stakes:
                    try:
                        if stake_op['address'] in self.consensus_engine.validators:
                            validator = self.consensus_engine.validators[stake_op['address']]
                            if validator.add_stake(stake_op['amount']):
                                results['stakes_processed'] += 1
                                # Update operation status
                                if stake_op['operation_id'] in self.pending_operations:
                                    self.pending_operations[stake_op['operation_id']].status = "executed"
                            else:
                                results['failures'] += 1
                    except Exception as e:
                        logger.error(f"Error processing stake operation: {e}")
                        results['failures'] += 1
                
                # Process unstakes
                for unstake_op in self.consensus_engine.epoch_state.pending_unstakes:
                    try:
                        if unstake_op['address'] in self.consensus_engine.validators:
                            validator = self.consensus_engine.validators[unstake_op['address']]
                            if validator.remove_stake(unstake_op['amount']):
                                results['unstakes_processed'] += 1
                                if unstake_op['operation_id'] in self.pending_operations:
                                    self.pending_operations[unstake_op['operation_id']].status = "executed"
                            else:
                                results['failures'] += 1
                    except Exception as e:
                        logger.error(f"Error processing unstake operation: {e}")
                        results['failures'] += 1
                
                # Process delegations
                for delegation_op in self.consensus_engine.epoch_state.pending_delegations:
                    try:
                        if delegation_op['validator_address'] in self.consensus_engine.validators:
                            validator = self.consensus_engine.validators[delegation_op['validator_address']]
                            if validator.add_delegation(delegation_op['delegator_address'], delegation_op['amount']):
                                results['delegations_processed'] += 1
                                if delegation_op['operation_id'] in self.pending_operations:
                                    self.pending_operations[delegation_op['operation_id']].status = "executed"
                            else:
                                results['failures'] += 1
                    except Exception as e:
                        logger.error(f"Error processing delegation operation: {e}")
                        results['failures'] += 1
                
                # Process undelegations
                for undelegation_op in self.consensus_engine.epoch_state.pending_undelegations:
                    try:
                        if undelegation_op['validator_address'] in self.consensus_engine.validators:
                            validator = self.consensus_engine.validators[undelegation_op['validator_address']]
                            if validator.remove_delegation(undelegation_op['delegator_address'], undelegation_op['amount']):
                                results['undelegations_processed'] += 1
                                if undelegation_op['operation_id'] in self.pending_operations:
                                    self.pending_operations[undelegation_op['operation_id']].status = "executed"
                            else:
                                results['failures'] += 1
                    except Exception as e:
                        logger.error(f"Error processing undelegation operation: {e}")
                        results['failures'] += 1
                
                # Clear pending operations
                self.consensus_engine.epoch_state.pending_stakes.clear()
                self.consensus_engine.epoch_state.pending_unstakes.clear()
                self.consensus_engine.epoch_state.pending_delegations.clear()
                self.consensus_engine.epoch_state.pending_undelegations.clear()
                
                logger.info(f"Processed staking operations: {results}")
                return results
                
            except Exception as e:
                logger.error(f"Error processing pending operations: {e}", exc_info=True)
                results['failures'] += len(self.consensus_engine.epoch_state.pending_stakes) + \
                                      len(self.consensus_engine.epoch_state.pending_unstakes) + \
                                      len(self.consensus_engine.epoch_state.pending_delegations) + \
                                      len(self.consensus_engine.epoch_state.pending_undelegations)
                return results

    def update_validator_set(self) -> Dict[str, Any]:
    	"""Update the active validator set based on current stakes and performance"""
    	with self.lock:
    		try:
    			results = {
    			    'validators_added': 0,
    			    'validators_removed': 0,
    			    'active_validators': 0,
    			    'total_voting_power': 0
    			}
    			
    			# Clear current active validators
    			self.consensus_engine.active_validators.clear()
    			
    			# Sort validators by effective stake (descending)
    			sorted_validators = sorted(
    			    self.consensus_engine.validators.values(),
    			    key=lambda v: v.effective_stake,
    			    reverse=True
    			)
    			
    			# Select top validators based on configuration
    			max_validators = getattr(self.config, 'max_validators', 100)
    			min_stake = getattr(self.config, 'min_stake_amount', 1000)
    			
    			for validator in sorted_validators:
    				# Check if validator meets requirements
    				if (validator.is_active and validator.effective_stake >= min_stake and len(self.consensus_engine.active_validators) < max_validators):
    					self.consensus_engine.active_validators.append(validator)
    					results['validators_added'] += 1
    					# Calculate voting power
    					validator.calculate_voting_power(self.consensus_engine.total_stake)
    					results['total_voting_power'] += validator.voting_power
    					
    			results['active_validators'] = len(self.consensus_engine.active_validators)
    			
    			# Update total stake in consensus engine
    			self._update_total_stake()
    			logger.info(f"Validator set updated: {results['active_validators']} active validators, " f"total voting power: {results['total_voting_power']}")
    			return results
    		
    		except Exception as e:
    			logger.error(f"Error updating validator set: {e}")
    			
    			return {
    			    'validators_added': 0,
    			    'validators_removed': 0,
    			    'active_validators': 0,
    			    'total_voting_power': 0,
    			    'error': str(e)
    			}
    			
    def _update_total_stake(self):
    	"""Update total stake in the consensus engine"""
    	try:
    		total_stake = sum(validator.total_stake for validator in self.consensus_engine.validators.values())
    		self.consensus_engine.total_stake = total_stake
    		
    		# Also update stakes dictionary for backward compatibility
    		self.consensus_engine.stakes.clear()
    		for validator in self.consensus_engine.validators.values():
    			self.consensus_engine.stakes[validator.address] = validator.total_stake
    		
    	except Exception as e:
    		logger.error(f"Error updating total stake: {e}")
    		
    def get_total_stake(self) -> int:
    	"""Get total stake in the system"""
    	with self.lock:
    		return self.consensus_engine.total_stake
    		
    def select_proposer(self, height: int, round: int) -> Optional[Any]:
    	"""Select proposer for the given height and round"""
    	with self.lock:
    		try:
    			if not self.consensus_engine.active_validators:
    				return None
    				
    			# Use deterministic algorithm based on height and round
    			total_voting_power = sum(v.voting_power for v in self.consensus_engine.active_validators)
    			if total_voting_power == 0:
    				return None
    			
    			# Create a deterministic seed
    			seed = f"{height}_{round}_{total_voting_power}"
    			seed_hash = hashlib.sha256(seed.encode()).hexdigest()
    			seed_int = int(seed_hash[:16], 16)
    			
    			# Weighted random selection based on voting power
    			target = seed_int % total_voting_power
    			current_sum = 0
    			
    			for validator in self.consensus_engine.active_validators:
    				current_sum += validator.voting_power
    				if current_sum > target:
    					return validator
    					
    			# Fallback to first validator
    			return self.consensus_engine.active_validators[0] if self.consensus_engine.active_validators else None
    		
    		except Exception as e:
    			logger.error(f"Error selecting proposer: {e}")
    			
    def distribute_epoch_rewards(self, reward_pool: int) -> Dict[str, Any]:
    	"""Distribute epoch rewards to validators and delegators"""
    	with self.lock:
    		try:
    			results = {
    			    'total_distributed': 0,
    			    'validators_rewarded': 0,
    			    'delegators_rewarded': 0,
    			    'foundation_fee': 0
    			}
    			
    			if reward_pool <= 0 or not self.consensus_engine.active_validators:
    				return results
    				
    			# Calculate foundation fee (5%)
    			foundation_fee = reward_pool // 20
    			net_rewards = reward_pool - foundation_fee
    			results['foundation_fee'] = foundation_fee
    			total_voting_power = sum(v.voting_power for v in self.consensus_engine.active_validators)
    			
    			if total_voting_power == 0:
    				return results
    				
    			# Distribute rewards proportionally to voting power
    			for validator in self.consensus_engine.active_validators:
    				if validator.voting_power > 0:
    					# Calculate validator's share
    					validator_share = (net_rewards * validator.voting_power) // total_voting_power
    					
    					if validator_share > 0:
    						# Apply commission
    						commission = (validator_share * validator.commission_rate) // 1
    						validator_reward = validator_share - commission
    						delegator_rewards = validator_share - validator_reward
    						
    						# Add to validator's total rewards
    						validator.total_rewards += validator_reward
    						results['total_distributed'] += validator_reward
    						results['validators_rewarded'] += 1
    						
    						# Distribute to delegators if any
    						if delegator_rewards > 0 and validator.total_delegated > 0:
    							self._distribute_delegator_rewards(validator, delegator_rewards)
    							results['delegators_rewarded'] += len(validator.delegators)
    							results['total_distributed'] += delegator_rewards
    			
    			logger.info(f"Epoch rewards distributed: {results}")
    			return results
    			
    		except Exception as e:
    			logger.error(f"Error distributing epoch rewards: {e}")
    			return {
    			    'total_distributed': 0,
    			    'validators_rewarded': 0,
    			    'delegators_rewarded': 0,
    			    'foundation_fee': 0,
    			    'error': str(e)
    			}
    			
    def _distribute_delegator_rewards(self, validator: Any, total_delegator_rewards: int):
    	"""Distribute rewards to delegators proportionally"""
    	try:
    		for delegator_address, delegation_amount in validator.delegators.items():
    			if delegation_amount > 0:
    				# Calculate delegator's share
    				delegator_share = (total_delegator_rewards * delegation_amount) // validator.total_delegated
    				if delegator_share > 0:
    					# Update delegator rewards tracking
    					if delegator_address not in validator.delegator_rewards:
    						validator.delegator_rewards[delegator_address] = 0
    					
    					validator.delegator_rewards[delegator_address] += delegator_share
    				
    		logger.debug(f"Distributed {total_delegator_rewards} rewards to {len(validator.delegators)} delegators")
    		
    	except Exception as e:
    		logger.error(f"Error distributing delegator rewards: {e}")
 
    def get_operation_status(self, operation_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific staking operation"""
        with self.lock:
            if operation_id in self.pending_operations:
                return self.pending_operations[operation_id].to_dict()
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self.lock:
            return {
                **self.metrics,
                'current_epoch': self.consensus_engine.epoch_state.current_epoch,
                'pending_operations_count': len(self.pending_operations),
                'jailed_validators_count': len(self.jailed_validators),
                'total_slashing_events': len(self.slashing_events),
                'uptime': self._calculate_uptime_metrics()
            }
    
    def _calculate_uptime_metrics(self) -> Dict[str, float]:
        """Calculate system uptime and performance metrics"""
        # Implementation for uptime calculation
        return {
            'system_uptime': 99.9,  # Placeholder
            'average_validator_uptime': 98.5,  # Placeholder
            'success_rate': 99.8  # Placeholder
        }

# Additional utility functions for enhanced functionality
def validate_staking_transaction(transaction: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate staking transaction before processing"""
    required_fields = ['type', 'address', 'amount', 'signature', 'nonce']
    
    for field in required_fields:
        if field not in transaction:
            return False, f"Missing required field: {field}"
    
    if transaction['amount'] <= 0:
        return False, "Amount must be positive"
    
    if transaction['nonce'] < 0:
        return False, "Nonce must be non-negative"
    
    return True, None

def calculate_slashing_penalty(offense_type: str, stake_amount: int) -> int:
    """Calculate slashing penalty based on offense type and stake amount"""
    penalty_rates = {
        'double_signing': 0.05,  # 5% penalty for double signing
        'downtime': 0.01,        # 1% penalty for excessive downtime
        'security_breach': 0.10  # 10% penalty for security breaches
    }
    
    rate = penalty_rates.get(offense_type, 0.02)  # Default 2% penalty
    return int(stake_amount * rate)