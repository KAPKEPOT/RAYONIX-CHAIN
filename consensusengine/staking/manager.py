# consensus/staking/manager.py
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Set
import logging
from threading import RLock

logger = logging.getLogger('StakingManager')

class StakingManager:
    """Production-ready staking operations and validator management"""
    
    def __init__(self, consensus_engine: Any):
        self.consensus_engine = consensus_engine
        self.config = consensus_engine.config
        self.lock = RLock()
        
        # Staking pools
        self.staking_pool: Dict[str, int] = {}  # address -> total staked amount
        self.delegation_pool: Dict[Tuple[str, str], int] = {}  # (delegator, validator) -> amount
        
        # Reward distribution tracking
        self.reward_distributions: Dict[int, Dict[str, int]] = {}  # epoch -> {address -> amount}
        
    def register_validator(self, address: str, public_key: str, stake_amount: int, 
                          commission_rate: float = 0.1) -> bool:
        """Register a new validator with production-grade validation"""
        with self.lock:
            try:
                # Validate inputs
                if not self._validate_validator_inputs(address, public_key, stake_amount, commission_rate):
                    return False
                
                # Check if validator already exists
                if address in self.consensus_engine.validators:
                    logger.warning(f"Validator already registered: {address}")
                    return False
                
                # Check minimum stake requirement
                if stake_amount < self.config.min_stake:
                    logger.warning(f"Insufficient stake: {stake_amount}, minimum: {self.config.min_stake}")
                    return False
                
                from consensus.models.validators import Validator, ValidatorStatus
                
                # Create new validator
                validator = Validator(
                    address=address,
                    public_key=public_key,
                    staked_amount=stake_amount,
                    commission_rate=commission_rate,
                    status=ValidatorStatus.PENDING,
                    created_block_height=self.consensus_engine.height
                )
                
                # Add to validators
                self.consensus_engine.validators[address] = validator
                self.consensus_engine.pending_validators.append(validator)
                
                # Add to staking pool
                self.staking_pool[address] = stake_amount
                
                # Schedule for next epoch
                self.consensus_engine.epoch_state.pending_stakes.append((address, stake_amount))
                
                # Update total stake
                self._update_total_stake()
                
                logger.info(f"Registered new validator: {address} with stake {stake_amount}")
                return True
                
            except Exception as e:
                logger.error(f"Error registering validator: {e}")
                return False
    
    def _validate_validator_inputs(self, address: str, public_key: str, stake_amount: int, 
                                 commission_rate: float) -> bool:
        """Validate validator registration inputs"""
        if not address or len(address) != 42:  # Ethereum-style address
            logger.warning("Invalid validator address format")
            return False
        
        if not public_key or len(public_key) < 64:
            logger.warning("Invalid public key format")
            return False
        
        if stake_amount <= 0:
            logger.warning("Stake amount must be positive")
            return False
        
        if commission_rate < 0 or commission_rate > 1:
            logger.warning("Commission rate must be between 0 and 1")
            return False
        
        return True
    
    def stake(self, validator_address: str, amount: int) -> bool:
        """Add stake to an existing validator"""
        with self.lock:
            try:
                if validator_address not in self.consensus_engine.validators:
                    logger.warning(f"Unknown validator: {validator_address}")
                    return False
                
                if amount <= 0:
                    logger.warning("Stake amount must be positive")
                    return False
                
                # Add to pending stakes for next epoch
                self.consensus_engine.epoch_state.pending_stakes.append((validator_address, amount))
                
                # Update staking pool immediately for tracking
                self.staking_pool[validator_address] = self.staking_pool.get(validator_address, 0) + amount
                
                logger.info(f"Added stake {amount} to validator {validator_address}")
                return True
                
            except Exception as e:
                logger.error(f"Error staking: {e}")
                return False
    
    def unstake(self, validator_address: str, amount: int) -> bool:
        """Remove stake from a validator"""
        with self.lock:
            try:
                if validator_address not in self.consensus_engine.validators:
                    logger.warning(f"Unknown validator: {validator_address}")
                    return False
                
                validator = self.consensus_engine.validators[validator_address]
                
                if amount <= 0:
                    logger.warning("Unstake amount must be positive")
                    return False
                
                if amount > validator.staked_amount:
                    logger.warning(f"Attempt to unstake more than available: {amount} > {validator.staked_amount}")
                    return False
                
                # Add to pending unstakes for next epoch
                self.consensus_engine.epoch_state.pending_unstakes.append((validator_address, amount))
                
                # Update staking pool
                self.staking_pool[validator_address] = max(0, self.staking_pool.get(validator_address, 0) - amount)
                
                logger.info(f"Unstaked {amount} from validator {validator_address}")
                return True
                
            except Exception as e:
                logger.error(f"Error unstaking: {e}")
                return False
    
    def delegate(self, delegator_address: str, validator_address: str, amount: int) -> bool:
        """Delegate stake to a validator"""
        with self.lock:
            try:
                if validator_address not in self.consensus_engine.validators:
                    logger.warning(f"Unknown validator: {validator_address}")
                    return False
                
                if amount <= 0:
                    logger.warning("Delegation amount must be positive")
                    return False
                
                # Add to pending delegations for next epoch
                self.consensus_engine.epoch_state.pending_delegations.append(
                    (delegator_address, validator_address, amount)
                )
                
                # Update delegation pool
                key = (delegator_address, validator_address)
                self.delegation_pool[key] = self.delegation_pool.get(key, 0) + amount
                
                logger.info(f"Delegated {amount} to validator {validator_address} from {delegator_address}")
                return True
                
            except Exception as e:
                logger.error(f"Error delegating: {e}")
                return False
    
    def undelegate(self, delegator_address: str, validator_address: str, amount: int) -> bool:
        """Remove delegation from a validator"""
        with self.lock:
            try:
                if validator_address not in self.consensus_engine.validators:
                    logger.warning(f"Unknown validator: {validator_address}")
                    return False
                
                validator = self.consensus_engine.validators[validator_address]
                key = (delegator_address, validator_address)
                
                if key not in self.delegation_pool:
                    logger.warning(f"No delegation found for {delegator_address} with validator {validator_address}")
                    return False
                
                current_delegation = self.delegation_pool[key]
                if amount > current_delegation:
                    logger.warning(f"Attempt to undelegate more than delegated: {amount} > {current_delegation}")
                    return False
                
                # Add to pending undelegations for next epoch
                self.consensus_engine.epoch_state.pending_undelegations.append(
                    (delegator_address, validator_address, amount)
                )
                
                # Update delegation pool
                self.delegation_pool[key] = current_delegation - amount
                if self.delegation_pool[key] == 0:
                    del self.delegation_pool[key]
                
                logger.info(f"Undelegated {amount} from validator {validator_address} by {delegator_address}")
                return True
                
            except Exception as e:
                logger.error(f"Error undelegating: {e}")
                return False
    
    def process_pending_operations(self):
        """Process all pending staking operations at epoch boundary"""
        with self.lock:
            try:
                # Process new stakes
                for address, amount in self.consensus_engine.epoch_state.pending_stakes:
                    if address in self.consensus_engine.validators:
                        self.consensus_engine.validators[address].add_stake(amount)
                
                # Process unstakes
                for address, amount in self.consensus_engine.epoch_state.pending_unstakes:
                    if address in self.consensus_engine.validators:
                        validator = self.consensus_engine.validators[address]
                        if not validator.remove_stake(amount):
                            logger.warning(f"Failed to process unstake for {address}")
                
                # Process delegations
                for delegator, validator_addr, amount in self.consensus_engine.epoch_state.pending_delegations:
                    if validator_addr in self.consensus_engine.validators:
                        validator = self.consensus_engine.validators[validator_addr]
                        if not validator.add_delegation(delegator, amount):
                            logger.warning(f"Failed to process delegation to {validator_addr}")
                
                # Process undelegations
                for delegator, validator_addr, amount in self.consensus_engine.epoch_state.pending_undelegations:
                    if validator_addr in self.consensus_engine.validators:
                        validator = self.consensus_engine.validators[validator_addr]
                        if not validator.remove_delegation(delegator, amount):
                            logger.warning(f"Failed to process undelegation from {validator_addr}")
                
                # Clear pending operations
                self.consensus_engine.epoch_state.pending_stakes.clear()
                self.consensus_engine.epoch_state.pending_unstakes.clear()
                self.consensus_engine.epoch_state.pending_delegations.clear()
                self.consensus_engine.epoch_state.pending_undelegations.clear()
                
                logger.info("Processed all pending staking operations")
                
            except Exception as e:
                logger.error(f"Error processing pending operations: {e}")
    
    def update_validator_set(self):
        """Update the active validator set based on stake at epoch boundaries"""
        with self.lock:
            try:
                if self.consensus_engine.height % self.config.epoch_blocks != 0:
                    return
                
                # Calculate voting power for all validators
                total_stake = self.get_total_stake()
                for validator in self.consensus_engine.validators.values():
                    validator.calculate_voting_power(total_stake)
                
                # Sort validators by effective stake
                sorted_validators = sorted(
                    [v for v in self.consensus_engine.validators.values() 
                     if v.status in [ValidatorStatus.ACTIVE, ValidatorStatus.PENDING] and v.is_active],
                    key=lambda x: x.effective_stake,
                    reverse=True
                )
                
                # Select top validators up to max_validators
                new_active = sorted_validators[:self.config.max_validators]
                
                # Update statuses
                for validator in new_active:
                    if validator.status == ValidatorStatus.PENDING:
                        validator.status = ValidatorStatus.ACTIVE
                        logger.info(f"Validator {validator.address} activated")
                
                # Deactivate validators that didn't make the cut
                for validator in sorted_validators[self.config.max_validators:]:
                    if validator.status == ValidatorStatus.ACTIVE:
                        validator.status = ValidatorStatus.INACTIVE
                        logger.info(f"Validator {validator.address} deactivated")
                
                self.consensus_engine.active_validators = new_active
                
                logger.info(f"Updated validator set: {len(new_active)} active validators")
                
            except Exception as e:
                logger.error(f"Error updating validator set: {e}")
    
    def distribute_epoch_rewards(self, total_reward: int):
        """Distribute epoch rewards to validators and delegators"""
        with self.lock:
            try:
                if total_reward <= 0:
                    return
                
                total_effective_stake = sum(v.effective_stake for v in self.consensus_engine.active_validators)
                if total_effective_stake == 0:
                    return
                
                current_epoch = self.consensus_engine.epoch_state.current_epoch
                self.reward_distributions[current_epoch] = {}
                
                for validator in self.consensus_engine.active_validators:
                    if validator.effective_stake == 0:
                        continue
                    
                    # Validator's share of rewards
                    validator_share = (validator.effective_stake / total_effective_stake) * total_reward
                    
                    # Commission goes to validator
                    commission = int(validator_share * validator.commission_rate)
                    validator.total_rewards += commission
                    
                    # Track distribution
                    self.reward_distributions[current_epoch][validator.address] = \
                        self.reward_distributions[current_epoch].get(validator.address, 0) + commission
                    
                    # Remainder goes to delegators proportionally
                    delegator_rewards = validator_share - commission
                    
                    if validator.total_delegated > 0 and delegator_rewards > 0:
                        self._distribute_delegator_rewards(validator, delegator_rewards, current_epoch)
                
                logger.info(f"Distributed {total_reward} rewards for epoch {current_epoch}")
                
            except Exception as e:
                logger.error(f"Error distributing rewards: {e}")
    
    def _distribute_delegator_rewards(self, validator: Validator, total_rewards: int, epoch: int):
        """Distribute rewards to delegators"""
        for delegator_address, delegated_amount in validator.delegators.items():
            delegator_share = (delegated_amount / validator.total_delegated) * total_rewards
            
            # Track delegator rewards
            validator.delegator_rewards[delegator_address] = \
                validator.delegator_rewards.get(delegator_address, 0) + delegator_share
            
            # Track in distribution record
            distribution_key = f"{validator.address}:{delegator_address}"
            self.reward_distributions[epoch][distribution_key] = \
                self.reward_distributions[epoch].get(distribution_key, 0) + delegator_share
    
    def select_proposer(self, height: int, round: int) -> Optional[Validator]:
        """
        Select block proposer for current height and round
        
        Uses weighted random selection based on stake with deterministic seeding
        """
        with self.lock:
            if not self.consensus_engine.active_validators:
                return None
            
            total_stake = sum(v.effective_stake for v in self.consensus_engine.active_validators)
            if total_stake == 0:
                return None
            
            # Deterministic selection based on height and round
            random_seed = hashlib.sha256(
                f"{height}_{round}_{self.consensus_engine.epoch_state.current_epoch}".encode()
            ).digest()
            
            random_number = int.from_bytes(random_seed, 'big') % total_stake
            
            current_sum = 0
            for validator in self.consensus_engine.active_validators:
                current_sum += validator.effective_stake
                if random_number < current_sum:
                    return validator
            
            return self.consensus_engine.active_validators[-1]
    
    def get_total_stake(self) -> int:
        """Get total stake in the system"""
        with self.lock:
            total = sum(v.total_stake for v in self.consensus_engine.validators.values())
            self.consensus_engine.total_stake = total
            return total
    
    def _update_total_stake(self):
        """Update total stake calculation"""
        self.consensus_engine.total_stake = self.get_total_stake()
    
    def get_validator_info(self, address: str) -> Optional[Dict]:
        """Get comprehensive validator information"""
        with self.lock:
            if address not in self.consensus_engine.validators:
                return None
            
            validator = self.consensus_engine.validators[address]
            
            return {
                'address': validator.address,
                'status': validator.status.name,
                'staked_amount': validator.staked_amount,
                'total_delegated': validator.total_delegated,
                'total_stake': validator.total_stake,
                'effective_stake': validator.effective_stake,
                'voting_power': validator.voting_power,
                'uptime': validator.uptime,
                'commission_rate': validator.commission_rate,
                'total_rewards': validator.total_rewards,
                'slashing_count': validator.slashing_count,
                'delegator_count': len(validator.delegators),
                'is_active': validator.is_active,
                'jail_until': validator.jail_until
            }
    
    def get_staking_statistics(self) -> Dict:
        """Get overall staking statistics"""
        with self.lock:
            total_validators = len(self.consensus_engine.validators)
            active_validators = len(self.consensus_engine.active_validators)
            total_stake = self.get_total_stake()
            total_delegated = sum(v.total_delegated for v in self.consensus_engine.validators.values())
            
            return {
                'total_validators': total_validators,
                'active_validators': active_validators,
                'total_stake': total_stake,
                'total_delegated': total_delegated,
                'average_uptime': sum(v.uptime for v in self.consensus_engine.validators.values()) / total_validators if total_validators > 0 else 0,
                'total_rewards_distributed': sum(v.total_rewards for v in self.consensus_engine.validators.values())
            }