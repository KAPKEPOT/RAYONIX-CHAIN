# consensus/core/consensus.py
import time
import threading
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass
import logging

from consensusengine.core.state import ConsensusState, RoundState, EpochState
from consensusengine.models.validators import Validator, ValidatorStatus
from consensusengine.models.blocks import BlockProposal
from consensusengine.models.votes import Vote, VoteType
from consensusengine.abci.interface import ABCIApplication
from consensusengine.staking.manager import StakingManager
from consensusengine.staking.slashing import SlashingManager
from network.core.p2p_network import AdvancedP2PNetwork
from consensusengine.crypto.signing import CryptoManager
from consensusengine.utils.database import DatabaseManager
from consensusengine.utils.timing import TimeoutManager
from consensusengine.utils.config.settings import ConsensusConfig

logger = logging.getLogger('ConsensusEngine')

class ProofOfStake:
    """Production-ready Proof-of-Stake consensus engine with BFT features"""
    
    def __init__(self, config: ConsensusConfig = None, network_config=None, **kwargs):
        """
        Initialize Proof-of-Stake consensus engine
        
        Args:
            config: Consensus configuration
            **kwargs: Override configuration parameters
        """
        if config is None:
        	from consensusengine.utils.config.settings import ConsensusConfig
        	self.config = ConsensusConfig(**kwargs)
        	
        elif isinstance(config, dict):
        	from consensusengine.utils.config.settings import ConsensusConfig
        	self.config = ConsensusConfig(**{**config, **kwargs})
        	
        elif hasattr(config, '__dataclass_fields__'):
        	self.config = config
        else:
        	from consensusengine.utils.config.settings import ConsensusConfig
        	self.config = ConsensusConfig(**vars(config))
        	
        # Add fallback for missing attributes
        self._ensure_config_attributes()
 
        # Core state management
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
        self.epoch_state = EpochState(self.config.epoch_blocks)
        
        # Manager instances
        self.db_manager = DatabaseManager(self.config.db_path)
        self.timeout_manager = TimeoutManager()
        self.crypto_manager = CryptoManager()
        self.staking_manager = StakingManager(self)
        self.slashing_manager = SlashingManager(self)
        
        #Network configuration
        self.network_config = network_config
        
        # Initialize network with proper config
        self.network = AdvancedP2PNetwork(config=self.network_config)
        
        # Block and vote storage
        self.block_proposals: Dict[str, BlockProposal] = {}
        self.votes: Dict[Tuple[int, int, VoteType], Dict[str, Vote]] = {}
        self.executed_blocks: Set[str] = set()
        
        # ABCI interface
        self.abci = ABCIApplication()
        
        # Validator sets
        self.validators: Dict[str, Validator] = {}
        self.active_validators: List[Validator] = []
        self.pending_validators: List[Validator] = []
        
        # Locks for thread safety
        self.lock = threading.RLock()
        self.validator_lock = threading.RLock()
        self.consensus_lock = threading.RLock()
        
        # Background task management
        self._running = True
        
        self._load_state()
        self._start_background_tasks()
        
    def _ensure_config_attributes(self):
    	"""Ensure config has all required attributes with defaults"""
    	defaults = {
    	    'epoch_blocks': 100,
    	    'timeout_propose': 3000,
    	    'timeout_prevote': 1000,
    	    'timeout_precommit': 1000,
    	    'timeout_commit': 1000,
    	    'db_path': './consensus_data',
    	    'max_validators': 100,
    	    'min_stake_amount': 1000,
    	    'unbonding_period': 86400 * 21,
    	    'slashing_percentage': 0.01,
    	    'jail_duration': 86400 * 2
    	}
    	for attr, default in defaults.items():
    		if not hasattr(self.config, attr):
    			setattr(self.config, attr, default)
    			logger.warning(f"Added missing config attribute {attr} with default value {default}")
    
    def _load_state(self):
        """Load consensus state from database"""
        try:
            state_data = self.db_manager.load_consensus_state()
            if state_data:
                self.height = state_data.get('height', 0)
                self.round = state_data.get('round', 0)
                self.step = ConsensusState(state_data.get('step', ConsensusState.NEW_HEIGHT.value))
                self.locked_round = state_data.get('locked_round', -1)
                self.valid_round = state_data.get('valid_round', -1)
                self.locked_value = state_data.get('locked_value')
                self.valid_value = state_data.get('valid_value')
                
                # Load validators
                validators_data = self.db_manager.load_validators()
                self.validators = {k: Validator.from_dict(v) for k, v in validators_data.items()}
                
                # Load active validators
                active_data = self.db_manager.load_active_validators()
                self.active_validators = [Validator.from_dict(v) for v in active_data]
                
                logger.info(f"Loaded consensus state: height={self.height}, round={self.round}")
            else:
                self._initialize_genesis_state()
                
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            self._initialize_genesis_state()
    
    def _initialize_genesis_state(self):
        """Initialize genesis state"""
        self.height = 0
        self.round = 0
        self.step = ConsensusState.NEW_HEIGHT
        self.locked_round = -1
        self.valid_round = -1
        self.locked_value = None
        self.valid_value = None
        self.validators = {}
        self.active_validators = []
        
        logger.info("Initialized genesis state")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        def epoch_processor():
            while self._running:
                time.sleep(30)
                try:
                    self._process_epoch_transition()
                except Exception as e:
                    logger.error(f"Epoch processor error: {e}")
        
        def validator_updater():
            while self._running:
                time.sleep(60)
                try:
                    self.staking_manager.update_validator_set()
                except Exception as e:
                    logger.error(f"Validator updater error: {e}")
        
        def jail_checker():
            while self._running:
                time.sleep(300)
                try:
                    self.slashing_manager.check_jailed_validators()
                except Exception as e:
                    logger.error(f"Jail checker error: {e}")
        
        def unavailability_checker():
            while self._running:
                time.sleep(self.config.epoch_blocks * 5)
                try:
                    self.slashing_manager.check_unavailability()
                except Exception as e:
                    logger.error(f"Unavailability checker error: {e}")
        
        # Start background threads
        threading.Thread(target=epoch_processor, daemon=True).start()
        threading.Thread(target=validator_updater, daemon=True).start()
        threading.Thread(target=jail_checker, daemon=True).start()
        threading.Thread(target=unavailability_checker, daemon=True).start()
        
        logger.info("Started background maintenance tasks")
    
    def _process_epoch_transition(self):
        """Process epoch transition and distribute rewards"""
        if self.height % self.config.epoch_blocks != 0:
            return
        
        with self.lock:
            try:
                # Process all pending staking operations
                self.staking_manager.process_pending_operations()
                
                # Update validator set for next epoch
                self.staking_manager.update_validator_set()
                
                # Distribute rewards
                if self.epoch_state.reward_pool > 0:
                    self.staking_manager.distribute_epoch_rewards(self.epoch_state.reward_pool)
                
                # Reset reward pool for next epoch
                self.epoch_state.reward_pool = 0
                self.epoch_state.current_epoch += 1
                
                self._save_state()
                logger.info(f"Processed epoch transition to epoch {self.epoch_state.current_epoch}")
                
            except Exception as e:
                logger.error(f"Epoch transition error: {e}")
    
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
            self.round_states.clear()
            
            # Start the propose step
            self._start_round(0)
            
            logger.info(f"Started new height: {height}")
    
    def _start_round(self, round: int):
        """Start a new round"""
        with self.consensus_lock:
            self.round = round
            self.step = ConsensusState.PROPOSE
            
            # Get the proposer for this round
            proposer = self.staking_manager.select_proposer(self.height, round)
            
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
            self.timeout_manager.set_timeout(
                self.height, round, ConsensusState.PROPOSE,
                self.config.timeout_propose, self._on_timeout
            )
            
            # If we're the proposer, create a proposal
            if proposer and self._is_current_validator(proposer.address):
                self._create_and_broadcast_proposal(round_state)
            
            logger.info(f"Started round {round} at height {self.height}, proposer: {proposer.address if proposer else 'None'}")
    
    def _is_current_validator(self, address: str) -> bool:
        """Check if the given address is the current validator"""
        # This would be implemented based on the node's identity
        # For now, return True for simulation
        return True
    
    def _create_and_broadcast_proposal(self, round_state: RoundState):
        """Create and broadcast a block proposal (for proposer only)"""
        try:
            # Create block proposal
            proposal = self._create_block_proposal(round_state)
            if proposal:
                # Store locally
                self.block_proposals[proposal.block_hash] = proposal
                round_state.proposal = proposal
                
                # Broadcast to network
                self.network_protocol.broadcast_proposal(proposal)
                
                logger.info(f"Created and broadcast proposal for height {self.height}, round {self.round}")
                
        except Exception as e:
            logger.error(f"Error creating proposal: {e}")
    
    def _create_block_proposal(self, round_state: RoundState) -> Optional[BlockProposal]:
        """Create a block proposal"""
        # This would create an actual block with transactions
        # For now, create a minimal proposal
        from cryptography.hazmat.primitives import hashes
        import hashlib
        
        block_data = f"{self.height}|{self.round}|{round_state.proposer.address}|{time.time()}"
        block_hash = hashlib.sha256(block_data.encode()).hexdigest()
        
        # Sign the proposal
        signature = self.crypto_manager.sign_data(block_data.encode())
        
        return BlockProposal(
            height=self.height,
            block_hash=block_hash,
            validator_address=round_state.proposer.address,
            timestamp=time.time(),
            signature=signature,
            view_number=0,
            round_number=self.round,
            parent_hash=self._get_parent_hash(),
            tx_hashes=[]  # Would include actual transaction hashes
        )
    
    def _get_parent_hash(self) -> str:
        """Get the parent block hash"""
        # This would retrieve the actual parent hash from block storage
        # For now, return a dummy hash
        return "0" * 64 if self.height == 0 else "parent_hash_placeholder"
    
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
    
    def _prevote(self, height: int, round: int, block_hash: Optional[str]):
        """Send a prevote for a block"""
        with self.consensus_lock:
            if height != self.height or round != self.round:
                return
            
            # Create and send vote
            vote = self._create_vote(height, round, block_hash, VoteType.PREVOTE)
            if vote:
                self._send_vote(vote)
                
                # Check if we have +2/3 prevotes for this block
                self._check_prevote_polka(height, round, block_hash)
    
    def _precommit(self, height: int, round: int, block_hash: Optional[str]):
        """Send a precommit for a block"""
        with self.consensus_lock:
            if height != self.height or round != self.round:
                return
            
            # Create and send vote
            vote = self._create_vote(height, round, block_hash, VoteType.PRECOMMIT)
            if vote:
                self._send_vote(vote)
                
                # Check if we have +2/3 precommits for this block
                self._check_precommit_polka(height, round, block_hash)
    
    def _create_vote(self, height: int, round: int, block_hash: Optional[str], vote_type: VoteType) -> Optional[Vote]:
        """Create a vote object"""
        try:
            # Get current validator address
            validator_address = self._get_current_validator_address()
            if not validator_address:
                return None
            
            # Create vote data
            vote_data = f"{height}|{round}|{vote_type.name}|{block_hash or 'nil'}"
            
            # Sign the vote
            signature = self.crypto_manager.sign_data(vote_data.encode())
            
            return Vote(
                height=height,
                block_hash=block_hash or "nil",
                validator_address=validator_address,
                timestamp=time.time(),
                signature=signature,
                round_number=round,
                vote_type=vote_type
            )
            
        except Exception as e:
            logger.error(f"Error creating vote: {e}")
            return None
    
    def _get_current_validator_address(self) -> Optional[str]:
        """Get the current validator's address"""
        # This would return the actual validator address of this node
        # For now, return a placeholder
        return "current_validator_address"
    
    def _send_vote(self, vote: Vote):
        """Send a vote to the network"""
        try:
            # Store vote locally
            vote_key = (vote.height, vote.round_number, vote.vote_type)
            if vote_key not in self.votes:
                self.votes[vote_key] = {}
            self.votes[vote_key][vote.validator_address] = vote
            
            # Broadcast to network
            self.network_protocol.broadcast_vote(vote)
            
            logger.info(f"Sent {vote.vote_type.name} for block {vote.block_hash} at height {vote.height}, round {vote.round_number}")
            
        except Exception as e:
            logger.error(f"Error sending vote: {e}")
    
    def _check_prevote_polka(self, height: int, round: int, block_hash: str):
        """Check if we have +2/3 prevotes for a block (polka)"""
        if not block_hash or block_hash == "nil":
            return
        
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
            self.timeout_manager.set_timeout(
                height, round, ConsensusState.PRECOMMIT,
                self.config.timeout_precommit, self._on_timeout
            )
            
            # Send precommit for this block
            self._precommit(height, round, block_hash)
            
            logger.info(f"Prevote polka achieved for block {block_hash} at height {height}, round {round}")
    
    def _check_precommit_polka(self, height: int, round: int, block_hash: str):
        """Check if we have +2/3 precommits for a block (commit)"""
        if not block_hash or block_hash == "nil":
            return
        
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
            
            logger.info(f"Precommit polka achieved for block {block_hash} at height {height}, round {round}")
    
    def _commit_block(self, height: int, block_hash: str):
        """Commit a block and move to the next height"""
        with self.consensus_lock:
            try:
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
                            # In production, this would trigger recovery procedures
                
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
                
            except Exception as e:
                logger.error(f"Error committing block: {e}")
                # Trigger recovery procedures
                self._start_recovery()
    
    def _start_recovery(self):
        """Start consensus recovery procedures"""
        logger.error("Starting consensus recovery")
        # Implementation would include view change, state sync, etc.
        self.step = ConsensusState.RECOVERY
    
    def _save_state(self):
        """Save consensus state to database"""
        with self.lock:
            try:
                state_data = {
                    'height': self.height,
                    'round': self.round,
                    'step': self.step.value,
                    'locked_round': self.locked_round,
                    'valid_round': self.valid_round,
                    'locked_value': self.locked_value,
                    'valid_value': self.valid_value
                }
                
                validators_data = {k: v.to_dict() for k, v in self.validators.items()}
                active_data = [v.to_dict() for v in self.active_validators]
                
                self.db_manager.save_consensus_state(state_data)
                self.db_manager.save_validators(validators_data)
                self.db_manager.save_active_validators(active_data)
                
            except Exception as e:
                logger.error(f"Error saving state: {e}")
    
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
            'total_stake': self.staking_manager.get_total_stake(),
            'active_validators_count': len(self.active_validators),
            'total_validators_count': len(self.validators)
        }
    
    def shutdown(self):
        """Shutdown the consensus engine"""
        self._running = False
        self.timeout_manager.shutdown()
        self.db_manager.close()
        logger.info("Consensus engine shutdown complete")