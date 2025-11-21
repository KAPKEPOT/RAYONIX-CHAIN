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
#from database.core.database import AdvancedDatabase
from consensusengine.utils.database import ConsensusDatabase
from consensusengine.utils.timing import TimeoutManager
from config.config_manager import ConfigManager

logger = logging.getLogger('ConsensusEngine')

class ProofOfStake:
    def __init__(self, config: ConfigManager = None, config_manager=None, **kwargs):
        
        self.config_manager = config_manager or config        
        if not self.config_manager:
        	self.config_manager = ConfigManager()
        	
        self.config = self.config_manager.config
        	
        if config_manager:
        	self.network = AdvancedP2PNetwork(config_manager=config_manager)
        
        else:
        	self.network = AdvancedP2PNetwork(config_manager=self.config_manager)

        # Core state management
        self.height = 0
        self.round = 0
        self.step = ConsensusState.NEW_HEIGHT
        self.locked_round = -1
        self.valid_round = -1
        self.locked_value: Optional[str] = None
        self.valid_value: Optional[str] = None
        self.stakes: Dict[str, int] = {}  # address -> stake amount
        self.total_stake: int = 0
        self.current_height: int = 0
        
        # Round states for each height and round
        self.round_states: Dict[Tuple[int, int], RoundState] = {}
        
        # Epoch management
        self.epoch_state = EpochState(self.config.consensus.epoch_blocks)
        
        # Manager instances
        self.db_manager = DatabaseManager(self.config.database.db_path)
        self.timeout_manager = TimeoutManager()
        self.crypto_manager = CryptoManager()
        self.staking_manager = StakingManager(self)
        self.slashing_manager = SlashingManager(self)
        self.state_manager = None
        
        #Network configuration
        self.config_manager = config_manager
        
        # Initialize network with proper config
        #self.network = AdvancedP2PNetwork(config=self.config_manager)
        
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
       
    def _initialize_with_real_data(self):
    	"""Initialize consensus state with real blockchain data"""
    	try:
    		if self.state_manager:
    			current_height = self.state_manager.get_current_height()
    			self.height = current_height
    			self.current_height = current_height
    			
    			# Load validators from state
    			self._load_validators_from_state()
    		
    		logger.info(f"Consensus engine initialized with height: {self.height}")
    	
    	except Exception as e:
    		logger.error(f"Failed to initialize consensus with real data: {e}")
    		raise RuntimeError(f"Consensus initialization failed: {e}")

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
           
    def set_state_manager(self, state_manager):
    	"""Set state manager after initialization to break circular dependency"""
    	self.state_manager = state_manager           
    
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
                time.sleep(self.config.consensus.epoch_blocks * 5)
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
        if self.height % self.config.consensus.epoch_blocks != 0:
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
                self.config.consensus.propose_timeout
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
                self.config.consensus.precommit_timeout
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
                
    def process_block(self, block: Any) -> bool:
    	try:
    		with self.consensus_lock:
    			# Comprehensive block validation
    			if not self._validate_block_consensus(block):
    				logger.error(f"Block {block.hash} failed consensus validation at height {block.header.height}")
    				return False
    			
    			if block.header.height == 0:
    				
    				return self._process_genesis_block(block)
    				
    				# Verify block proposer is current round validator
    				expected_proposer = self.staking_manager.select_proposer(block.header.height, 0)
    				if not expected_proposer or expected_proposer.address != block.header.validator:
    					logger.error(f"Block proposer {block.header.validator} not authorized for height {block.header.height}")
    					return False
    				# Update consensus state machine
    				self.current_height = block.header.height
    				
    				# Process validator performance metrics
    				validator_address = block.header.validator
    				
    				if validator_address in self.validators:
    					validator = self.validators[validator_address]
    					validator.last_active = time.time()
    					validator.signed_blocks += 1
    					
    					# Calculate rolling uptime with 1000-block window
    					total_recent_blocks = validator.missed_blocks + validator.signed_blocks
    					if total_recent_blocks > 1000:
    						# Maintain rolling window of 1000 blocks
    						excess_blocks = total_recent_blocks - 1000
    						if validator.missed_blocks > excess_blocks:
    							validator.missed_blocks -= excess_blocks
    						else:
    							validator.signed_blocks = 1000 - (excess_blocks - validator.missed_blocks)
    							validator.missed_blocks = 0
    					total_blocks = validator.missed_blocks + validator.signed_blocks
    					if total_blocks > 0:
    						validator.uptime = (validator.signed_blocks / total_blocks) * 100.0
    				# Calculate and distribute block rewards
    				if block.header.height >= 1:
    					block_reward = self._calculate_block_reward(block.header.height)
    					transaction_fees = self._calculate_transaction_fees(block.transactions)
    					total_reward = block_reward + transaction_fees
    					
    					# Add to epoch reward pool for distribution
    					self.epoch_state.reward_pool += total_reward
    					
    					# Update monetary supply with precise accounting
    					self.total_supply += total_reward
    					
    				# Update staking manager state (starting from block 1)
    				if block.header.height >= 1:
    					self.staking_manager.process_block_production(validator_address, block.header.height)
    					
    					# Execute slashing conditions for missed blocks
    					self._check_missed_blocks_slashing(block.header.height)
    					
    					# Update network difficulty based on recent block times
    					self._adjust_network_difficulty(block.header.timestamp)
    				
    				# Persist consensus state changes
    				logger.info(f"Successfully processed block {block.hash[:16]} at height {block.header.height}")
    				return True
    				
    	except Exception as e:
    		logger.error(f"Critical error processing block {block.hash if hasattr(block, 'hash') else 'unknown'}: {e}", exc_info=True)
    		return False
    		
    def _process_genesis_block(self, block: Any) -> bool:
    	"""Special handling for genesis block at height 0"""
    	try:
    		logger.info(f"Processing genesis block at height {block.header.height}")
    		
    		# Initialize consensus state with genesis block
    		self.current_height = 0
    		self.height = 0
    		
    		# Initialize validator set for genesis if needed
    		if not self.validators:
    			logger.info("Initializing empty validator set for genesis")
    			
    		# Initialize epoch state
    		self.epoch_state.current_epoch = 0
    		self.epoch_state.reward_pool = 0
    		
    		# Initialize other consensus state
    		self.round = 0
    		self.step = ConsensusState.NEW_HEIGHT
    		self.locked_round = -1
    		self.valid_round = -1
    		self.locked_value = None
    		self.valid_value = None
    		
    		logger.info("Genesis block (height 0) consensus processing completed")
    		return True
    	
    	except Exception as e:
    		logger.error(f"Error processing genesis block: {e}")
    		return False
 
    def revert_block(self, block: Any) -> bool:
    	"""Production-grade block reversion with complete state rollback"""
    	try:
    		with self.consensus_lock:
    			# Calculate and remove block rewards
    			block_reward = self._calculate_block_reward(block.header.height)
    			transaction_fees = self._calculate_transaction_fees(block.transactions)
    			total_reward = block_reward + transaction_fees
    			
    			# Revert reward pool with bounds checking
    			self.epoch_state.reward_pool = max(0, self.epoch_state.reward_pool - total_reward)
    			
    			# Revert total supply with bounds checking
    			self.total_supply = max(0, self.total_supply - total_reward)
    			
    			# Revert validator statistics
    			validator_address = block.header.validator
    			if validator_address in self.validators:
    				validator = self.validators[validator_address]
    				validator.signed_blocks = max(0, validator.signed_blocks - 1)
    				
    				# Recalculate uptime
    				total_blocks = validator.missed_blocks + validator.signed_blocks
    				if total_blocks > 0:
    					validator.uptime = (validator.signed_blocks / total_blocks) * 100.0
    					
    			# Revert height
    			self.current_height = max(0, block.header.height - 1)
    			
    			# Revert staking manager state
    			self.staking_manager.revert_block_production(validator_address, block.header.height)
    			
    			# Persist reverted state
    			self._save_state()
    			logger.warning(f"Reverted block {block.hash[:16]} at height {block.header.height}")
    			
    			return True
    			
    	except Exception as e:
    		logger.error(f"Critical error reverting block {block.hash if hasattr(block, 'hash') else 'unknown'}: {e}", exc_info=True)
    		return False
    		
    def _validate_block_consensus(self, block: Any) -> bool:
    	try:
    		# Validate block structure integrity
    		if not hasattr(block, 'header') or not hasattr(block.header, 'height'):
    			logger.error("Block missing required header structure")
    			return False
    			
    		# ENHANCED VALIDATION: Special handling for genesis block
    		if block.header.height == 0:
    			# Skip height sequencing validation for genesis
    			if not hasattr(block.header, 'validator') or not block.header.validator:
    				logger.error("Genesis block missing validator identification")
    				return False
    			# Validate genesis-specific requirements
    			if block.header.previous_hash != '0' * 64:
    				logger.error("Genesis block must have all-zero previous hash")
    				return False
    			
    			# Set current height to 0 so genesis becomes height 1
    			if self.current_height == 0:
    				logger.info("Genesis block validation passed")
    				return True
    			else:
    				logger.error(f"Genesis block processed at incorrect current height: {self.current_height}")
    				return False
    			
    		# Validate block height sequencing
    		if block.header.height != self.current_height + 1:
    			logger.error(f"Block height sequencing violation: expected {self.current_height + 1}, got {block.header.height}")
    			return False
    		
    		# Validate proposer authorization
    		if not hasattr(block.header, 'validator') or not block.header.validator:
    			logger.error("Block missing validator identification")
    			return False
    			
    		validator_address = block.header.validator
    		
    		# Verify validator exists and is authorized
    		if validator_address not in self.validators:
    			logger.error(f"Block validator {validator_address} not in registered validator set")
    			return False
    			
    		validator = self.validators[validator_address]
    		
    		# Comprehensive validator status checks
    		
    		if validator.status != ValidatorStatus.ACTIVE:
    			logger.error(f"Block validator {validator_address} status is {validator.status.name}, not ACTIVE")
    			return False
    			
    		if validator.jail_until and time.time() < validator.jail_until:
    			logger.error(f"Block validator {validator_address} is jailed until {validator.jail_until}")
    			return False
    			
    		if validator.total_stake < self.config.staking.min_stake:
    			logger.error(f"Block validator {validator_address} has insufficient stake: {validator.total_stake} < {self.config.min_stake_amount}")
    			return False
    		
    		# Validate block timestamp constraints
    		current_time = time.time()
    		max_future_tolerance = self.config.get('max_future_block_time', 15)  # 15 seconds
    		max_past_tolerance = self.config.get('max_past_block_time', 300)     # 5 minutes
    		
    		if block.header.timestamp > current_time + max_future_tolerance:
    			logger.error(f"Block timestamp {block.header.timestamp} exceeds future tolerance limit")
    			return False
    		
    		if block.header.timestamp < current_time - max_past_tolerance:
    			logger.error(f"Block timestamp {block.header.timestamp} exceeds past tolerance limit")
    			return False
    		
    		# Validate block signature using crypto manager
    		if hasattr(block.header, 'signature') and block.header.signature:
    			signing_data = self._get_block_signing_data(block)
    			if not self.crypto_manager.verify_signature(signing_data, block.header.signature, validator.public_key):
    				logger.error(f"Block signature verification failed for validator {validator_address}")
    				return False
    			
    			# Validate gas limits and block size
    			if hasattr(block, 'transactions'):
    				total_gas = sum(getattr(tx, 'gas_limit', 0) for tx in block.transactions)
    				max_block_gas = self.config.get('max_block_gas', 8000000)
    				if total_gas > max_block_gas:
    					logger.error(f"Block gas limit exceeded: {total_gas} > {max_block_gas}")
    					return False
    				
    				block_size = len(pickle.dumps(block))
    				max_block_size = self.config.get('max_block_size', 4194304)  # 4MB
    				if block_size > max_block_size:
    					logger.error(f"Block size limit exceeded: {block_size} > {max_block_size}")
    					return False
    					
    			# Validate difficulty adjustment
    			expected_difficulty = self._calculate_expected_difficulty(block.header.height)
    			if block.header.difficulty != expected_difficulty:
    				logger.error(f"Block difficulty mismatch: expected {expected_difficulty}, got {block.header.difficulty}")
    				return False
    				
    			# Enhanced Merkle validation
    			if not self._validate_merkle_tree_comprehensive(block):
    				logger.error(f"Block {block.hash} failed comprehensive Merkle validation")
    				return False
    			
    			return True
    	except Exception as e:
    		logger.error(f"Block consensus validation error: {e}")
    		return False
    		
    def _validate_merkle_tree_comprehensive(self, block: Any) -> bool:
    	"""Comprehensive Merkle tree validation"""
    	try:
    		# Verify Merkle root matches calculated root
    		calculated_root = block.calculate_merkle_root()
    		
    		if calculated_root != block.header.merkle_root:
    			logger.error(f"Merkle root validation failed: calculated {calculated_root}, got {block.header.merkle_root}")
    			return False
    		
    		# For blocks with transactions, verify proof generation works
    		if block.transactions:
    			# Test proof generation for first, middle, and last transactions
    			test_indices = [0, len(block.transactions) // 2, len(block.transactions) - 1]
    			for idx in test_indices:
    				if idx < len(block.transactions):
    					tx_hash = block.transactions[idx].tx_hash
    					proof = block.get_merkle_proof(tx_hash)
    					
    					if not proof:
    						logger.error(f"Failed to generate Merkle proof for transaction {tx_hash}")
    						return False
    						
    					# Verify the proof
    					if not block.validate_merkle_proof(tx_hash, proof):
    						logger.error(f"Merkle proof verification failed for transaction {tx_hash}")
    						return False
    						
    		# Additional security: Verify no duplicate transactions
    		tx_hashes = [tx.tx_hash for tx in block.transactions]
    		if len(tx_hashes) != len(set(tx_hashes)):
    			logger.error("Duplicate transaction hashes detected in Merkle tree")
    			return False
    		
    		return True
    	
    	except Exception as e:
    		logger.error(f"Comprehensive Merkle validation error: {e}")
    		return False
    
    def _calculate_block_reward(self, height: int) -> int:
    	# Base monetary policy parameters
    	INITIAL_BLOCK_REWARD = 50 * 10**8  # 50 coins in satoshis
    	HALVING_INTERVAL = 210000  # Blocks between halvings
    	MINIMUM_BLOCK_REWARD = 1 * 10**8  # 1 coin minimum
    	
    	# Calculate halving period
    	halvings = height // HALVING_INTERVAL
    	
    	# Apply exponential reward reduction
    	reward = INITIAL_BLOCK_REWARD // (2 ** halvings)
    	
    	# Enforce minimum reward
    	
    	reward = max(reward, MINIMUM_BLOCK_REWARD)
    	
    	# Apply foundation fee (5%)
    	foundation_fee = reward // 20
    	net_reward = reward - foundation_fee
    	
    	return net_reward
    
    def _calculate_transaction_fees(self, transactions: List[Any]) -> int:
    	"""Calculate total transaction fees from block transactions"""
    	total_fees = 0
    	for tx in transactions:
    		if hasattr(tx, 'fee'):
    			total_fees += tx.fee
    		elif hasattr(tx, 'inputs') and hasattr(tx, 'outputs'):
    			# Calculate fee as input sum - output sum
    			input_sum = sum(getattr(inp, 'amount', 0) for inp in tx.inputs)
    			output_sum = sum(getattr(out, 'amount', 0) for out in tx.outputs)
    			total_fees += max(0, input_sum - output_sum)
    	return total_fees
    	
    def _get_block_signing_data(self, block: Any) -> bytes:
    	"""Generate deterministic signing data for block validation"""
    	import struct
    	signing_data = b''
    	
    	# Pack core header fields
    	signing_data += struct.pack('>I', block.header.version)
    	signing_data += struct.pack('>Q', block.header.height)
    	signing_data += bytes.fromhex(block.header.previous_hash)
    	signing_data += bytes.fromhex(block.header.merkle_root)
    	signing_data += struct.pack('>d', block.header.timestamp)
    	signing_data += struct.pack('>Q', block.header.difficulty)
    	signing_data += struct.pack('>Q', block.header.nonce)
    	signing_data += block.header.validator.encode('utf-8')
    	
    	# Include transaction commitments
    	if hasattr(block, 'transactions'):
    		for tx in block.transactions:
    			if hasattr(tx, 'hash'):
    				signing_data += bytes.fromhex(tx.hash)
    	return signing_data
    	
    def _calculate_expected_difficulty(self, height: int) -> int:
    	if height == 0:
    		return 1  # Genesis block difficulty
    	
    	# Use moving average of recent block times for difficulty adjustment
    	recent_blocks = self._get_recent_block_times(height)
    	if len(recent_blocks) < 10:
    		return 1
    	
    	average_block_time = sum(recent_blocks) / len(recent_blocks)
    	target_block_time = getattr(self.config, 'block_time_target', 30)
    	
    	# Adjust difficulty based on block time ratio
    	time_ratio = average_block_time / target_block_time
    	current_difficulty = self._get_current_difficulty()
    	
    	# Limit adjustment to ±25% per period
    	adjustment_factor = max(0.75, min(1.25, time_ratio))
    	new_difficulty = int(current_difficulty * adjustment_factor)
    	
    	# Ensure minimum difficulty
    	
    	return max(1, new_difficulty)
  
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
        
    def to_bytes(self) -> bytes:
    	"""Convert consensus state to bytes"""
    	state_data = {
    	    'validators': self.validators,
    	    'stakes': self.stakes,
    	    'total_stake': self.total_stake,
    	    'current_height': self.current_height
    	}
    	return pickle.dumps(state_data)
    	
    def from_bytes(self, data: bytes):
    	"""Restore consensus state from bytes"""
    	state_data = pickle.loads(data)
    	self.validators = state_data.get('validators', {})
    	self.stakes = state_data.get('stakes', {})
    	self.total_stake = state_data.get('total_stake', 0)
    	self.current_height = state_data.get('current_height', 0)
    	
    def create_snapshot(self):
        """Create a snapshot of current consensus state"""
        return {
            'validators': {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                          for k, v in self.validators.items()},
            'stakes': self.stakes.copy(),
            'total_stake': self.total_stake,
            'current_height': self.current_height,
            'timestamp': time.time()
        }
    	
    def restore_snapshot(self, snapshot):
        """Restore consensus state from snapshot"""
        try:
            self.validators = snapshot.get('validators', {})
            self.stakes = snapshot.get('stakes', {}).copy()
            self.total_stake = snapshot.get('total_stake', 0)
            self.current_height = snapshot.get('current_height', 0)
            
            # Rebuild active_validators list if needed
            if hasattr(self, 'active_validators'):
                self.active_validators = [
                    v for v in self.validators.values() 
                    if hasattr(v, 'status') and getattr(v, 'status') == 'active'
                ]
        except Exception as e:
            logger.error(f"Error restoring consensus snapshot: {e}")
            raise
    	
    def calculate_hash(self) -> str:
        """Calculate hash of consensus state"""
        import hashlib
        import json
        
        state_data = {
            'validators_hash': hashlib.sha256(
                str(sorted(self.validators.items())).encode()
            ).hexdigest(),
            'stakes_hash': hashlib.sha256(
                str(sorted(self.stakes.items())).encode()
            ).hexdigest(),
            'total_stake': self.total_stake,
            'current_height': self.current_height
        }
        return hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
    	
    def verify_integrity(self) -> bool:
        """Verify consensus state integrity"""
        try:
            # Basic validation
            if self.total_stake < 0:
                return False
            
            # Validate stakes consistency
            calculated_total = sum(self.stakes.values())
            if abs(calculated_total - self.total_stake) > 1:  # Allow small rounding differences
                logger.warning(f"Stakes inconsistency: calculated={calculated_total}, total={self.total_stake}")
                return False
            
            # Validate validators
            for validator_addr, validator in self.validators.items():
                if not hasattr(validator, 'address'):
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Consensus integrity verification failed: {e}")
            return False
            
    def add_stake(self, address: str, amount: int) -> bool:
        """Add stake for a validator"""
        try:
            if amount <= 0:
                return False
            
            self.stakes[address] = self.stakes.get(address, 0) + amount
            self.total_stake += amount
            
            # Update validator if exists
            if address in self.validators:
                self.validators[address].total_stake = self.stakes[address]
            
            return True
        except Exception as e:
            logger.error(f"Error adding stake: {e}")
            return False
    
    def remove_stake(self, address: str, amount: int) -> bool:
        """Remove stake for a validator"""
        try:
            if address not in self.stakes or self.stakes[address] < amount:
                return False
            
            self.stakes[address] -= amount
            self.total_stake -= amount
            
            if self.stakes[address] == 0:
                del self.stakes[address]
            
            # Update validator if exists
            if address in self.validators:
                self.validators[address].total_stake = self.stakes.get(address, 0)
            
            return True
        except Exception as e:
            logger.error(f"Error removing stake: {e}")
            return False
            
    def get_validator_count(self) -> int:
    	"""Get the number of active validators"""
    	with self.validator_lock:
    		if not hasattr(self, 'active_validators'):
    			raise AttributeError("active_validators not initialized in consensus engine")
    		return len(self.active_validators)
    	
    def get_total_stake(self) -> int:
    	"""Get total stake in the system"""
    	with self.lock:
    		if not hasattr(self, 'total_stake'):
    			raise AttributeError("total_stake not initialized in consensus engine")
    		return self.total_stake
    		
    def get_total_supply(self) -> int:
    	"""Calculate total coin supply based on actual blockchain state"""
    	if not hasattr(self, 'state_manager') or not self.state_manager:
    		raise RuntimeError("Consensus engine: state_manager not available for supply calculation")
    	
    	if not hasattr(self.state_manager, 'get_total_supply'):
    		raise AttributeError("Consensus engine: state_manager missing get_total_supply method")
    	
    	return self.state_manager.get_total_supply()
    		
    def get_circulating_supply(self) -> int:
    	"""Get circulating supply from state manager"""
    	if not hasattr(self, 'state_manager') or not self.state_manager:
    		raise RuntimeError("Consensus engine: state_manager not available for supply calculation")
    	
    	if not hasattr(self.state_manager, 'get_circulating_supply'):
    		raise AttributeError("Consensus engine: state_manager missing get_circulating_supply method")
    	
    	return self.state_manager.get_circulating_supply()
    
    def _load_validators_from_state(self):
    	"""Load validators from state manager"""
    	try:
    		# This should load validators from persistent storage
    		# For now, initialize empty if no validators exist
    		if not hasattr(self, 'validators'):
    			self.validators = {}
    		
    		if not hasattr(self, 'active_validators'):
    			self.active_validators = []
    		
    		if not hasattr(self, 'total_stake'):
    			self.total_stake = 0
    		
    		logger.info(f"Loaded {len(self.validators)} validators from state")
    	
    	except Exception as e:
    		logger.error(f"Failed to load validators from state: {e}")
    		raise
 
    def shutdown(self):
        """Shutdown the consensus engine"""
        self._running = False
        self.timeout_manager.shutdown()
        self.db_manager.close()
        logger.info("Consensus engine shutdown complete")