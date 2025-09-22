"""
Consensus engine implementation - BFT Proof-of-Stake consensus
"""

import time
import threading
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
from dataclasses import dataclass, field
import logging

from ..storage import BlockStore, VoteStore
from ..validator import ValidatorManager
from ..crypto import CryptoService
from ..network import NetworkLayer
from ..metrics import MetricsCollector
from ..config import ConsensusConfig
from ..utils import synchronized, RateLimiter, Backoff
from ..exceptions import ConsensusError, ValidationError, TimeoutError

logger = logging.getLogger('consensus.engine')

class ConsensusEngine:
    """BFT Proof-of-Stake Consensus Engine"""
    
    def __init__(self, config: ConsensusConfig, crypto_service: CryptoService,
                 validator_manager: ValidatorManager, block_store: BlockStore,
                 vote_store: VoteStore, network_layer: NetworkLayer,
                 metrics_collector: MetricsCollector):
        self.config = config
        self.crypto = crypto_service
        self.validators = validator_manager
        self.blocks = block_store
        self.votes = vote_store
        self.network = network_layer
        self.metrics = metrics_collector
        
        # Consensus state
        self.height = 0
        self.round = 0
        self.step = "NEW_HEIGHT"
        self.locked_round = -1
        self.valid_round = -1
        self.locked_value: Optional[str] = None
        self.valid_value: Optional[str] = None
        
        # Round management
        self.current_round_start = 0.0
        self.round_timeouts: Dict[Tuple[int, int], Any] = {}
        
        # Locks
        self.consensus_lock = threading.RLock()
        self.proposal_lock = threading.Lock()
        
        # Rate limiting
        self.proposal_limiter = RateLimiter(10, 1.0)  # 10 proposals per second
        self.vote_limiter = RateLimiter(100, 1.0)  # 100 votes per second
        
        # Backoff for retries
        self.backoff = Backoff(base_delay=0.1, max_delay=5.0)
        
        # ABCI callbacks
        self.check_tx_callback: Optional[Callable[[str], bool]] = None
        self.deliver_tx_callback: Optional[Callable[[str], bool]] = None
        self.commit_callback: Optional[Callable[[], str]] = None
        
        logger.info("Consensus engine initialized")
    
    def start(self) -> None:
        """Start consensus engine"""
        logger.info("Starting consensus engine")
        self._load_state()
        self._start_new_height(self.height)
        self._start_background_tasks()
    
    def stop(self) -> None:
        """Stop consensus engine"""
        logger.info("Stopping consensus engine")
        self._save_state()
        self._cancel_all_timeouts()
    
    @synchronized(consensus_lock)
    def _load_state(self) -> None:
        """Load consensus state from storage"""
        try:
            state = self.blocks.load_consensus_state()
            if state:
                self.height = state.get('height', 0)
                self.round = state.get('round', 0)
                self.step = state.get('step', 'NEW_HEIGHT')
                self.locked_round = state.get('locked_round', -1)
                self.valid_round = state.get('valid_round', -1)
                self.locked_value = state.get('locked_value')
                self.valid_value = state.get('valid_value')
                
                logger.info(f"Loaded consensus state: height={self.height}, round={self.round}")
        except Exception as e:
            logger.error(f"Failed to load consensus state: {e}")
            # Start fresh
            self.height = 0
            self.round = 0
            self.step = "NEW_HEIGHT"
    
    @synchronized(consensus_lock)
    def _save_state(self) -> None:
        """Save consensus state to storage"""
        state = {
            'height': self.height,
            'round': self.round,
            'step': self.step,
            'locked_round': self.locked_round,
            'valid_round': self.valid_round,
            'locked_value': self.locked_value,
            'valid_value': self.valid_value
        }
        try:
            self.blocks.save_consensus_state(state)
        except Exception as e:
            logger.error(f"Failed to save consensus state: {e}")
    
    def _start_new_height(self, height: int) -> None:
        """Start consensus for new block height"""
        with self.consensus_lock:
            self.height = height
            self.round = 0
            self.step = "PROPOSE"
            self.current_round_start = time.time()
            
            logger.info(f"Starting new height: {height}")
            self.metrics.record_height_start(height)
            
            self._start_round(0)
    
    def _start_round(self, round_num: int) -> None:
        """Start new consensus round"""
        with self.consensus_lock:
            self.round = round_num
            self.step = "PROPOSE"
            self.current_round_start = time.time()
            
            # Select proposer for this round
            proposer = self.validators.select_proposer(self.height, round_num)
            
            logger.info(f"Starting round {round_num} at height {self.height}, proposer: {proposer.address if proposer else 'None'}")
            self.metrics.record_round_start(self.height, round_num)
            
            # Set timeout for propose step
            self._set_timeout("PROPOSE", self.config.timeout_propose)
            
            # If we're the proposer, create block proposal
            if proposer and self.validators.is_current_validator(proposer.address):
                self._create_proposal()
    
    def _set_timeout(self, step: str, timeout_ms: int) -> None:
        """Set timeout for current step"""
        timeout_sec = timeout_ms / 1000.0
        round_key = (self.height, self.round)
        
        # Cancel existing timeout
        if round_key in self.round_timeouts:
            self.round_timeouts[round_key].cancel()
        
        def timeout_handler():
            self._on_timeout(step)
        
        timer = threading.Timer(timeout_sec, timeout_handler)
        timer.daemon = True
        timer.start()
        
        self.round_timeouts[round_key] = timer
    
    def _on_timeout(self, step: str) -> None:
        """Handle consensus timeout"""
        with self.consensus_lock:
            logger.warning(f"Timeout in {step} step at height {self.height}, round {self.round}")
            self.metrics.record_timeout(step)
            
            if step == "PROPOSE":
                self._prevote(None)
            elif step == "PREVOTE":
                self._precommit(None)
            elif step == "PRECOMMIT":
                # Move to next round
                next_round = self.round + 1
                if next_round < self.config.max_rounds:
                    self._start_round(next_round)
                else:
                    logger.error(f"Max rounds reached at height {self.height}")
                    self._start_new_height(self.height + 1)
    
    def _create_proposal(self) -> None:
        """Create block proposal (if we're the proposer)"""
        if not self.proposal_limiter.acquire():
            logger.warning("Proposal rate limit exceeded")
            return
        
        try:
            # Create block with transactions from mempool
            block_data = self._create_block_data()
            
            # Sign the proposal
            signature = self.crypto.sign_message(block_data)
            
            # Create proposal
            proposal = {
                'height': self.height,
                'round': self.round,
                'block_data': block_data,
                'signature': signature,
                'timestamp': time.time()
            }
            
            # Broadcast proposal
            self.network.broadcast_proposal(proposal)
            
            logger.info(f"Created and broadcast proposal for height {self.height}, round {self.round}")
            
        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")
    
    def _create_block_data(self) -> bytes:
        """Create block data for proposal"""
        # This would collect transactions from mempool and create block
        # For now, return empty data
        return b"block_data"
    
    def receive_proposal(self, proposal: Dict) -> bool:
        """Process received block proposal"""
        try:
            # Validate proposal
            if not self._validate_proposal(proposal):
                return False
            
            # Store proposal
            self.blocks.store_proposal(proposal)
            
            # Move to prevote step
            with self.consensus_lock:
                if (self.height == proposal['height'] and 
                    self.round == proposal['round'] and 
                    self.step == "PROPOSE"):
                    
                    self.step = "PREVOTE"
                    self._set_timeout("PREVOTE", self.config.timeout_prevote)
                    
                    # Send prevote for this block
                    self._prevote(proposal['block_hash'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing proposal: {e}")
            return False
    
    def _validate_proposal(self, proposal: Dict) -> bool:
        """Validate block proposal"""
        # Check basic structure
        required_fields = {'height', 'round', 'block_data', 'signature', 'timestamp'}
        if not all(field in proposal for field in required_fields):
            return False
        
        # Check if proposer is valid for this round
        proposer_addr = self.crypto.recover_address(proposal['block_data'], proposal['signature'])
        expected_proposer = self.validators.select_proposer(proposal['height'], proposal['round'])
        
        if not expected_proposer or expected_proposer.address != proposer_addr:
            return False
        
        # Verify signature
        if not self.crypto.verify_signature(proposer_addr, proposal['block_data'], proposal['signature']):
            return False
        
        # Check timeliness
        current_time = time.time()
        if abs(current_time - proposal['timestamp']) > self.config.timeout_propose / 1000:
            return False
        
        return True
    
    def _prevote(self, block_hash: Optional[str]) -> None:
        """Send prevote for block"""
        if not self.vote_limiter.acquire():
            return
        
        try:
            vote = self._create_vote(block_hash, "PREVOTE")
            self.votes.store_vote(vote)
            self.network.broadcast_vote(vote)
            
            logger.info(f"Sent prevote for block {block_hash or 'nil'}")
            
        except Exception as e:
            logger.error(f"Failed to create prevote: {e}")
    
    def _precommit(self, block_hash: Optional[str]) -> None:
        """Send precommit for block"""
        if not self.vote_limiter.acquire():
            return
        
        try:
            vote = self._create_vote(block_hash, "PRECOMMIT")
            self.votes.store_vote(vote)
            self.network.broadcast_vote(vote)
            
            logger.info(f"Sent precommit for block {block_hash or 'nil'}")
            
        except Exception as e:
            logger.error(f"Failed to create precommit: {e}")
    
    def _create_vote(self, block_hash: Optional[str], vote_type: str) -> Dict:
        """Create vote message"""
        vote_data = self._get_vote_signing_data(block_hash, vote_type)
        signature = self.crypto.sign_message(vote_data)
        
        return {
            'height': self.height,
            'round': self.round,
            'vote_type': vote_type,
            'block_hash': block_hash or "nil",
            'signature': signature,
            'timestamp': time.time()
        }
    
    def _get_vote_signing_data(self, block_hash: Optional[str], vote_type: str) -> bytes:
        """Get data to sign for vote"""
        data = f"{self.height}|{self.round}|{vote_type}|{block_hash or 'nil'}"
        return data.encode()
    
    def receive_vote(self, vote: Dict) -> bool:
        """Process received vote"""
        try:
            # Validate vote
            if not self._validate_vote(vote):
                return False
            
            # Store vote
            self.votes.store_vote(vote)
            
            # Check for +2/3 majority
            if vote['vote_type'] == "PREVOTE":
                self._check_prevote_polka(vote['block_hash'])
            else:
                self._check_precommit_polka(vote['block_hash'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing vote: {e}")
            return False
    
    def _validate_vote(self, vote: Dict) -> bool:
        """Validate vote"""
        required_fields = {'height', 'round', 'vote_type', 'block_hash', 'signature', 'timestamp'}
        if not all(field in vote for field in required_fields):
            return False
        
        # Verify signature
        vote_data = self._get_vote_signing_data(vote['block_hash'], vote['vote_type'])
        voter_addr = self.crypto.recover_address(vote_data, vote['signature'])
        
        if not self.crypto.verify_signature(voter_addr, vote_data, vote['signature']):
            return False
        
        # Check if voter is active validator
        if not self.validators.is_active_validator(voter_addr):
            return False
        
        return True
    
    def _check_prevote_polka(self, block_hash: str) -> None:
        """Check for +2/3 prevotes (polka)"""
        votes = self.votes.get_prevotes(self.height, self.round, block_hash)
        voting_power = sum(self.validators.get_voting_power(vote['voter']) for vote in votes)
        total_power = self.validators.get_total_voting_power()
        
        if voting_power > (2 * total_power) / 3:
            with self.consensus_lock:
                self.step = "PRECOMMIT"
                self._set_timeout("PRECOMMIT", self.config.timeout_precommit)
                
                # Update locked value
                if self.locked_round < self.round:
                    self.locked_value = block_hash
                    self.locked_round = self.round
                
                self.valid_value = block_hash
                self.valid_round = self.round
                
                # Send precommit
                self._precommit(block_hash)
    
    def _check_precommit_polka(self, block_hash: str) -> None:
        """Check for +2/3 precommits (commit)"""
        votes = self.votes.get_precommits(self.height, self.round, block_hash)
        voting_power = sum(self.validators.get_voting_power(vote['voter']) for vote in votes)
        total_power = self.validators.get_total_voting_power()
        
        if voting_power > (2 * total_power) / 3:
            # Commit the block
            self._commit_block(block_hash)
    
    def _commit_block(self, block_hash: str) -> None:
        """Commit block and move to next height"""
        try:
            # Get block data
            block_data = self.blocks.get_proposal(block_hash)
            if not block_data:
                raise ConsensusError(f"Block data not found for hash: {block_hash}")
            
            # Execute transactions (ABCI)
            if self.deliver_tx_callback:
                # This would iterate through transactions in the block
                pass
            
            # Update app state
            if self.commit_callback:
                app_hash = self.commit_callback()
            
            # Update validator statistics
            proposer_addr = self.crypto.recover_address(block_data['block_data'], block_data['signature'])
            self.validators.record_block_creation(proposer_addr)
            
            # Move to next height
            with self.consensus_lock:
                self.step = "COMMITTED"
                self.metrics.record_block_commit(self.height, block_hash)
                self._start_new_height(self.height + 1)
            
            logger.info(f"Committed block {block_hash} at height {self.height}")
            
        except Exception as e:
            logger.error(f"Failed to commit block: {e}")
            self.metrics.record_block_failure(self.height)
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        def timeout_checker():
            while True:
                time.sleep(1)
                self._check_timeouts()
        
        def state_persister():
            while True:
                time.sleep(30)
                self._save_state()
        
        # Start background threads
        threading.Thread(target=timeout_checker, daemon=True).start()
        threading.Thread(target=state_persister, daemon=True).start()
    
    def _check_timeouts(self) -> None:
        """Check for round timeouts"""
        current_time = time.time()
        round_duration = current_time - self.current_round_start
        
        timeout_map = {
            "PROPOSE": self.config.timeout_propose / 1000,
            "PREVOTE": self.config.timeout_prevote / 1000,
            "PRECOMMIT": self.config.timeout_precommit / 1000
        }
        
        if self.step in timeout_map and round_duration > timeout_map[self.step]:
            self._on_timeout(self.step)
    
    def _cancel_all_timeouts(self) -> None:
        """Cancel all pending timeouts"""
        for timer in self.round_timeouts.values():
            timer.cancel()
        self.round_timeouts.clear()
    
    def set_check_tx_callback(self, callback: Callable[[str], bool]) -> None:
        """Set transaction validation callback"""
        self.check_tx_callback = callback
    
    def set_deliver_tx_callback(self, callback: Callable[[str], bool]) -> None:
        """Set transaction delivery callback"""
        self.deliver_tx_callback = callback
    
    def set_commit_callback(self, callback: Callable[[], str]) -> None:
        """Set commit callback"""
        self.commit_callback = callback