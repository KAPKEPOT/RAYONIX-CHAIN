# blockchain/state/state_manager.py
import os
import hashlib
import json
import time
import threading
import pickle
import zlib
from typing import Dict, List, Any, Optional, Deque, Tuple, Set, Generator
from dataclasses import dataclass, asdict, field
from collections import deque
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
import logging
import uuid

from blockchain.models.block import Block
from utxo_system.database.core import UTXOSet
from consensusengine.core.consensus import ProofOfStake
from smart_contract.core.contract_manager import ContractManager

logger = logging.getLogger(__name__)

class TransactionState(Enum):
    """Transaction state enumeration"""
    PENDING = "pending"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"

class StateTransitionType(Enum):
    """State transition type enumeration"""
    BLOCK_APPLY = "block_apply"
    BLOCK_REVERT = "block_revert"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    STATE_RECOVERY = "state_recovery"

@dataclass
class StateSnapshot:
    """Comprehensive state snapshot container"""
    utxo_set: Any
    consensus_state: Any
    contract_states: Any
    timestamp: float
    state_checksum: str
    block_height: int
    transaction_id: str
    snapshot_hash: str = ""

@dataclass
class StateTransition:
    """State transition record"""
    transaction_id: str
    type: StateTransitionType
    block_hash: Optional[str] = None
    block_height: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    state_before: Optional[StateSnapshot] = None
    state_after: Optional[StateSnapshot] = None
    duration: float = 0.0
    success: bool = False
    error: Optional[str] = None

@dataclass
class StateMetrics:
    """State performance metrics container"""
    total_transitions: int = 0
    successful_transitions: int = 0
    failed_transitions: int = 0
    total_processing_time: float = 0.0
    average_transition_time: float = 0.0
    last_transition_time: float = 0.0
    checkpoint_creations: int = 0
    checkpoint_restorations: int = 0
    state_verifications: int = 0
    state_corruptions: int = 0

class StateIntegrityError(Exception):
    """State integrity verification failed"""
    pass

class StateRecoveryError(Exception):
    """State recovery failed"""
    pass

class StateManager:
    """Manages blockchain state transitions atomically with enterprise-grade features"""
    
    def __init__(self, database: Any, utxo_set: UTXOSet, consensus: ProofOfStake, 
                 contract_manager: ContractManager, state_path: Optional[str] = None):
        self.database = database
        self.utxo_set = utxo_set
        self.consensus = consensus
        self.contract_manager = contract_manager
        
        # State storage configuration
        self.state_path = Path(state_path) if state_path else Path("state_data")
        self.state_path.mkdir(exist_ok=True)
        
        # Threading and synchronization
        self.lock = threading.RLock()
        self.transaction_locks: Dict[str, threading.RLock] = {}
        
        # State management
        self.state_transition_log: Deque[StateTransition] = deque(maxlen=50000)
        self.active_transactions: Set[str] = set()
        self.last_checkpoint: Optional[str] = None
        self.state_checksum: Optional[str] = None
        self.last_persist_time: float = 0
        self.last_verification_time: float = 0
        
        # Configuration
        self.persist_interval: int = 60  # seconds
        self.verification_interval: int = 300  # seconds
        self.max_transaction_age: int = 3600  # seconds
        self.snapshot_compression: bool = True
        self.enable_state_verification: bool = True
        self.state_recovery_enabled: bool = True
        
        # Performance tracking
        self.metrics = StateMetrics()
        self.startup_time: float = time.time()
        
        # Initialize state
        self._initialize_state()
        
        logger.info("StateManager initialized with enterprise features")

    def _initialize_state(self):
        """Initialize or recover state on startup"""
        try:
            # Try to load persisted state
            if self._load_persisted_state():
                logger.info("State loaded from persistence")
                if self.enable_state_verification:
                    if not self.verify_state_integrity(force=True):
                        logger.warning("State integrity check failed on startup")
                        if self.state_recovery_enabled:
                            self._attempt_state_recovery()
            else:
                logger.info("No persisted state found, starting fresh")
                self._persist_state()  # Persist initial state
                
        except Exception as e:
            logger.error(f"State initialization failed: {e}")
            if self.state_recovery_enabled:
                self._attempt_state_recovery()

    @contextmanager
    def atomic_state_transition(self, transition_type: StateTransitionType = StateTransitionType.BLOCK_APPLY,
                               block: Optional[Block] = None) -> Generator[str, None, None]:
        """Context manager for atomic state transitions with enhanced features"""
        transaction_id = self._start_transaction(transition_type, block)
        start_time = time.time()
        
        try:
            yield transaction_id
            self._commit_transaction(transaction_id, block, time.time() - start_time)
            
        except Exception as e:
            duration = time.time() - start_time
            self._rollback_transaction(transaction_id, str(e), duration)
            raise

    def _start_transaction(self, transition_type: StateTransitionType, 
                          block: Optional[Block] = None) -> str:
        """Start a new state transaction with comprehensive snapshot"""
        transaction_id = str(uuid.uuid4())
        
        with self.lock:
            # Create transaction-specific lock
            self.transaction_locks[transaction_id] = threading.RLock()
            self.active_transactions.add(transaction_id)
            
            # Create comprehensive state snapshot
            snapshot = StateSnapshot(
                utxo_set=self.utxo_set.create_snapshot(),
                consensus_state=self.consensus.create_snapshot(),
                contract_states=self.contract_manager.create_snapshot(),
                timestamp=time.time(),
                state_checksum=self.state_checksum,
                block_height=self.get_current_height(),
                transaction_id=transaction_id
            )
            
            # Calculate snapshot hash for integrity
            snapshot.snapshot_hash = self._calculate_snapshot_hash(snapshot)
            
            # Create transition record
            transition = StateTransition(
                transaction_id=transaction_id,
                type=transition_type,
                block_hash=block.hash if block else None,
                block_height=block.header.height if block else None,
                state_before=snapshot,
                timestamp=time.time()
            )
            
            self.state_transition_log.append(transition)
            
            logger.debug(f"Started transaction {transaction_id} for {transition_type.value}")
            
            return transaction_id

    def _commit_transaction(self, transaction_id: str, block: Optional[Block], duration: float):
        """Commit a state transaction with verification"""
        with self.lock:
            if transaction_id not in self.active_transactions:
                raise StateIntegrityError(f"Transaction {transaction_id} not active")
            
            # Find the transaction record
            transition = self._find_transition(transaction_id)
            if not transition:
                raise StateIntegrityError(f"Transaction record not found: {transaction_id}")
            
            # Update transition record
            transition.state_after = StateSnapshot(
                utxo_set=self.utxo_set.create_snapshot(),
                consensus_state=self.consensus.create_snapshot(),
                contract_states=self.contract_manager.create_snapshot(),
                timestamp=time.time(),
                state_checksum=self._calculate_state_hash(),
                block_height=self.get_current_height(),
                transaction_id=transaction_id
            )
            transition.state_after.snapshot_hash = self._calculate_snapshot_hash(transition.state_after)
            transition.duration = duration
            transition.success = True
            
            # Verify state integrity if enabled
            if self.enable_state_verification:
                if not self.verify_state_integrity():
                    raise StateIntegrityError("State integrity verification failed after commit")
            
            # Persist state if needed
            current_time = time.time()
            if current_time - self.last_persist_time >= self.persist_interval:
                self._persist_state()
                self.last_persist_time = current_time
            
            # Cleanup transaction
            self.active_transactions.remove(transaction_id)
            if transaction_id in self.transaction_locks:
                del self.transaction_locks[transaction_id]
            
            # Update metrics
            self.metrics.successful_transitions += 1
            self.metrics.total_transitions += 1
            self.metrics.total_processing_time += duration
            self.metrics.average_transition_time = (
                self.metrics.total_processing_time / self.metrics.total_transitions
            )
            self.metrics.last_transition_time = current_time
            
            logger.debug(f"Committed transaction {transaction_id} in {duration:.3f}s")

    def _rollback_transaction(self, transaction_id: str, error: str, duration: float):
        """Rollback a state transaction with comprehensive recovery"""
        with self.lock:
            try:
                # Find the transaction start and restore state
                transition = self._find_transition(transaction_id)
                if not transition or not transition.state_before:
                    raise StateRecoveryError("Cannot rollback - no snapshot available")
                
                # Verify snapshot integrity before restoration
                if not self._verify_snapshot_integrity(transition.state_before):
                    raise StateIntegrityError("Snapshot integrity verification failed")
                
                # Restore state from snapshot
                self.utxo_set.restore_snapshot(transition.state_before.utxo_set)
                self.consensus.restore_snapshot(transition.state_before.consensus_state)
                self.contract_manager.restore_snapshot(transition.state_before.contract_states)
                self.state_checksum = transition.state_before.state_checksum
                
                # Update transition record
                transition.duration = duration
                transition.success = False
                transition.error = error
                
                logger.warning(f"Rolled back transaction {transaction_id} due to: {error}")
                
            except Exception as rollback_error:
                logger.error(f"Critical rollback failure for {transaction_id}: {rollback_error}")
                # Emergency state recovery
                if self.state_recovery_enabled:
                    self._emergency_state_recovery()
                raise StateRecoveryError(f"Rollback failed: {rollback_error}") from rollback_error
            
            finally:
                # Cleanup transaction
                if transaction_id in self.active_transactions:
                    self.active_transactions.remove(transaction_id)
                if transaction_id in self.transaction_locks:
                    del self.transaction_locks[transaction_id]
                
                # Update metrics
                self.metrics.failed_transitions += 1
                self.metrics.total_transitions += 1

    def apply_block(self, block: Block, verify_after: bool = True) -> bool:
        """Apply a block to the state atomically with enhanced validation"""
        with self.atomic_state_transition(StateTransitionType.BLOCK_APPLY, block) as transaction_id:
            # Pre-validation
            if not self._validate_block_state_changes(block):
                raise ValueError("Block state changes validation failed")
            
            # Update UTXO set with batch processing
            utxo_results = []
            for tx in block.transactions:
                result = self.utxo_set.process_transaction(tx)
                utxo_results.append((tx.hash, result))
                if not result:
                    raise ValueError(f"Failed to process transaction {tx.hash}")
            
            # Update consensus state
            if not self.consensus.process_block(block):
                # Revert UTXO changes
                for tx_hash, result in reversed(utxo_results):
                    if result:
                        self.utxo_set.revert_transaction_by_hash(tx_hash)
                raise ValueError("Failed to process block in consensus")
            
            # Execute smart contracts with gas limiting
            contract_results = []
            for tx in block.transactions:
                if tx.is_contract_call():
                    result = self.contract_manager.execute_transaction(tx)
                    contract_results.append((tx.hash, result))
                    if not result.success:
                        # Revert all changes
                        for tx_hash, contract_result in reversed(contract_results):
                            if contract_result.success:
                                self.contract_manager.revert_transaction_by_hash(tx_hash)
                        for tx_hash, utxo_result in reversed(utxo_results):
                            if utxo_result:
                                self.utxo_set.revert_transaction_by_hash(tx_hash)
                        self.consensus.revert_block(block)
                        raise ValueError(f"Contract execution failed: {result.error}")
            
            # Update state checksum
            new_checksum = self._calculate_state_hash()
            self.state_checksum = new_checksum
            
            # Post-application verification
            if verify_after and self.enable_state_verification:
                if not self.verify_state_integrity():
                    raise StateIntegrityError("State integrity verification failed after block application")
            
            logger.info(f"Successfully applied block {block.hash} at height {block.header.height}")
            return True

    def revert_block(self, block: Block, verify_after: bool = True) -> bool:
        """Revert a block from the state with comprehensive rollback"""
        with self.atomic_state_transition(StateTransitionType.BLOCK_REVERT, block) as transaction_id:
            # Revert contract states first (reverse order)
            contract_reverts = []
            for tx in reversed(block.transactions):
                if tx.is_contract_call():
                    success = self.contract_manager.revert_transaction(tx)
                    contract_reverts.append((tx.hash, success))
                    if not success:
                        raise ValueError(f"Failed to revert contract transaction {tx.hash}")
            
            # Revert consensus state
            if not self.consensus.revert_block(block):
                # Restore contract states
                for tx_hash, success in contract_reverts:
                    if success:
                        # Note: This would require more sophisticated restoration logic
                        logger.warning(f"Partial revert - contract state may be inconsistent for {tx_hash}")
                raise ValueError("Failed to revert block in consensus")
            
            # Revert UTXO set (reverse order)
            utxo_reverts = []
            for tx in reversed(block.transactions):
                success = self.utxo_set.revert_transaction(tx)
                utxo_reverts.append((tx.hash, success))
                if not success:
                    raise ValueError(f"Failed to revert transaction {tx.hash}")
            
            # Update state checksum
            self.state_checksum = self._calculate_state_hash()
            
            # Post-reversion verification
            if verify_after and self.enable_state_verification:
                if not self.verify_state_integrity():
                    raise StateIntegrityError("State integrity verification failed after block reversion")
            
            logger.info(f"Successfully reverted block {block.hash} at height {block.header.height}")
            return True

    def _validate_block_state_changes(self, block: Block) -> bool:
        """Validate that block state changes are safe to apply"""
        # Check for double spends in the block itself
        spent_outputs = set()
        for tx in block.transactions:
            for input in tx.inputs:
                output_key = f"{input.tx_hash}:{input.output_index}"
                if output_key in spent_outputs:
                    return False
                spent_outputs.add(output_key)
        
        # Check gas limits for contract executions
        total_gas = 0
        for tx in block.transactions:
            if tx.is_contract_call():
                total_gas += tx.gas_limit
                if total_gas > self.contract_manager.max_block_gas:
                    return False
        
        return True

    def create_checkpoint(self, name: Optional[str] = None) -> str:
        """Create a comprehensive state checkpoint with integrity verification"""
        checkpoint_id = name or f"checkpoint_{int(time.time())}_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}"
        
        with self.lock:
            try:
                # Create comprehensive checkpoint data
                checkpoint_data = StateSnapshot(
                    utxo_set=self.utxo_set.create_snapshot(),
                    consensus=self.consensus.create_snapshot(),
                    contracts=self.contract_manager.create_snapshot(),
                    timestamp=time.time(),
                    state_checksum=self.state_checksum,
                    block_height=self.get_current_height(),
                    transaction_id=checkpoint_id
                )
                
                checkpoint_data.snapshot_hash = self._calculate_snapshot_hash(checkpoint_data)
                
                # Compress checkpoint data if enabled
                checkpoint_bytes = pickle.dumps(checkpoint_data)
                if self.snapshot_compression:
                    checkpoint_bytes = zlib.compress(checkpoint_bytes)
                
                # Store in database
                self.database.put(f"checkpoint_{checkpoint_id}", checkpoint_bytes)
                
                # Also store in file system for redundancy
                self._store_checkpoint_file(checkpoint_id, checkpoint_bytes)
                
                self.last_checkpoint = checkpoint_id
                self.metrics.checkpoint_creations += 1
                
                logger.info(f"Checkpoint created successfully: {checkpoint_id} at height {checkpoint_data.block_height}")
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Failed to create checkpoint: {e}")
                raise

    def restore_checkpoint(self, checkpoint_id: str, verify_integrity: bool = True) -> bool:
        """Restore state from checkpoint with comprehensive validation"""
        with self.lock:
            try:
                # Try to load from database first
                checkpoint_bytes = self.database.get(f"checkpoint_{checkpoint_id}")
                if not checkpoint_bytes:
                    # Fallback to file system
                    checkpoint_bytes = self._load_checkpoint_file(checkpoint_id)
                    if not checkpoint_bytes:
                        logger.error(f"Checkpoint not found: {checkpoint_id}")
                        return False
                
                # Decompress if necessary
                try:
                    if self.snapshot_compression:
                        checkpoint_bytes = zlib.decompress(checkpoint_bytes)
                    checkpoint_data = pickle.loads(checkpoint_bytes)
                except (zlib.error, pickle.PickleError) as e:
                    logger.error(f"Checkpoint data corrupted: {e}")
                    return False
                
                # Verify checkpoint integrity
                if verify_integrity:
                    if not self._verify_snapshot_integrity(checkpoint_data):
                        logger.error(f"Checkpoint integrity verification failed: {checkpoint_id}")
                        return False
                
                # Restore state using atomic transaction
                with self.atomic_state_transition(StateTransitionType.CHECKPOINT_RESTORE) as transaction_id:
                    self.utxo_set.restore_snapshot(checkpoint_data.utxo_set)
                    self.consensus.restore_snapshot(checkpoint_data.consensus)
                    self.contract_manager.restore_snapshot(checkpoint_data.contracts)
                    self.state_checksum = checkpoint_data.state_checksum
                
                self.last_checkpoint = checkpoint_id
                self.metrics.checkpoint_restorations += 1
                
                logger.info(f"Successfully restored checkpoint: {checkpoint_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
                return False

    def verify_state_integrity(self, force: bool = False) -> bool:
        """Verify the integrity of the current state with comprehensive checks"""
        current_time = time.time()
        
        if not force and current_time - self.last_verification_time < self.verification_interval:
            return True  # Skip if not forced and within interval
        
        try:
            current_checksum = self._calculate_state_hash()
            stored_checksum = self.database.get('state_checksum')
            
            # Verify against stored checksum
            if stored_checksum and current_checksum != stored_checksum:
                logger.error(f"State integrity check failed: checksum mismatch")
                self.metrics.state_corruptions += 1
                return False
            
            # Verify component integrity
            if not self.utxo_set.verify_integrity():
                logger.error("UTXO set integrity verification failed")
                return False
            
            if not self.consensus.verify_integrity():
                logger.error("Consensus state integrity verification failed")
                return False
            
            if not self.contract_manager.verify_integrity():
                logger.error("Contract manager integrity verification failed")
                return False
            
            self.last_verification_time = current_time
            self.metrics.state_verifications += 1
            
            return True
            
        except Exception as e:
            logger.error(f"State integrity verification error: {e}")
            return False

    def _persist_state(self):
        """Persist current state to database with compression and verification"""
        try:
            # Create state snapshot
            state_snapshot = {
                'utxo_set': self.utxo_set.to_bytes(),
                'consensus_state': self.consensus.to_bytes(),
                'contract_states': self.contract_manager.to_bytes(),
                'state_checksum': self._calculate_state_hash(),
                'timestamp': time.time(),
                'block_height': self.get_current_height()
            }
            
            # Compress state data
            state_data = pickle.dumps(state_snapshot)
            if self.snapshot_compression:
                state_data = zlib.compress(state_data)
            
            # Store in database
            self.database.put('state_snapshot', state_data)
            self.database.put('state_checksum', state_snapshot['state_checksum'])
            self.database.put('last_persist_time', time.time())
            
            # Also store in file system
            self._store_state_file(state_data)
            
            logger.debug("State persisted successfully")
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
            raise

    def _load_persisted_state(self) -> bool:
        """Load persisted state from database"""
        try:
            state_data = self.database.get('state_snapshot')
            if not state_data:
                return False
            
            # Decompress if necessary
            try:
                if self.snapshot_compression:
                    state_data = zlib.decompress(state_data)
                state_snapshot = pickle.loads(state_data)
            except (zlib.error, pickle.PickleError) as e:
                logger.error(f"Persisted state data corrupted: {e}")
                return False
            
            # Restore state
            self.utxo_set.from_bytes(state_snapshot['utxo_set'])
            self.consensus.from_bytes(state_snapshot['consensus_state'])
            self.contract_manager.from_bytes(state_snapshot['contract_states'])
            self.state_checksum = state_snapshot['state_checksum']
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load persisted state: {e}")
            return False

    def _calculate_state_hash(self) -> str:
        """Calculate comprehensive hash of current state for integrity checking"""
        state_components = {
            'utxo_set_hash': self.utxo_set.calculate_hash(),
            'consensus_hash': self.consensus.calculate_hash(),
            'contracts_hash': self.contract_manager.calculate_hash(),
            'block_height': self.get_current_height(),
            'timestamp': time.time()
        }
        
        # Sort keys for consistent hashing
        state_string = json.dumps(state_components, sort_keys=True)
        return hashlib.sha256(state_string.encode()).hexdigest()

    def _calculate_snapshot_hash(self, snapshot: StateSnapshot) -> str:
        """Calculate hash of a state snapshot"""
        snapshot_data = {
            'utxo_hash': hashlib.sha256(pickle.dumps(snapshot.utxo_set)).hexdigest(),
            'consensus_hash': hashlib.sha256(pickle.dumps(snapshot.consensus_state)).hexdigest(),
            'contracts_hash': hashlib.sha256(pickle.dumps(snapshot.contract_states)).hexdigest(),
            'timestamp': snapshot.timestamp,
            'block_height': snapshot.block_height
        }
        return hashlib.sha256(json.dumps(snapshot_data, sort_keys=True).encode()).hexdigest()

    def _verify_snapshot_integrity(self, snapshot: StateSnapshot) -> bool:
        """Verify the integrity of a state snapshot"""
        calculated_hash = self._calculate_snapshot_hash(snapshot)
        return calculated_hash == snapshot.snapshot_hash

    def _find_transition(self, transaction_id: str) -> Optional[StateTransition]:
        """Find a state transition by transaction ID"""
        for transition in reversed(self.state_transition_log):
            if transition.transaction_id == transaction_id:
                return transition
        return None

    def _store_checkpoint_file(self, checkpoint_id: str, data: bytes):
        """Store checkpoint data in file system"""
        try:
            file_path = self.state_path / f"{checkpoint_id}.checkpoint"
            with open(file_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            logger.warning(f"Failed to store checkpoint file: {e}")

    def _load_checkpoint_file(self, checkpoint_id: str) -> Optional[bytes]:
        """Load checkpoint data from file system"""
        try:
            file_path = self.state_path / f"{checkpoint_id}.checkpoint"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}")
        return None

    def _store_state_file(self, data: bytes):
        """Store state data in file system"""
        try:
            file_path = self.state_path / f"state_{int(time.time())}.snapshot"
            with open(file_path, 'wb') as f:
                f.write(data)
        except Exception as e:
            logger.warning(f"Failed to store state file: {e}")

    def _attempt_state_recovery(self):
        """Attempt to recover state from available checkpoints"""
        logger.warning("Attempting state recovery...")
        # Implementation would depend on specific recovery strategy
        pass

    def _emergency_state_recovery(self):
        """Emergency state recovery procedure"""
        logger.error("Performing emergency state recovery")
        # Implementation would depend on specific recovery requirements
        pass

    def get_current_height(self) -> int:
        """Get current blockchain height"""
        return self.database.get('current_height', 0)

    def get_state_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the current state"""
        current_time = time.time()
        uptime = current_time - self.startup_time
        
        return {
            'utxo_count': self.utxo_set.get_utxo_count(),
            'validators_count': self.consensus.get_validator_count(),
            'contracts_count': self.contract_manager.get_contract_count(),
            'state_size_bytes': self._estimate_state_size(),
            'last_checkpoint': self.last_checkpoint,
            'state_checksum': self.state_checksum,
            'active_transactions': len(self.active_transactions),
            'total_transitions': len(self.state_transition_log),
            'uptime_seconds': uptime,
            'state_metrics': asdict(self.metrics),
            'integrity_verified': self.verify_state_integrity(),
            'last_verification': self.last_verification_time
        }

    def _estimate_state_size(self) -> int:
        """Estimate the size of the current state in bytes"""
        utxo_size = self.utxo_set.get_utxo_count() * 200
        consensus_size = self.consensus.get_validator_count() * 500
        contract_size = self.contract_manager.get_contract_count() * 1000
        return utxo_size + consensus_size + contract_size

    def cleanup_old_data(self, max_transition_age: int = 86400):
        """Clean up old state transition data"""
        current_time = time.time()
        cutoff_time = current_time - max_transition_age
        
        with self.lock:
            # Remove old transitions
            self.state_transition_log = deque(
                [t for t in self.state_transition_log if t.timestamp > cutoff_time],
                maxlen=50000
            )
            
            # Cleanup old transaction locks
            expired_transactions = [
                tid for tid in self.transaction_locks.keys() 
                if tid not in self.active_transactions
            ]
            for tid in expired_transactions:
                del self.transaction_locks[tid]
            
            logger.info(f"Cleaned up state data older than {max_transition_age} seconds")