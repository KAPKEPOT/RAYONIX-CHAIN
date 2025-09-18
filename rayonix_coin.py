# rayonix_coin.py
import os
import hashlib
import json
import time
import threading
import asyncio
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
import ecdsa
from ecdsa import SECP256k1, SigningKey, VerifyingKey
import base58
import bech32
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sortedcontainers import SortedDict, SortedList
from collections import defaultdict, deque
import heapq
import msgpack
import zlib
import lmdb
from contextlib import contextmanager
from functools import lru_cache
import cachetools
import aiohttp
from aioprocessing import AioQueue, AioProcess
import uvloop

# Configure uvloop for better async performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import internal modules
from merkle import MerkleTree, CompactMerkleTree, MerkleTreeConfig, HashAlgorithm, ProofFormat, MerkleTreeFactory, MerkleTreeStats, global_stats, create_merkle_tree_from_file, create_merkle_tree_from_large_file, batch_verify_proofs, batch_verify_proofs_async, create_merkle_mountain_range
from utxo import UTXOSet, Transaction, UTXO
from consensus import ProofOfStake, Validator, Delegation, SlashingEvidence
from smart_contract import ContractManager, SmartContract, ContractState, ContractExecutionResult
from database import AdvancedDatabase, DatabaseConfig, DatabaseType, TransactionalDatabase
from wallet import RayonixWallet, WalletConfig, WalletType, AddressType, HDWallet, KeyManager
from p2p_network import AdvancedP2PNetwork, NodeConfig, PeerManager, NetworkProtocol
from crypto import CryptoUtils, SignatureVerifier, KeyDerivation, HashFunctions

class BlockchainState(Enum):
    SYNCING = auto()
    SYNCED = auto()
    FORKED = auto()
    RECOVERING = auto()
    STOPPED = auto()

class ValidationLevel(Enum):
    BASIC = auto()
    STANDARD = auto()
    FULL = auto()
    CONSENSUS = auto()

@dataclass
class BlockHeader:
    version: int
    height: int
    previous_hash: str
    merkle_root: str
    timestamp: int
    difficulty: int
    nonce: int
    validator: str
    signature: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Block:
    header: BlockHeader
    transactions: List[Transaction]
    hash: str
    chainwork: int
    size: int
    received_time: float = field(default_factory=time.time)

@dataclass
class ChainState:
    total_supply: int
    circulating_supply: int
    staking_rewards_distributed: int
    foundation_funds: int
    active_validators: int
    total_stake: int
    average_block_time: float
    current_difficulty: int
    last_block_time: float

@dataclass
class ForkResolution:
    common_ancestor: int
    old_chain_length: int
    new_chain_length: int
    chainwork_difference: int
    resolution_time: float
    blocks_rolled_back: int
    blocks_applied: int

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    execution_time: float
    validation_level: ValidationLevel

@dataclass
class TransactionCreationResult:
    success: bool
    transaction: Optional[Transaction]
    fee_estimate: int
    selected_utxos: List[UTXO]
    change_amount: int
    error_message: Optional[str] = None

@dataclass
class FeeEstimate:
    low: int
    medium: int
    high: int
    urgent: int
    timestamp: float
    confidence: float
    mempool_size: int

class StateManager:
    """Manages blockchain state transitions atomically"""
    
    def __init__(self, database: AdvancedDatabase, utxo_set: UTXOSet, consensus: ProofOfStake, 
                 contract_manager: ContractManager):
        self.database = database
        self.utxo_set = utxo_set
        self.consensus = consensus
        self.contract_manager = contract_manager
        self.lock = threading.RLock()
        self.state_transition_log = deque(maxlen=10000)
        self.last_checkpoint = None
        
    @contextmanager
    def atomic_state_transition(self):
        """Context manager for atomic state transitions"""
        transaction_id = self._start_transaction()
        try:
            yield transaction_id
            self._commit_transaction(transaction_id)
        except Exception as e:
            self._rollback_transaction(transaction_id)
            raise
    
    def _start_transaction(self) -> str:
        """Start a new state transaction"""
        transaction_id = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
        with self.lock:
            # Create transaction snapshot
            snapshot = {
                'utxo_set': self.utxo_set.snapshot(),
                'consensus_state': self.consensus.snapshot(),
                'contract_states': self.contract_manager.snapshot(),
                'timestamp': time.time()
            }
            self.state_transition_log.append((transaction_id, 'start', snapshot))
        return transaction_id
    
    def _commit_transaction(self, transaction_id: str):
        """Commit a state transaction"""
        with self.lock:
            self.state_transition_log.append((transaction_id, 'commit', None))
            # Persist state to database
            self._persist_state()
    
    def _rollback_transaction(self, transaction_id: str):
        """Rollback a state transaction"""
        with self.lock:
            # Find the transaction start and restore state
            for i, (tid, action, snapshot) in enumerate(reversed(self.state_transition_log)):
                if tid == transaction_id and action == 'start':
                    self.utxo_set.restore(snapshot['utxo_set'])
                    self.consensus.restore(snapshot['consensus_state'])
                    self.contract_manager.restore(snapshot['contract_states'])
                    break
            self.state_transition_log.append((transaction_id, 'rollback', None))
    
    def apply_block(self, block: Block) -> bool:
        """Apply a block to the state atomically"""
        with self.atomic_state_transition() as transaction_id:
            # Update UTXO set
            for tx in block.transactions:
                if not self.utxo_set.process_transaction(tx):
                    raise ValueError(f"Failed to process transaction {tx.hash}")
            
            # Update consensus state
            if not self.consensus.process_block(block):
                raise ValueError("Failed to process block in consensus")
            
            # Execute smart contracts
            for tx in block.transactions:
                if tx.is_contract_call():
                    result = self.contract_manager.execute_transaction(tx)
                    if not result.success:
                        raise ValueError(f"Contract execution failed: {result.error}")
            
            return True
    
    def revert_block(self, block: Block) -> bool:
        """Revert a block from the state"""
        with self.atomic_state_transition() as transaction_id:
            # Revert transactions in reverse order
            for tx in reversed(block.transactions):
                if not self.utxo_set.revert_transaction(tx):
                    raise ValueError(f"Failed to revert transaction {tx.hash}")
            
            # Revert consensus state
            if not self.consensus.revert_block(block):
                raise ValueError("Failed to revert block in consensus")
            
            # Revert contract states
            for tx in reversed(block.transactions):
                if tx.is_contract_call():
                    if not self.contract_manager.revert_transaction(tx):
                        raise ValueError(f"Failed to revert contract transaction {tx.hash}")
            
            return True
    
    def _persist_state(self):
        """Persist current state to database"""
        try:
            # Save UTXO set
            self.database.put('utxo_set_state', self.utxo_set.to_bytes())
            
            # Save consensus state
            self.database.put('consensus_state', self.consensus.to_bytes())
            
            # Save contract states
            self.database.put('contract_states', self.contract_manager.to_bytes())
            
            # Update state checksum
            state_hash = self._calculate_state_hash()
            self.database.put('state_checksum', state_hash)
            
        except Exception as e:
            logger.error(f"Failed to persist state: {e}")
            raise
    
    def _calculate_state_hash(self) -> str:
        """Calculate hash of current state for integrity checking"""
        state_data = {
            'utxo_set_hash': self.utxo_set.calculate_hash(),
            'consensus_hash': self.consensus.calculate_hash(),
            'contracts_hash': self.contract_manager.calculate_hash(),
            'timestamp': time.time()
        }
        return hashlib.sha256(json.dumps(state_data, sort_keys=True).encode()).hexdigest()
    
    def create_checkpoint(self) -> str:
        """Create a state checkpoint"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        checkpoint_data = {
            'utxo_set': self.utxo_set.snapshot(),
            'consensus': self.consensus.snapshot(),
            'contracts': self.contract_manager.snapshot(),
            'timestamp': time.time(),
            'block_height': self.get_current_height()
        }
        self.database.put(f"checkpoint_{checkpoint_id}", checkpoint_data)
        self.last_checkpoint = checkpoint_id
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint"""
        try:
            checkpoint_data = self.database.get(f"checkpoint_{checkpoint_id}")
            if not checkpoint_data:
                return False
            
            self.utxo_set.restore(checkpoint_data['utxo_set'])
            self.consensus.restore(checkpoint_data['consensus'])
            self.contract_manager.restore(checkpoint_data['contracts'])
            
            logger.info(f"Restored state from checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False

class ValidationManager:
    """Modular validation system with validation pipelines"""
    
    def __init__(self, state_manager: StateManager, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.config = config
        self.validation_pipelines = {
            ValidationLevel.BASIC: self._create_basic_pipeline(),
            ValidationLevel.STANDARD: self._create_standard_pipeline(),
            ValidationLevel.FULL: self._create_full_pipeline(),
            ValidationLevel.CONSENSUS: self._create_consensus_pipeline()
        }
        self.cache = cachetools.LRUCache(maxsize=10000)
        self.validation_stats = defaultdict(int)
    
    def validate_block(self, block: Block, level: ValidationLevel = ValidationLevel.FULL) -> ValidationResult:
        """Validate a block with specified validation level"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            pipeline = self.validation_pipelines[level]
            
            for validator_name, validator_func in pipeline:
                try:
                    result = validator_func(block)
                    if not result['valid']:
                        errors.extend(result['errors'])
                    if result['warnings']:
                        warnings.extend(result['warnings'])
                    
                    if errors and level != ValidationLevel.CONSENSUS:
                        break
                        
                except Exception as e:
                    errors.append(f"{validator_name} failed: {str(e)}")
                    break
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Validation pipeline failed: {str(e)}")
            is_valid = False
        
        execution_time = time.time() - start_time
        self.validation_stats[level.name] += 1
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            execution_time=execution_time,
            validation_level=level
        )
    
    def validate_transaction(self, transaction: Transaction, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a transaction with specified validation level"""
        # Similar implementation to validate_block but for transactions
        pass
    
    def _create_basic_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create basic validation pipeline"""
        return [
            ('block_structure', self._validate_block_structure),
            ('block_hash', self._validate_block_hash),
            ('previous_block', self._validate_previous_block)
        ]
    
    def _create_standard_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create standard validation pipeline"""
        pipeline = self._create_basic_pipeline()
        pipeline.extend([
            ('merkle_root', self._validate_merkle_root),
            ('timestamp', self._validate_timestamp),
            ('difficulty', self._validate_difficulty),
            ('signature', self._validate_signature)
        ])
        return pipeline
    
    def _create_full_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create full validation pipeline"""
        pipeline = self._create_standard_pipeline()
        pipeline.extend([
            ('transactions_basic', self._validate_transactions_basic),
            ('gas_limit', self._validate_gas_limit),
            ('block_size', self._validate_block_size)
        ])
        return pipeline
    
    def _create_consensus_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create consensus-level validation pipeline"""
        pipeline = self._create_full_pipeline()
        pipeline.extend([
            ('transactions_full', self._validate_transactions_full),
            ('state_transition', self._validate_state_transition),
            ('consensus_rules', self._validate_consensus_rules)
        ])
        return pipeline
    
    def _validate_block_structure(self, block: Block) -> Dict[str, Any]:
        """Validate block structure"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['version', 'height', 'previous_hash', 'merkle_root', 
                          'timestamp', 'difficulty', 'nonce', 'validator']
        
        for field in required_fields:
            if not hasattr(block.header, field) or getattr(block.header, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Check block size
        if block.size > self.config['max_block_size']:
            errors.append(f"Block size {block.size} exceeds maximum {self.config['max_block_size']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_block_hash(self, block: Block) -> Dict[str, Any]:
        """Validate block hash"""
        errors = []
        calculated_hash = self._calculate_block_hash(block)
        
        if calculated_hash != block.hash:
            errors.append(f"Invalid block hash. Calculated: {calculated_hash}, Expected: {block.hash}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_previous_block(self, block: Block) -> Dict[str, Any]:
        """Validate previous block reference"""
        errors = []
        
        try:
            previous_block = self.state_manager.database.get_block(block.header.previous_hash)
            if not previous_block:
                errors.append(f"Previous block not found: {block.header.previous_hash}")
            elif previous_block.header.height != block.header.height - 1:
                errors.append(f"Height mismatch with previous block")
                
        except Exception as e:
            errors.append(f"Error validating previous block: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_merkle_root(self, block: Block) -> Dict[str, Any]:
        """Validate merkle root"""
        errors = []
        
        try:
            tx_hashes = [tx.hash for tx in block.transactions]
            calculated_root = MerkleTree(tx_hashes).get_root_hash()
            
            if calculated_root != block.header.merkle_root:
                errors.append(f"Invalid merkle root. Calculated: {calculated_root}, Expected: {block.header.merkle_root}")
                
        except Exception as e:
            errors.append(f"Error calculating merkle root: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_timestamp(self, block: Block) -> Dict[str, Any]:
        """Validate block timestamp"""
        errors = []
        warnings = []
        
        current_time = time.time()
        max_future_time = current_time + self.config['max_future_block_time']
        
        if block.header.timestamp > max_future_time:
            errors.append(f"Block timestamp is too far in the future")
        elif block.header.timestamp < current_time - self.config['max_past_block_time']:
            warnings.append("Block timestamp is very old")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_difficulty(self, block: Block) -> Dict[str, Any]:
        """Validate block difficulty"""
        errors = []
        
        try:
            expected_difficulty = self.state_manager.consensus.calculate_difficulty(block.header.height)
            if block.header.difficulty != expected_difficulty:
                errors.append(f"Invalid difficulty. Expected: {expected_difficulty}, Got: {block.header.difficulty}")
                
        except Exception as e:
            errors.append(f"Error calculating difficulty: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_signature(self, block: Block) -> Dict[str, Any]:
        """Validate block signature"""
        errors = []
        
        try:
            if not self.state_manager.consensus.validate_block_signature(block):
                errors.append("Invalid block signature")
                
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_transactions_basic(self, block: Block) -> Dict[str, Any]:
        """Basic transaction validation"""
        errors = []
        warnings = []
        
        for tx in block.transactions:
            result = self.validate_transaction(tx, ValidationLevel.BASIC)
            if not result.is_valid:
                errors.extend([f"Transaction {tx.hash}: {e}" for e in result.errors])
            if result.warnings:
                warnings.extend([f"Transaction {tx.hash}: {w}" for w in result.warnings])
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_transactions_full(self, block: Block) -> Dict[str, Any]:
        """Full transaction validation including state checks"""
        errors = []
        warnings = []
        
        for tx in block.transactions:
            
            result = self.validate_transaction(tx, ValidationLevel.FULL)
            if not result.is_valid:
                errors.extend([f"Transaction {tx.hash}: {e}" for e in result.errors])
            if result.warnings:
                warnings.extend([f"Transaction {tx.hash}: {w}" for w in result.warnings])
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_state_transition(self, block: Block) -> Dict[str, Any]:
        """Validate state transition caused by block"""
        errors = []
        
        try:
            # Create temporary state manager for validation
            temp_state = self.state_manager.__class__(
                self.state_manager.database,
                self.state_manager.utxo_set.__class__(),
                self.state_manager.consensus.__class__(),
                self.state_manager.contract_manager.__class__()
            )
            
            # Copy current state
            temp_state.utxo_set.restore(self.state_manager.utxo_set.snapshot())
            temp_state.consensus.restore(self.state_manager.consensus.snapshot())
            temp_state.contract_manager.restore(self.state_manager.contract_manager.snapshot())
            
            # Try to apply block
            if not temp_state.apply_block(block):
                errors.append("State transition validation failed")
                
        except Exception as e:
            errors.append(f"State transition error: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_consensus_rules(self, block: Block) -> Dict[str, Any]:
        """Validate consensus-specific rules"""
        errors = []
        
        try:
            if not self.state_manager.consensus.validate_block_consensus(block):
                errors.append("Block violates consensus rules")
                
        except Exception as e:
            errors.append(f"Consensus validation error: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _calculate_block_hash(self, block: Block) -> str:
        """Calculate block hash for validation"""
        header_data = json.dumps(asdict(block.header), sort_keys=True).encode()
        return hashlib.sha256(header_data).hexdigest()

class TransactionManager:
    """Advanced transaction creation and fee estimation"""
    
    def __init__(self, state_manager: StateManager, wallet: RayonixWallet, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.wallet = wallet
        self.config = config
        self.fee_estimator = FeeEstimator(state_manager, config)
        self.coin_selection_strategies = {
            'default': self._default_coin_selection,
            'privacy': self._privacy_coin_selection,
            'efficiency': self._efficiency_coin_selection,
            'consolidation': self._consolidation_coin_selection
        }
        self.mempool = SortedDict()  # tx_hash -> (transaction, timestamp, fee_rate)
    
    def create_transaction(self, from_address: str, to_address: str, amount: int,
                         fee_strategy: str = 'default', coin_selection: str = 'default',
                         memo: Optional[str] = None, locktime: int = 0) -> TransactionCreationResult:
        """Create a transaction with advanced options"""
        try:
            # Get UTXOs for sender
            utxos = self.state_manager.utxo_set.get_utxos_for_address(from_address)
            if not utxos:
                return TransactionCreationResult(
                    success=False,
                    transaction=None,
                    fee_estimate=0,
                    selected_utxos=[],
                    change_amount=0,
                    error_message="No spendable funds"
                )
            
            # Estimate fee
            fee_estimate = self.fee_estimator.estimate_fee(fee_strategy)
            
            # Select coins based on strategy
            coin_selector = self.coin_selection_strategies.get(coin_selection, self._default_coin_selection)
            selected_utxos, total_input, change_amount = coin_selector(utxos, amount, fee_estimate)
            
            if total_input < amount + fee_estimate:
                return TransactionCreationResult(
                    success=False,
                    transaction=None,
                    fee_estimate=fee_estimate,
                    selected_utxos=selected_utxos,
                    change_amount=change_amount,
                    error_message="Insufficient funds"
                )
            
            # Create transaction inputs
            inputs = []
            for utxo in selected_utxos:
                inputs.append(TransactionInput(
                    tx_hash=utxo.tx_hash,
                    output_index=utxo.output_index,
                    signature=None,  # Will be signed later
                    public_key=None  # Will be set during signing
                ))
            
            # Create transaction outputs
            outputs = [
                TransactionOutput(
                    address=to_address,
                    amount=amount,
                    locktime=locktime
                )
            ]
            
            # Add change output if needed
            if change_amount > 0:
                change_address = self.wallet.get_change_address()
                outputs.append(TransactionOutput(
                    address=change_address,
                    amount=change_amount,
                    locktime=0
                ))
            
            # Create transaction
            transaction = Transaction(
                inputs=inputs,
                outputs=outputs,
                locktime=locktime,
                version=2,
                memo=memo
            )
            
            # Sign transaction
            signed_transaction = self.wallet.sign_transaction(transaction)
            
            return TransactionCreationResult(
                success=True,
                transaction=signed_transaction,
                fee_estimate=fee_estimate,
                selected_utxos=selected_utxos,
                change_amount=change_amount,
                error_message=None
            )
            
        except Exception as e:
            return TransactionCreationResult(
                success=False,
                transaction=None,
                fee_estimate=0,
                selected_utxos=[],
                change_amount=0,
                error_message=str(e)
            )
    
    def _default_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Default coin selection strategy (largest first)"""
        sorted_utxos = sorted(utxos, key=lambda x: x.amount, reverse=True)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            if total >= amount + fee:
                break
            selected.append(utxo)
            total += utxo.amount
        
        change = total - amount - fee
        return selected, total, change
    
    def _privacy_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Privacy-focused coin selection (minimize address reuse)"""
        # Prefer UTXOs that haven't been spent from recently
        sorted_utxos = sorted(utxos, key=lambda x: x.age, reverse=True)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            if total >= amount + fee:
                break
            selected.append(utxo)
            total += utxo.amount
        
        change = total - amount - fee
        return selected, total, change
    
    def _efficiency_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Efficiency-focused coin selection (minimize UTXO count)"""
        # Try to find a single UTXO that covers the amount
        for utxo in sorted(utxos, key=lambda x: x.amount, reverse=True):
            if utxo.amount >= amount + fee:
                change = utxo.amount - amount - fee
                return [utxo], utxo.amount, change
        
        # Fall back to default strategy
        return self._default_coin_selection(utxos, amount, fee)
    
    def _consolidation_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Consolidation strategy (spend many small UTXOs)"""
        sorted_utxos = sorted(utxos, key=lambda x: x.amount)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            selected.append(utxo)
            total += utxo.amount
            if total >= amount + fee:
                break
        
        change = total - amount - fee
        return selected, total, change
    
    def add_to_mempool(self, transaction: Transaction) -> bool:
        """Add transaction to mempool"""
        try:
            # Validate transaction
            validation_result = self.validate_transaction(transaction)
            if not validation_result.is_valid:
                return False
            
            # Calculate fee rate
            fee = sum(inp.amount for inp in transaction.inputs) - sum(out.amount for out in transaction.outputs)
            size = len(transaction.to_bytes())
            fee_rate = fee / size if size > 0 else 0
            
            # Add to mempool sorted by fee rate
            self.mempool[transaction.hash] = (transaction, time.time(), fee_rate)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add transaction to mempool: {e}")
            return False
    
    def get_mempool_transactions(self, limit: int = 1000) -> List[Transaction]:
        """Get transactions from mempool sorted by fee rate"""
        transactions = []
        for tx_hash, (tx, timestamp, fee_rate) in self.mempool.items():
            transactions.append(tx)
            if len(transactions) >= limit:
                break
        return transactions
    
    def remove_from_mempool(self, transaction_hashes: List[str]):
        """Remove transactions from mempool"""
        for tx_hash in transaction_hashes:
            if tx_hash in self.mempool:
                del self.mempool[tx_hash]

class FeeEstimator:
    """Dynamic fee estimation based on network conditions"""
    
    def __init__(self, state_manager: StateManager, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.config = config
        self.fee_history = deque(maxlen=1000)
        self.mempool_stats = deque(maxlen=100)
        self.last_estimate = FeeEstimate(0, 0, 0, 0, time.time(), 0.0, 0)
    
    def estimate_fee(self, strategy: str = 'medium') -> int:
        """Estimate transaction fee based on strategy"""
        current_stats = self._get_current_mempool_stats()
        historical_stats = self._get_historical_stats()
        
        if strategy == 'low':
            return self._calculate_low_fee(current_stats, historical_stats)
        elif strategy == 'medium':
            return self._calculate_medium_fee(current_stats, historical_stats)
        elif strategy == 'high':
            return self._calculate_high_fee(current_stats, historical_stats)
        elif strategy == 'urgent':
            return self._calculate_urgent_fee(current_stats, historical_stats)
        else:
            return self.config['min_transaction_fee']
    
    def _get_current_mempool_stats(self) -> Dict[str, Any]:
        """Get current mempool statistics"""
        # This would query the actual mempool state
        return {
            'size': 0,  # Placeholder
            'average_fee_rate': 0,
            'capacity_usage': 0,
            'priority_count': 0
        }
    
    def _get_historical_stats(self) -> Dict[str, Any]:
        """Get historical fee statistics"""
        # This would analyze historical fee data
        return {
            'average_fees': [],
            'congestion_patterns': {},
            'seasonal_trends': {}
        }
    
    def _calculate_low_fee(self, current_stats: Dict, historical_stats: Dict) -> int:
        """Calculate low priority fee"""
        base_fee = self.config['min_transaction_fee']
        # Add congestion adjustment
        congestion_factor = max(1.0, current_stats['capacity_usage'] / 0.5)
        return int(base_fee * congestion_factor)
    
    def _calculate_medium_fee(self, current_stats: Dict, historical_stats: Dict) -> int:
        """Calculate medium priority fee"""
        base_fee = self.config['min_transaction_fee'] * 2
        congestion_factor = max(1.0, current_stats['capacity_usage'] / 0.3)
        return int(base_fee * congestion_factor)
    
    def _calculate_high_fee(self, current_stats: Dict, historical_stats: Dict) -> int:
        """Calculate high priority fee"""
        base_fee = self.config['min_transaction_fee'] * 5
        congestion_factor = max(1.0, current_stats['capacity_usage'] / 0.1)
        return int(base_fee * congestion_factor)
    
    def _calculate_urgent_fee(self, current_stats: Dict, historical_stats: Dict) -> int:
        """Calculate urgent priority fee"""
        base_fee = self.config['min_transaction_fee'] * 10
        congestion_factor = max(1.0, current_stats['capacity_usage'] / 0.05)
        return int(base_fee * congestion_factor)
    
    def update_fee_history(self, block: Block):
        """Update fee history with new block data"""
        block_fees = sum(tx.fee for tx in block.transactions if tx.fee > 0)
        average_fee = block_fees / len(block.transactions) if block.transactions else 0
        
        self.fee_history.append({
            'height': block.header.height,
            'timestamp': block.header.timestamp,
            'average_fee': average_fee,
            'total_fees': block_fees,
            'transaction_count': len(block.transactions)
        })

class ForkManager:
    """Manages blockchain forks and reconciliation"""
    
    def __init__(self, state_manager: StateManager, validation_manager: ValidationManager, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.validation_manager = validation_manager
        self.config = config
        self.fork_history = deque(maxlen=100)
        self.reorganization_count = 0
        self.last_reorganization = 0
        self.fork_detection_threshold = config.get('fork_detection_threshold', 6)
    
    async def handle_possible_fork(self, new_block: Block) -> Optional[ForkResolution]:
        """Handle a potential fork caused by a new block"""
        current_head = self.state_manager.get_chain_head()
        
        # Check if this block causes a fork
        if new_block.header.previous_hash != current_head.hash:
            logger.warning(f"Potential fork detected at height {new_block.header.height}")
            return await self._resolve_fork(new_block, current_head)
        
        return None
    
    async def _resolve_fork(self, new_block: Block, current_head: Block) -> ForkResolution:
        """Resolve a fork between two chains"""
        start_time = time.time()
        
        try:
            # Find common ancestor
            common_ancestor = await self._find_common_ancestor(new_block, current_head)
            if not common_ancestor:
                logger.error("Could not find common ancestor for fork resolution")
                return None
            
            # Get both chains from common ancestor
            old_chain = await self._get_chain_segment(common_ancestor.hash, current_head.hash)
            new_chain = await self._get_chain_segment(common_ancestor.hash, new_block.hash)
            
            # Calculate chainwork for both chains
            old_chainwork = sum(block.chainwork for block in old_chain)
            new_chainwork = sum(block.chainwork for block in new_chain)
            
            # Decide which chain to keep
            if new_chainwork > old_chainwork:
                # New chain has more work, reorganize
                resolution = await self._reorganize_to_new_chain(old_chain, new_chain, common_ancestor)
                self.reorganization_count += 1
                self.last_reorganization = time.time()
                return resolution
            else:
                # Old chain has more or equal work, keep it
                logger.info("Keeping existing chain (equal or more chainwork)")
                return ForkResolution(
                    common_ancestor=common_ancestor.height,
                    old_chain_length=len(old_chain),
                    new_chain_length=len(new_chain),
                    chainwork_difference=old_chainwork - new_chainwork,
                    resolution_time=time.time() - start_time,
                    blocks_rolled_back=0,
                    blocks_applied=0
                )
                
        except Exception as e:
            logger.error(f"Fork resolution failed: {e}")
            raise
    
    async def _find_common_ancestor(self, block1: Block, block2: Block) -> Optional[Block]:
        """Find common ancestor of two blocks"""
        # If blocks are at different heights, walk back the longer chain
        height_diff = abs(block1.header.height - block2.header.height)
        
        if block1.header.height > block2.header.height:
            walk_block = block1
            for _ in range(height_diff):
                walk_block = self.state_manager.database.get_block(walk_block.header.previous_hash)
                if not walk_block:
                	                    return None
        else:
            walk_block = block2
            for _ in range(height_diff):
                walk_block = self.state_manager.database.get_block(walk_block.header.previous_hash)
                if not walk_block:
                    return None
        
        # Now both blocks are at same height, walk back until we find common hash
        block_a = block1 if block1.header.height <= block2.header.height else walk_block
        block_b = block2 if block2.header.height <= block1.header.height else walk_block
        
        while block_a and block_b:
            if block_a.hash == block_b.hash:
                return block_a
            
            block_a = self.state_manager.database.get_block(block_a.header.previous_hash)
            block_b = self.state_manager.database.get_block(block_b.header.previous_hash)
        
        return None
    
    async def _get_chain_segment(self, from_hash: str, to_hash: str) -> List[Block]:
        """Get chain segment between two blocks"""
        segment = []
        current_block = self.state_manager.database.get_block(to_hash)
        
        while current_block and current_block.hash != from_hash:
            segment.append(current_block)
            current_block = self.state_manager.database.get_block(current_block.header.previous_hash)
        
        segment.reverse()  # Return from ancestor to tip
        return segment
    
    async def _reorganize_to_new_chain(self, old_chain: List[Block], new_chain: List[Block], 
                                     common_ancestor: Block) -> ForkResolution:
        """Reorganize to new chain by rolling back old blocks and applying new ones"""
        rolled_back_blocks = 0
        applied_blocks = 0
        
        try:
            # Roll back old chain blocks
            for block in reversed(old_chain):
                if block.header.height > common_ancestor.height:
                    if not self.state_manager.revert_block(block):
                        raise ValueError(f"Failed to revert block {block.hash}")
                    rolled_back_blocks += 1
            
            # Apply new chain blocks
            for block in new_chain:
                if block.header.height > common_ancestor.height:
                    validation_result = self.validation_manager.validate_block(block, ValidationLevel.CONSENSUS)
                    if not validation_result.is_valid:
                        raise ValueError(f"New chain block validation failed: {validation_result.errors}")
                    
                    if not self.state_manager.apply_block(block):
                        raise ValueError(f"Failed to apply block {block.hash}")
                    applied_blocks += 1
            
            resolution = ForkResolution(
                common_ancestor=common_ancestor.height,
                old_chain_length=len(old_chain),
                new_chain_length=len(new_chain),
                chainwork_difference=sum(b.chainwork for b in new_chain) - sum(b.chainwork for b in old_chain),
                resolution_time=time.time() - start_time,
                blocks_rolled_back=rolled_back_blocks,
                blocks_applied=applied_blocks
            )
            
            logger.info(f"Chain reorganization completed: {rolled_back_blocks} blocks rolled back, {applied_blocks} blocks applied")
            return resolution
            
        except Exception as e:
            # Emergency recovery: restore from checkpoint
            logger.critical(f"Reorganization failed, restoring from checkpoint: {e}")
            if not self.state_manager.restore_checkpoint(self.state_manager.last_checkpoint):
                logger.critical("Checkpoint restoration failed! Node state may be inconsistent")
            raise
    
    def monitor_fork_risk(self) -> Dict[str, Any]:
        """Monitor and report fork risk metrics"""
        current_height = self.state_manager.get_current_height()
        fork_risk = {
            'current_height': current_height,
            'reorganization_count': self.reorganization_count,
            'time_since_last_reorg': time.time() - self.last_reorganization,
            'fork_probability': self._calculate_fork_probability(),
            'network_health': self._assess_network_health(),
            'recommended_actions': self._get_fork_prevention_actions()
        }
        return fork_risk
    
    def _calculate_fork_probability(self) -> float:
        """Calculate probability of fork based on network conditions"""
        # This would use statistical models based on:
        # - Network latency
        # - Validator distribution
        # - Block time variance
        # - Historical fork data
        return 0.05  # Placeholder
    
    def _assess_network_health(self) -> str:
        """Assess overall network health regarding forks"""
        if self.reorganization_count > self.config['max_reorganizations_per_hour']:
            return 'critical'
        elif time.time() - self.last_reorganization < 3600:
            return 'degraded'
        else:
            return 'healthy'
    
    def _get_fork_prevention_actions(self) -> List[str]:
        """Get recommended actions to prevent forks"""
        actions = []
        if self.reorganization_count > 0:
            actions.append("Increase network connectivity")
            actions.append("Monitor validator performance")
            actions.append("Consider increasing block time")
        return actions
        
    def get_network_risk_assessment(self) -> Dict[str, Any]:
        """Comprehensive network-level risk assessment"""
        return {
            'fork_probability': self._calculate_fork_probability(),
            'slashing_risk_index': self._calculate_network_slashing_risk(),
            'network_health_score': self._calculate_network_health(),
            'recommended_actions': self._get_risk_mitigation_strategies()
        }        

class RayonixCoin:
    """Production-ready RAYONIX blockchain engine"""
    
    def __init__(self, network_type: str = "mainnet", data_dir: str = "./rayonix_data"):
        self.network_type = network_type
        self.data_dir = data_dir
        self.state = BlockchainState.STOPPED
        self.config = self._load_configuration()
        
        # Initialize components
        self.database = self._initialize_database()
        self.utxo_set = UTXOSet()
        self.consensus = ProofOfStake(self.config['consensus'])
        self.contract_manager = ContractManager()
        self.wallet = self._initialize_wallet()
        self.network = self._initialize_network()
        self.gas_price = GAS_PRICE_CONFIG['base_gas_price']
        self.gas_price_history = deque(maxlen=1000)
        self.update_errors = 0
        
        # Initialize core managers
        self.state_manager = StateManager(self.database, self.utxo_set, self.consensus, self.contract_manager)
        self.validation_manager = ValidationManager(self.state_manager, self.config['validation'])
        self.transaction_manager = TransactionManager(self.state_manager, self.wallet, self.config['transactions'])
        self.fork_manager = ForkManager(self.state_manager, self.validation_manager, self.config['forks'])
        
        # State variables
        self.chain_head = None
        self.mempool_size = 0
        self.sync_progress = 0
        self.performance_metrics = defaultdict(list)
        
        # Background tasks
        self.background_tasks = []
        self.running = False
        
        # Initialize blockchain
        self._initialize_blockchain()
        
        logger.info(f"RAYONIX node initialized for {network_type} network")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load node configuration"""
        config_path = os.path.join(self.data_dir, 'rayonix.yaml')
        default_config = {
            'network': {
                'type': self.network_type,
                'port': 30303,
                'max_connections': 50,
                'bootstrap_nodes': self._get_bootstrap_nodes()
            },
            'consensus': {
                'block_reward': 50,
                'halving_interval': 210000,
                'difficulty_adjustment_blocks': 2016,
                'stake_minimum': 1000,
                'block_time_target': 30,
                'max_supply': 21000000,
                'developer_fee_percent': 0.05
            },
            'validation': {
                'max_block_size': 4000000,
                'max_transaction_size': 100000,
                'min_transaction_fee': 1,
                'max_future_block_time': 7200,
                'max_past_block_time': 86400
            },
            'transactions': {
                'max_mempool_size': 10000,
                'mempool_expiry_time': 3600,
                'fee_estimation_window': 100
            },
            'forks': {
                'fork_detection_threshold': 6,
                'max_reorganizations_per_hour': 3,
                'reorganization_depth_limit': 100
            },
            'database': {
                'type': 'plyvel',
                'cache_size': 128 * 1024 * 1024,
                'compression': 'snappy',
                'max_open_files': 1000
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"Failed to load config file: {e}")
        
        return default_config
    
    def _initialize_database(self) -> AdvancedDatabase:
        """Initialize database with proper configuration"""
        db_config = DatabaseConfig(
            db_type=DatabaseType.PLYVEL,
            compression=CompressionType.SNAPPY,
            cache_size=self.config['database']['cache_size'],
            max_open_files=self.config['database']['max_open_files']
        )
        
        db_path = os.path.join(self.data_dir, 'blockchain_db')
        return AdvancedDatabase(db_path, db_config)
    
    def _initialize_wallet(self) -> RayonixWallet:
        """Initialize wallet system"""
        wallet_config = WalletConfig(
            network=self.network_type,
            address_type=AddressType.RAYONIX,
            encryption=True,
            hd_wallet=True
        )
        
        wallet_path = os.path.join(self.data_dir, 'wallets')
        return RayonixWallet(wallet_config, wallet_path)
    
    def _initialize_network(self) -> AdvancedP2PNetwork:
        """Initialize P2P network"""
        network_config = NodeConfig(
            network_type=self.network_type.upper(),
            listen_port=self.config['network']['port'],
            max_connections=self.config['network']['max_connections'],
            bootstrap_nodes=self.config['network']['bootstrap_nodes']
        )
        
        return AdvancedP2PNetwork(network_config)
    
    def _get_bootstrap_nodes(self) -> List[str]:
        """Get bootstrap nodes for network"""
        if self.network_type == "mainnet":
            return [
                "node1.rayonix.com:30303",
                "node2.rayonix.com:30303",
                "node3.rayonix.com:30303",
                "node4.rayonix.com:30303"
            ]
        elif self.network_type == "testnet":
            return [
                "testnet-node1.rayonix.com:30304",
                "testnet-node2.rayonix.com:30304",
                "testnet-node3.rayonix.com:30304"
            ]
        else:
            return []
    
    def _initialize_blockchain(self):
        """Initialize or load blockchain state"""
        try:
            # Try to load chain head from database
            self.chain_head = self.database.get('chain_head')
            
            if not self.chain_head:
                # Create genesis block if not exists
                genesis_block = self._create_genesis_block()
                self._process_genesis_block(genesis_block)
            else:
                # Load full state
                self._load_blockchain_state()
                
        except Exception as e:
            logger.error(f"Blockchain initialization failed: {e}")
            raise
    
    def _create_genesis_block(self) -> Block:
        """Create genesis block"""
        genesis_tx = Transaction(
            inputs=[],
            outputs=[TransactionOutput(
                address=self.config['consensus']['foundation_address'],
                amount=self.config['consensus']['premine_amount'],
                locktime=0
            )],
            locktime=0,
            version=1
        )
        
        header = BlockHeader(
            version=1,
            height=0,
            previous_hash='0' * 64,
            merkle_root=MerkleTree([genesis_tx.hash]).get_root_hash(),
            timestamp=int(time.time()),
            difficulty=1,
            nonce=0,
            validator='genesis'
        )
        
        return Block(
            header=header,
            transactions=[genesis_tx],
            hash=self._calculate_block_hash(header),
            chainwork=1,
            size=len(json.dumps(asdict(header)).encode()) + len(genesis_tx.to_bytes())
        )
    
    def _process_genesis_block(self, genesis_block: Block):
        """Process genesis block and initialize state"""
        # Validate genesis block
        validation_result = self.validation_manager.validate_block(genesis_block, ValidationLevel.CONSENSUS)
        if not validation_result.is_valid:
            raise ValueError(f"Genesis block validation failed: {validation_result.errors}")
        
        # Apply to state
        if not self.state_manager.apply_block(genesis_block):
            raise ValueError("Failed to apply genesis block")
        
        # Save to database
        self.database.put_block(genesis_block.hash, genesis_block)
        self.database.put('chain_head', genesis_block.hash)
        self.chain_head = genesis_block.hash
        
        logger.info("Genesis block created and processed")
    
    def _load_blockchain_state(self):
        """Load full blockchain state from database"""
        # This would load the entire state from the database
        # For now, we assume the state manager handles this
        logger.info("Blockchain state loaded from database")
    
    def start(self):
        """Start the blockchain node"""
        if self.running:
            logger.warning("Node is already running")
            return
        
        self.running = True
        self.state = BlockchainState.SYNCING
        
        # Start background tasks
        self._start_background_tasks()
        
        # Connect to network
        self.network.start()
        
        # Start syncing
        asyncio.create_task(self._sync_with_network())
        
        logger.info("RAYONIX node started")
    
    def stop(self):
        """Stop the blockchain node"""
        if not self.running:
            return
        
        self.running = False
        self.state = BlockchainState.STOPPED
        
        # Stop background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Disconnect from network
        self.network.stop()
        
        # Save state
        self._save_state()
        
        logger.info("RAYONIX node stopped")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        tasks = [
            self._block_production_loop,
            self._mempool_management_loop,
            self._state_pruning_loop,
            self._performance_monitoring_loop,
            self._fork_monitoring_loop,
            self.start_gas_price_updater  # Add gas price updater
        ]
        
        for task_func in tasks:
            task = asyncio.create_task(task_func())
            self.background_tasks.append(task)
            
        for task_func in tasks:
            if task_func == self.start_gas_price_updater:
                # Start gas price updater directly
                task_func()
            else:
                task = asyncio.create_task(task_func())
                self.background_tasks.append(task)
    
    async def _block_production_loop(self):
        """Proof-of-Stake block production loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED and self._should_produce_block():
                    block = await self._create_new_block()
                    if block:
                        await self._process_new_block(block)
                        await self._broadcast_block(block)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Block production error: {e}")
                await asyncio.sleep(5)
    
    async def _mempool_management_loop(self):
        """Mempool management loop"""
        while self.running:
            try:
                # Clean expired transactions
                self._clean_mempool()
                
                # Validate mempool transactions
                await self._validate_mempool()
                
                # Broadcast transactions
                await self._broadcast_mempool()
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Mempool management error: {e}")
                await asyncio.sleep(30)
    
    async def _state_pruning_loop(self):
        """State pruning and compaction loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED:
                    # Prune old state data
                    await self._prune_old_state()
                    
                    # Compact database
                    if self._should_compact_database():
                        await self._compact_database()
                
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"State pruning error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.running:
            try:
                metrics = await self._collect_performance_metrics()
                self.performance_metrics[time.time()] = metrics
                
                # Keep only last 24 hours of metrics
                cutoff = time.time() - 86400
                self.performance_metrics = {k: v for k, v in self.performance_metrics.items() if k > cutoff}
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _fork_monitoring_loop(self):
        """Fork monitoring loop"""
        while self.running:
            try:
                if self.state == BlockchainState.SYNCED:
                    fork_risk = self.fork_manager.monitor_fork_risk()
                    if fork_risk['fork_probability'] > 0.1:
                        logger.warning(f"High fork risk detected: {fork_risk}")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Fork monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _sync_with_network(self):
        """Synchronize with network peers"""
        try:
            logger.info("Starting blockchain synchronization")
            
            # Get best block from peers
            best_block = await self._get_best_block_from_peers()
            if not best_block:
                logger.warning("No peers available for synchronization")
                return
            
            current_height = self.state_manager.get_current_height()
            
            if best_block.header.height > current_height:
                # We need to sync
                logger.info(f"Synchronizing from height {current_height} to {best_block.header.height}")
                await self._download_blocks(current_height + 1, best_block.header.height)
            
            self.state = BlockchainState.SYNCED
            logger.info("Blockchain synchronization completed")
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            self.state = BlockchainState.SYNCING
    
    async def _download_blocks(self, start_height: int, end_height: int):
        """Download blocks in parallel"""
        batch_size = 100
        semaphore = asyncio.Semaphore(10)  # Limit concurrent downloads
        
        async def download_batch(heights: List[int]):
            async with semaphore:
                blocks = await self._request_blocks_from_peers(heights)
                for block in blocks:
                    if block:
                        await self._process_block(block)
        
        tasks = []
        for batch_start in range(start_height, end_height + 1, batch_size):
            batch_end = min(batch_start + batch_size, end_height + 1)
            batch_heights = list(range(batch_start, batch_end))
            tasks.append(download_batch(batch_heights))
        
        await asyncio.gather(*tasks)
    
    async def _process_block(self, block: Block):
        """Process a downloaded block"""
        try:
            # Validate block
            validation_result = self.validation_manager.validate_block(block, ValidationLevel.FULL)
            if not validation_result.is_valid:
                logger.warning(f"Invalid block received: {validation_result.errors}")
                return False
            
            # Check for forks
            fork_resolution = await self.fork_manager.handle_possible_fork(block)
            if fork_resolution:
                logger.info(f"Fork resolved: {fork_resolution}")
            
            # Apply to state
            if not self.state_manager.apply_block(block):
                logger.error(f"Failed to apply block {block.hash}")
                return False
            
            # Update chain head
            self.chain_head = block.hash
            self.database.put('chain_head', block.hash)
            
            # Update sync progress
            self.sync_progress = (block.header.height / self.state_manager.get_current_height()) * 100
            
            return True
            
        except Exception as e:
            logger.error(f"Block processing failed: {e}")
            return False
    
    def _should_produce_block(self) -> bool:
        """Check if we should produce a block"""
        if not self.wallet or not self.wallet.addresses:
            return False
        
        validator_address = list(self.wallet.addresses.keys())[0]
        return self.consensus.should_produce_block(validator_address, time.time())
    
    async def _create_new_block(self) -> Optional[Block]:
        """Create a new block"""
        try:
            # Get transactions from mempool
            transactions = self.transaction_manager.get_mempool_transactions(
                self.config['validation']['max_block_size'] // 1000  # Approximate count
            )
            
            # Get current validator
            validator_address = list(self.wallet.addresses.keys())[0]
            current_head = self.database.get_block(self.chain_head)
            
            # Create block header
            header = BlockHeader(
                version=2,
                height=current_head.header.height + 1,
                previous_hash=current_head.hash,
                merkle_root=self._calculate_merkle_root([tx.hash for tx in transactions]),
                timestamp=int(time.time()),
                difficulty=self.consensus.calculate_difficulty(current_head.header.height + 1),
                nonce=0,
                validator=validator_address
            )
            
            # Create block
            block = Block(
                header=header,
                transactions=transactions,
                hash=self._calculate_block_hash(header),
                chainwork=current_head.chainwork + self._calculate_block_work(header.difficulty),
                size=self._calculate_block_size(header, transactions)
            )
            
            # Sign block
            block = await self._sign_block(block)
            
            return block
            
        except Exception as e:
            logger.error(f"Block creation failed: {e}")
            return None
    
    async def _sign_block(self, block: Block) -> Block:
        """Sign a block"""
        try:
            signature = self.wallet.sign_data(block.hash.encode())
            block.header.signature = signature
            return block
        except Exception as e:
            logger.error(f"Block signing failed: {e}")
            raise
    
    async def _process_new_block(self, block: Block):
        """Process a newly created block"""
        try:
            # Validate block
            validation_result = self.validation_manager.validate_block(block, ValidationLevel.CONSENSUS)
            if not validation_result.is_valid:
                logger.error(f"Self-created block validation failed: {validation_result.errors}")
                return False
            
            # Apply to state
            if not self.state_manager.apply_block(block):
                logger.error("Failed to apply self-created block")
                return False
            
            # Update chain head
            self.chain_head = block.hash
            self.database.put('chain_head', block.hash)
            self.database.put_block(block.hash, block)
            
            # Remove transactions from mempool
            self.transaction_manager.remove_from_mempool([tx.hash for tx in block.transactions])
            
            logger.info(f"New block created: #{block.header.height} - {block.hash[:16]}...")
            return True
            
        except Exception as e:
            logger.error(f"New block processing failed: {e}")
            return False
    
    async def _broadcast_block(self, block: Block):
        """Broadcast block to network"""
        try:
            await self.network.broadcast_message('block', block.to_dict())
        except Exception as e:
            logger.error(f"Block broadcast failed: {e}")
    
    def _clean_mempool(self):
        """Clean expired transactions from mempool"""
        current_time = time.time()
        expired_hashes = []
        
        for tx_hash, (tx, timestamp, fee_rate) in self.transaction_manager.mempool.items():
            if current_time - timestamp > self.config['transactions']['mempool_expiry_time']:
                expired_hashes.append(tx_hash)
        
        self.transaction_manager.remove_from_mempool(expired_hashes)
    
    async def _validate_mempool(self):
        """Validate all transactions in mempool"""
        invalid_hashes = []
        
        for tx_hash, (tx, timestamp, fee_rate) in self.transaction_manager.mempool.items():
            validation_result = self.validation_manager.validate_transaction(tx, ValidationLevel.STANDARD)
            if not validation_result.is_valid:
                invalid_hashes.append(tx_hash)
        
        self.transaction_manager.remove_from_mempool(invalid_hashes)
    
    async def _broadcast_mempool(self):
        """Broadcast mempool transactions to network"""
        transactions = self.transaction_manager.get_mempool_transactions(100)  # Broadcast top 100
        
        for tx in transactions:
            try:
                await self.network.broadcast_message('transaction', tx.to_dict())
            except Exception as e:
                logger.error(f"Transaction broadcast failed: {e}")
    
    async def _prune_old_state(self):
        """Prune old state data to save space"""
        current_height = self.state_manager.get_current_height()
        prune_height = current_height - self.config['forks']['reorganization_depth_limit']
        
        if prune_height > 0:
            # Prune old blocks and state data
            await self.database.prune_blocks_before(prune_height)
            logger.info(f"Pruned state data before height {prune_height}")
    
    async def _compact_database(self):
        """Compact database"""
        try:
            await self.database.compact()
            logger.info("Database compaction completed")
        except Exception as e:
            logger.error(f"Database compaction failed: {e}")
    
    def _should_compact_database(self) -> bool:
        """Check if database should be compacted"""
        # Check database size and fragmentation
        stats = self.database.get_stats()
        fragmentation = stats.get('fragmentation_ratio', 0)
        return fragmentation > 0.7  # Compact if fragmentation > 70%
    
    async def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics"""
        return {
            'block_height': self.state_manager.get_current_height(),
            'mempool_size': len(self.transaction_manager.mempool),
            'active_connections': self.network.get_connection_count(),
            'memory_usage': self._get_memory_usage(),
            'cpu_usage': self._get_cpu_usage(),
            'disk_usage': self._get_disk_usage(),
            'validation_stats': dict(self.validation_manager.validation_stats),
            'fork_count': self.fork_manager.reorganization_count
        }
    
    def _get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        import psutil
        return psutil.cpu_percent()
    
    def _get_disk_usage(self) -> float:
        """Get disk usage in GB"""
        import psutil
        usage = psutil.disk_usage(self.data_dir)
        return usage.used / 1024 / 1024 / 1024
    
    def _save_state(self):
        """Save current state to disk"""
        try:
            self.state_manager.create_checkpoint()
            self.database.sync()
            logger.info("Node state saved successfully")
        except Exception as e:
            logger.error(f"Failed to save node state: {e}")
    
    def _calculate_block_hash(self, header: BlockHeader) -> str:
        """Calculate block hash"""
        header_data = json.dumps(asdict(header), sort_keys=True).encode()
        return hashlib.sha256(header_data).hexdigest()
    
    def _calculate_merkle_root(self, tx_hashes: List[str]) -> str:
        """Calculate merkle root"""
        if not tx_hashes:
            return '0' * 64
        return MerkleTree(tx_hashes).get_root_hash()
    
    def _calculate_block_work(self, difficulty: int) -> int:
        """Calculate block work"""
        return 2 ** 256 // (difficulty + 1)
    
    def _calculate_block_size(self, header: BlockHeader, transactions: List[Transaction]) -> int:
        """Calculate block size in bytes"""
        header_size = len(json.dumps(asdict(header)).encode())
        transactions_size = sum(len(tx.to_bytes()) for tx in transactions)
        return header_size + transactions_size
    
    async def _get_best_block_from_peers(self) -> Optional[Block]:
        """Get best block information from peers"""
        try:
            # Query multiple peers and return the block with most chainwork
            peers = self.network.get_peers()
            if not peers:
                return None
            
            best_block = None
            best_chainwork = 0
            
            for peer in peers:
                try:
                    block_info = await peer.get_best_block()
                    if block_info and block_info.chainwork > best_chainwork:
                        best_block = block_info
                        best_chainwork = block_info.chainwork
                except Exception as e:
                    continue
            
            return best_block
            
        except Exception as e:
            logger.error(f"Failed to get best block from peers: {e}")
            return None
    
    async def _request_blocks_from_peers(self, heights: List[int]) -> List[Block]:
        """Request blocks from peers in parallel"""
        tasks = []
        for height in heights:
            tasks.append(self._request_block_from_peers(height))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [result for result in results if result and not isinstance(result, Exception)]
    
    async def _request_block_from_peers(self, height: int) -> Optional[Block]:
        """Request a specific block from peers"""
        try:
            peers = self.network.get_peers()
            for peer in peers:
                try:
                    block = await peer.get_block(height)
                    if block:
                        return block
                except Exception as e:
                    continue
            return None
        except Exception as e:
            logger.error(f"Failed to request block {height}: {e}")
            return None

    # Public API methods
    def get_balance(self, address: str) -> int:
        """Get balance for address"""
        return self.state_manager.utxo_set.get_balance(address)
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction by hash"""
        # Check mempool first
        if tx_hash in self.transaction_manager.mempool:
            return self.transaction_manager.mempool[tx_hash][0]
        
        # Check blockchain
        return self.database.get_transaction(tx_hash)
    
    def get_block(self, height_or_hash: Union[int, str]) -> Optional[Block]:
        """Get block by height or hash"""
        if isinstance(height_or_hash, int):
            return self.database.get_block_by_height(height_or_hash)
        else:
            return self.database.get_block(height_or_hash)
    
    def send_transaction(self, from_address: str, to_address: str, amount: int,
                       fee_strategy: str = 'medium', **kwargs) -> TransactionCreationResult:
        """Send a transaction"""
        return self.transaction_manager.create_transaction(
            from_address, to_address, amount, fee_strategy, **kwargs
        )
    
    def get_blockchain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        current_height = self.state_manager.get_current_height()
        return {
            'height': current_height,
            'chain_head': self.chain_head,
            'total_supply': self.state_manager.consensus.total_supply,
            'circulating_supply': self.state_manager.consensus.circulating_supply,
            'difficulty': self.consensus.calculate_difficulty(current_height),
            'block_reward': self.consensus.get_block_reward(current_height),
            'mempool_size': len(self.transaction_manager.mempool),
            'network_state': self.state.name,
            'sync_progress': self.sync_progress
        }
    
    def get_validator_info(self, address: Optional[str] = None) -> Dict[str, Any]:
        """Get validator information"""
        if not address and self.wallet.addresses:
            address = list(self.wallet.addresses.keys())[0]
        
        if not address:
            return {'error': 'No validator address provided'}
        
        return self.consensus.get_validator_info(address)
    
    def register_validator(self, stake_amount: int) -> bool:
        """Register as validator"""
        if not self.wallet.addresses:
            return False
        
        validator_address = list(self.wallet.addresses.keys())[0]
        return self.consensus.register_validator(validator_address, stake_amount)
    
    def deploy_contract(self, contract_code: str, initial_balance: int = 0) -> Optional[str]:
        """Deploy smart contract"""
        if not self.wallet.addresses:
            return None
        
        deployer_address = list(self.wallet.addresses.keys())[0]
        return self.contract_manager.deploy_contract(deployer_address, contract_code, initial_balance)
    
    def call_contract(self, contract_address: str, function_name: str,
                     args: List[Any], value: int = 0) -> Any:
        """Call smart contract function"""
        if not self.wallet.addresses:
            return None
        
        caller_address = list(self.wallet.addresses.keys())[0]
        return self.contract_manager.execute_contract(
            contract_address, function_name, args, caller_address, value
        )
@classmethod
def generate_genesis_block(cls, config_manager: Any = None, custom_config: Optional[Dict] = None) -> Dict:
    """Generate genesis block with custom configuration"""
    try:
    	# Use the global config manager if available, otherwise create a minimal one
    	if config_manager is None:
    		try:
    			from config import get_config
    			config_manager = get_config()
    		except ImportError:
    			config_manager = None
    	# Get configuration values from config manager
    	premine_amount = config_manager.get('consensus.block_reward', 50) * 20000 # 20,000 blocks worth
    	foundation_address = config_manager.get('wallet.foundation_address', 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX')
    	block_reward = config_manager.get('consensus.block_reward', 50)
    	network_id = config_manager.get('network.network_id', 1)
    	block_time = config_manager.get('consensus.block_time', 30)
    	max_supply = config_manager.get('consensus.max_supply', 21000000)
    	developer_fee_percent = config_manager.get('consensus.developer_fee_percent', 0.05)
    	
    	# Merge with custom configuration if provided
    	genesis_config = {
    	    'premine_amount': premine_amount,
    	    'foundation_address': foundation_address,
    	    'block_reward': block_reward,
    	    'timestamp': int(time.time()),
    	    'network_id': network_id,
    	    'validator': 'genesis',
    	    'version': 1,
    	    'difficulty': 1,
    	    'nonce': 0,
    	    'block_time_target': block_time,
    	    'max_supply': max_supply,
    	    'developer_fee_percent': developer_fee_percent
    	}
    	if custom_config:
    		genesis_config.update(custom_config)
    		
    	# Create premine transaction
    	premine_tx = Transaction(
    	    inputs=[],
    	    outputs=[{
    	        'address': genesis_config['foundation_address'],
    	        'amount': genesis_config['premine_amount'],
    	        'locktime': 0
    	    }],
    	    locktime=0
    	)
    	
    	genesis_transactions = [premine_tx.to_dict()]
    	
    	# Create genesis block
    	genesis_block = {
    	    'height': 0,
    	    'hash': '0' * 64,  # Temporary placeholder
    	    'previous_hash': '0' * 64,
    	    'merkle_root': '',
    	    'timestamp': genesis_config['timestamp'],
    	    'difficulty': genesis_config['difficulty'],
    	    'nonce': genesis_config['nonce'],
    	    'validator': genesis_config['validator'],
    	    'transactions': genesis_transactions,
    	    'version': genesis_config['version'],
    	    'chainwork': 1,
    	    'network_id': genesis_config['network_id'],
    	    'config_hash': hashlib.sha256(json.dumps(genesis_config, sort_keys=True).encode()).hexdigest()
    	}
    	
    	# Calculate merkle root using the instance method
    	temp_instance = cls.__new__(cls)
    	temp_instance.config = {'block_time_target': genesis_config['block_time_target']}
    	genesis_block['merkle_root'] = temp_instance._calculate_merkle_root(genesis_transactions)
    	# Calculate block hash
    	block_data = temp_instance._get_block_signing_data(genesis_block)
    	genesis_block['hash'] = hashlib.sha256(block_data).hexdigest()
    	
    	# Add signature if private key is provided in custom config
    	if custom_config and 'private_key' in custom_config:
    		try:
    			signing_key = SigningKey.from_string(bytes.fromhex(custom_config['private_key']), curve=SECP256k1)
    			signature = signing_key.sign(block_data)
    			genesis_block['signature'] = signature.hex()
    			genesis_block['public_key'] = signing_key.get_verifying_key().to_string().hex()
    		except Exception as e:
    			logger.warning(f"Could not sign genesis block: {e}")
    	
    	logger.info(f"Generated genesis block for network {genesis_config['network_id']}")
    	logger.info(f"Premine: {genesis_config['premine_amount']} to {genesis_config['foundation_address']}")
    	
    	return genesis_block
    	
    except ImportError as e:
    	logger.error(f"Config module not available: {e}")
    	raise RuntimeError("Configuration module is required for genesis block generation")
    	
    except Exception as e:
    	logger.error(f"Error generating genesis block: {e}")
    	raise        
        
        
# Add to rayonix_coin.py
class ValidatorAnalyticsManager:
    def __init__(self, state_manager: StateManager, consensus: ProofOfStake, database: AdvancedDatabase):
        self.state_manager = state_manager
        self.consensus = consensus
        self.database = database
        self.cache = cachetools.TTLCache(maxsize=1000, ttl=300)
        
    def _get_basic_validator_info(self, validator_address: str) -> Dict:
    	try:
    		info = self.consensus.get_validator_info(validator_address)
    		
    		# Enhanced validation of returned data
    		required_fields = ['registered', 'active', 'registration_height']
    		for field in required_fields:
    			if field not in info:
    				raise ValueError(f"Missing required field: {field}")
    				
    		# Additional derived information
    		current_height = len(self.blockchain) - 1
    		age_blocks = current_height - info.get('registration_height', current_height)
    		return {
    		    'is_registered': bool(info.get('registered', False)),
    		    'is_active': bool(info.get('active', False)),
    		    'registration_height': info.get('registration_height'),
    		    'last_active_height': info.get('last_active_height'),
    		    'validator_age_blocks': max(0, age_blocks),
    		    'validator_age_days': self._blocks_to_days(age_blocks),
    		    'public_key': info.get('public_key'),
    		    'commission_rate': info.get('commission_rate', 0.0),
    		    'max_commission_rate': info.get('max_commission_rate', 0.0),
    		    'commission_change_rate': info.get('commission_change_rate', 0.0),
    		    'identity': info.get('identity', {}),
    		    'description': info.get('description', {}),
    		    'status': self._determine_validator_status(info)
    		}
    	except Exception as e:
    		logger.error(f"Failed to get basic validator info: {e}")
    		return {
    		    'error': f'basic_info_failure: {str(e)}',
    		    'is_registered': False,
    		    'is_active': False,
    		    'status': 'unknown'
    		}    	         	

    def _get_staking_info(self, validator_address: str) -> Dict:
    	try:
    	    staked_amount = self.consensus.get_stake_amount(validator_address)
    	    total_staked = self.consensus.get_total_stake()
    	    delegated_stake = self.consensus.get_delegated_stake(validator_address)
    	    
    	    # Calculate percentages with safety checks 
    	    stake_percentage = (staked_amount / total_staked * 100) if total_staked > 0 else 0
    	    voting_power = self.consensus.get_voting_power(validator_address) if hasattr(
            self.consensus, 'get_voting_power') else staked_amount
    	    return {
    	        'self_stake': staked_amount,
    	        'delegated_stake': delegated_stake,
    	        'total_stake': staked_amount + delegated_stake,
    	        'total_network_stake': total_staked,
    	        'stake_percentage': round(stake_percentage, 4),
    	        'voting_power': voting_power,
    	        'voting_power_percentage': (voting_power / total_staked * 100) if total_staked > 0 else 0,
    	        'min_self_stake_requirement': self.config['stake_minimum'],
    	        'meets_min_stake': staked_amount >= self.config['stake_minimum'],
    	        'stake_rank': self.consensus.get_validator_rank(validator_address) if hasattr(
                self.consensus, 'get_validator_rank') else None,
                'delegators_count': self.consensus.get_delegators_count(validator_address) if hasattr(
                self.consensus, 'get_delegators_count') else 0
    	    }
    	except Exception as e:
    	    logger.error(f"Failed to get staking info: {e}")
    	    return {
    	        'error': f'staking_info_failure: {str(e)}',
    	        'self_stake': 0,
    	        'total_network_stake': 0,
    	        'stake_percentage': 0
    	    }	
    	    
    def _calculate_validator_performance(self, validator_address: str) -> Dict:
        """Calculate comprehensive performance metrics with caching."""
        cache_key = f"validator_perf_{validator_address}"
        cached = self._get_cached_performance(cache_key)
        if cached and not self._is_performance_data_stale(cached):
            return cached
        try:
            # Count blocks validated with efficient scanning
            blocks_validated = 0
            recent_blocks = 0
            total_blocks = max(1, len(self.blockchain) - 1)
            
            # Use efficient block scanning for large blockchains
            if len(self.blockchain) > 1000:
            	# For large chains, sample recent blocks and extrapolate
            	sample_size = min(1000, total_blocks)
            	sample_blocks = self.blockchain[-sample_size:]
            	sample_validated = sum(1 for block in sample_blocks if block.get('validator') == validator_address)
            	blocks_validated = int(sample_validated * (total_blocks / sample_size))
            else:
                # For small chains, scan all blocks
                blocks_validated = sum(1 for block in self.blockchain[1:] if block.get('validator') == validator_address)
            # Get detailed performance metrics
            uptime = self.consensus.get_validator_uptime(validator_address)
            block_times = self._get_validator_block_times(validator_address)
            recent_activity = self._get_recent_validator_activity(validator_address)
            performance_data = {
                'blocks_validated': blocks_validated,
                'validation_success_rate': round(
                    (blocks_validated / total_blocks * 100) if total_blocks > 0 else 0, 2
                ),
                'uptime_percentage': round(uptime or 0, 2),
                'average_block_time': round(
                    sum(block_times) / len(block_times) if block_times else 0, 3
                ),
                'block_time_std_dev': round(
                    statistics.stdev(block_times) if len(block_times) > 1 else 0, 3
                ),
                'missed_blocks': self.consensus.get_missed_blocks(validator_address),
                'last_missed_block': self.consensus.get_last_missed_block(validator_address),
                'consecutive_successes': self.consensus.get_consecutive_successes(validator_address),
                'recent_activity': recent_activity,
                'performance_score': self._calculate_performance_score(
                    blocks_validated, uptime, block_times
                ),
                'reliability_rating': self._get_reliability_rating(uptime or 0),
                'efficiency_rating': self._get_efficiency_rating(block_times)
            
            }
            # Cache the performance data
            self._cache_performance_data(cache_key, performance_data)
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Failed to calculate performance: {e}")
            return {
                'error': f'performance_calculation_failure: {str(e)}',
                'blocks_validated': 0,
                'validation_success_rate': 0,
                'uptime_percentage': 0
            }                                              
                                 
    def _get_validator_economics(self, validator_address: str) -> Dict:
    	"""Get comprehensive economic information with forecasting."""
    	try:
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		pending_rewards = self._calculate_pending_rewards(validator_address)
    		total_stake = self.consensus.get_total_stake()
    		
    		# Calculate advanced metrics
    		apr = self._calculate_apr(validator_address, total_rewards)
    		roi = self._calculate_roi(validator_address, total_rewards, staked_amount)
    		annual_earnings = self._estimate_annual_earnings(validator_address)
    		return {
    		    'staked_amount': staked_amount,
    		    'total_rewards_earned': total_rewards,
    		    'pending_rewards': pending_rewards,
    		    'unclaimed_rewards': self.consensus.get_unclaimed_rewards(validator_address),
    		    'reward_history': self._get_reward_history(validator_address),
    		    'apr': round(apr, 4),
    		    'apy': round(self._calculate_apy(apr), 4),
    		    'roi': round(roi, 4),
    		    'annual_earnings_estimate': annual_earnings,
    		    'fee_structure': self._get_fee_structure(),
    		    'commission_earnings': self._calculate_commission_earnings(validator_address),
    		    
    		    'delegator_rewards': self._calculate_delegator_rewards(validator_address),
    		    'economic_viability': self._assess_economic_viability(
                staked_amount, apr, total_stake
            ),
    		    'break_even_estimation': self._calculate_break_even(validator_address)
        
    		    
    		}
    	except Exception as e:
    		logger.error(f"Failed to get economics info: {e}")
    		return {
    		    'error': f'economics_calculation_failure: {str(e)}',
    		    'staked_amount': 0,
    		    'total_rewards_earned': 0,
    		    'apr': 0
    		}  
    
    def _assess_slashing_risk(self, validator_address: str) -> Dict:    	
    	                      
        try:
        	# Get comprehensive risk factors
        	risk_factors = self._collect_risk_factors(validator_address)
        	
        	# Calculate risk score using weighted factors
        	risk_score = self._calculate_risk_score(risk_factors)
        	risk_level = self._determine_risk_level(risk_score)
        	
        	# Get mitigation recommendations
        	recommendations = self._get_risk_recommendations(risk_level, risk_factors)
        	
        	# Calculate financial impact
        	financial_impact = self._calculate_slashing_impact(validator_address, risk_score)
        	return {
        	    'risk_level': risk_level,
        	    'risk_score': round(risk_score, 4),
        	    'risk_factors': risk_factors,
        	    'financial_impact': financial_impact,
        	    'recommendations': recommendations,
        	    'monitoring_required': risk_level in ['high', 'critical'],
        	    'last_risk_assessment': int(time.time()),
        	    'risk_trend': self._get_risk_trend(validator_address),
        	    'insurance_coverage': self._get_insurance_coverage(validator_address)
        	}
        except Exception as e:
            logger.error(f"Failed to assess slashing risk: {e}")
            return {
                
                'risk_level': 'unknown',
                'error': f'risk_assessment_failure: {str(e)}'
            }	        	                   

    def _validate_address_format(self, address: str) -> bool:
    	"""Validate RAYONIX address format with comprehensive checks"""
    	if not address or not isinstance(address, str):
    		return False
    		
    	# Check basic format requirements
    	if not address.startswith('ryx1') or len(address) < 42 or len(address) > 90:
    		return False
    		
    		# Bech32 validation with detailed error checking
    		try:
    			hrp, data = bech32.bech32_decode(address)
    			if hrp != 'ryx' or data is None:
    				return False
    				
    			# Additional format validation
    			if not bech32.bech32_verify_checksum(hrp, data):
    				return False
    				
    			# Check for valid witness version and program length
    			decoded = bech32.decode(hrp, address)
    			if decoded is None:
    				return False
    				
    			witness_version, witness_program = decoded
    			if witness_version not in [0, 1]:
    				return False
    				
    			# Validate program length based on witness version
    			if witness_version == 0:
    				if len(witness_program) not in [20, 32]:
    					return False
    			elif witness_version == 1:
    				if len(witness_program) != 32:
    					return False
    					
    			return True
    			
    		except (ValueError, TypeError, Exception) as e:
    			logger.debug(f"Address validation failed for {address}: {e}")
    			return False
    			
    def _create_error_response(self, error_code: str, message: str, metrics: Dict, extra: Dict = None) -> Dict:
    	"""Create standardized error response."""
    	error_data = {
    	    'code': error_code,
    	    'message': message,
    	    'timestamp': int(time.time()),
    	    'request_id': hashlib.sha256(f"{time.time()}{error_code}".encode()).hexdigest()[:16]
    	}
    	if extra:
    	    error_data.update(extra)
    	response = {
    	    'error': error_data,
    	    'system_info': {
    	        'current_height': len(self.blockchain) - 1,
    	        'blockchain_hash': self.blockchain[-1]['hash'] if self.blockchain else '0' * 64,
    	        'network_id': self.config['network_id'],
    	        'node_version': '1.0.0',
    	        'timestamp': int(time.time()),
    	        'uptime_seconds': int(time.time() - self.start_time) if hasattr(self, 'start_time') else 0
    	    },
    	    '_metrics': metrics,
    	    'metadata': {
    	        'version': '2.0',
    	        'format': 'standardized_error',
    	        'suggested_actions': self._get_error_suggestions(error_code)
    	    }
    	}
    	return response
    	    
    def _get_error_suggestions(self, error_code: str) -> List[str]:
    	"""Get suggested actions for different error codes"""
    	suggestions = {
    	    'invalid_address_format': [
    	        "Check the address format starts with 'ryx1'",
    	        "Verify the address length is between 42-90 characters",
    	        "Ensure the address uses valid Bech32 encoding"
    	    ],
    	    'address_resolution_failed': [
    	        "Provide a specific validator address",
    	        "Ensure your wallet is properly initialized",
    	        "Check if the node has validator registration"
    	    ],
    	    'validator_not_found': [
    	        "Verify the validator address is correct",
    	        "Check if the validator is registered in consensus",
    	        "Ensure the validator has active stake"
    	    ],
    	    'internal_error': [
    	        "Check node logs for detailed error information",
    	        "Restart the node if the issue persists",
    	        "Verify database integrity"
    	    ]
    	}
    	return suggestions.get(error_code, ["Check system logs for details"])   

    def _get_with_timeout(self, future, timeout: int, operation: str) -> Any:
    	"""Robust future result retrieval with comprehensive timeout handling"""
    	try:
    		start_time = time.time()
    		result = future.result(timeout=timeout)
    		elapsed = time.time() - start_time
    		
    		if elapsed > timeout * 0.8:
    			logger.warning(f"Operation {operation} completed in {elapsed:.3f}s (close to timeout {timeout}s)")
    			
    		return result
    		
    	except TimeoutError:
    		logger.warning(f"Timeout in {operation} operation after {timeout}s")
    		return {
    		    'error': f'timeout_in_{operation}',
    		    'timed_out': True,
    		    'timeout_seconds': timeout,
    		    'suggestion': 'Consider increasing timeout or optimizing the operation'
    		}
    	except Exception as e:
    		logger.error(f"Error in {operation} operation: {e}", exc_info=True)
    		return {
    		    'error': f'error_in_{operation}',
    		    'exception_type': type(e).__name__,
    		    'exception_message': str(e),
    		    'timestamp': int(time.time())
    		}    
    		
    def _blocks_to_days(self, blocks: int) -> float:
    	"""Convert blocks to days with precise calculation"""
    	block_time = self.config['block_time_target']
    	return (blocks * block_time) / 86400.0 
    	# precise seconds per day    			 	        	 	    	 	        	 	    
    def _determine_validator_status(self, info: Dict) -> str:
    	"""Comprehensive validator status determination"""
    	if not info.get('registered', False):
    		return 'unregistered'
    		
    	if not info.get('active', False):
    		# Check specific reasons for inactivity
    		if info.get('jailed', False):
    			return 'jailed'
    		elif info.get('tombstoned', False):
    			return 'tombstoned'
    		elif info.get('unbonding', False):
    			return 'unbonding'
    		else:
    			return 'inactive'
    	# Check additional active status conditions
    	if info.get('missed_too_many_blocks', False):
    		return 'warning'
    	if info.get('low_commission', False):
    		return 'competitive'
    	if info.get('top_performer', False):
    		return 'elite'
    	return 'active'  
    	
    def _calculate_performance_score(self, blocks_validated: int, uptime: float, block_times: List[float]) -> float:
    	"""Calculate comprehensive performance score with multiple factors"""
    	if not block_times or blocks_validated == 0:
    		return 0.0
    	# Weighted scoring components
    	weights = {
    	    'uptime': 0.4,
    	    'consistency': 0.3,
    	    'efficiency': 0.2,
    	    'reliability': 0.1
    	}
    	
    	# Uptime component (0-1)
    	uptime_score = uptime
    	
    	# Consistency component (block time standard deviation)
    	avg_block_time = statistics.mean(block_times)
    	std_dev = statistics.stdev(block_times) if len(block_times) > 1 else 0
    	target_time = self.config['block_time_target']
    	consistency_score = max(0, 1 - (std_dev / target_time))
    	
    	# Efficiency component (how close to target block time)
    	efficiency_ratio = avg_block_time / target_time
    	efficiency_score = 1.0 / efficiency_ratio if efficiency_ratio > 1 else efficiency_ratio
    	
    	# Reliability component (consecutive successful blocks)
    	max_consecutive = self.consensus.get_consecutive_successes(self.validator_address)
    	reliability_score = min(1.0, max_consecutive / 100.0)
    	
    	# Calculate weighted total score
    	total_score = (
    	    weights['uptime'] * uptime_score +
    	    weights['consistency'] * consistency_score +
    	    weights['efficiency'] * efficiency_score +
    	    weights['reliability'] * reliability_score
    	)
    	return round(total_score, 4)

    def _get_reliability_rating(self, uptime: float) -> str:
    	"""Get detailed reliability rating with tiered thresholds"""
    	rating_map = [
    	    (0.999, 'platinum', 1.0),
    	    (0.995, 'gold', 0.9),
    	    (0.99, 'silver', 0.8),
    	    (0.98, 'bronze', 0.7),
    	    (0.95, 'standard', 0.6),
    	    (0.90, 'basic', 0.5),
    	    (0.80, 'unreliable', 0.3),
    	    (0.00, 'poor', 0.0)
    	]
    	
    	for threshold, rating, _ in rating_map:
    		if uptime >= threshold:
    			return rating 
    	return 'poor'    
  
    def _get_efficiency_rating(self, block_times: List[float]) -> str:
    	"""Get efficiency rating based on block time performance"""
    	if not block_times:
    		return 'unknown'
    		
    	avg_time = statistics.mean(block_times)
    	target = self.config['block_time_target']
    	ratio = avg_time / target
    	
    	efficiency_map = [
    	    (0.80, 'exceptional'),
    	    (0.90, 'excellent'),
    	    (0.95, 'very_good'),
    	    (1.00, 'good'),
    	    (1.05, 'average'),
    	    (1.10, 'below_average'),
    	    (1.20, 'poor'),
    	    (float('inf'), 'inefficient')
    	]
    	for threshold, rating, _ in rating_map:
    		if uptime >= threshold:
    			return rating 
    	return 'inefficient'
            
    def _calculate_apy(self, apr: float) -> float:
    	"""Calculate APY from APR with compound interest"""
    	compounding_periods = 365
    	apy = ((1 + (apr / compounding_periods)) ** compounding_periods) - 1
    	return round(apy, 6)
    	
    def _calculate_commission_earnings(self, validator_address: str) -> int:
    	"""Calculate total commission earnings for validator"""
    	try:
    		# Get commission rate from consensus
    		commission_rate = self.consensus.get_commission_rate(validator_address)
    		
    		# Get total rewards generated by validator
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		# Calculate commission earnings
    		commission_earnings = int(total_rewards * commission_rate)
    		
    		# Adjust for any commission caps or limits
    		max_commission = self.consensus.get_max_commission(validator_address)
    		if max_commission is not None:
    			commission_earnings = min(commission_earnings, max_commission)
    			
    		return commission_earnings
    		
    	except Exception as e:
    		logger.error(f"Error calculating commission earnings: {e}")
    		return 0
    		 
    def _calculate_delegator_rewards(self, validator_address: str) -> int:
    	"""Calculate total rewards distributed to delegators"""
    	try:
    		total_rewards = self._calculate_validator_rewards(validator_address)
    		
    		# Delegator rewards are total rewards minus commission
    		delegator_rewards = total_rewards - commission_earnings
    		
    		# Ensure non-negative
    		return max(0, delegator_rewards)
    		
    	except Exception as e:
    		logger.error(f"Error calculating delegator rewards: {e}")
    		return 0 
    		
    def _assess_economic_viability(self, staked_amount: int, apr: float, total_stake: int) -> str:
    	"""Comprehensive economic viability assessment"""
    	if total_stake == 0:
    		return 'unknown'
    	stake_percentage = staked_amount / total_stake
    	annual_earnings = staked_amount * apr
    	
    	# Minimum viable earnings threshold (configurable)
    	min_viable_earnings = 1000  # 1000 RXY per year minimum
    	
    	viability_factors = []
    	
    	# Stake percentage factor
    	if stake_percentage < 0.0001:
    		viability_factors.append('very_low_stake')
    	elif stake_percentage < 0.001:
    		viability_factors.append('low_stake')
    		
    	# Earnings factor
    	if annual_earnings < min_viable_earnings:
    		viability_factors.append('low_earnings')
    		
    	# APR factor
    	if apr < 0.05:  # 5% minimum APR
    		viability_factors.append('low_apr')
    		
    	# Determine overall viability
    	if not viability_factors:
    		return 'highly_viable'
    	elif 'very_low_stake' in viability_factors:
    		return 'marginally_viable'
    	else:
    		return 'viable'
    		
    def _calculate_break_even(self, validator_address: str) -> Dict:
    	"""Calculate break-even analysis for validator operation"""
    	try:
    		# Get operational costs (could be configurable or from consensus)
    		operational_costs = {
    		    'server_costs': 200,  # monthly server costs in RXY
    		    'maintenance': 100,   # monthly maintenance
    		    'insurance': 50,      # monthly insurance
    		}
    		monthly_costs = sum(operational_costs.values())
    		annual_costs = monthly_costs * 12
    		
    		# Get expected annual earnings
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		apr = self._calculate_apr(validator_address, 0)  # 0 for current APR
    		annual_earnings = staked_amount * apr
    		# Calculate break-even
    		if annual_earnings <= annual_costs:
    			return {
    			    'status': 'not_profitable',
    			    'annual_costs': annual_costs,
    			    'annual_earnings': annual_earnings,
    			    'deficit': annual_costs - annual_earnings,
    			    'months_to_breakeven': float('inf')
    			}
    		monthly_earnings = annual_earnings / 12
    		monthly_profit = monthly_earnings - monthly_costs
    		
    		# Time to recover initial investment (simplified)
    		initial_investment = staked_amount
    		months_to_breakeven = initial_investment / monthly_profit
    		
    		return {
    		    'status': 'profitable',
    		    'annual_costs': annual_costs,
    		    'annual_earnings': annual_earnings,
    		    'monthly_profit': monthly_profit,
    		    'months_to_breakeven': round(months_to_breakeven, 1),
    		    'roi_percentage': (annual_earnings - annual_costs) / initial_investment * 100
    		}
    	except Exception as e:
    		logger.error(f"Error calculating break-even: {e}")
    		return {'status': 'calculation_error', 'error': str(e)}
    		
    def _collect_risk_factors(self, validator_address: str) -> Dict:
    	"""Collect comprehensive risk factors for validator"""
    	risk_factors = {}
    	
    	try:
    		# Uptime risk
    		uptime = self.consensus.get_validator_uptime(validator_address)
    		if uptime < 0.95:
    			risk_factors['low_uptime'] = {
    			    'severity': 'high' if uptime < 0.90 else 'medium',
    			    'value': uptime,
    			    'threshold': 0.95
    			}
    			
    		# Slashing history
    		slashing_events = self.consensus.get_slashing_history(validator_address)
    		if slashing_events:
    			risk_factors['slashing_history'] = {
    			    'severity': 'high',
    			    'count': len(slashing_events),
    			    'last_event': max(event['height'] for event in slashing_events)
    			}
    			
    		# Commission change risk
    		commission_history = self.consensus.get_commission_history(validator_address)
    		if commission_history and len(commission_history) > 3:
    			recent_changes = commission_history[-3:]
    			change_rate = statistics.stdev([chg['rate'] for chg in recent_changes])
    			if change_rate > 0.05:
    				risk_factors['volatile_commission'] = {
    				    'severity': 'medium',
    				    'change_rate': change_rate,
    				    'threshold': 0.05
    				}
    		# Delegation concentration risk
    		delegators = self.consensus.get_delegators(validator_address)
    		if delegators:
    			total_delegated = sum(d['amount'] for d in delegators)
    			if total_delegated > 0:
    				top_delegator = max(delegators, key=lambda x: x['amount'])
    				concentration = top_delegator['amount'] / total_delegated
    				if concentration > 0.3:
    					risk_factors['delegation_concentration'] = {
    					    'severity': 'medium',
    					    'concentration': concentration,
    					    'threshold': 0.3
    					    
    					}
    		# Network connectivity risk
    		connectivity = self.consensus.get_connectivity_metrics(validator_address)
    		if connectivity and connectivity.get('latency', 0) > 1000:
    			risk_factors['high_latency'] = {
    			    
    			    'severity': 'medium',
    			    'latency_ms': connectivity['latency'],
    			    'threshold': 1000
    			}
    	except Exception as e:
    		logger.error(f"Error collecting risk factors: {e}")
    		risk_factors['collection_error'] = {'error': str(e)}
    	return risk_factors
    	
    def _calculate_risk_score(self, risk_factors: Dict) -> float:
    	"""Calculate comprehensive risk score from risk factors"""
    	if not risk_factors:
    		return 0.1  # Low risk baseline
    	severity_weights = {
    	    'critical': 1.0,
    	    'high': 0.7,
    	    'medium': 0.4,
    	    'low': 0.2
    	}
    	
    	total_weight = 0
    	weighted_score = 0
    	
    	for factor_name, factor_data in risk_factors.items():
    		severity = factor_data.get('severity', 'medium')
    		weight = severity_weights.get(severity, 0.4)
    		
    		# Additional weight based on factor type
    		if 'slashing' in factor_name:
    			weight *= 1.5
    		elif 'uptime' in factor_name:
    			weight *= 1.2
    		weighted_score += weight
    		total_weight += weight
    		
    	if total_weight == 0:
    		return 0.1
    		
    	# Normalize to 0-1 scale
    	base_score = weighted_score / total_weight
    	
    	# Apply non-linear scaling to emphasize high risks
    	risk_score = min(1.0, base_score * 1.5)
    	return round(risk_score, 4)  

    def _determine_risk_level(self, risk_score: float) -> str:
    	"""Determine risk level with detailed thresholds"""
    	risk_levels = [
    	    (0.20, 'very_low'),
    	    (0.40, 'low'),
    	    (0.60, 'moderate'),
    	    (0.75, 'elevated'),
    	    (0.85, 'high'),
    	    (0.95, 'very_high'),
    	    (1.00, 'extreme')
    	]
    	for threshold, level in risk_levels:
    		if risk_score <= threshold:
    			return level
    			 
    	return 'extreme'    	  	    		         		  
    def _calculate_slashing_impact(self, validator_address: str, risk_score: float) -> Dict:
    	"""Calculate potential slashing impact"""
    	try:
    		staked_amount = self.consensus.get_stake_amount(validator_address)
    		delegated_stake = self.consensus.get_delegated_stake(validator_address)
    		total_stake = staked_amount + delegated_stake
    		
    		# Base slashing percentages from consensus parameters
    		slashing_params = self.consensus.get_slashing_parameters()
    		double_sign_penalty = slashing_params.get('double_sign_penalty', 0.05)  # 5%
    		downtime_penalty = slashing_params.get('downtime_penalty', 0.0001)     # 0.01%
    		
    		# Calculate potential losses
    		potential_losses = {
    		    'double_sign': total_stake * double_sign_penalty * risk_score,
    		    'downtime': total_stake * downtime_penalty * risk_score,
    		    'maximum_possible': total_stake * min(1.0, risk_score * 2)  # Cap at 100%
    		}
    		
    		# Calculate probability-adjusted expected loss
    		expected_loss = (
    		    potential_losses['double_sign'] * 0.1 +  # 10% probability of double sign
    		    potential_losses['downtime'] * 0.3       # 30% probability of downtime
    		)
    		return {
    		    'potential_losses': {k: int(v) for k, v in potential_losses.items()},
    		    'expected_annual_loss': int(expected_loss),
    		    'risk_adjusted_apr_impact': (expected_loss / total_stake) * 100 if total_stake > 0 else 0,
    		    'insurance_coverage_required': int(expected_loss * 1.1)  # 10% buffer
    		}
    		
    	except Exception as e:
    		logger.error(f"Error calculating slashing impact: {e}")
    		return {'error': str(e)}	  
    		
    def _get_risk_trend(self, validator_address: str) -> str:
    	"""Analyze risk trend over time"""
    	try:
    		# Get historical risk scores (last 30 days)
    		historical_scores = self.consensus.get_historical_risk_scores(validator_address, 30)
    		if not historical_scores or len(historical_scores) < 2:
    			return 'insufficient_data'
    		# Calculate trend using linear regression
    		x = list(range(len(historical_scores)))
    		y = historical_scores
    		
    		slope, intercept = statistics.linear_regression(x, y)
    		
    		# Determine trend direction and strength
    		if abs(slope) < 0.001:
    			return 'stable'
    		elif slope > 0.01:
    			return 'increasing_rapidly'
    		elif slope > 0.001:
    			return 'increasing'
    		elif slope < -0.01:
    			return 'decreasing_rapidly'
    		elif slope < -0.001:
    			return 'decreasing'
    		else:
    			return 'stable'
    			
    	except Exception as e:
    		logger.error(f"Error calculating risk trend: {e}")
    		return 'unknown'
    		
    def _get_insurance_coverage(self, validator_address: str) -> Dict:
    	"""Get insurance coverage information"""
    	try:
    		# Check if validator has active insurance
    		insurance_info = self.consensus.get_validator_insurance(validator_address)
    		if not insurance_info:
    			return {
    			    'covered': False,
    			    'recommended_coverage': self._calculate_slashing_impact(validator_address, 0.5)['insurance_coverage_required']
    			    
    			}
    		return {
    		    'covered': True,
    		    'provider': insurance_info.get('provider', 'unknown'),
    		    'coverage_amount': insurance_info.get('coverage_amount', 0),
    		    'premium': insurance_info.get('premium', 0),
    		    'expiration_date': insurance_info.get('expiration_date'),
    		    'deductible': insurance_info.get('deductible', 0),
    		    'coverage_percentage': min(100, (insurance_info.get('coverage_amount', 0) / 
                self.consensus.get_stake_amount(validator_address)) * 100) if self.consensus.get_stake_amount(validator_address) > 0 else 0
    		}
    		
    	except Exception as e:
    		logger.error(f"Error getting insurance coverage: {e}")
    		return {'covered': False, 'error': str(e)}    		
 
    def _get_validation_schedule(self, validator_address: str) -> Dict:
    	"""Get detailed validation schedule"""
    	try:
    		schedule = self.consensus.get_validation_schedule(validator_address)
    		if not schedule:
    			# Calculate estimated schedule based on stake and network parameters
    			total_stake = self.consensus.get_total_stake()
    			validator_stake = self.consensus.get_stake_amount(validator_address)
    			
    			if total_stake > 0 and validator_stake > 0:
    				stake_percentage = validator_stake / total_stake
    				expected_blocks_per_day = (86400 / self.config['block_time_target']) * stake_percentage
    				
    				schedule = {
    				    'estimated_blocks_per_day': round(expected_blocks_per_day, 2),
    				    'estimated_blocks_per_week': round(expected_blocks_per_day * 7, 2),
    				    'validation_frequency': 'variable_based_on_stake',
    				    'next_expected_block': 'unknown'
    				}
    				
    		return schedule or {'status': 'no_schedule_available'}
    		
    	except Exception as e:
    		logger.error(f"Error getting validation schedule: {e}")
    		return {'error': str(e)}
    		
    def _get_network_validator_status(self) -> Dict:
    	"""Get comprehensive network validator status"""
    	try:
    		network_status = self.consensus.get_network_status()
    		if not network_status:
    			# Fallback calculation
    			active_validators = len(self.consensus.get_active_validators())
    			total_stake = self.consensus.get_total_stake()
    			avg_uptime = statistics.mean([
    			    self.consensus.get_validator_uptime(v) 
                for v in self.consensus.get_active_validators()[:10]  # Sample first
    			]) if active_validators > 0 else 0
    			
    			network_status = {
    			    'active_validators': active_validators,
    			    'total_stake': total_stake,
    			    'average_uptime': round(avg_uptime, 4),
    			    'network_health': 'good' if avg_uptime > 0.95 else 'degraded',
    			    'block_time_variance': self._calculate_network_block_time_variance(),
    			    'governance_participation': self.consensus.get_governance_participation_rate()
    			}
    			
    		return network_status
    		
    	except Exception as e:
    	    logger.error(f"Error getting network status: {e}")
    	    return {
    	        'active_validators': 'unknown',
    	        'network_health': 'unknown',
    	        'error': str(e)
    	    }		
  
    def _calculate_network_block_time_variance(self) -> float:
    	"""Calculate network-wide block time variance"""
    	try:
    		# Sample recent blocks for variance calculation
    		recent_blocks = self.blockchain[-100:]  # Last 100 blocks
    		if len(recent_blocks) < 2:
    			return 0.0
    			
    		block_times = []
    		for i in range(1, len(recent_blocks)):
    			time_diff = recent_blocks[i]['timestamp'] - recent_blocks[i-1]['timestamp']
    			block_times.append(time_diff)
    			
    		if len(block_times) < 2:
    			return 0.0
    			
    		variance = statistics.variance(block_times)
    		return round(variance, 4)
    	except Exception as e:
    	    logger.error(f"Error calculating block time variance: {e}")
    	    return 0.0
    	    
    def _get_cached_performance(self, cache_key: str) -> Optional[Dict]:
        """Get cached performance data with validation"""
        try:
        	cached_data = self.database.get(cache_key)
        	if not cached_data or not isinstance(cached_data, dict):
        		return None
        		
        	# Validate cache structure
        	required_fields = ['data', 'timestamp', 'expires_at']
        	if not all(field in cached_data for field in required_fields):
        		return None
        		
        	# Check if cache is expired
        	if time.time() > cached_data['expires_at']:
        		return None
        		
        	# Validate data integrity
        	if 'signature' in cached_data:
        		# Verify cryptographic signature if present
        		data_hash = hashlib.sha256(
        		    json.dumps(cached_data['data'], sort_keys=True).encode()
        		).hexdigest()
        		
        		if not self._verify_cache_signature(data_hash, cached_data['signature']):
        			logger.warning(f"Cache signature verification failed for {cache_key}")
        			return None
        			
        	return cached_data['data']
        	
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
            return None
            		
    def _verify_cache_signature(self, data_hash: str, signature: str) -> bool:
        """Verify cache data signature"""
        # Implementation would depend on your signing mechanism
        # For now, return True as placeholder
        return True
        
    def _is_performance_data_stale(self, cached_data: Dict) -> bool:
        """Check if performance data is stale with multiple criteria"""
        if not isinstance(cached_data, dict):  
            return True
            
        current_time = time.time()
        cached_time = cached_data.get('timestamp', 0)
        
        # Absolute time-based staleness
        if current_time - cached_time > 300:
        	return True
        	
        # Blockchain progress-based staleness
        cached_height = cached_data.get('block_height', 0)
        current_height = len(self.blockchain) - 1
        if current_height - cached_height > 10:
        	# More than 10 blocks since cache
        	return True
        	
        # Network state-based staleness
        if cached_data.get('network_hash') != self._calculate_network_state_hash():
        	return True
        return False
        
    def _calculate_network_state_hash(self) -> str:
      """Calculate hash of current network state for cache validation"""
      network_state = {
          'height': len(self.blockchain) - 1,
          'total_stake': self.consensus.get_total_stake(),
          'active_validators': len(self.consensus.get_active_validators()),
          'timestamp': int(time.time())
      }
      return hashlib.sha256(json.dumps(network_state, sort_keys=True).encode()).hexdigest()
      
    def _cache_performance_data(self, cache_key: str, data: Dict):
        """Cache performance data with comprehensive metadata"""
        try:
        	cache_data = {
        	    'data': data,
        	    'timestamp': time.time(),
        	    'expires_at': time.time() + 300,  # 5 minutes
        	    'block_height': len(self.blockchain) - 1,
        	    'network_hash': self._calculate_network_state_hash(),
        	    'node_id': getattr(self, 'node_id', 'unknown'),
        	    'version': '1.0'
        	}
        	
        	# Add cryptographic signature for data integrity
        	data_hash = hashlib.sha256(
        	    json.dumps(data, sort_keys=True).encode()
        	).hexdigest()
        	cache_data['signature'] = self._sign_cache_data(data_hash)
        	
        	self.database.put(cache_key, cache_data)
        	
        except Exception as e:
        	logger.error(f"Failed to cache performance data: {e}")
        	            
    def _sign_cache_data(self, data_hash: str) -> str:
        """Sign cache data for integrity verification"""
        # Implementation would depend on your signing mechanism
        # Return placeholder for now
        return f"signed_{data_hash[:16]}"            
      
    def _sign_validator_info(self, data: Dict) -> Dict:
    	"""Sign validator information for data integrity."""
    	try:
    		# Create hash of the data
    		data_hash = hashlib.sha256(
    		    json.dumps(data, sort_keys=True).encode()
    		).hexdigest()
    		
    		# Sign with node key if available
    		signature = None
    		if hasattr(self, 'node_key'):
    			signature = self.node_key.sign(data_hash.encode()).hex()
    			return {
    			    'data_hash': data_hash,
    			    'signature': signature,
    			    'signing_node': getattr(self, 'node_id', 'unknown'),
    			    'timestamp': int(time.time())
    			}
    	except Exception as e:
    		logger.warning(f"Failed to sign validator info: {e}")
    		return {'error': f'signing_failed: {str(e)}'}
    		        
    def _cache_validator_info(self, address: str, data: Dict):
    	"""Cache validator information with expiration."""
    	try:
    		cache_data = {
    		    'data': data,
    		    'timestamp': time.time(),
    		    'expires_at': time.time() + 300,
    		    'height': len(self.blockchain) - 1
    		}
    		cache_key = f"validator_info_{address}"
    		self.database.put(cache_key, cache_data)
    	except Exception as e:
    		logger.warning(f"Failed to cache validator info: {e}")
  
    		
    def _get_data_freshness(self) -> str:
    	"""Get data freshness indicator."""
    	height = len(self.blockchain) - 1
    	if height < 10:
    		return 'very_fresh'
    	elif height < 100:
    		return 'fresh'
    	elif height < 1000:
    		return 'recent'
    	else:
    		return 'established'
        
    # Add all the helper methods: _calculate_apr, _assess_economic_viability, etc.        

# Utility functions
def create_rayonix_network(network_type: str = "mainnet") -> RayonixCoin:
    """Create RAYONIX network instance"""
    return RayonixCoin(network_type)

def validate_rayonix_address(address: str) -> bool:
    """Validate RAYONIX address format"""
    if not address or not isinstance(address, str):
        return False
    
    if not address.startswith('ryx1') or len(address) < 42 or len(address) > 90:
        return False
    
    try:
        hrp, data = bech32.bech32_decode(address)
        return hrp == 'ryx' and data is not None
    except:
        return False

def calculate_mining_reward(height: int, base_reward: int = 50, halving_interval: int = 210000) -> int:
    """Calculate mining reward at given height"""
    halvings = height // halving_interval
    reward = base_reward >> halvings
    return max(reward, 1)