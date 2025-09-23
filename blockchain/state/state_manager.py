# blockchain/state/state_manager.py
import os
import hashlib
import json
import time
import threading
from typing import Dict, List, Any, Optional, Deque
from dataclasses import asdict
from collections import deque
from contextlib import contextmanager
import logging

from blockchain.models.block import Block
from utxo_system.database.utxo import UTXOSet
from rules.core.consensus import ConsensusEngine
from smart_contract.core.contract_manager import ContractManager

logger = logging.getLogger(__name__)

class StateManager:
    """Manages blockchain state transitions atomically"""
    
    def __init__(self, database: Any, utxo_set: UTXOSet, consensus: ProofOfStake, 
                 contract_manager: ContractManager):
        self.database = database
        self.utxo_set = utxo_set
        self.consensus = consensus
        self.contract_manager = contract_manager
        self.lock = threading.RLock()
        self.state_transition_log: Deque = deque(maxlen=10000)
        self.last_checkpoint: Optional[str] = None
        self.state_checksum: Optional[str] = None
        self.last_persist_time: float = 0
        self.persist_interval: int = 60  # seconds
        
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
                'timestamp': time.time(),
                'state_checksum': self.state_checksum
            }
            self.state_transition_log.append((transaction_id, 'start', snapshot))
        return transaction_id
    
    def _commit_transaction(self, transaction_id: str):
        """Commit a state transaction"""
        with self.lock:
            self.state_transition_log.append((transaction_id, 'commit', None))
            # Persist state to database if needed
            current_time = time.time()
            if current_time - self.last_persist_time >= self.persist_interval:
                self._persist_state()
                self.last_persist_time = current_time
    
    def _rollback_transaction(self, transaction_id: str):
        """Rollback a state transaction"""
        with self.lock:
            # Find the transaction start and restore state
            snapshot = None
            for i, (tid, action, snap) in enumerate(reversed(self.state_transition_log)):
                if tid == transaction_id and action == 'start':
                    snapshot = snap
                    break
            
            if snapshot:
                self.utxo_set.restore(snapshot['utxo_set'])
                self.consensus.restore(snapshot['consensus_state'])
                self.contract_manager.restore(snapshot['contract_states'])
                self.state_checksum = snapshot['state_checksum']
            
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
            
            # Update state checksum
            self.state_checksum = self._calculate_state_hash()
            
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
            
            # Update state checksum
            self.state_checksum = self._calculate_state_hash()
            
            return True
    
    def _persist_state(self):
        """Persist current state to database"""
        try:
            # Save UTXO set
            utxo_state = self.utxo_set.to_bytes()
            self.database.put('utxo_set_state', utxo_state)
            
            # Save consensus state
            consensus_state = self.consensus.to_bytes()
            self.database.put('consensus_state', consensus_state)
            
            # Save contract states
            contract_states = self.contract_manager.to_bytes()
            self.database.put('contract_states', contract_states)
            
            # Update state checksum
            self.state_checksum = self._calculate_state_hash()
            self.database.put('state_checksum', self.state_checksum)
            
            # Save state transition log (for debugging)
            log_data = list(self.state_transition_log)
            self.database.put('state_transition_log', log_data)
            
            logger.info("State persisted successfully")
            
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
    
    def create_checkpoint(self, name: Optional[str] = None) -> str:
        """Create a state checkpoint"""
        checkpoint_id = name or f"checkpoint_{int(time.time())}"
        
        checkpoint_data = {
            'utxo_set': self.utxo_set.snapshot(),
            'consensus': self.consensus.snapshot(),
            'contracts': self.contract_manager.snapshot(),
            'timestamp': time.time(),
            'block_height': self.get_current_height(),
            'state_checksum': self.state_checksum
        }
        
        self.database.put(f"checkpoint_{checkpoint_id}", checkpoint_data)
        self.last_checkpoint = checkpoint_id
        
        logger.info(f"Checkpoint created: {checkpoint_id}")
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint"""
        try:
            checkpoint_data = self.database.get(f"checkpoint_{checkpoint_id}")
            if not checkpoint_data:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return False
            
            self.utxo_set.restore(checkpoint_data['utxo_set'])
            self.consensus.restore(checkpoint_data['consensus'])
            self.contract_manager.restore(checkpoint_data['contracts'])
            self.state_checksum = checkpoint_data['state_checksum']
            
            logger.info(f"Restored state from checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore checkpoint: {e}")
            return False
    
    def get_current_height(self) -> int:
        """Get current blockchain height"""
        # This would be implemented based on your specific database structure
        return self.database.get('current_height', 0)
    
    def verify_state_integrity(self) -> bool:
        """Verify the integrity of the current state"""
        current_checksum = self._calculate_state_hash()
        stored_checksum = self.database.get('state_checksum')
        
        if stored_checksum and current_checksum != stored_checksum:
            logger.error(f"State integrity check failed: {current_checksum} != {stored_checksum}")
            return False
        
        return True
    
    def get_state_stats(self) -> Dict[str, Any]:
        """Get statistics about the current state"""
        return {
            'utxo_count': self.utxo_set.get_utxo_count(),
            'validators_count': self.consensus.get_validator_count(),
            'contracts_count': self.contract_manager.get_contract_count(),
            'state_size_bytes': self._estimate_state_size(),
            'last_checkpoint': self.last_checkpoint,
            'state_checksum': self.state_checksum,
            'transaction_count': len(self.state_transition_log)
        }
    
    def _estimate_state_size(self) -> int:
        """Estimate the size of the current state in bytes"""
        # This is a simplified estimation
        utxo_size = self.utxo_set.get_utxo_count() * 200  # Approx 200 bytes per UTXO
        consensus_size = self.consensus.get_validator_count() * 500  # Approx 500 bytes per validator
        contract_size = self.contract_manager.get_contract_count() * 1000  # Approx 1KB per contract
        
        return utxo_size + consensus_size + contract_size
    
    def cleanup_old_checkpoints(self, keep_last: int = 10):
        """Clean up old checkpoints, keeping only the most recent ones"""
        try:
            checkpoints = []
            for key in self.database.keys():
                if key.startswith('checkpoint_'):
                    checkpoints.append(key)
            
            # Sort by timestamp (assuming checkpoint names include timestamp)
            checkpoints.sort(reverse=True)
            
            # Remove old checkpoints
            for checkpoint in checkpoints[keep_last:]:
                self.database.delete(checkpoint)
                logger.info(f"Removed old checkpoint: {checkpoint}")
                
        except Exception as e:
            logger.error(f"Failed to clean up checkpoints: {e}")