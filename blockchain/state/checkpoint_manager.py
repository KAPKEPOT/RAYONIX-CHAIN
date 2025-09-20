# blockchain/state/checkpoint_manager.py
import time
import threading
import logging
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from blockchain.models.block import Block

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Manages blockchain checkpoints for fast synchronization and recovery"""
    
    def __init__(self, database: Any, state_manager: Any):
        self.database = database
        self.state_manager = state_manager
        self.lock = threading.RLock()
        self.checkpoint_interval: int = 1000  # blocks
        self.max_checkpoints: int = 10
        self.last_checkpoint_height: int = -1
    
    def should_create_checkpoint(self, block: Block) -> bool:
        """Check if a checkpoint should be created for this block"""
        if block.header.height == 0:  # Always checkpoint genesis
            return True
        
        if block.header.height % self.checkpoint_interval == 0:
            return True
        
        # Check if we've had significant state changes
        if self._has_significant_state_changes():
            return True
        
        return False
    
    def create_checkpoint_if_needed(self, block: Block) -> Optional[str]:
        """Create a checkpoint if needed for the given block"""
        if not self.should_create_checkpoint(block):
            return None
        
        checkpoint_name = f"height_{block.header.height}_{int(time.time())}"
        checkpoint_id = self.state_manager.create_checkpoint(checkpoint_name)
        
        # Store checkpoint metadata
        checkpoint_meta = {
            'height': block.header.height,
            'hash': block.hash,
            'timestamp': time.time(),
            'name': checkpoint_name,
            'state_size': self.state_manager.get_state_stats()['state_size_bytes']
        }
        
        self.database.put(f"checkpoint_meta_{checkpoint_name}", checkpoint_meta)
        self.last_checkpoint_height = block.header.height
        
        logger.info(f"Created checkpoint at height {block.header.height}: {checkpoint_name}")
        return checkpoint_name
    
    def _has_significant_state_changes(self) -> bool:
        """Check if there have been significant state changes since last checkpoint"""
        # This could be implemented based on various metrics:
        # - Number of UTXO changes
        # - Number of contract executions
        # - Validator set changes
        # - State size growth
        
        # For now, use a simple time-based approach
        if self.last_checkpoint_height == -1:
            return True
        
        # Check if we've processed many blocks since last checkpoint
        current_height = self.state_manager.get_current_height()
        if current_height - self.last_checkpoint_height > self.checkpoint_interval * 2:
            return True
        
        return False
    
    def get_best_checkpoint(self, target_height: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get the best checkpoint for the given target height"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        if target_height is None:
            # Return the most recent checkpoint
            return max(checkpoints, key=lambda x: x['height'])
        
        # Find the checkpoint closest to but not exceeding the target height
        suitable_checkpoints = [c for c in checkpoints if c['height'] <= target_height]
        
        if not suitable_checkpoints:
            return None
        
        return max(suitable_checkpoints, key=lambda x: x['height'])
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints"""
        checkpoints = []
        
        for key in self.database.keys():
            if key.startswith('checkpoint_meta_'):
                meta = self.database.get(key)
                if meta:
                    checkpoints.append(meta)
        
        return checkpoints
    
    def restore_from_checkpoint(self, checkpoint_name: str) -> bool:
        """Restore state from a specific checkpoint"""
        return self.state_manager.restore_checkpoint(checkpoint_name)
    
    def cleanup_old_checkpoints(self):
        """Clean up old checkpoints, keeping only the most recent ones"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by height (most recent first)
        checkpoints.sort(key=lambda x: x['height'], reverse=True)
        
        # Keep only the most recent checkpoints
        checkpoints_to_keep = checkpoints[:self.max_checkpoints]
        checkpoints_to_remove = checkpoints[self.max_checkpoints:]
        
        for checkpoint in checkpoints_to_remove:
            checkpoint_name = checkpoint['name']
            
            # Remove checkpoint data
            self.database.delete(f"checkpoint_{checkpoint_name}")
            self.database.delete(f"checkpoint_meta_{checkpoint_name}")
            
            logger.info(f"Removed old checkpoint: {checkpoint_name}")
    
    def verify_checkpoint_integrity(self, checkpoint_name: str) -> bool:
        """Verify the integrity of a checkpoint"""
        try:
            # Restore checkpoint temporarily
            original_state = {
                'utxo_set': self.state_manager.utxo_set.snapshot(),
                'consensus': self.state_manager.consensus.snapshot(),
                'contracts': self.state_manager.contract_manager.snapshot()
            }
            
            # Restore checkpoint
            success = self.restore_from_checkpoint(checkpoint_name)
            if not success:
                return False
            
            # Verify state integrity
            integrity_ok = self.state_manager.verify_state_integrity()
            
            # Restore original state
            self.state_manager.utxo_set.restore(original_state['utxo_set'])
            self.state_manager.consensus.restore(original_state['consensus'])
            self.state_manager.contract_manager.restore(original_state['contracts'])
            
            return integrity_ok
            
        except Exception as e:
            logger.error(f"Checkpoint integrity check failed: {e}")
            return False
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints"""
        checkpoints = self.list_checkpoints()
        
        return {
            'total_checkpoints': len(checkpoints),
            'latest_checkpoint_height': max(c['height'] for c in checkpoints) if checkpoints else -1,
            'earliest_checkpoint_height': min(c['height'] for c in checkpoints) if checkpoints else -1,
            'average_checkpoint_interval': self._calculate_average_interval(checkpoints),
            'total_state_size': sum(c.get('state_size', 0) for c in checkpoints),
            'checkpoints': checkpoints
        }
    
    def _calculate_average_interval(self, checkpoints: List[Dict[str, Any]]) -> float:
        """Calculate average checkpoint interval"""
        if len(checkpoints) < 2:
            return 0.0
        
        heights = sorted(c['height'] for c in checkpoints)
        intervals = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        
        return sum(intervals) / len(intervals) if intervals else 0.0