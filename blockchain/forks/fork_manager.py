# blockchain/forks/fork_manager.py
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional, Deque
from collections import deque

from blockchain.models.block import Block
from blockchain.models.fork import ForkResolution
from blockchain.validation.validation_manager import ValidationManager

logger = logging.getLogger(__name__)

class ForkManager:
    """Manages blockchain forks and reconciliation"""
    
    def __init__(self, state_manager: Any, validation_manager: ValidationManager, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.validation_manager = validation_manager
        self.config = config
        self.fork_history: Deque = deque(maxlen=100)
        self.reorganization_count = 0
        self.last_reorganization = 0
        self.fork_detection_threshold = config.get('fork_detection_threshold', 6)
        self.reorganization_depth_limit = config.get('reorganization_depth_limit', 100)
        self.fork_monitoring_interval = config.get('fork_monitoring_interval', 30)
    
    async def handle_possible_fork(self, new_block: Block) -> Optional[ForkResolution]:
        """Handle a potential fork caused by a new block"""
        current_head = self.state_manager.database.get('chain_head')
        if not current_head:
            logger.error("No current chain head found")
            return None
        
        # Check if this block causes a fork
        if new_block.header.previous_hash != current_head.hash:
            logger.warning(f"Potential fork detected at height {new_block.header.height}")
            return await self._resolve_fork(new_block, current_head)
        
        return None
    
    async def _resolve_fork(self, new_block: Block, current_head: Block) -> Optional[ForkResolution]:
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
            
            # Check reorganization depth limit
            if len(old_chain) > self.reorganization_depth_limit or len(new_chain) > self.reorganization_depth_limit:
                logger.warning(f"Reorganization depth exceeds limit: {len(old_chain)} > {self.reorganization_depth_limit}")
                return None
            
            # Calculate chainwork for both chains
            old_chainwork = sum(block.chainwork for block in old_chain)
            new_chainwork = sum(block.chainwork for block in new_chain)
            
            # Decide which chain to keep
            if new_chainwork > old_chainwork:
                # New chain has more work, reorganize
                resolution = await self._reorganize_to_new_chain(old_chain, new_chain, common_ancestor)
                self.reorganization_count += 1
                self.last_reorganization = time.time()
                
                # Add to fork history
                self.fork_history.append(resolution)
                
                return resolution
            else:
                # Old chain has more or equal work, keep it
                logger.info("Keeping existing chain (equal or more chainwork)")
                return ForkResolution(
                    common_ancestor_height=common_ancestor.height,
                    common_ancestor_hash=common_ancestor.hash,
                    old_chain_length=len(old_chain),
                    new_chain_length=len(new_chain),
                    chainwork_difference=old_chainwork - new_chainwork,
                    resolution_time=time.time() - start_time,
                    blocks_rolled_back=0,
                    blocks_applied=0,
                    resolution_strategy="chainwork"
                )
                
        except Exception as e:
            logger.error(f"Fork resolution failed: {e}", exc_info=True)
            return None
    
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
        elif block2.header.height > block1.header.height:
            walk_block = block2
            for _ in range(height_diff):
                walk_block = self.state_manager.database.get_block(walk_block.header.previous_hash)
                if not walk_block:
                    return None
        else:
            walk_block = block2
        
        # Now both blocks are at same height, walk back until we find common hash
        block_a = block1
        block_b = walk_block
        
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
            if not current_block:
                break
        
        segment.reverse()  # Return from ancestor to tip
        return segment
    
    async def _reorganize_to_new_chain(self, old_chain: List[Block], new_chain: List[Block], 
                                     common_ancestor: Block) -> ForkResolution:
        """Reorganize to new chain by rolling back old blocks and applying new ones"""
        rolled_back_blocks = 0
        applied_blocks = 0
        rolled_back_hashes = []
        applied_hashes = []
        
        try:
            # Roll back old chain blocks
            for block in reversed(old_chain):
                if block.header.height > common_ancestor.height:
                    if not self.state_manager.revert_block(block):
                        raise ValueError(f"Failed to revert block {block.hash}")
                    rolled_back_blocks += 1
                    rolled_back_hashes.append(block.hash)
            
            # Apply new chain blocks
            for block in new_chain:
                if block.header.height > common_ancestor.height:
                    validation_result = self.validation_manager.validate_block(block, ValidationLevel.CONSENSUS)
                    if not validation_result.is_valid:
                        raise ValueError(f"New chain block validation failed: {validation_result.errors}")
                    
                    if not self.state_manager.apply_block(block):
                        raise ValueError(f"Failed to apply block {block.hash}")
                    applied_blocks += 1
                    applied_hashes.append(block.hash)
            
            resolution = ForkResolution(
                common_ancestor_height=common_ancestor.height,
                common_ancestor_hash=common_ancestor.hash,
                old_chain_length=len(old_chain),
                new_chain_length=len(new_chain),
                chainwork_difference=sum(b.chainwork for b in new_chain) - sum(b.chainwork for b in old_chain),
                resolution_time=time.time() - start_time,
                blocks_rolled_back=rolled_back_blocks,
                blocks_applied=applied_blocks,
                rolled_back_hashes=rolled_back_hashes,
                applied_hashes=applied_hashes,
                resolution_strategy="chainwork"
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
            'recommended_actions': self._get_fork_prevention_actions(),
            'fork_history_size': len(self.fork_history)
        }
        return fork_risk
    
    def _calculate_fork_probability(self) -> float:
        """Calculate probability of fork based on network conditions"""
        # This would use statistical models based on:
        # - Network latency
        # - Validator distribution
        # - Block time variance
        # - Historical fork data
        
        base_probability = 0.01  # Base 1% probability
        
        # Increase probability based on recent reorganizations
        if self.reorganization_count > 0:
            base_probability += min(0.1, self.reorganization_count * 0.02)
        
        # Adjust based on time since last reorganization
        time_since_reorg = time.time() - self.last_reorganization
        if time_since_reorg < 3600:  # Less than 1 hour
            base_probability += 0.05
        elif time_since_reorg < 86400:  # Less than 1 day
            base_probability += 0.02
        
        return min(0.5, base_probability)  # Cap at 50%
    
    def _assess_network_health(self) -> str:
        """Assess overall network health regarding forks"""
        if self.reorganization_count > self.config.get('max_reorganizations_per_hour', 3):
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
        
        if time.time() - self.last_reorganization < 3600:
            actions.append("Check network stability")
            actions.append("Verify validator synchronization")
        
        return actions
    
    def get_fork_history(self, limit: int = 10) -> List[ForkResolution]:
        """Get recent fork resolution history"""
        return list(self.fork_history)[-limit:]
    
    def get_reorganization_stats(self) -> Dict[str, Any]:
        """Get statistics about reorganizations"""
        if not self.fork_history:
            return {
                'total_reorganizations': 0,
                'average_resolution_time': 0,
                'max_blocks_reorganized': 0,
                'success_rate': 1.0
            }
        
        successful_reorgs = [r for r in self.fork_history if r.was_successful()]
        resolution_times = [r.resolution_time for r in successful_reorgs]
        block_counts = [r.blocks_rolled_back + r.blocks_applied for r in successful_reorgs]
        
        return {
            'total_reorganizations': len(self.fork_history),
            'successful_reorganizations': len(successful_reorgs),
            'average_resolution_time': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            'max_resolution_time': max(resolution_times) if resolution_times else 0,
            'max_blocks_reorganized': max(block_counts) if block_counts else 0,
            'success_rate': len(successful_reorgs) / len(self.fork_history) if self.fork_history else 1.0
        }
    
    async def emergency_rollback(self, target_height: int) -> bool:
        """Emergency rollback to specific height"""
        try:
            current_height = self.state_manager.get_current_height()
            if target_height >= current_height:
                logger.warning("Target height must be less than current height")
                return False
            
            # Find block at target height
            target_block = self.state_manager.database.get_block_by_height(target_height)
            if not target_block:
                logger.error(f"Block not found at height {target_height}")
                return False
            
            # Roll back blocks from current height to target height
            blocks_to_rollback = []
            current_block = self.state_manager.database.get('chain_head')
            
            while current_block and current_block.header.height > target_height:
                blocks_to_rollback.append(current_block)
                current_block = self.state_manager.database.get_block(current_block.header.previous_hash)
            
            # Perform rollback
            for block in reversed(blocks_to_rollback):
                if not self.state_manager.revert_block(block):
                    raise ValueError(f"Failed to revert block {block.hash}")
            
            logger.warning(f"Emergency rollback to height {target_height} completed")
            return True
            
        except Exception as e:
            logger.critical(f"Emergency rollback failed: {e}")
            return False