# blockchain/state/checkpoint_manager.py
import time
import threading
import logging
import hashlib
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import pickle
import zlib

from blockchain.models.block import Block

logger = logging.getLogger(__name__)

class CheckpointState(Enum):
    """Checkpoint state enumeration"""
    CREATED = "created"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"
    PENDING_DELETION = "pending_deletion"

@dataclass
class CheckpointMetadata:
    """Checkpoint metadata container"""
    height: int
    hash: str
    timestamp: float
    name: str
    state_size: int
    state_hash: str
    block_count_since_previous: int = 0
    state_change_count: int = 0
    status: CheckpointState = CheckpointState.CREATED
    version: str = "1.0"

class CheckpointCreationStrategy:
    """Strategy pattern for checkpoint creation criteria"""
    
    def __init__(self, 
                 interval_blocks: int = 1000,
                 max_blocks_without_checkpoint: int = 2000,
                 min_state_changes: int = 50000,
                 time_interval_seconds: int = 3600):
        self.interval_blocks = interval_blocks
        self.max_blocks_without_checkpoint = max_blocks_without_checkpoint
        self.min_state_changes = min_state_changes
        self.time_interval_seconds = time_interval_seconds

class CheckpointIntegrityError(Exception):
    """Checkpoint integrity verification failed"""
    pass

class CheckpointManager:
    """Manages blockchain checkpoints for fast synchronization and recovery"""
    
    def __init__(self, database: Any, state_manager: Any, storage_path: Optional[str] = None):
        self.database = database
        self.state_manager = state_manager
        self.lock = threading.RLock()
        self.storage_path = Path(storage_path) if storage_path else Path("checkpoints")
        self.storage_path.mkdir(exist_ok=True)
        
        # Configuration
        self.creation_strategy = CheckpointCreationStrategy()
        self.max_checkpoints: int = 10
        self.min_checkpoints: int = 3
        self.auto_cleanup: bool = True
        self.verify_on_create: bool = True
        self.compression_enabled: bool = True
        
        # State tracking
        self.last_checkpoint_height: int = -1
        self.last_checkpoint_time: float = 0
        self.blocks_since_last_checkpoint: int = 0
        self.state_changes_since_last_checkpoint: int = 0
        
        # Performance metrics
        self.metrics = {
            'checkpoints_created': 0,
            'checkpoints_restored': 0,
            'checkpoints_deleted': 0,
            'failed_restorations': 0,
            'total_creation_time': 0.0,
            'total_restoration_time': 0.0
        }
        
        logger.info("CheckpointManager initialized")

    def should_create_checkpoint(self, block: Block, state_change_count: int = 0) -> bool:
        """Check if a checkpoint should be created for this block using multiple criteria"""
        current_height = block.header.height
        current_time = time.time()
        
        # Always checkpoint genesis block
        if current_height == 0:
            return True
        
        # Interval-based checkpointing
        if current_height % self.creation_strategy.interval_blocks == 0:
            return True
        
        # Maximum blocks without checkpoint
        if self.blocks_since_last_checkpoint >= self.creation_strategy.max_blocks_without_checkpoint:
            return True
        
        # Significant state changes
        if state_change_count >= self.creation_strategy.min_state_changes:
            return True
        
        # Time-based checkpointing
        if (self.last_checkpoint_time > 0 and 
            current_time - self.last_checkpoint_time >= self.creation_strategy.time_interval_seconds):
            return True
        
        # Emergency checkpoint if state is growing too large
        if self._is_state_growing_excessively():
            return True
            
        return False

    def create_checkpoint(self, block: Block, state_change_count: int = 0) -> Optional[str]:
        """Create a checkpoint for the given block"""
        with self.lock:
            if not self.should_create_checkpoint(block, state_change_count):
                return None
            
            start_time = time.time()
            checkpoint_name = self._generate_checkpoint_name(block)
            
            try:
                # Create state snapshot
                checkpoint_id = self.state_manager.create_checkpoint(checkpoint_name)
                if not checkpoint_id:
                    raise CheckpointIntegrityError("Failed to create state checkpoint")
                
                # Calculate state hash for integrity verification
                state_hash = self._calculate_state_hash()
                
                # Create metadata
                checkpoint_meta = CheckpointMetadata(
                    height=block.header.height,
                    hash=block.hash,
                    timestamp=time.time(),
                    name=checkpoint_name,
                    state_size=self.state_manager.get_state_stats()['state_size_bytes'],
                    state_hash=state_hash,
                    block_count_since_previous=self.blocks_since_last_checkpoint,
                    state_change_count=state_change_count,
                    status=CheckpointState.CREATED
                )
                
                # Store metadata in database
                meta_key = f"checkpoint_meta_{checkpoint_name}"
                self.database.put(meta_key, asdict(checkpoint_meta))
                
                # Also store in file system for redundancy
                self._store_metadata_file(checkpoint_name, checkpoint_meta)
                
                # Update state tracking
                self.last_checkpoint_height = block.header.height
                self.last_checkpoint_time = time.time()
                self.blocks_since_last_checkpoint = 0
                self.state_changes_since_last_checkpoint = 0
                
                # Verify checkpoint integrity if enabled
                if self.verify_on_create:
                    if not self.verify_checkpoint_integrity(checkpoint_name):
                        raise CheckpointIntegrityError("Checkpoint verification failed after creation")
                    checkpoint_meta.status = CheckpointState.VERIFIED
                    self.database.put(meta_key, asdict(checkpoint_meta))
                
                # Update metrics
                creation_time = time.time() - start_time
                self.metrics['checkpoints_created'] += 1
                self.metrics['total_creation_time'] += creation_time
                
                logger.info(f"Created checkpoint at height {block.header.height}: {checkpoint_name} "
                           f"(took {creation_time:.2f}s, state_size={checkpoint_meta.state_size})")
                
                # Cleanup old checkpoints if enabled
                if self.auto_cleanup:
                    self.cleanup_old_checkpoints()
                
                return checkpoint_name
                
            except Exception as e:
                logger.error(f"Failed to create checkpoint at height {block.header.height}: {e}")
                # Cleanup partially created checkpoint
                self._cleanup_failed_checkpoint(checkpoint_name)
                return None

    def restore_from_checkpoint(self, checkpoint_name: str, verify_integrity: bool = True) -> bool:
        """Restore state from a specific checkpoint"""
        with self.lock:
            start_time = time.time()
            
            try:
                # Verify checkpoint exists and is valid
                meta = self.get_checkpoint_metadata(checkpoint_name)
                if not meta:
                    logger.error(f"Checkpoint not found: {checkpoint_name}")
                    return False
                
                if meta.status == CheckpointState.CORRUPTED:
                    logger.error(f"Cannot restore corrupted checkpoint: {checkpoint_name}")
                    return False
                
                # Verify integrity before restoration
                if verify_integrity and not self.verify_checkpoint_integrity(checkpoint_name):
                    logger.error(f"Checkpoint integrity verification failed: {checkpoint_name}")
                    return False
                
                # Perform restoration
                success = self.state_manager.restore_checkpoint(checkpoint_name)
                if not success:
                    self.metrics['failed_restorations'] += 1
                    return False
                
                # Update state tracking
                self.last_checkpoint_height = meta.height
                self.last_checkpoint_time = time.time()
                self.blocks_since_last_checkpoint = 0
                
                # Update metrics
                restoration_time = time.time() - start_time
                self.metrics['checkpoints_restored'] += 1
                self.metrics['total_restoration_time'] += restoration_time
                
                logger.info(f"Successfully restored from checkpoint: {checkpoint_name} "
                           f"(height: {meta.height}, took {restoration_time:.2f}s)")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to restore from checkpoint {checkpoint_name}: {e}")
                self.metrics['failed_restorations'] += 1
                return False

    def get_checkpoint_metadata(self, checkpoint_name: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint"""
        try:
            meta_key = f"checkpoint_meta_{checkpoint_name}"
            meta_dict = self.database.get(meta_key)
            if meta_dict:
                return CheckpointMetadata(**meta_dict)
            
            # Fallback to file system
            return self._load_metadata_file(checkpoint_name)
        except Exception as e:
            logger.error(f"Failed to get metadata for checkpoint {checkpoint_name}: {e}")
            return None

    def list_checkpoints(self, include_corrupted: bool = False) -> List[CheckpointMetadata]:
        """List all available checkpoints"""
        checkpoints = []
        
        try:
            # Fix: Use proper database iteration for LevelDB
            if hasattr(self.database, 'iterator'):
            	# For LevelDB
            	with self.database.iterator() as it:
            		for key, value in it:
            			key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            			if key_str.startswith('checkpoint_meta_'):
            				try:
            					meta_dict = json.loads(value) if isinstance(value, (bytes, str)) else value
            					meta = CheckpointMetadata(**meta_dict)
            					if include_corrupted or meta.status != CheckpointState.CORRUPTED:
            						checkpoints.append(meta)
            				except Exception as e:
            					logger.warning(f"Failed to parse checkpoint metadata: {e}")
            else:
            	# For other databases
            	# You'll need to implement appropriate iteration
            	pass
            # Sort by height
            checkpoints.sort(key=lambda x: x.height)
        except Exception as e:
        	logger.error(f"Failed to list checkpoints: {e}")
        return checkpoints
 
    def get_best_checkpoint(self, target_height: Optional[int] = None) -> Optional[CheckpointMetadata]:
        """Get the best checkpoint for the given target height"""
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        if target_height is None:
            # Return the most recent verified checkpoint
            verified_checkpoints = [c for c in checkpoints if c.status == CheckpointState.VERIFIED]
            return max(verified_checkpoints, key=lambda x: x.height) if verified_checkpoints else None
        
        # Find the checkpoint closest to but not exceeding the target height
        suitable_checkpoints = [c for c in checkpoints 
                              if c.height <= target_height and c.status == CheckpointState.VERIFIED]
        
        if not suitable_checkpoints:
            return None
        
        return max(suitable_checkpoints, key=lambda x: x.height)

    def verify_checkpoint_integrity(self, checkpoint_name: str) -> bool:
        """Verify the integrity of a checkpoint"""
        with self.lock:
            try:
                # Get current state for restoration
                original_state = self.state_manager.create_snapshot()
                
                # Restore checkpoint temporarily
                success = self.state_manager.restore_checkpoint(checkpoint_name)
                if not success:
                    return False
                
                # Verify state integrity
                integrity_ok = self.state_manager.verify_state_integrity()
                
                # Verify state hash matches
                if integrity_ok:
                    meta = self.get_checkpoint_metadata(checkpoint_name)
                    if meta:
                        current_state_hash = self._calculate_state_hash()
                        integrity_ok = (current_state_hash == meta.state_hash)
                
                # Restore original state
                self.state_manager.restore_snapshot(original_state)
                
                # Update checkpoint status
                if integrity_ok:
                    self._update_checkpoint_status(checkpoint_name, CheckpointState.VERIFIED)
                else:
                    self._update_checkpoint_status(checkpoint_name, CheckpointState.CORRUPTED)
                
                return integrity_ok
                
            except Exception as e:
                logger.error(f"Checkpoint integrity check failed for {checkpoint_name}: {e}")
                self._update_checkpoint_status(checkpoint_name, CheckpointState.CORRUPTED)
                return False

    def cleanup_old_checkpoints(self) -> int:
        """Clean up old checkpoints, keeping only the most recent ones"""
        with self.lock:
            checkpoints = self.list_checkpoints(include_corrupted=True)
            
            if len(checkpoints) <= self.min_checkpoints:
                return 0
            
            # Sort by height (most recent first)
            checkpoints.sort(key=lambda x: x.height, reverse=True)
            
            # Keep the most recent checkpoints, prioritizing verified ones
            checkpoints_to_keep = []
            verified_count = 0
            
            for checkpoint in checkpoints:
                if len(checkpoints_to_keep) < self.max_checkpoints:
                    checkpoints_to_keep.append(checkpoint)
                    if checkpoint.status == CheckpointState.VERIFIED:
                        verified_count += 1
                elif checkpoint.status == CheckpointState.VERIFIED and verified_count < self.min_checkpoints:
                    # Ensure we keep minimum number of verified checkpoints
                    checkpoints_to_keep.append(checkpoint)
                    verified_count += 1
            
            checkpoints_to_remove = [c for c in checkpoints if c not in checkpoints_to_keep]
            removed_count = 0
            
            for checkpoint in checkpoints_to_remove:
                if self._delete_checkpoint(checkpoint.name):
                    removed_count += 1
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} old checkpoints, keeping {len(checkpoints_to_keep)}")
            
            return removed_count

    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about checkpoints"""
        checkpoints = self.list_checkpoints(include_corrupted=True)
        verified_checkpoints = [c for c in checkpoints if c.status == CheckpointState.VERIFIED]
        corrupted_checkpoints = [c for c in checkpoints if c.status == CheckpointState.CORRUPTED]
        
        stats = {
            'total_checkpoints': len(checkpoints),
            'verified_checkpoints': len(verified_checkpoints),
            'corrupted_checkpoints': len(corrupted_checkpoints),
            'latest_checkpoint_height': max(c.height for c in checkpoints) if checkpoints else -1,
            'earliest_checkpoint_height': min(c.height for c in checkpoints) if checkpoints else -1,
            'average_checkpoint_interval': self._calculate_average_interval(verified_checkpoints),
            'total_state_size': sum(c.state_size for c in checkpoints),
            'average_creation_time': (self.metrics['total_creation_time'] / self.metrics['checkpoints_created'] 
                                    if self.metrics['checkpoints_created'] > 0 else 0),
            'average_restoration_time': (self.metrics['total_restoration_time'] / self.metrics['checkpoints_restored'] 
                                       if self.metrics['checkpoints_restored'] > 0 else 0),
            'performance_metrics': self.metrics.copy()
        }
        
        return stats

    def _generate_checkpoint_name(self, block: Block) -> str:
        """Generate a unique checkpoint name"""
        timestamp = int(time.time())
        return f"height_{block.header.height}_{timestamp}_{block.hash[:8]}"

    def _calculate_state_hash(self) -> str:
        """Calculate a hash of the current state for integrity verification"""
        state_snapshot = self.state_manager.create_snapshot()
        state_data = pickle.dumps(state_snapshot)
        return hashlib.sha256(state_data).hexdigest()

    def _is_state_growing_excessively(self) -> bool:
        """Check if state is growing excessively since last checkpoint"""
        if self.last_checkpoint_height == -1:
            return False
        
        current_stats = self.state_manager.get_state_stats()
        previous_size = self._get_previous_state_size()
        
        if previous_size > 0:
            growth_factor = current_stats['state_size_bytes'] / previous_size
            return growth_factor > 2.0  # State size doubled
        
        return False

    def _get_previous_state_size(self) -> int:
        """Get the state size from the previous checkpoint"""
        checkpoints = self.list_checkpoints()
        if len(checkpoints) >= 2:
            return checkpoints[-2].state_size
        return 0

    def _store_metadata_file(self, checkpoint_name: str, metadata: CheckpointMetadata) -> bool:
        """Store metadata in file system for redundancy"""
        try:
            file_path = self.storage_path / f"{checkpoint_name}.meta"
            with open(file_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)
            return True
        except Exception as e:
            logger.warning(f"Failed to store metadata file for {checkpoint_name}: {e}")
            return False

    def _load_metadata_file(self, checkpoint_name: str) -> Optional[CheckpointMetadata]:
        """Load metadata from file system"""
        try:
            file_path = self.storage_path / f"{checkpoint_name}.meta"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    meta_dict = json.load(f)
                return CheckpointMetadata(**meta_dict)
        except Exception as e:
            logger.warning(f"Failed to load metadata file for {checkpoint_name}: {e}")
        return None

    def _update_checkpoint_status(self, checkpoint_name: str, status: CheckpointState) -> bool:
        """Update checkpoint status"""
        try:
            meta = self.get_checkpoint_metadata(checkpoint_name)
            if meta:
                meta.status = status
                meta_key = f"checkpoint_meta_{checkpoint_name}"
                self.database.put(meta_key, asdict(meta))
                
                # Update file system copy
                self._store_metadata_file(checkpoint_name, meta)
                return True
        except Exception as e:
            logger.error(f"Failed to update checkpoint status for {checkpoint_name}: {e}")
        return False

    def _delete_checkpoint(self, checkpoint_name: str) -> bool:
        """Delete a checkpoint and its metadata"""
        try:
            # Remove from database
            self.database.delete(f"checkpoint_{checkpoint_name}")
            self.database.delete(f"checkpoint_meta_{checkpoint_name}")
            
            # Remove from file system
            meta_file = self.storage_path / f"{checkpoint_name}.meta"
            if meta_file.exists():
                meta_file.unlink()
            
            self.metrics['checkpoints_deleted'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {checkpoint_name}: {e}")
            return False

    def _cleanup_failed_checkpoint(self, checkpoint_name: str):
        """Cleanup a partially created checkpoint"""
        try:
            self._delete_checkpoint(checkpoint_name)
            logger.info(f"Cleaned up failed checkpoint: {checkpoint_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup failed checkpoint {checkpoint_name}: {e}")

    def _calculate_average_interval(self, checkpoints: List[CheckpointMetadata]) -> float:
        """Calculate average checkpoint interval"""
        if len(checkpoints) < 2:
            return 0.0
        
        heights = sorted(c.height for c in checkpoints)
        intervals = [heights[i+1] - heights[i] for i in range(len(heights)-1)]
        
        return sum(intervals) / len(intervals) if intervals else 0.0

    def update_creation_strategy(self, strategy: CheckpointCreationStrategy):
        """Update the checkpoint creation strategy"""
        with self.lock:
            self.creation_strategy = strategy
            logger.info("Checkpoint creation strategy updated")

    def mark_checkpoint_corrupted(self, checkpoint_name: str) -> bool:
        """Manually mark a checkpoint as corrupted"""
        return self._update_checkpoint_status(checkpoint_name, CheckpointState.CORRUPTED)

    def get_checkpoint_chain(self, from_height: int, to_height: int) -> List[CheckpointMetadata]:
        """Get a chain of checkpoints between specified heights"""
        checkpoints = self.list_checkpoints()
        return [c for c in checkpoints if from_height <= c.height <= to_height]

    def export_checkpoint(self, checkpoint_name: str, export_path: str) -> bool:
        """Export checkpoint to external storage"""
        # Implementation would depend on specific storage requirements
        logger.info(f"Exporting checkpoint {checkpoint_name} to {export_path}")
        return True

    def import_checkpoint(self, import_path: str) -> Optional[str]:
        """Import checkpoint from external storage"""
        # Implementation would depend on specific storage requirements
        logger.info(f"Importing checkpoint from {import_path}")
        return None