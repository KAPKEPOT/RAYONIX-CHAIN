# network/sync_manager.py - Block synchronization logic
import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass
logger = logging.getLogger("rayonix_node.sync")
from config.sync_config import SyncConfig

class SyncManager:
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.syncing = False
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        self.last_sync_attempt = 0
        self.min_retry_interval = 30
        self.peer_timeout = 10
        
        # Circuit breaker state
        self.circuit_open = False
        self.circuit_open_until = 0
        
        # Sync state tracking
        self.current_sync_height = 0
        self.target_sync_height = 0
        self.sync_start_time = 0
        
        # Peer reliability tracking
        self.peer_reliability: Dict[str, float] = {}
        
        logger.info("SyncManager initialized")
        
        self.sync_config = SyncConfig()
        self.user_paused = False
        self.user_cancelled = False
        self.download_stats = {
            'blocks_downloaded': 0,
            'bytes_downloaded': 0,
            'download_speed': 0,
            'estimated_time_remaining': 0
        }
    
    async def start_sync_with_options(self, sync_mode: SyncMode = None,config_options: Dict = None) -> bool:
    	"""Start synchronization with user-selected options"""
    	if sync_mode:
    		self.sync_config.mode = sync_mode
    	
    	if config_options:
    		for key, value in config_options.items():
    			if hasattr(self.sync_config, key):
    				setattr(self.sync_config, key, value)
    	
    	logger.info(f"Starting sync in {self.sync_config.mode.value} mode")
    	return await self.sync_blocks()
    
    async def pause_sync(self):
    	"""Pause synchronization"""
    	self.user_paused = True
    	logger.info("Sync paused by user")
    
    async def resume_sync(self):
    	"""Resume synchronization"""
    	self.user_paused = False
    	logger.info("Sync resumed by user")
    
    async def cancel_sync(self):
    	"""Cancel synchronization"""
    	self.user_cancelled = True
    	self.user_paused = False
    	logger.info("Sync cancelled by user")
    
    def get_sync_progress_detailed(self) -> Dict[str, Any]:
    	"""Get detailed progress information for UI"""
    	progress = self.get_sync_progress()
    	
    	return {
    	    'mode': self.sync_config.mode.value,
    	    'progress_percentage': progress,
    	    'current_height': self.current_sync_height,
    	    'target_height': self.target_sync_height,
    	    'blocks_remaining': self.target_sync_height - self.current_sync_height,
    	    'paused': self.user_paused,
    	    'cancelled': self.user_cancelled,
    	    'download_stats': self.download_stats.copy(),
    	    'peer_count': len(self.peer_reliability),
    	    'reliable_peers': len([p for p in self.peer_reliability.values() if p > 0.7])
    	}
    	
    async def sync_blocks(self):
        """Complete sync loop implementation"""
        while self.node.running:
            try:
                if self._is_circuit_open():
                    await asyncio.sleep(60)
                    continue
                
                if not await self._should_sync():
                    await asyncio.sleep(30)
                    continue
                
                if not self._should_retry():
                    await asyncio.sleep(30)
                    continue
                
                success = await self._execute_complete_sync_cycle()
                
                if success:
                    self.consecutive_failures = 0
                    await asyncio.sleep(30)
                else:
                    self.consecutive_failures += 1
                    backoff = self._calculate_backoff()
                    logger.warning(f"Sync failed, backing off for {backoff}s")
                    await asyncio.sleep(backoff)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.consecutive_failures += 1
                logger.error(f"Sync loop error: {e}")
                backoff = self._calculate_backoff()
                await asyncio.sleep(backoff)

    def _is_circuit_open(self) -> bool:
        if self.circuit_open:
            if time.time() > self.circuit_open_until:
                self.circuit_open = False
                logger.info("Circuit breaker closed")
                return False
            return True
        return False

    def _should_retry(self) -> bool:
        if self.consecutive_failures == 0:
            return True
            
        min_interval = self.min_retry_interval * (2 ** (self.consecutive_failures - 1))
        time_since_last_attempt = time.time() - self.last_sync_attempt
        
        if time_since_last_attempt < min_interval:
            return False
            
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.circuit_open = True
            self.circuit_open_until = time.time() + 300
            logger.error("Circuit breaker opened due to excessive failures")
            return False
            
        return True

    def _calculate_backoff(self) -> int:
        base_backoff = min(300, self.min_retry_interval * (2 ** self.consecutive_failures))
        jitter = base_backoff * 0.1
        return int(base_backoff + jitter)

    async def _should_sync(self) -> bool:
        if not self.node.get_config_value('network.enabled', True):
            return False
        if self.syncing:
            return False
        if not self.node.running:
            return False
        if not await self._check_network_ready():
            return False
        return True

    async def _check_network_ready(self) -> bool:
        try:
            if not self.node.network:
                return False
            if not hasattr(self.node.network, 'is_running') or not self.node.network.is_running:
                return False
            peers = await self._get_available_peers()
            return len(peers) >= 1
        except Exception as e:
            logger.debug(f"Network readiness check failed: {e}")
            return False

    async def _get_available_peers(self) -> List[Dict]:
        try:
            if hasattr(self.node.network, 'get_peers'):
                peers = await self.node.network.get_peers()
                return peers if peers else []
            return []
        except Exception as e:
            logger.debug(f"Failed to get peers: {e}")
            return []

    async def _execute_complete_sync_cycle(self) -> bool:
        self.last_sync_attempt = time.time()
        self.syncing = True
        self.sync_start_time = time.time()
        
        try:
            logger.info("Starting complete sync cycle")
            
            best_height = await self._get_consensus_block_height()
            if best_height is None:
                logger.warning("Could not determine consensus block height")
                return False
            
            current_height = self.node.rayonix_chain.get_block_count()
            self.current_sync_height = current_height  # Track current height
            self.target_sync_height = best_height      # Track target height
            
            if current_height >= best_height:
                logger.info(f"Node synchronized at height {current_height}")
                self._update_node_sync_state(current_height, best_height, 100.0, False)
                return True
                
            logger.info(f"Syncing from {current_height} to {best_height}")
            
            success = await self._synchronize_block_range(current_height, best_height)
            
            if success:
                final_height = self.node.rayonix_chain.get_block_count()
                logger.info(f"Sync completed successfully to height {final_height}")
                self._update_node_sync_state(final_height, best_height, 100.0, False)
            else:
                logger.warning("Block synchronization failed")
                self._update_node_sync_state(current_height, best_height, 0.0, False)
                
            return success
            
        except asyncio.TimeoutError:
            logger.error("Sync operation timed out")
            return False
        except Exception as e:
            logger.error(f"Sync execution failed: {e}")
            return False
        finally:
            self.syncing = False

    async def _get_consensus_block_height(self) -> Optional[int]:
        try:
            async with asyncio.timeout(30):
                peers = await self._get_available_peers()
                if not peers:
                    return None
                
                heights = []
                reliable_peers = self._get_reliable_peers(peers)
                
                for peer in reliable_peers[:5]:  # Limit to 5 reliable peers
                    try:
                        height = await self._query_peer_block_height(peer)
                        if height is not None and height > 0:
                            heights.append(height)
                            await self._update_peer_reliability(peer['id'], True)
                    except Exception as e:
                        logger.debug(f"Failed to query peer {peer.get('id')}: {e}")
                        await self._update_peer_reliability(peer['id'], False)
                        continue
                
                if not heights:
                    return None
                
                # Use median to avoid outliers
                sorted_heights = sorted(heights)
                median_height = sorted_heights[len(sorted_heights) // 2]
                
                # Verify consensus (at least 60% of peers within 10 blocks)
                consensus_threshold = len(heights) * 0.6
                consensus_count = sum(1 for h in heights if abs(h - median_height) <= 10)
                
                if consensus_count >= consensus_threshold:
                    return median_height
                else:
                    logger.warning(f"Insufficient height consensus: {consensus_count}/{len(heights)}")
                    return None
                
        except asyncio.TimeoutError:
            logger.warning("Block height consensus timed out")
            return None
        except Exception as e:
            logger.error(f"Block height consensus failed: {e}")
            return None

    def _get_reliable_peers(self, peers: List[Dict]) -> List[Dict]:
        reliable_peers = []
        for peer in peers:
            peer_id = peer.get('id')
            reliability = self.peer_reliability.get(peer_id, 0.5)
            if reliability >= 0.3:  # Include somewhat reliable peers
                reliable_peers.append(peer)
        return reliable_peers

    async def _update_peer_reliability(self, peer_id: str, success: bool):
        current_score = self.peer_reliability.get(peer_id, 0.5)
        alpha = 0.2  # Learning rate
        
        if success:
            new_score = alpha * 1.0 + (1 - alpha) * current_score
        else:
            new_score = alpha * 0.0 + (1 - alpha) * current_score
            
        self.peer_reliability[peer_id] = max(0.0, min(1.0, new_score))

    async def _query_peer_block_height(self, peer: Dict) -> Optional[int]:
        try:
            peer_id = peer.get('id')
            if not peer_id:
                return None
                
            async with asyncio.timeout(self.peer_timeout):
                response = await self.node.network.send_message(
                    peer_id, 'GET_BLOCKS', {}
                )
                return response.get('height') if response else None
                
        except asyncio.TimeoutError:
            logger.debug(f"Timeout querying peer {peer_id}")
            return None
        except Exception as e:
            logger.debug(f"Failed to query peer {peer_id}: {e}")
            return None

    async def _synchronize_block_range(self, start_height: int, end_height: int) -> bool:
        if start_height > end_height:
            logger.error(f"Invalid sync range: {start_height}-{end_height}")
            return False

        logger.info(f"Synchronizing blocks {start_height} to {end_height}")
        
        try:
            batch_size = 50  # Conservative batch size
            current_height = start_height
            successful_batches = 0
            
            while current_height <= end_height and self.node.running:
                batch_end = min(current_height + batch_size - 1, end_height)
                
                logger.debug(f"Fetching blocks {current_height}-{batch_end}")
                blocks = await self._fetch_blocks_from_peers(current_height, batch_end)
                
                if not blocks:
                    logger.error(f"No blocks received for range {current_height}-{batch_end}")
                    return False
                
                if not await self._validate_and_process_blocks(blocks, current_height):
                    logger.error(f"Block processing failed for range {current_height}-{batch_end}")
                    return False
                
                successful_batches += 1
                current_height = batch_end + 1
                self.current_sync_height = current_height
                
                # Update progress
                progress = (current_height / end_height) * 100
                self._update_node_sync_state(current_height, end_height, progress, True)
                
                # Adaptive batch sizing
                if successful_batches > 3:
                    batch_size = min(batch_size * 2, 200)  # Gradually increase
                elif successful_batches == 0:
                    batch_size = max(batch_size // 2, 10)  # Reduce on failure
                
                # Rate limiting
                await asyncio.sleep(0.05)
            
            logger.info(f"Successfully synchronized {successful_batches} batches to height {end_height}")
            return True
            
        except Exception as e:
            logger.error(f"Block range synchronization failed: {e}")
            return False

    async def _fetch_blocks_from_peers(self, start: int, end: int) -> List[Dict]:
        peers = await self._get_available_peers()
        if not peers:
            return []
        
        # Try reliable peers first
        reliable_peers = self._get_reliable_peers(peers)
        all_peers = reliable_peers + [p for p in peers if p not in reliable_peers]
        
        for peer in all_peers:
            try:
                async with asyncio.timeout(15):
                    response = await self.node.network.send_message(
                        peer['id'], 
                        'GET_BLOCKS', 
                        {
                            'start_height': start,
                            'end_height': end,
                            'batch_size': end - start + 1
                        }
                    )
                    
                    if response and 'blocks' in response:
                        blocks = response['blocks']
                        if (blocks and 
                            self._validate_block_batch(blocks, start, end)):
                            await self._update_peer_reliability(peer['id'], True)
                            return blocks
                        else:
                            logger.debug(f"Invalid block batch from peer {peer['id']}")
                            await self._update_peer_reliability(peer['id'], False)
                            
            except asyncio.TimeoutError:
                logger.debug(f"Peer {peer.get('id')} timeout fetching blocks {start}-{end}")
                await self._update_peer_reliability(peer['id'], False)
            except Exception as e:
                logger.debug(f"Peer {peer.get('id')} failed: {e}")
                await self._update_peer_reliability(peer['id'], False)
                continue
        
        return []

    def _validate_block_batch(self, blocks: List[Dict], expected_start: int, expected_end: int) -> bool:
        if not blocks:
            return False
        
        # Check batch continuity
        if blocks[0].get('height') != expected_start:
            return False
        if blocks[-1].get('height') != expected_end:
            return False
        
        # Check sequential order
        for i, block in enumerate(blocks):
            expected_height = expected_start + i
            if block.get('height') != expected_height:
                return False
            if not self._validate_block_structure(block):
                return False
        
        return True

    def _validate_block_structure(self, block: Dict) -> bool:
        required_fields = ['hash', 'height', 'previous_hash', 'transactions', 'timestamp']
        if not all(field in block for field in required_fields):
            return False
        
        # Basic sanity checks
        if not isinstance(block['transactions'], list):
            return False
        if block['height'] < 0:
            return False
        if len(block['hash']) != 64:  # Assuming 64-char hex hash
            return False
            
        return True

    async def _validate_and_process_blocks(self, blocks: List[Dict], expected_start: int) -> bool:
        for i, block in enumerate(blocks):
            expected_height = expected_start + i
            
            # Deep block validation
            if not await self._validate_block_completely(block, expected_height):
                logger.error(f"Block validation failed at height {expected_height}")
                return False
            
            # Apply to blockchain
            if not await self._apply_block_to_blockchain(block):
                logger.error(f"Failed to apply block {block.get('hash')} at height {expected_height}")
                return False
            
            # Update mempool
            await self._update_mempool(block)
            
            logger.debug(f"Successfully processed block {expected_height}")
        
        return True

    async def _validate_block_completely(self, block: Dict, expected_height: int) -> bool:
        if block.get('height') != expected_height:
            return False
        
        # Use blockchain's native validation
        if hasattr(self.node.rayonix_chain, '_validate_block'):
            return self.node.rayonix_chain._validate_block(block)
        elif hasattr(self.node.rayonix_chain, 'validate_block'):
            return self.node.rayonix_chain.validate_block(block)
        else:
            # Fallback validation
            return self._validate_block_structure(block)

    async def _apply_block_to_blockchain(self, block: Dict) -> bool:
        try:
            # Use the blockchain's native block application
            if hasattr(self.node.rayonix_chain, '_process_block'):
                return self.node.rayonix_chain._process_block(block)
            elif hasattr(self.node.rayonix_chain, 'add_block'):
                return self.node.rayonix_chain.add_block(block)
            else:
                logger.error("No block application method found in blockchain")
                return False
        except Exception as e:
            logger.error(f"Block application failed: {e}")
            return False

    async def _update_mempool(self, block: Dict):
        try:
            # Remove transactions from mempool that are now in a block
            if hasattr(self.node.rayonix_chain, '_remove_from_mempool'):
                for tx in block.get('transactions', []):
                    tx_hash = tx.get('hash')
                    if tx_hash:
                        self.node.rayonix_chain._remove_from_mempool(tx_hash)
        except Exception as e:
            logger.debug(f"Mempool update failed: {e}")

    def _update_node_sync_state(self, current: int, target: int, progress: float, syncing: bool):
        try:
            self.node.state_manager.update_sync_state(
                syncing=syncing,
                current_block=current,
                target_block=target,
                sync_progress=progress,
                last_sync_time=time.time()
            )
        except Exception as e:
            logger.debug(f"Failed to update sync state: {e}")

    def get_status(self) -> Dict:
        """Complete status reporting"""
        return {
            "syncing": self.syncing,
            "consecutive_failures": self.consecutive_failures,
            "circuit_open": self.circuit_open,
            "last_sync_attempt": self.last_sync_attempt,
            "current_height": self.node.rayonix_chain.get_block_count() if self.node.rayonix_chain else 0,
            "reliable_peers": len([p for p in self.peer_reliability.values() if p > 0.7]),
            "total_peers_tracked": len(self.peer_reliability),
            "retry_backoff": self._calculate_backoff() if self.consecutive_failures > 0 else 0
        }
       
    def get_sync_progress(self) -> float:
    	"""Get current sync progress percentage"""
    	if not self.syncing:
    		return 100.0
    	if self.target_sync_height == 0:
    		return 0.0
    	
    	current_height = self.current_sync_height or (self.node.rayonix_chain.get_block_count() if self.node.rayonix_chain else 0)
    	progress = (current_height / self.target_sync_height) * 100
    	return min(100.0, max(0.0, progress))
    
    def is_syncing(self) -> bool:
    	"""Check if currently syncing"""
    	return self.syncing
    
    def get_current_sync_height(self) -> int:
    	"""Get current sync height"""
    	return self.current_sync_height
    
    def get_target_sync_height(self) -> int:
    	"""Get target sync height"""
    	return self.target_sync_height
    