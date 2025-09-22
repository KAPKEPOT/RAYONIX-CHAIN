# network/sync_manager.py - Block synchronization logic

import asyncio
import time
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("rayonix_node.sync")

class SyncManager:
    """Manages blockchain synchronization with peers"""
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.syncing = False
        self.sync_start_time = 0
        self.last_sync_update = 0
        self.sync_peers: List[str] = []
        self.current_sync_peer: Optional[str] = None
    
    async def sync_blocks(self):
        """Main synchronization loop"""
        while self.node.running:
            try:
                if not self.node.network:
                    await asyncio.sleep(5)
                    continue
                
                # Check if we need to sync
                if not self._should_sync():
                    await asyncio.sleep(10)
                    continue
                
                # Start sync process
                await self._start_sync()
                
                # Perform sync
                await self._perform_sync()
                
                # End sync process
                await self._end_sync()
                
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                await asyncio.sleep(10)
    
    def _should_sync(self) -> bool:
        """Check if synchronization is needed"""
        # Don't sync if network is disabled
        if not self.node.get_config_value('network.enabled', True):
            return False
        
        # Don't sync if already syncing
        if self.syncing:
            return False
        
        # Check if we have peers
        if not self.node.network or not self.node.network.get_connected_peers():
            return False
        
        return True
    
    async def _start_sync(self):
        """Start synchronization process"""
        self.syncing = True
        self.sync_start_time = time.time()
        self.last_sync_update = time.time()
        
        logger.info("Starting blockchain synchronization")
        
        # Update state manager
        self.node.state_manager.update_sync_state(
            syncing=True,
            current_block=self.node.rayonix_coin.get_block_count(),
            target_block=0,  # Will be updated during sync
            last_sync_time=time.time()
        )
    
    async def _perform_sync(self):
        """Perform the actual synchronization"""
        try:
            # Get best block from peers
            best_block_height = await self._get_best_block_height()
            if best_block_height is None:
                logger.warning("Could not determine best block height from peers")
                return
            
            current_height = self.node.rayonix_coin.get_block_count()
            
            # Update target block
            self.node.state_manager.update_sync_state(target_block=best_block_height)
            
            if current_height >= best_block_height:
                logger.info("Blockchain is already synchronized")
                return
            
            logger.info(f"Synchronizing blocks {current_height} to {best_block_height}")
            
            # Sync blocks in batches
            batch_size = 100
            while current_height < best_block_height and self.node.running:
                end_height = min(current_height + batch_size, best_block_height)
                
                # Get blocks from peers
                blocks = await self._get_blocks_from_peers(current_height + 1, end_height)
                if not blocks:
                    logger.warning(f"Failed to get blocks {current_height + 1}-{end_height}")
                    break
                
                # Process blocks
                for block in blocks:
                    if not self.node.rayonix_coin._process_block(block):
                        logger.warning(f"Failed to process block {block.get('height')}")
                        break
                    
                    current_height = block.get('height')
                    
                    # Update sync progress
                    if time.time() - self.last_sync_update > 1.0:  # Update every second
                        self.node.state_manager.update_sync_state(
                            current_block=current_height,
                            sync_progress=(current_height / best_block_height) * 100
                        )
                        self.last_sync_update = time.time()
                
                logger.info(f"Synced to block {current_height}/{best_block_height}")
                
                # Adjust batch size based on performance
                if len(blocks) == batch_size:
                    batch_size = min(batch_size * 2, 1000)  # Double batch size, max 1000
                else:
                    batch_size = max(batch_size // 2, 10)  # Halve batch size, min 10
                
                await asyncio.sleep(0.1)  # Brief pause
            
            logger.info(f"Synchronization completed at block {current_height}")
            
        except Exception as e:
            logger.error(f"Error during synchronization: {e}")
    
    async def _end_sync(self):
        """End synchronization process"""
        self.syncing = False
        current_height = self.node.rayonix_coin.get_block_count()
        
        # Update state manager
        self.node.state_manager.update_sync_state(
            syncing=False,
            current_block=current_height,
            sync_progress=100.0,
            last_sync_time=time.time()
        )
        
        sync_duration = time.time() - self.sync_start_time
        logger.info(f"Synchronization completed in {sync_duration:.2f} seconds")
    
    async def _get_best_block_height(self) -> Optional[int]:
        """Get the best block height from connected peers"""
        if not self.node.network:
            return None
        
        try:
            peers = await self.node.network.get_peers()
            if not peers:
                return None
            
            # Request block height from all peers
            heights = []
            for peer in peers:
                try:
                    response = await self.node.network.send_message(
                        peer['id'], 
                        'get_block_height', 
                        {}
                    )
                    if response and 'height' in response:
                        heights.append(response['height'])
                except Exception as e:
                    logger.debug(f"Failed to get block height from peer {peer['id']}: {e}")
            
            if not heights:
                return None
            
            # Return the maximum height (most common consensus approach)
            return max(heights)
            
        except Exception as e:
            logger.error(f"Error getting best block height: {e}")
            return None
    
    async def _get_blocks_from_peers(self, start_height: int, end_height: int) -> List[Dict]:
        """Get blocks from peers"""
        if not self.node.network:
            return []
        
        try:
            peers = await self.node.network.get_peers()
            if not peers:
                return []
            
            # Try each peer until we get the blocks
            for peer in peers:
                try:
                    response = await self.node.network.send_message(
                        peer['id'],
                        'get_blocks',
                        {
                            'start_height': start_height,
                            'end_height': end_height
                        }
                    )
                    
                    if response and 'blocks' in response:
                        blocks = response['blocks']
                        if blocks and len(blocks) == (end_height - start_height + 1):
                            return blocks
                            
                except Exception as e:
                    logger.debug(f"Failed to get blocks from peer {peer['id']}: {e}")
                    continue
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting blocks from peers: {e}")
            return []
    
    def is_syncing(self) -> bool:
        """Check if synchronization is in progress"""
        return self.syncing
    
    def get_sync_status(self) -> Dict:
        """Get current synchronization status"""
        return self.node.state_manager.get_sync_state()