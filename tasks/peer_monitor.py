# tasks/peer_monitor.py - Peer monitoring

import asyncio
import time
import logging
from typing import Dict, List

logger = logging.getLogger("rayonix_node.peer_monitor")

class PeerMonitor:
    """Monitors and manages peer connections"""
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.peer_stats: Dict[str, Dict] = {}
        self.last_peer_discovery = time.time()
    
    async def monitor_peers(self):
        """Main peer monitoring loop"""
        while self.node.running:
            try:
                if not self.node.network:
                    await asyncio.sleep(10)
                    continue
                
                # Monitor peer connections
                await self._check_peer_connections()
                
                # Discover new peers periodically
                if time.time() - self.last_peer_discovery > 300:  # Every 5 minutes
                    await self._discover_new_peers()
                    self.last_peer_discovery = time.time()
                
                # Update peer statistics
                await self._update_peer_stats()
                
                # Update node state with peer count
                peers = await self.node.network.get_peers()
                self.node.state_manager.update_sync_state(peers_connected=len(peers))
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in peer monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _check_peer_connections(self):
        """Check and maintain peer connections"""
        if not self.node.network:
            return
        
        try:
            current_peers = await self.node.network.get_peers()
            max_connections = self.node.get_config_value('network.max_connections', 50)
            
            # Check if we need more connections
            if len(current_peers) < max_connections // 2:
                logger.info(f"Low peer count ({len(current_peers)}), discovering new peers")
                await self._discover_new_peers()
            
            # Check peer health
            for peer in current_peers:
                peer_id = peer.get('id')
                if peer_id and peer_id not in self.peer_stats:
                    self.peer_stats[peer_id] = {
                        'first_seen': time.time(),
                        'last_seen': time.time(),
                        'messages_sent': 0,
                        'messages_received': 0,
                        'connection_time': 0
                    }
                
                # Update peer stats
                if peer_id in self.peer_stats:
                    self.peer_stats[peer_id]['last_seen'] = time.time()
                    self.peer_stats[peer_id]['connection_time'] = (
                        time.time() - self.peer_stats[peer_id]['first_seen']
                    )
            
            # Remove stale peers from stats
            stale_time = time.time() - 3600  # 1 hour
            for peer_id in list(self.peer_stats.keys()):
                if self.peer_stats[peer_id]['last_seen'] < stale_time:
                    del self.peer_stats[peer_id]
                    
        except Exception as e:
            logger.error(f"Error checking peer connections: {e}")
    
    async def _discover_new_peers(self):
        """Discover new peers to connect to"""
        if not self.node.network:
            return
        
        try:
            # Get current peers
            current_peers = await self.node.network.get_peers()
            current_peer_addresses = {f"{p.get('ip')}:{p.get('port')}" for p in current_peers}
            
            # Ask current peers for their peer lists
            new_peers = set()
            for peer in current_peers:
                try:
                    response = await self.node.network.send_message(
                        peer['id'],
                        'get_peers',
                        {}
                    )
                    
                    if response and 'peers' in response:
                        for new_peer in response['peers']:
                            peer_addr = f"{new_peer.get('ip')}:{new_peer.get('port')}"
                            if peer_addr not in current_peer_addresses:
                                new_peers.add(peer_addr)
                                
                except Exception as e:
                    logger.debug(f"Failed to get peers from {peer['id']}: {e}")
                    continue
            
            # Connect to new peers (limit to avoid connection storms)
            max_new_connections = min(10, self.node.get_config_value('network.max_connections', 50) - len(current_peers))
            connected_count = 0
            
            for peer_addr in list(new_peers)[:max_new_connections]:
                try:
                    if await self.node.network.connect_to_peer(peer_addr):
                        connected_count += 1
                        logger.info(f"Connected to new peer: {peer_addr}")
                    else:
                        logger.debug(f"Failed to connect to new peer: {peer_addr}")
                except Exception as e:
                    logger.debug(f"Error connecting to new peer {peer_addr}: {e}")
            
            if connected_count > 0:
                logger.info(f"Discovered and connected to {connected_count} new peers")
                
        except Exception as e:
            logger.error(f"Error discovering new peers: {e}")
    
    async def _update_peer_stats(self):
        """Update peer statistics"""
        if not self.node.network:
            return
        
        try:
            # Get network statistics
            stats = await self.node.network.get_stats()
            if stats:
                # Update peer stats with network info
                for peer_id, peer_stat in self.peer_stats.items():
                    if peer_id in stats.get('peer_stats', {}):
                        peer_stat.update(stats['peer_stats'][peer_id])
                        
        except Exception as e:
            logger.error(f"Error updating peer stats: {e}")
    
    def get_peer_statistics(self) -> Dict:
        """Get peer statistics"""
        return self.peer_stats.copy()
    
    async def ban_peer(self, peer_id: str, reason: str = "Unknown", duration: int = 3600) -> bool:
        """Ban a peer temporarily"""
        if not self.node.network:
            return False
        
        try:
            # Disconnect from peer
            await self.node.network.disconnect_peer(peer_id)
            
            # Add to ban list (implementation would depend on network layer)
            logger.warning(f"Banned peer {peer_id} for {duration}s: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error banning peer {peer_id}: {e}")
            return False