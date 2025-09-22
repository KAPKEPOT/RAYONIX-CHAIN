# network/network_manager.py - Network initialization and management

import asyncio
import logging
from typing import Dict, List, Optional
from network.core.p2p_network import AdvancedP2PNetwork

logger = logging.getLogger("rayonix_node.network")

class NetworkManager:
    """Manages P2P network initialization and operations"""
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.network = None
        self.connected_peers: Dict[str, Dict] = {}
    
    async def initialize_network(self) -> bool:
        """Initialize the P2P network"""
        try:       
            
            network_config = {
                'listen_ip': self.node.get_config_value('network.listen_ip', '0.0.0.0'),
                'listen_port': self.node.get_config_value('network.listen_port', 9333),
                'max_connections': self.node.get_config_value('network.max_connections', 50),
                'bootstrap_nodes': self.node.get_config_value('network.bootstrap_nodes', []),
                'enable_encryption': self.node.get_config_value('network.enable_encryption', True),
                'enable_compression': self.node.get_config_value('network.enable_compression', True),
                'connection_timeout': self.node.get_config_value('network.connection_timeout', 30),
                'message_timeout': self.node.get_config_value('network.message_timeout', 10),
                'network_id': self.node.get_config_value('network.network_id', 1)
            }
            
            self.network = AdvancedP2PNetwork(network_config)
            
            # Register message handlers
            self._register_message_handlers()
            
            logger.info("P2P network initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize network: {e}")
            return False
    
    def _register_message_handlers(self):
        """Register message handlers for different message types"""
        if not self.network:
            return
        
        from network.message_handlers import (
            handle_block_message,
            handle_transaction_message,
            handle_peer_list_message,
            handle_sync_request_message,
            handle_ping_message
        )
        
        self.network.register_message_handler('block', handle_block_message)
        self.network.register_message_handler('transaction', handle_transaction_message)
        self.network.register_message_handler('peer_list', handle_peer_list_message)
        self.network.register_message_handler('sync_request', handle_sync_request_message)
        self.network.register_message_handler('ping', handle_ping_message)
    
    async def connect_to_bootstrap_nodes(self):
        """Connect to bootstrap nodes"""
        if not self.network:
            return False
        
        bootstrap_nodes = self.node.get_config_value('network.bootstrap_nodes', [])
        if not bootstrap_nodes:
            logger.warning("No bootstrap nodes configured")
            return False
        
        connected = 0
        for node_address in bootstrap_nodes:
            try:
                if await self.network.connect_to_peer(node_address):
                    connected += 1
                    logger.info(f"Connected to bootstrap node: {node_address}")
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap node {node_address}: {e}")
        
        logger.info(f"Connected to {connected} bootstrap nodes")
        return connected > 0
    
    async def broadcast_block(self, block_data: Dict) -> bool:
        """Broadcast block to all connected peers"""
        if not self.network:
            return False
        
        try:
            await self.network.broadcast_message('block', block_data)
            logger.info(f"Block {block_data.get('hash', 'unknown')} broadcast to network")
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast block: {e}")
            return False
    
    async def broadcast_transaction(self, tx_data: Dict) -> bool:
        """Broadcast transaction to all connected peers"""
        if not self.network:
            return False
        
        try:
            await self.network.broadcast_message('transaction', tx_data)
            logger.info(f"Transaction {tx_data.get('hash', 'unknown')} broadcast to network")
            return True
        except Exception as e:
            logger.error(f"Failed to broadcast transaction: {e}")
            return False
    
    async def get_peer_info(self) -> List[Dict]:
        """Get information about connected peers"""
        if not self.network:
            return []
        
        try:
            return await self.network.get_peers()
        except Exception as e:
            logger.error(f"Failed to get peer info: {e}")
            return []
    
    async def disconnect_peer(self, peer_id: str) -> bool:
        """Disconnect from a specific peer"""
        if not self.network:
            return False
        
        try:
            return await self.network.disconnect_peer(peer_id)
        except Exception as e:
            logger.error(f"Failed to disconnect peer {peer_id}: {e}")
            return False
    
    async def get_network_stats(self) -> Dict:
        """Get network statistics"""
        if not self.network:
            return {}
        
        try:
            stats = await self.network.get_stats()
            return {
                'total_peers': stats.get('total_peers', 0),
                'connected_peers': stats.get('connected_peers', 0),
                'messages_sent': stats.get('messages_sent', 0),
                'messages_received': stats.get('messages_received', 0),
                'bytes_sent': stats.get('bytes_sent', 0),
                'bytes_received': stats.get('bytes_received', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get network stats: {e}")
            return {}