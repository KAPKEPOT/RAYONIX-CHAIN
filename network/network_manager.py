# network/network_manager.py - Network initialization and management

import asyncio
import logging
from typing import Dict, List, Optional
from network.core.p2p_network import AdvancedP2PNetwork
from network.config.network_types import MessageType  # Add this import

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
            # Extract individual parameters instead of passing a dict
            network_id = self.node.get_config_value('network.network_id', 1)
            listen_port = self.node.get_config_value('network.listen_port', 9333)
            max_connections = self.node.get_config_value('network.max_connections', 50)
            
            # Create network config object if needed, but pass individual params to constructor
            self.network = AdvancedP2PNetwork(
                network_id=network_id,
                port=listen_port,
                max_connections=max_connections,
                node_id=None,  # Let it generate its own
                config=None    # Don't pass the dict directly
            )
            
            # Register message handlers
            self._register_message_handlers()
            
            logger.info("P2P network initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize network: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Use MessageType enum values directly
        self.network.register_message_handler(MessageType.BLOCK, handle_block_message)
        self.network.register_message_handler(MessageType.TRANSACTION, handle_transaction_message)
        self.network.register_message_handler(MessageType.PEER_LIST, handle_peer_list_message)
        self.network.register_message_handler(MessageType.SYNC_REQUEST, handle_sync_request_message)
        self.network.register_message_handler(MessageType.PING, handle_ping_message)
    
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
            return await self.network.get_stats()
        except Exception as e:
            logger.error(f"Failed to get network stats: {e}")
            return {}