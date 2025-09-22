import asyncio
import logging
import time
from typing import Callable, Any, Dict, List
from ..interfaces.processor_interface import IMessageProcessor
from ..exceptions import MessageError
from ..models.network_message import NetworkMessage
from ..config.network_types import MessageType

logger = logging.getLogger("MessageProcessor")

class MessageProcessor(IMessageProcessor):
    """Message processing implementation"""
    
    def __init__(self, network):
        self.network = network
        self.handlers: Dict[MessageType, List[Callable]] = {}
    
    async def process_message(self, connection_id: str, message: NetworkMessage):
        """Process incoming message"""
        try:
            # Update connection activity
            self.network.connection_manager.update_connection_activity(connection_id)
            
            # Log message receipt
            logger.debug(f"Processing {message.message_type.name} from {connection_id}")
            
            # Call registered handlers
            if message.message_type in self.handlers:
                for handler in self.handlers[message.message_type]:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Handler error for {message.message_type}: {e}")
            
            # Handle specific message types
            if message.message_type == MessageType.PING:
                await self._handle_ping(connection_id, message)
            elif message.message_type == MessageType.PONG:
                await self._handle_pong(connection_id, message)
            elif message.message_type == MessageType.PEER_LIST:
                await self._handle_peer_list(connection_id, message)
            elif message.message_type == MessageType.HANDSHAKE:
                await self._handle_handshake(connection_id, message)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            await self.network.penalize_peer(connection_id, -5)
    
    async def _handle_ping(self, connection_id: str, message: NetworkMessage):
        """Handle ping message"""
        try:
            # Send pong response
            pong_message = NetworkMessage(
                message_id=f"pong_{time.time()}",
                message_type=MessageType.PONG,
                payload={"original_timestamp": message.payload.get("timestamp")},
                source_node=self.network.node_id
            )
            
            await self.send_message(connection_id, pong_message)
            
        except Exception as e:
            logger.error(f"Ping handling error: {e}")
    
    async def _handle_pong(self, connection_id: str, message: NetworkMessage):
        """Handle pong message"""
        try:
            # Calculate latency
            original_timestamp = message.payload.get("original_timestamp")
            if original_timestamp:
                latency = time.time() - original_timestamp
                
                # Update connection metrics
                self.network.metrics_collector.update_connection_metrics(
                    connection_id, latency=latency
                )
                
                # Update peer info if available
                if connection_id in self.network.connections:
                    peer_info = self.network.connections[connection_id].get('peer_info')
                    if peer_info:
                        peer_info.latency = latency
                        peer_info.last_seen = time.time()
            
        except Exception as e:
            logger.error(f"Pong handling error: {e}")
    
    async def _handle_peer_list(self, connection_id: str, message: NetworkMessage):
        """Handle peer list message"""
        try:
            if message.payload.get("request"):
                # Send our peer list
                peers_to_share = self.network.peer_discovery.get_best_peers(10)
                peer_data = []
                
                for peer in peers_to_share:
                    peer_data.append({
                        "address": peer.address,
                        "port": peer.port,
                        "protocol": peer.protocol.name.lower(),
                        "capabilities": peer.capabilities,
                        "reputation": peer.reputation
                    })
                
                response = NetworkMessage(
                    message_id=f"peer_resp_{time.time()}",
                    message_type=MessageType.PEER_LIST,
                    payload={"peers": peer_data},
                    source_node=self.network.node_id
                )
                
                await self.send_message(connection_id, response)
                
            else:
                # Process received peer list
                peers = message.payload.get("peers", [])
                
                for peer_data in peers:
                    try:
                        peer_info = PeerInfo(
                            node_id="",  # Will be set during connection
                            address=peer_data["address"],
                            port=peer_data["port"],
                            protocol=ProtocolType[peer_data["protocol"].upper()],
                            version="1.0.0",  # Default
                            capabilities=peer_data.get("capabilities", []),
                            reputation=peer_data.get("reputation", 50),
                            last_seen=time.time()
                        )
                        
                        self.network.peer_discovery.add_peer(peer_info)
                        
                    except Exception as e:
                        logger.debug(f"Invalid peer data: {e}")
            
        except Exception as e:
            logger.error(f"Peer list handling error: {e}")
    
    async def _handle_handshake(self, connection_id: str, message: NetworkMessage):
        """Handle handshake message"""
        try:
            # Verify handshake and establish secure connection
            success = await self.network.security_manager.process_handshake(
                connection_id, message
            )
            
            if success:
                # Update connection state
                if connection_id in self.network.connections:
                    peer_info = self.network.connections[connection_id].get('peer_info')
                    if peer_info:
                        peer_info.state = ConnectionState.READY
                        peer_info.node_id = message.source_node
                        peer_info.capabilities = message.payload.get("capabilities", [])
                        peer_info.version = message.payload.get("version", "1.0.0")
            
        except Exception as e:
            logger.error(f"Handshake handling error: {e}")
            await self.network.penalize_peer(connection_id, -10)
    
    def register_handler(self, message_type: Any, handler: Callable):
        """Register message handler"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        
        self.handlers[message_type].append(handler)
        logger.debug(f"Registered handler for {message_type.name}")
    
    def unregister_handler(self, message_type: Any, handler: Callable):
        """Unregister message handler"""
        if message_type in self.handlers and handler in self.handlers[message_type]:
            self.handlers[message_type].remove(handler)
            logger.debug(f"Unregistered handler for {message_type.name}")
    
    async def send_message(self, connection_id: str, message: NetworkMessage) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.network.connections:
            logger.warning(f"Connection {connection_id} not found for message sending")
            return False
        
        try:
            connection = self.network.connections[connection_id]
            protocol = connection.get('protocol')
            
            if protocol == 'tcp':
                return await self.network.tcp_handler.send_message(connection_id, message)
            elif protocol == 'udp':
                # For UDP, we need the address info
                peer_info = connection.get('peer_info')
                if peer_info:
                    # Serialize and send via UDP
                    serialized = self.network.utils.serialization.serialize_message(message)
                    self.network.udp_handler.sendto(serialized, (peer_info.address, peer_info.port))
                    return True
                return False
            elif protocol == 'websocket':
                return await self.network.websocket_handler.send_message(connection_id, message)
            elif protocol == 'http':
                # HTTP requires the URL
                peer_info = connection.get('peer_info')
                if peer_info:
                    protocol_str = "https" if self.network.config.enable_encryption else "http"
                    url = f"{protocol_str}://{peer_info.address}:{peer_info.port}/message"
                    return await self.network.http_handler.send_message(url, message)
                return False
            else:
                logger.error(f"Unknown protocol: {protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            return False
    
    async def broadcast_message(self, message: NetworkMessage, exclude: list = None):
        """Broadcast message to all connected peers"""
        exclude = exclude or []
        
        for connection_id in list(self.network.connections.keys()):
            if connection_id not in exclude:
                try:
                    await self.send_message(connection_id, message)
                except Exception as e:
                    logger.debug(f"Failed to broadcast to {connection_id}: {e}")