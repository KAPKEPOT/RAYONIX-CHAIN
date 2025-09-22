import asyncio
import logging
from typing import Dict, Any
from network.interfaces.protocol_interface import IProtocolHandler
from network.utils.serialization import serialize_message, deserialize_message
from network.utils.compression import compress_data, decompress_data
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger("WebSocketHandler")

class WebSocketHandler(IProtocolHandler):
    """WebSocket protocol implementation"""
    
    def __init__(self, network, config, ssl_context):
        self.network = network
        self.config = config
        self.ssl_context = ssl_context
        self.server = None
        self.connections: Dict[str, WebSocketServerProtocol] = {}
    
    async def start_server(self):
        """Start WebSocket server"""
        try:
            self.server = await websockets.serve(
                self.handle_connection,
                self.config.listen_ip,
                self.config.listen_port,
                ssl=self.ssl_context if self.config.enable_encryption else None,
                max_size=self.config.max_message_size
            )
            
            logger.info(f"WebSocket server listening on {self.config.listen_ip}:{self.config.listen_port}")
            return self.server
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
    
    async def stop_server(self):
        """Stop WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection"""
        peer_addr = websocket.remote_address
        if not peer_addr:
            return
            
        connection_id = f"ws_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await self.network.ban_manager.is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting WebSocket from banned peer: {peer_addr[0]}")
            await websocket.close()
            return
        
        try:
            # Store connection
            self.connections[connection_id] = websocket
            
            # Start message processing
            await self.process_messages(connection_id, websocket)
            
            logger.info(f"WebSocket connection established with {peer_addr}")
            
        except Exception as e:
            logger.error(f"WebSocket connection error with {peer_addr}: {e}")
            await websocket.close()
    
    async def process_messages(self, connection_id: str, websocket: WebSocketServerProtocol):
        """Process incoming WebSocket messages"""
        try:
            async for message in websocket:
                if connection_id not in self.connections:
                    break
                
                # Check message size
                if len(message) > self.config.max_message_size:
                    logger.warning(f"Oversized message from {connection_id}")
                    await self.network.penalize_peer(connection_id, -20)
                    continue
                
                # Check rate limit
                if not await self.network.rate_limiter.check_rate_limit(connection_id, len(message)):
                    logger.warning(f"Rate limit exceeded for {connection_id}")
                    await self.network.penalize_peer(connection_id, -5)
                    continue
                
                # Decrypt if encryption enabled
                if self.config.enable_encryption:
                    try:
                        message = self.network.security_manager.decrypt_data(message, connection_id)
                    except Exception as e:
                        logger.error(f"Decryption failed for {connection_id}: {e}")
                        await self.network.penalize_peer(connection_id, -15)
                        continue
                
                # Decompress if compression enabled
                if self.config.enable_compression:
                    try:
                        message = decompress_data(message)
                    except Exception as e:
                        logger.error(f"Decompression failed for {connection_id}: {e}")
                        await self.network.penalize_peer(connection_id, -5)
                        continue
                
                # Deserialize message
                try:
                    message_obj = deserialize_message(message)
                except Exception as e:
                    logger.error(f"Deserialization failed for {connection_id}: {e}")
                    await self.network.penalize_peer(connection_id, -10)
                    continue
                
                # Update metrics
                self.network.metrics_collector.update_connection_metrics(
                    connection_id, bytes_received=len(message), messages_received=1
                )
                
                # Process message
                await self.network.message_processor.process_message(connection_id, message_obj)
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket message processing error for {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                await self.close_connection(connection_id)
    
    async def send_message(self, connection_id: str, message: Any) -> bool:
        """Send message via WebSocket"""
        if connection_id not in self.connections:
            logger.warning(f"WebSocket connection {connection_id} not found")
            return False
        
        try:
            # Serialize message
            serialized = serialize_message(message)
            
            # Compress if enabled
            if self.config.enable_compression:
                serialized = compress_data(serialized)
            
            # Encrypt if enabled
            if self.config.enable_encryption:
                serialized = self.network.security_manager.encrypt_data(serialized, connection_id)
            
            # Check rate limit
            if not await self.network.rate_limiter.check_rate_limit(connection_id, len(serialized)):
                logger.warning(f"Rate limit exceeded for sending to {connection_id}")
                return False
            
            # Send message
            websocket = self.connections[connection_id]
            await websocket.send(serialized)
            
            # Update metrics
            self.network.metrics_collector.update_connection_metrics(
                connection_id, bytes_sent=len(serialized), messages_sent=1
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send WebSocket message to {connection_id}: {e}")
            await self.close_connection(connection_id)
            return False
    
    async def close_connection(self, connection_id: str):
        """Close WebSocket connection"""
        if connection_id not in self.connections:
            return
            
        websocket = self.connections[connection_id]
        
        try:
            await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket connection {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
            self.network.metrics_collector.remove_connection_metrics(connection_id)
            self.network.rate_limiter.remove_connection(connection_id)
            
            logger.debug(f"WebSocket connection {connection_id} closed")