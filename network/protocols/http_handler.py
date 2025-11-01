import asyncio
import logging
from typing import Dict, Any
from network.interfaces.protocol_interface import IProtocolHandler
from network.utils.serialization import serialize_message, deserialize_message
from network.utils.compression import compress_data, decompress_data
from aiohttp import web, ClientSession

logger = logging.getLogger("HTTPHandler")

class HTTPHandler(IProtocolHandler):
    """HTTP protocol implementation"""
    
    def __init__(self, network, config, ssl_context):
        self.network = network
        self.config = config
        self.ssl_context = ssl_context
        self.app = web.Application()
        self.runner = None
        self.site = None
        self.http_session = None
        self.connections: Dict[str, Any] = {}
    
    async def start_server(self):
        """Start HTTP server"""
        try:
            # Setup routes
            self.app.router.add_post('/message', self.handle_message)
            self.app.router.add_get('/health', self.handle_health)
            
            # Start server
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(
                self.runner,
                self.config.listen_ip,
                self.config.http_port,
                ssl_context=self.ssl_context if self.config.enable_encryption else None
            )
            
            await self.site.start()
            
            # Create HTTP session for outgoing requests
            self.http_session = ClientSession()
            
            logger.info(f"HTTP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            return self.site
            
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            raise
    
    async def stop_server(self):
        """Stop HTTP server"""
        if self.runner:
            await self.runner.cleanup()
        
        if self.http_session:
            await self.http_session.close()
    
    async def handle_message(self, request):
        """Handle incoming HTTP message"""
        peer_addr = request.remote
        connection_id = f"http_{peer_addr}_{id(request)}"
        
        # Check if peer is banned
        if await self.network.ban_manager.is_peer_banned(peer_addr):
            logger.warning(f"Rejecting HTTP from banned peer: {peer_addr}")
            return web.Response(status=403, text="Peer banned")
        
        try:
            # Read message data
            data = await request.read()
            
            # Check message size
            if len(data) > self.config.max_message_size:
                logger.warning(f"Oversized message from {connection_id}")
                return web.Response(status=413, text="Message too large")
            
            # Check rate limit
            if not await self.network.rate_limiter.check_rate_limit(connection_id, len(data)):
                logger.warning(f"Rate limit exceeded for {connection_id}")
                return web.Response(status=429, text="Rate limit exceeded")
            
            # Decrypt if encryption enabled
            if self.config.enable_encryption:
                try:
                    data = self.network.security_manager.decrypt_data(data, connection_id)
                except Exception as e:
                    logger.error(f"Decryption failed for {connection_id}: {e}")
                    return web.Response(status=400, text="Decryption failed")
            
            # Decompress if compression enabled
            if self.config.enable_compression:
                try:
                    data = decompress_data(data)
                except Exception as e:
                    logger.error(f"Decompression failed for {connection_id}: {e}")
                    return web.Response(status=400, text="Decompression failed")
            
            # Deserialize message
            try:
                message = deserialize_message(data)
            except Exception as e:
                logger.error(f"Deserialization failed for {connection_id}: {e}")
                return web.Response(status=400, text="Deserialization failed")
            
            # Update metrics
            self.network.metrics_collector.update_connection_metrics(
                connection_id, bytes_received=len(data), messages_received=1
            )
            
            # Process message
            await self.network.message_processor.process_message(connection_id, message)
            
            return web.Response(status=200, text="OK")
            
        except Exception as e:
            logger.error(f"HTTP message handling error: {e}")
            return web.Response(status=500, text="Internal server error")
    
    async def handle_health(self, request):
        """Handle health check"""
        return web.Response(status=200, text="OK")
    
    async def send_message(self, connection_id: str, message: Any) -> bool:
        """Send message via HTTP"""
        # HTTP is connectionless, so connection_id should be a URL
        if not self.http_session:
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
            
            # Send POST request
            async with self.http_session.post(connection_id, data=serialized) as response:
                if response.status != 200:
                    logger.warning(f"HTTP send failed with status {response.status}")
                    return False
                
                # Update metrics
                self.network.metrics_collector.update_connection_metrics(
                    connection_id, bytes_sent=len(serialized), messages_sent=1
                )
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to send HTTP message to {connection_id}: {e}")
            return False
    
    async def handle_connection(self, *args, **kwargs):
        """HTTP doesn't maintain persistent connections like TCP/WebSocket"""
        pass