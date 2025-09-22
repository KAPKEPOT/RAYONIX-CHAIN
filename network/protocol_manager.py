import asyncio
import logging
import websockets
from typing import Dict, Any, Optional, Tuple
from asyncio import DatagramProtocol, Transport
import ssl

from .models import ProtocolType, ConnectionState
from .exceptions import ProtocolError
from .utils import NetworkUtils

logger = logging.getLogger("AdvancedNetwork")

class UDPProtocol:
    """UDP protocol handler"""
    
    def __init__(self, network):
        self.network = network
        self.transport = None
    
    def connection_made(self, transport: DatagramProtocol):
        self.transport = transport
    
    def datagram_received(self, data: bytes, addr: tuple):
        asyncio.create_task(self.network.protocol_manager.handle_udp_message(data, addr))
    
    def error_received(self, exc: Exception):
        logger.error(f"UDP error: {exc}")
    
    def connection_lost(self, exc: Optional[Exception]):
        logger.info("UDP connection closed")

class ProtocolManager:
    """Manages protocol-specific communication details"""
    
    def __init__(self, config, connection_manager):
        self.config = config
        self.connection_manager = connection_manager
        self.servers = {}
        self.udp_transport = None
        self.udp_protocol = None
    
    async def start_servers(self):
        """Start all server listeners"""
        server_tasks = [
            self.start_tcp_server(),
            self.start_udp_server(),
            self.start_websocket_server(),
            self.start_http_server()
        ]
        
        try:
            await asyncio.gather(*server_tasks)
            logger.info("All protocol servers started successfully")
        except Exception as e:
            logger.error(f"Failed to start servers: {e}")
            raise
    
    async def stop_servers(self):
        """Stop all server listeners"""
        for server_name, server in self.servers.items():
            try:
                server.close()
                await server.wait_closed()
                logger.info(f"Stopped {server_name} server")
            except Exception as e:
                logger.error(f"Error stopping {server_name} server: {e}")
        
        if self.udp_transport:
            self.udp_transport.close()
            logger.info("Stopped UDP server")
    
    async def start_tcp_server(self):
        """Start TCP server"""
        try:
            server = await asyncio.start_server(
                self.handle_tcp_connection,
                self.config.listen_ip,
                self.config.listen_port,
                reuse_address=True,
                reuse_port=True,
                ssl=self.connection_manager.ssl_context if self.config.enable_encryption else None,
                backlog=100  # Allow more pending connections
            )
            
            self.servers['tcp'] = server
            logger.info(f"TCP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            
        except Exception as e:
            logger.error(f"TCP server error: {e}")
            raise
    
    async def start_udp_server(self):
        """Start UDP server"""
        try:
            loop = asyncio.get_running_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: UDPProtocol(self),
                local_addr=(self.config.listen_ip, self.config.listen_port),
                reuse_port=True
            )
            
            self.udp_transport = transport
            self.udp_protocol = protocol
            
            logger.info(f"UDP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            
        except Exception as e:
            logger.error(f"UDP server error: {e}")
            raise
    
    async def start_websocket_server(self):
        """Start WebSocket server"""
        try:
            server = await websockets.serve(
                self.handle_websocket_connection,
                self.config.listen_ip,
                self.config.listen_port + 1,  # Different port for WS
                ssl=self.connection_manager.ssl_context if self.config.enable_encryption else None,
                ping_interval=None,  # We handle our own pings
                max_size=self.config.max_message_size,
                compression=None,  # We handle our own compression
                max_queue=1000  # Limit pending connections
            )
            
            self.servers['websocket'] = server
            logger.info(f"WebSocket server listening on {self.config.listen_ip}:{self.config.listen_port + 1}")
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
    
    async def start_http_server(self):
        """Start HTTP server"""
        try:
            server = await asyncio.start_server(
                self.handle_http_connection,
                self.config.listen_ip,
                self.config.listen_port + 2,  # Different port for HTTP
                reuse_address=True,
                reuse_port=True,
                ssl=self.connection_manager.ssl_context if self.config.enable_encryption else None,
                backlog=50  # Limit pending HTTP connections
            )
            
            self.servers['http'] = server
            logger.info(f"HTTP server listening on {self.config.listen_ip}:{self.config.listen_port + 2}")
            
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            raise
    
    async def handle_tcp_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        peer_addr = writer.get_extra_info('peername')
        if not peer_addr:
            return
            
        connection_id = f"tcp_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await network.peer_manager.is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting connection from banned peer: {peer_addr[0]}")
            writer.close()
            await writer.wait_closed()
            return
        
        try:
            # Perform handshake with timeout
            async with asyncio.timeout(self.config.connection_timeout):
                await network.security_manager.perform_handshake(reader, writer, connection_id, ProtocolType.TCP)
            
            # Add to connections
            network.connection_manager.add_connection(connection_id, {
                'reader': reader,
                'writer': writer,
                'protocol': ProtocolType.TCP,
                'metrics': ConnectionMetrics(),
                'address': peer_addr,
                'state': ConnectionState.READY,
                'last_activity': time.time()
            })
            
            # Start message processing
            asyncio.create_task(self.process_tcp_messages(connection_id))
            
            logger.info(f"TCP connection established with {peer_addr}")
            
        except asyncio.TimeoutError:
            logger.warning(f"TCP handshake timeout with {peer_addr}")
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"TCP connection error with {peer_addr}: {e}")
            writer.close()
            await writer.wait_closed()
    
    async def handle_udp_message(self, data: bytes, addr: tuple):
        """Handle incoming UDP datagram"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        connection_id = f"udp_{addr[0]}_{addr[1]}"
        
        # Check if peer is banned
        if await network.peer_manager.is_peer_banned(addr[0]):
            logger.warning(f"Rejecting UDP from banned peer: {addr[0]}")
            return
        
        try:
            # Parse message header
            header, payload = NetworkUtils.parse_message_header(data)
            
            # Verify magic number
            if header['magic'] != network.security_manager.magic:
                logger.warning(f"Invalid magic number from {addr}")
                return
            
            # Verify checksum
            expected_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            if header['checksum'] != expected_checksum:
                logger.warning(f"Invalid checksum from {addr}")
                return
            
            # Check message size
            if len(payload) > self.config.max_message_size:
                logger.warning(f"Oversized message from {addr}")
                return
            
            # For UDP, we don't maintain persistent connections, so we need to
            # handle the message directly without a connection context
            
            # Process message (simplified for UDP)
            try:
                # Decrypt if enabled
                if self.config.enable_encryption:
                    # For UDP, we might not have a session key, so we need to handle this
                    # In a real implementation, you'd need a way to establish session keys for UDP
                    logger.debug("UDP encryption not fully implemented")
                    return
                
                # Decompress if enabled
                if self.config.enable_compression:
                    from .utils import CompressionUtils
                    payload = CompressionUtils.decompress_data(payload)
                
                # Deserialize message
                message = network.message_manager.deserialize_message(payload)
                
                # Handle the message directly
                await network.message_manager.handle_message(connection_id, message)
                
            except Exception as e:
                logger.error(f"UDP message processing error: {e}")
            
        except Exception as e:
            logger.error(f"UDP handling error from {addr}: {e}")
    
    async def handle_websocket_connection(self, websocket, path: str):
        """Handle incoming WebSocket connection"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        peer_addr = websocket.remote_address
        if not peer_addr:
            return
            
        connection_id = f"ws_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await network.peer_manager.is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting WebSocket from banned peer: {peer_addr[0]}")
            await websocket.close()
            return
        
        try:
            # Perform handshake with timeout
            async with asyncio.timeout(self.config.connection_timeout):
                await network.security_manager.perform_websocket_handshake(websocket, connection_id)
            
            # Add to connections
            network.connection_manager.add_connection(connection_id, {
                'websocket': websocket,
                'protocol': ProtocolType.WEBSOCKET,
                'metrics': ConnectionMetrics(),
                'address': peer_addr,
                'state': ConnectionState.READY,
                'last_activity': time.time()
            })
            
            # Start message processing
            asyncio.create_task(self.process_websocket_messages(connection_id))
            
            logger.info(f"WebSocket connection established with {peer_addr}")
            
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket handshake timeout with {peer_addr}")
            await websocket.close()
        except Exception as e:
            logger.error(f"WebSocket connection error with {peer_addr}: {e}")
            await websocket.close()
    
    async def handle_http_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming HTTP connection"""
        try:
            request = await reader.read(8192)
            if not request:
                return
            
            # Parse HTTP request
            request_str = request.decode('utf-8')
            lines = request_str.split('\r\n')
            
            if not lines:
                response = self._create_http_response(400, "Bad Request")
                writer.write(response)
                await writer.drain()
                return
            
            # Parse request line
            request_line = lines[0].split()
            if len(request_line) < 3:
                response = self._create_http_response(400, "Bad Request")
                writer.write(response)
                await writer.drain()
                return
            
            method, path, version = request_line
            
            # Handle different endpoints
            if method == 'GET':
                if path == '/peers':
                    from .core import AdvancedP2PNetwork
                    network = AdvancedP2PNetwork.instance()
                    
                    peers_data = json.dumps([{
                        'node_id': peer.node_id,
                        'address': peer.address,
                        'port': peer.port,
                        'protocol': peer.protocol.name,
                        'reputation': peer.reputation,
                        'last_seen': peer.last_seen
                    } for peer in network.peer_manager.peers.values()])
                    response = self._create_http_response(200, "OK", peers_data, 'application/json')
                elif path == '/status':
                    from .core import AdvancedP2PNetwork
                    network = AdvancedP2PNetwork.instance()
                    
                    status = {
                        'node_id': network.config.node_id,
                        'connections': len(network.connection_manager.connections),
                        'peers': len(network.peer_manager.peers),
                        'status': 'running',
                        'network': network.config.network_type.name,
                        'version': '1.0'
                    }
                    response = self._create_http_response(200, "OK", json.dumps(status), 'application/json')
                elif path == '/health':
                    response = self._create_http_response(200, "OK", '{"status": "healthy"}', 'application/json')
                else:
                    response = self._create_http_response(404, "Not Found")
            else:
                response = self._create_http_response(405, "Method Not Allowed")
            
            writer.write(response)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")
            response = self._create_http_response(500, "Internal Server Error")
            writer.write(response)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
    
    def _create_http_response(self, status_code: int, status_message: str, 
                             content: str = "", content_type: str = "text/plain") -> bytes:
        """Create HTTP response"""
        response = f"HTTP/1.1 {status_code} {status_message}\r\n"
        response += f"Content-Type: {content_type}\r\n"
        response += f"Content-Length: {len(content)}\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"
        response += content
        
        return response.encode('utf-8')
    
    async def process_tcp_messages(self, connection_id: str):
        """Process incoming messages for a TCP connection"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        if connection_id not in network.connection_manager.connections:
            return
            
        connection = network.connection_manager.connections[connection_id]
        reader = connection['reader']
        
        try:
            while network.running and connection_id in network.connection_manager.connections:
                # Read message with header
                data = await NetworkUtils.receive_data(reader)
                if not data:
                    break
                
                # Process message
                success = await network.message_manager.process_message(
                    connection_id, data, network.security_manager, connection['metrics']
                )
                
                if not success:
                    await network.peer_manager.update_peer_reputation(connection_id, -10)
                
        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by peer: {connection_id}")
        except Exception as e:
            logger.error(f"Message processing error for {connection_id}: {e}")
        finally:
            if connection_id in network.connection_manager.connections:
                await network.connection_manager.close_connection(connection_id)
    
    async def process_websocket_messages(self, connection_id: str):
        """Process WebSocket messages"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        if connection_id not in network.connection_manager.connections:
            return
            
        connection = network.connection_manager.connections[connection_id]
        websocket = connection['websocket']
        
        try:
            async for message_data in websocket:
                if not network.running or connection_id not in network.connection_manager.connections:
                    break
                
                # Process message
                success = await network.message_manager.process_message(
                    connection_id, message_data, network.security_manager, connection['metrics']
                )
                
                if not success:
                    await network.peer_manager.update_peer_reputation(connection_id, -10)
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket message processing error for {connection_id}: {e}")
        finally:
            if connection_id in network.connection_manager.connections:
                await network.connection_manager.close_connection(connection_id)
    
    async def create_websocket_connection(self, uri: str, ssl_context=None):
        """Create a WebSocket connection"""
        return await websockets.connect(
            uri, 
            ssl=ssl_context, 
            ping_interval=None,
            max_size=self.config.max_message_size,
            compression=None
        )
    
    def get_server_stats(self) -> Dict:
        """Get server statistics"""
        stats = {
            'tcp_connections': 0,
            'websocket_connections': 0,
            'udp_active': self.udp_transport is not None,
            'servers_running': len(self.servers)
        }
        
        for conn in self.connection_manager.connections.values():
            if conn['protocol'] == ProtocolType.TCP:
                stats['tcp_connections'] += 1
            elif conn['protocol'] == ProtocolType.WEBSOCKET:
                stats['websocket_connections'] += 1
        
        return stats