import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from network.interfaces.connection_interface import IConnectionManager
from network.exceptions import ConnectionError
from network.models.peer_info import PeerInfo
from network.config.network_types import ConnectionState, ProtocolType

logger = logging.getLogger("ConnectionManager")

class ConnectionManager(IConnectionManager):
    """Connection management implementation"""
    
    def __init__(self, network):
        self.network = network
        self.connections: Dict[str, Dict[str, Any]] = {}
    
    async def start(self):
        """Start connection manager"""
        logger.info("Connection manager started")
    
    async def stop(self):
        """Stop connection manager"""
        await self.close_all_connections()
        logger.info("Connection manager stopped")
    
    async def connect_to_peer(self, address: str, port: int, protocol: str) -> Optional[str]:
        """Connect to a peer"""
        connection_id = f"{protocol}_{address}_{port}"
        
        # Check if already connected
        if connection_id in self.connections:
            logger.debug(f"Already connected to {connection_id}")
            return connection_id
        
        # Check if peer is banned
        if await self.network.ban_manager.is_peer_banned(address):
            logger.warning(f"Cannot connect to banned peer: {address}")
            return None
        
        try:
            # Create peer info
            peer_info = PeerInfo(
                node_id="",  # Will be set during handshake
                address=address,
                port=port,
                protocol=ProtocolType[protocol.upper()],
                version="1.0.0",
                capabilities=[],
                state=ConnectionState.CONNECTING
            )
            
            # Store connection
            self.connections[connection_id] = {
                'peer_info': peer_info,
                'protocol': protocol,
                'created_at': time.time(),
                'last_activity': time.time()
            }
            
            # Connect using appropriate protocol handler
            if protocol == 'tcp':
                await self._connect_tcp(connection_id, address, port)
            elif protocol == 'udp':
                # UDP is connectionless, just store the address
                pass
            elif protocol == 'websocket':
                await self._connect_websocket(connection_id, address, port)
            elif protocol == 'http':
                # HTTP is connectionless
                pass
            else:
                raise ConnectionError(f"Unsupported protocol: {protocol}")
            
            # Update connection state
            self.connections[connection_id]['peer_info'].state = ConnectionState.CONNECTED
            self.connections[connection_id]['last_activity'] = time.time()
            
            logger.info(f"Connected to {connection_id}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            if connection_id in self.connections:
                del self.connections[connection_id]
            return None
    
    async def _connect_tcp(self, connection_id: str, address: str, port: int):
        """Connect via TCP"""
        try:
            reader, writer = await asyncio.open_connection(
                address, port,
                ssl=self.network.ssl_context if self.network.config.enable_encryption else None
            )
            
            self.connections[connection_id]['reader'] = reader
            self.connections[connection_id]['writer'] = writer
            
            # Start message processing
            asyncio.create_task(self.network.tcp_handler.process_messages(connection_id))
            
        except Exception as e:
            raise ConnectionError(f"TCP connection failed: {e}")
    
    async def _connect_websocket(self, connection_id: str, address: str, port: int):
        """Connect via WebSocket"""
        try:
            import websockets
            
            uri = f"ws://{address}:{port}"
            if self.network.config.enable_encryption:
                uri = f"wss://{address}:{port}"
            
            websocket = await websockets.connect(uri)
            self.connections[connection_id]['websocket'] = websocket
            
            # Start message processing
            asyncio.create_task(
                self.network.websocket_handler.process_messages(connection_id, websocket)
            )
            
        except Exception as e:
            raise ConnectionError(f"WebSocket connection failed: {e}")
    
    async def close_connection(self, connection_id: str):
        """Close a connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            protocol = connection.get('protocol')
            
            if protocol == 'tcp':
                await self.network.tcp_handler.close_connection(connection_id)
            elif protocol == 'websocket':
                await self.network.websocket_handler.close_connection(connection_id)
            else:
                # For UDP and HTTP, just remove from connections
                if connection_id in self.connections:
                    del self.connections[connection_id]
                self.network.metrics_collector.remove_connection_metrics(connection_id)
                self.network.rate_limiter.remove_connection(connection_id)
                
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
    
    async def close_all_connections(self):
        """Close all connections"""
        for connection_id in list(self.connections.keys()):
            await self.close_connection(connection_id)
    
    def is_connected(self) -> bool:
        """Check if network has active connections"""
        return len(self.connections) > 0
    
    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection details"""
        return self.connections.get(connection_id)
    
    async def check_connection_health(self):
        """Check health of all connections"""
        current_time = time.time()
        dead_connections = []
        
        for connection_id, connection in self.connections.items():
            # Check for stale connections
            if current_time - connection['last_activity'] > self.network.config.connection_timeout:
                logger.warning(f"Connection {connection_id} is stale, closing")
                dead_connections.append(connection_id)
                continue
            
            # Check if connection is still alive
            protocol = connection.get('protocol')
            if protocol == 'tcp':
                writer = connection.get('writer')
                if writer and writer.is_closing():
                    dead_connections.append(connection_id)
            elif protocol == 'websocket':
                websocket = connection.get('websocket')
                if websocket and websocket.closed:
                    dead_connections.append(connection_id)
        
        # Close dead connections
        for connection_id in dead_connections:
            await self.close_connection(connection_id)
    
    def update_connection_activity(self, connection_id: str):
        """Update connection activity timestamp"""
        if connection_id in self.connections:
            self.connections[connection_id]['last_activity'] = time.time()