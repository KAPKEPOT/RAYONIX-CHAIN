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
        """Connect to a peer with resource limits"""
        current_connections = len(self.connections)
        if current_connections >= self.network.config.max_connections:
        	logger.warning(f"Connection limit reached ({current_connections})")
        	return None
        
        # Validate input parameters
        if not self._validate_address(address):
        	logger.error(f"Invalid address: {address}")
        	return None
        	
        if not (0 < port < 65536):
        	logger.error(f"Invalid port: {port}")
        	return None
        	
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
        	# Create peer info with timeout
        	peer_info = PeerInfo(
        	    node_id="",
        	    address=address,
        	    port=port,
        	    protocol=ProtocolType[protocol.upper()],
        	    version="1.0.0",
        	    capabilities=[],
        	    state=ConnectionState.CONNECTING
        	)
        	
        	# Store connection with timeout
        	self.connections[connection_id] = {
        	    'peer_info': peer_info,
        	    'protocol': protocol,
        	    'created_at': time.time(),
        	    'last_activity': time.time(),
        	    'connection_timeout': 30  # seconds
        	}
        	
        	# Connect with timeout
        	if protocol == 'tcp':
        		await asyncio.wait_for(
        		    
        		    self._connect_tcp(connection_id, address, port), 
        		    timeout=10.0
        		)
        	
        	elif protocol == 'websocket':
        		await asyncio.wait_for(
        		    self._connect_websocket(connection_id, address, port),
        		    timeout=10.0
        		)
        	
        	else:
        		raise ConnectionError(f"Unsupported protocol: {protocol}")
        		
        	# Update connection state
        	self.connections[connection_id]['peer_info'].state = ConnectionState.CONNECTED
        	self.connections[connection_id]['last_activity'] = time.time()
        	logger.info(f"Connected to {connection_id}")
        	return connection_id
        	
        except asyncio.TimeoutError:
        	logger.error(f"Connection timeout to {address}:{port}")
        	await self._cleanup_connection(connection_id)
        	return None
        	
        except Exception as e:
        	logger.error(f"Failed to connect to {address}:{port}: {e}")
        	await self._cleanup_connection(connection_id)
        	return None
        	
    async def _cleanup_connection(self, connection_id: str):
    	"""Safely cleanup connection resources"""
    	if connection_id in self.connections:
    		try:
    			connection = self.connections[connection_id]
    			protocol = connection.get('protocol')
    			
    			if protocol == 'tcp':
    				writer = connection.get('writer')
    				if writer and not writer.is_closing():
    					writer.close()
    					try:
    						await asyncio.wait_for(writer.wait_closed(), timeout=5.0)
    					except asyncio.TimeoutError:
    						pass
    						
    			elif protocol == 'websocket':
    				websocket = connection.get('websocket')
    				if websocket and not websocket.closed:
    					await websocket.close()
    			
    			del self.connections[connection_id]
    			
    		except Exception as e:
    			logger.debug(f"Error during connection cleanup: {e}")
    	
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
                if not hasattr(self.network, 'tcp_handler') or not self.network.tcp_handler:
                	logger.warning(f"TCP handler not available for connection {connection_id}")
                await self.network.tcp_handler.close_connection(connection_id)
                
            elif protocol == 'websocket':
                if not hasattr(self.network, 'websocket_handler') or not self.network.websocket_handler:
                	logger.warning(f"WebSocket handler not available for connection {connection_id}")
                await self.network.websocket_handler.close_connection(connection_id)
            else:
                raise ConnectionError(f"Unsupported protocol: {protocol}")
                
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
        finally:
            # Always clean up internal tracking
            if connection_id in self.connections:
                del self.connections[connection_id]
                
            # REQUIRE metrics collector
            self.network.metrics_collector.remove_connection_metrics(connection_id)
            # REQUIRE rate limiter  
            await self.network.rate_limiter.remove_connection(connection_id)
            
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
        
        # PROPER FIX: Require config, no fallbacks
        if not hasattr(self.network, 'config'):
        	raise ConnectionError("Network configuration not available")
        
        connection_timeout = self.network.config.connection_timeout
        
        for connection_id, connection in self.connections.items():
            # Check for stale connections
            if current_time - connection['last_activity'] > connection_timeout:
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
            
    def get_connected_peers_count(self) -> int:
        """Get number of connected peers"""
        return len(self.connections)
    
    def get_connection_ids(self) -> List[str]:
        """Get list of all connection IDs"""
        return list(self.connections.keys())
    
    async def get_connection_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed connection information"""
        if connection_id not in self.connections:
            return None
        
        connection = self.connections[connection_id]
        peer_info = connection.get('peer_info')
        
        if not peer_info:
            return None
        
        # Require PeerInfo objects
        if not hasattr(peer_info, 'address'):
        	raise ValueError(f"Invalid peer_info type for {connection_id}: expected PeerInfo object, got {type(peer_info)}")
        	       
        return {
            'connection_id': connection_id,
            'address': peer_info.address,
            'port': peer_info.port,
            'protocol': peer_info.protocol.name,
            'state': peer_info.state.name,
            'reputation': peer_info.reputation,
            'latency': peer_info.latency,
            'last_seen': peer_info.last_seen,
            'connection_time': time.time() - connection['created_at'],
            'last_activity': connection['last_activity']
        }            