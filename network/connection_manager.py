import asyncio
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
import ssl

from .models import ProtocolType, ConnectionState, ConnectionMetrics, PeerInfo
from .exceptions import ConnectionError, RateLimitError
from .protocol_manager import ProtocolManager
from .security_manager import SecurityManager
from .utils import NetworkUtils

logger = logging.getLogger("AdvancedNetwork")

class ConnectionManager:
    """Manages connections and peer health"""
    
    def __init__(self, config):
        self.config = config
        self.connections: Dict[str, Any] = {}
        self.ssl_context = self._create_ssl_context()
        self.protocol_manager = ProtocolManager(config, self)
        self.security_manager = SecurityManager(config)
        self.connection_attempts: Dict[str, int] = {}  # Track connection attempts by address
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for secure connections"""
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Set more secure options
        ssl_context.options |= ssl.OP_NO_SSLv2
        ssl_context.options |= ssl.OP_NO_SSLv3
        ssl_context.options |= ssl.OP_NO_TLSv1
        ssl_context.options |= ssl.OP_NO_TLSv1_1
        ssl_context.options |= ssl.OP_NO_COMPRESSION
        
        # Set preferred ciphers
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return ssl_context
    
    async def start_servers(self):
        """Start all server listeners"""
        await self.protocol_manager.start_servers()
    
    async def stop_servers(self):
        """Stop all server listeners"""
        await self.protocol_manager.stop_servers()
    
    def add_connection(self, connection_id: str, connection_data: Dict):
        """Add a new connection to the manager"""
        self.connections[connection_id] = connection_data
    
    def get_connection(self, connection_id: str) -> Optional[Dict]:
        """Get connection data by ID"""
        return self.connections.get(connection_id)
    
    def remove_connection(self, connection_id: str):
        """Remove a connection from the manager"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # Remove session key
        self.security_manager.remove_session_key(connection_id)
    
    async def close_connection(self, connection_id: str):
        """Close a connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        
        try:
            if connection['protocol'] == ProtocolType.TCP:
                writer = connection.get('writer')
                if writer:
                    writer.close()
                    await writer.wait_closed()
            elif connection['protocol'] == ProtocolType.WEBSOCKET:
                websocket = connection.get('websocket')
                if websocket:
                    await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing connection {connection_id}: {e}")
        finally:
            self.remove_connection(connection_id)
            logger.debug(f"Connection {connection_id} closed")
    
    async def connect_to_peer(self, address: str, port: int, 
                             protocol: ProtocolType = ProtocolType.TCP) -> Optional[str]:
        """Connect to a peer"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        # Check connection attempts
        attempt_key = f"{address}:{port}"
        self.connection_attempts[attempt_key] = self.connection_attempts.get(attempt_key, 0) + 1
        
        if self.connection_attempts[attempt_key] > 3:
            logger.warning(f"Too many connection attempts to {address}:{port}")
            return None
        
        try:
            connection_id = f"{protocol.name.lower()}_{address}_{port}"
            
            # Check if already connected
            if connection_id in self.connections:
                return connection_id
            
            # Check if banned
            if await network.peer_manager.is_peer_banned(address):
                logger.warning(f"Cannot connect to banned peer: {address}")
                return None
            
            # Connect based on protocol
            if protocol == ProtocolType.TCP:
                # Set connection timeout
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(
                            address, port,
                            ssl=self.ssl_context if self.config.enable_encryption else None
                        ),
                        timeout=self.config.connection_timeout
                    )
                    
                    # Perform handshake
                    await self.security_manager.perform_handshake(reader, writer, connection_id, protocol)
                    
                    # Store connection
                    self.add_connection(connection_id, {
                        'reader': reader,
                        'writer': writer,
                        'protocol': protocol,
                        'metrics': ConnectionMetrics(),
                        'address': (address, port),
                        'state': ConnectionState.READY,
                        'last_activity': time.time()
                    })
                    
                    # Start message processing
                    asyncio.create_task(self.protocol_manager.process_tcp_messages(connection_id))
                    
                except asyncio.TimeoutError:
                    raise ConnectionError(f"Connection timeout to {address}:{port}")
                
            elif protocol == ProtocolType.WEBSOCKET:
                ssl_context = self.ssl_context if self.config.enable_encryption else None
                websocket = await self.protocol_manager.create_websocket_connection(
                    f"ws://{address}:{port}", ssl_context
                )
                
                # Perform handshake
                await self.security_manager.perform_websocket_handshake(websocket, connection_id)
                
                # Store connection
                self.add_connection(connection_id, {
                    'websocket': websocket,
                    'protocol': protocol,
                    'metrics': ConnectionMetrics(),
                    'address': (address, port),
                    'state': ConnectionState.READY,
                    'last_activity': time.time()
                })
                
                # Start message processing
                asyncio.create_task(self.protocol_manager.process_websocket_messages(connection_id))
            
            # Reset connection attempts on success
            self.connection_attempts.pop(attempt_key, None)
            
            logger.info(f"Connected to peer {address}:{port} via {protocol.name}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            await network.peer_manager.update_peer_reputation_by_address(address, -5)
            return None
    
    async def maintain_connections(self):
        """Maintain target number of connections"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        current_count = len(self.connections)
        target_count = min(self.config.max_connections, len(network.peer_manager.peers))
        
        if current_count < target_count // 2:
            # Need more connections
            await self.discover_and_connect_peers(target_count - current_count)
        elif current_count > target_count * 1.2:
            # Too many connections, close some
            await self.prune_connections(current_count - target_count)
    
    async def discover_and_connect_peers(self, count: int):
        """Discover and connect to new peers"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        try:
            # Get candidate peers sorted by reputation
            candidate_peers = sorted(
                network.peer_manager.peers.values(),
                key=lambda p: p.reputation,
                reverse=True
            )
            
            # Filter out already connected and banned peers
            connected_addresses = {
                conn['address'][0] for conn in self.connections.values()
            }
            
            candidates = [
                peer for peer in candidate_peers
                if peer.address not in connected_addresses
                and not await network.peer_manager.is_peer_banned(peer.address)
                and peer.next_attempt <= time.time()
            ]
            
            # Connect to top candidates
            connection_tasks = []
            for peer in candidates[:min(count, len(candidates))]:
                task = asyncio.create_task(
                    self.connect_to_peer(peer.address, peer.port, peer.protocol)
                )
                connection_tasks.append(task)
            
            if connection_tasks:
                results = await asyncio.gather(*connection_tasks, return_exceptions=True)
                successful = sum(1 for r in results if r is not None)
                logger.info(f"Connection attempt: {successful} successful, {len(results) - successful} failed")
                
        except Exception as e:
            logger.error(f"Peer discovery and connection error: {e}")
    
    async def prune_connections(self, count: int):
        """Prune excess connections"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        # Sort connections by reputation and activity
        connections_to_prune = sorted(
            self.connections.items(),
            key=lambda x: (
                network.peer_manager.get_peer_by_address(x[1]['address'][0]).reputation
                if network.peer_manager.get_peer_by_address(x[1]['address'][0]) 
                else 0,
                x[1]['metrics'].last_activity
            )
        )[:count]
        
        for connection_id, _ in connections_to_prune:
            await self.close_connection(connection_id)
    
    async def check_connection_health(self):
        """Check connection health and send periodic pings"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        current_time = time.time()
        for connection_id in list(self.connections.keys()):
            connection = self.connections[connection_id]
            
            # Check for stale connections
            if current_time - connection['metrics'].last_activity > self.config.connection_timeout * 2:
                logger.warning(f"Closing stale connection: {connection_id}")
                await self.close_connection(connection_id)
                continue
            
            # Check for unresponsive connections (no pong response)
            if (connection['metrics'].last_ping_time > 0 and 
                connection['metrics'].last_pong_time < connection['metrics'].last_ping_time and
                current_time - connection['metrics'].last_ping_time > self.config.ping_interval * 2):
                logger.warning(f"Closing unresponsive connection: {connection_id}")
                await self.close_connection(connection_id)
                continue
            
            # Send periodic ping
            if current_time - connection['metrics'].last_activity > self.config.ping_interval:
                ping_message = NetworkMessage(
                    message_id=NetworkUtils.generate_node_id(),
                    message_type=MessageType.PING,
                    payload={
                        'timestamp': current_time,
                        'node_id': network.config.node_id
                    }
                )
                
                connection['metrics'].last_ping_time = current_time
                await network.send_message(connection_id, ping_message)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        stats = {
            'total': len(self.connections),
            'by_protocol': defaultdict(int),
            'by_state': defaultdict(int),
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'total_messages_sent': 0,
            'total_messages_received': 0
        }
        
        for conn in self.connections.values():
            stats['by_protocol'][conn['protocol'].name] += 1
            stats['by_state'][conn['state'].name] += 1
            stats['total_bytes_sent'] += conn['metrics'].bytes_sent
            stats['total_bytes_received'] += conn['metrics'].bytes_received
            stats['total_messages_sent'] += conn['metrics'].messages_sent
            stats['total_messages_received'] += conn['metrics'].messages_received
        
        return stats
    
    async def cleanup_connection_attempts(self):
        """Clean up old connection attempts"""
        while True:
            await asyncio.sleep(300)  # Clean up every 5 minutes
            current_time = time.time()
            # Remove attempts older than 1 hour
            for key in list(self.connection_attempts.keys()):
                # We don't track timestamps for attempts, so we'll just clear periodically
                del self.connection_attempts[key]