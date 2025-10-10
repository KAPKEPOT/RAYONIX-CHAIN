import asyncio
import logging
import ssl
import time
from typing import Dict, List, Optional, Any
from network.config.node_config import NodeConfig
from network.config.network_types import NetworkType, ProtocolType, ConnectionState, MessageType
from network.models.peer_info import PeerInfo
from network.models.network_message import NetworkMessage
from network.exceptions import NetworkError, ConnectionError
from network.core.connection_manager import ConnectionManager
from network.core.peer_discovery import PeerDiscovery
from network.core.message_processor import MessageProcessor
from network.core.security_manager import SecurityManager
from network.utils.rate_limiter import RateLimiter
from network.utils.ban_manager import BanManager
from network.utils.metrics_collector import MetricsCollector
from network.protocols.tcp_handler import TCPHandler
from network.protocols.udp_handler import UDPHandler
from network.protocols.websocket_handler import WebSocketHandler
from network.protocols.http_handler import HTTPHandler
from network.protocols.udp_protocol import UDPProtocol

logger = logging.getLogger("AdvancedP2PNetwork")

class AdvancedP2PNetwork:
    """Main P2P network class"""
    
    def __init__(self, network_id: int = 1, port: int = 30303, max_connections: int = 50, node_id: str = None, config: Optional[NodeConfig] = None):
        # Use provided parameters or create config from them
        if config is None:
            config = NodeConfig(
                network_type=NetworkType.TESTNET,
                listen_port=port,
                max_connections=max_connections,
                listen_ip="0.0.0.0",
                public_ip=None,
                public_port=None,
                max_peers=1000,
                connection_timeout=30,
                message_timeout=10,
                ping_interval=60,
                bootstrap_nodes=[],
                enable_nat_traversal=True,
                enable_encryption=False,  # Disable for initial testing
                enable_compression=True,
                enable_dht=True,
                enable_gossip=True,
                enable_syncing=True,
                max_message_size=10 * 1024 * 1024,  # 10MB
                rate_limit_per_peer=1000,
                ban_threshold=-100,
                ban_duration=3600,
                dht_bootstrap_nodes=[],
                dns_seeds=[]
            )
        
        self.config = config
        self.node_id = node_id or self._generate_node_id()
        self.magic = self._get_magic_number(network_id)  # Network magic number
        
        # Core components
        self.connection_manager = ConnectionManager(self)
        self.peer_discovery = PeerDiscovery(self)
        self.message_processor = MessageProcessor(self)
        self.security_manager = SecurityManager(self)
        
        # Utility components
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_peer)
        self.ban_manager = BanManager(self.config.ban_threshold, self.config.ban_duration)
        self.metrics_collector = MetricsCollector()
        
        # Protocol handlers
        self.ssl_context = self._create_ssl_context()
        self.tcp_handler = TCPHandler(self, self.config, self.ssl_context)
        self.udp_handler = UDPHandler(self, self.config)
        self.websocket_handler = WebSocketHandler(self, self.config, self.ssl_context)
        self.http_handler = HTTPHandler(self, self.config, self.ssl_context)
        self.udp_protocol = UDPProtocol(self.udp_handler)
        
        # State - Initialize with empty dicts using string keys
        self.is_running = False
        self.start_time = 0
        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, Any] = {}
        self.message_handlers: Dict[MessageType, List] = {}
        
        # Task references
        self.maintenance_task = None
        self.metrics_task = None
        
    def _get_magic_number(self, network_id: int) -> bytes:
        """Get magic number based on network ID"""
        magic_numbers = {
            1: b'RAYX',  # Mainnet
            2: b'RAYT',  # Testnet
            3: b'RAYD',  # Devnet
        }
        return magic_numbers.get(network_id, b'RAYX')
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context if encryption is enabled"""
        if not self.config.enable_encryption:
            return None
        
        try:
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            # In production, you would load proper certificates here
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        except Exception as e:
            logger.warning(f"SSL context creation failed: {e}")
            return None
    
    async def start(self):
        """Start the P2P network"""
        if self.is_running:
            logger.warning("Network is already running")
            return
        
        try:
            logger.info(f"Starting P2P network (Node ID: {self.node_id})")
            
            # Start protocol handlers
            await self.tcp_handler.start_server()
            await self.udp_handler.start_server()
            await self.websocket_handler.start_server()
            await self.http_handler.start_server()
            
            # Start core components
            await self.security_manager.initialize()
            await self.peer_discovery.bootstrap_network()
            
            # Start maintenance tasks
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            self.metrics_task = asyncio.create_task(self._metrics_loop())
            
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("P2P network started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start P2P network: {e}")
            await self.stop()
            raise NetworkError(f"Failed to start network: {e}")
    
    async def stop(self):
        """Stop the P2P network"""
        if not self.is_running:
            return
        
        logger.info("Stopping P2P network")
        
        # Cancel maintenance tasks
        if self.maintenance_task:
            self.maintenance_task.cancel()
        if self.metrics_task:
            self.metrics_task.cancel()
        
        # Stop protocol handlers
        await self.tcp_handler.stop_server()
        await self.udp_handler.stop_server()
        await self.websocket_handler.stop_server()
        await self.http_handler.stop_server()
        
        # Close all connections
        await self.connection_manager.close_all_connections()
        
        self.is_running = False
        logger.info("P2P network stopped")
    
    async def _maintenance_loop(self):
        """Maintenance loop for network operations"""
        while self.is_running:
            try:
                # Peer discovery and maintenance
                await self.peer_discovery.maintain_peer_list()
                await self.peer_discovery.request_peer_lists()
                
                # Clean up expired bans
                await self.ban_manager.cleanup_expired_bans()
                
                # Check connection health
                await self.connection_manager.check_connection_health()
                
                # Send periodic pings
                await self._send_pings()
                
                await asyncio.sleep(30)  # Run every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(30)
    
    async def _metrics_loop(self):
        """Metrics collection and reporting loop"""
        while self.is_running:
            try:
                # Log metrics periodically
                banned_count = len(self.ban_manager.banned_peers)
                self.metrics_collector.log_metrics(
                    len(self.connections),
                    len(self.peers),
                    banned_count
                )
                
                await asyncio.sleep(60)  # Log every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics loop error: {e}")
                await asyncio.sleep(60)
    
    async def _send_pings(self):
        """Send ping messages to connected peers"""
        for connection_id in list(self.connections.keys()):
            try:
                ping_message = NetworkMessage(
                    message_id=f"ping_{time.time()}",
                    message_type=MessageType.PING,
                    payload={"timestamp": time.time()},
                    source_node=self.node_id
                )
                
                await self.message_processor.send_message(connection_id, ping_message)
                
            except Exception as e:
                logger.debug(f"Failed to send ping to {connection_id}: {e}")
    
    async def penalize_peer(self, connection_id: str, penalty: int):
        """Penalize a peer (negative) or reward (positive)"""
        if connection_id not in self.connections:
            return
        
        peer_info = self.connections[connection_id].get('peer_info')
        if not peer_info:
            return
        
        peer_info.reputation += penalty
        
        # Check if peer should be banned
        if peer_info.reputation <= self.config.ban_threshold:
            logger.warning(f"Banning peer {connection_id} for low reputation")
            await self.ban_manager.ban_peer(peer_info.address)
            await self.connection_manager.close_connection(connection_id)
    
    def register_message_handler(self, message_type: MessageType, handler):
        """Register a message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for {message_type.name}")
    
    def unregister_message_handler(self, message_type: MessageType, handler):
        """Unregister a message handler"""
        if message_type in self.message_handlers and handler in self.message_handlers[message_type]:
            self.message_handlers[message_type].remove(handler)
            logger.debug(f"Unregistered handler for {message_type.name}")
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "node_id": self.node_id,
            "uptime": time.time() - self.start_time,
            "peers_count": len(self.peers),
            "connections_count": len(self.connections),
            "bytes_sent": self.metrics_collector.get_global_metrics().bytes_sent,
            "bytes_received": self.metrics_collector.get_global_metrics().bytes_received,
            "messages_sent": self.metrics_collector.get_global_metrics().messages_sent,
            "messages_received": self.metrics_collector.get_global_metrics().messages_received,
            "banned_peers": len(self.ban_manager.banned_peers),
            "is_running": self.is_running
        }
    
    async def broadcast_message(self, message: NetworkMessage, exclude: List[str] = None):
        """Broadcast message to all connected peers"""
        exclude = exclude or []
        
        for connection_id in list(self.connections.keys()):
            if connection_id not in exclude:
                try:
                    await self.message_processor.send_message(connection_id, message)
                except Exception as e:
                    logger.debug(f"Failed to broadcast to {connection_id}: {e}")
[file content end]