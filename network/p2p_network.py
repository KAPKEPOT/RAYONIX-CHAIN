import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any

from .models import NodeConfig, NetworkMessage, MessageType
from .connection_manager import ConnectionManager
from .peer_manager import PeerManager
from .security_manager import SecurityManager
from .message_manager import MessageManager
from .metrics_manager import MetricsManager
from .protocol_manager import ProtocolManager
from .exceptions import NetworkError, ConnectionError
from .dht import KademliaDHT

logger = logging.getLogger("AdvancedNetwork")

class AdvancedP2PNetwork:
    """Advanced P2P network implementation with multiple protocols and security"""
    
    _instance = None
    
    @classmethod
    def instance(cls):
        """Get the singleton instance"""
        return cls._instance
    
    def __init__(self, config: NodeConfig, node_id: Optional[str] = None):
        if AdvancedP2PNetwork._instance is not None:
            raise NetworkError("AdvancedP2PNetwork is a singleton class")
        
        AdvancedP2PNetwork._instance = self
        
        self.config = config
        self.config.node_id = node_id or NetworkUtils.generate_node_id()
        self.running = False
        self.start_time = time.time()
        
        # Initialize managers
        self.connection_manager = ConnectionManager(self.config)
        self.peer_manager = PeerManager(self.config)
        self.security_manager = SecurityManager(self.config)
        self.message_manager = MessageManager(self.config)
        self.metrics_manager = MetricsManager(self.config)
        self.protocol_manager = ProtocolManager(self.config, self.connection_manager)
        
        # Initialize DHT if enabled
        if config.enable_dht:
        	self.dht = KademliaDHT(config, self.config.node_id)
        else:
        	self.dht = None
        
        logger.info(f"Initialized P2P network with node ID: {self.config.node_id}")
    
    async def start(self):
        """Start the network node"""
        if self.running:
            raise NetworkError("Network is already running")
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # Start server listeners
            await self.connection_manager.start_servers()
            
            # Start background tasks
            background_tasks = [
                self._connection_manager_loop(),
                self._peer_discovery_loop(),
                self._metrics_collection_loop(),
                self._gossip_broadcast_loop(),
                self._rate_limiter_loop(),
                self._ban_management_loop(),
                self._cleanup_loops(),
                self._security_cleanup_loop()
            ]
            
            # Start priority message processors
            for priority in self.message_manager.priority_queues:
                asyncio.create_task(self.message_manager.priority_message_processor(priority))
            
            # Bootstrap to network
            await self.peer_manager.bootstrap_network()
            
            logger.info("Network started successfully")
            if self.dht:
            	asyncio.create_task(self.dht.bootstrap())
            	asyncio.create_task(self.dht.maintain_dht())
            
            # Run all tasks
            await asyncio.gather(*background_tasks)
            
        except asyncio.CancelledError:
            logger.info("Network shutdown requested")
        except Exception as e:
            logger.error(f"Failed to start network: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the network node"""
        if not self.running:
            return
        
        self.running = False
        
        try:
            # Close all connections
            for connection_id in list(self.connection_manager.connections.keys()):
                await self.connection_manager.close_connection(connection_id)
            
            # Stop servers
            await self.connection_manager.stop_servers()
            
            logger.info("Network stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during network shutdown: {e}")
    
    def is_connected(self) -> bool:
        """Check if network has active connections"""
        return len(self.connection_manager.connections) > 0
    
    async def send_message(self, connection_id: str, message: NetworkMessage, priority: int = 1) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.connection_manager.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        try:
            connection = self.connection_manager.connections[connection_id]
            protocol = connection['protocol']
            
            # Prepare message for sending
            message_data = await self.message_manager.prepare_message_for_sending(
                message, connection_id, self.security_manager
            )
            
            # Check rate limit
            if not await self.metrics_manager.check_rate_limit(connection_id, len(message_data)):
                logger.warning(f"Rate limit exceeded for sending to {connection_id}")
                return False
            
            # Send based on protocol
            if protocol == ProtocolType.TCP:
                writer = connection['writer']
                await NetworkUtils.send_data(writer, message_data)
            elif protocol == ProtocolType.WEBSOCKET:
                websocket = connection['websocket']
                await websocket.send(message_data)
            elif protocol == ProtocolType.UDP:
                addr = connection['address']
                if hasattr(self.protocol_manager, 'udp_transport'):
                    self.protocol_manager.udp_transport.sendto(message_data, addr)
            
            # Update metrics
            connection['metrics'].messages_sent += 1
            connection['metrics'].bytes_sent += len(message_data)
            connection['metrics'].last_activity = time.time()
            
            # Update rate limit metrics
            self.metrics_manager.update_sent_metrics(connection_id, len(message_data))
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.connection_manager.close_connection(connection_id)
            return False
    
    async def broadcast_message(self, message: NetworkMessage, exclude: List[str] = None, priority: int = 1):
        """Broadcast message to all connected peers"""
        exclude = exclude or []
        tasks = []
        
        for connection_id in list(self.connection_manager.connections.keys()):
            if connection_id not in exclude:
                task = asyncio.create_task(self.send_message(connection_id, message, priority))
                tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if r is True)
            logger.info(f"Broadcast complete: {successful} successful, {len(results) - successful} failed")
    
    async def rpc_call(self, connection_id: str, method: str, params: Any, timeout: int = 30) -> Any:
        """Make RPC call to peer"""
        if connection_id not in self.connection_manager.connections:
            raise ConnectionError("Connection not found")
        
        request_id = NetworkUtils.generate_node_id()
        future = asyncio.get_event_loop().create_future()
        self.message_manager.pending_requests[request_id] = future
        
        rpc_message = NetworkMessage(
            message_id=request_id,
            message_type=MessageType.RPC_REQUEST,
            payload={'method': method, 'params': params}
        )
        
        try:
            success = await self.send_message(connection_id, rpc_message)
            if not success:
                raise ConnectionError("Failed to send RPC request")
            
            # Wait for response with timeout
            return await asyncio.wait_for(future, timeout)
            
        except asyncio.TimeoutError:
            if request_id in self.message_manager.pending_requests:
                del self.message_manager.pending_requests[request_id]
            raise ConnectionError("RPC call timeout")
        except Exception as e:
            if request_id in self.message_manager.pending_requests:
                del self.message_manager.pending_requests[request_id]
            raise ConnectionError(f"RPC call failed: {e}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_manager.register_message_handler(message_type, handler)
        
    def register_dht_handlers(self):
    	if self.dht:
    		self.message_manager.register_message_handler(
    		    MessageType.DHT,
    		    self.dht.protocol.handle_dht_message 
    		)
    
    def unregister_message_handler(self, message_type: MessageType, handler: Callable):
        """Unregister message handler"""
        self.message_manager.unregister_message_handler(message_type, handler)
    
    async def _connection_manager_loop(self):
        """Connection management background task"""
        while self.running:
            try:
                # Check connection health
                await self.connection_manager.check_connection_health()
                
                # Maintain target number of connections
                await self.connection_manager.maintain_connections()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                await asyncio.sleep(10)
    
    async def _peer_discovery_loop(self):
        """Peer discovery background task"""
        while self.running:
            try:
                await self.peer_manager.discover_peers()
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Peer discovery error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collection_loop(self):
        """Metrics collection background task"""
        await self.metrics_manager.collect_metrics()
    
    async def _gossip_broadcast_loop(self):
        """Gossip broadcasting background task"""
        if not self.config.enable_gossip:
            return
            
        while self.running:
            try:
                # Create gossip message
                gossip_message = NetworkMessage(
                    message_id=NetworkUtils.generate_node_id(),
                    message_type=MessageType.GOSSIP,
                    payload={
                        'node_id': self.config.node_id,
                        'timestamp': time.time(),
                        'content': 'network_heartbeat',
                        'version': '1.0'
                    },
                    ttl=5  # Limit propagation
                )
                
                await self.broadcast_message(gossip_message, priority=0)
                
                await asyncio.sleep(30)  # Broadcast every 30 seconds
                
            except Exception as e:
                logger.error(f"Gossip broadcasting error: {e}")
                await asyncio.sleep(60)
    
    async def _rate_limiter_loop(self):
        """Rate limiting background task"""
        await self.metrics_manager.manage_rate_limits()
    
    async def _ban_management_loop(self):
        """Ban management background task"""
        await self.metrics_manager.manage_bans()
    
    async def _cleanup_loops(self):
        """Various cleanup tasks"""
        while self.running:
            try:
                # Clean up connection attempts
                if hasattr(self.connection_manager, 'cleanup_connection_attempts'):
                    await self.connection_manager.cleanup_connection_attempts()
                
                # Clean up old messages
                self.message_manager.reset_processing_stats()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(600)
    
    async def _security_cleanup_loop(self):
        """Security-related cleanup tasks"""
        while self.running:
            try:
                # Clean up used nonces
                if hasattr(self.security_manager, 'cleanup_used_nonces'):
                    await self.security_manager.cleanup_used_nonces()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Security cleanup error: {e}")
                await asyncio.sleep(600)
    
    def get_status(self) -> Dict:
        """Get current network status"""
        return {
            'running': self.running,
            'node_id': self.config.node_id,
            'network': self.config.network_type.name,
            'uptime': time.time() - self.start_time,
            'connections': len(self.connection_manager.connections),
            'peers': len(self.peer_manager.peers),
            'version': '1.0'
        }
    
    async def get_detailed_status(self) -> Dict:
        """Get detailed network status"""
        metrics = self.metrics_manager.get_current_metrics()
        connection_stats = self.connection_manager.get_connection_stats()
        peer_stats = self.peer_manager.get_peer_stats()
        
        return {
            **self.get_status(),
            'metrics': metrics,
            'connection_stats': connection_stats,
            'peer_stats': peer_stats,
            'timestamp': time.time()
        }