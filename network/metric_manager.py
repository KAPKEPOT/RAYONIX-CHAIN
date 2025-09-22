import asyncio
import time
import logging
from typing import Dict, Any
from collections import defaultdict

from .models import ConnectionMetrics
from .utils import NetworkUtils, TimeUtils

logger = logging.getLogger("AdvancedNetwork")

class MetricsManager:
    """Manages network metrics and rate limiting"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = ConnectionMetrics()
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # connection_id -> rate limit data
        self.metrics_history: Dict[str, List] = {
            'connections': [],
            'throughput': [],
            'latency': [],
            'errors': []
        }
        self.history_size = 1000  # Keep last 1000 data points
    
    async def collect_metrics(self):
        """Collect and log network metrics"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        while network.running:
            try:
                total_connections = len(network.connection_manager.connections)
                total_peers = len(network.peer_manager.peers)
                banned_peers = len([p for p in network.peer_manager.banned_peers.values() if p > time.time()])
                
                # Calculate overall metrics
                total_bytes_sent = sum(
                    c['metrics'].bytes_sent for c in network.connection_manager.connections.values()
                )
                total_bytes_received = sum(
                    c['metrics'].bytes_received for c in network.connection_manager.connections.values()
                )
                total_messages_sent = sum(
                    c['metrics'].messages_sent for c in network.connection_manager.connections.values()
                )
                total_messages_received = sum(
                    c['metrics'].messages_received for c in network.connection_manager.connections.values()
                )
                
                # Calculate rates (per second)
                current_time = time.time()
                time_diff = current_time - self.metrics.last_activity
                
                if time_diff > 0:
                    send_rate = total_bytes_sent - self.metrics.bytes_sent / time_diff
                    receive_rate = total_bytes_received - self.metrics.bytes_received / time_diff
                    message_send_rate = total_messages_sent - self.metrics.messages_sent / time_diff
                    message_receive_rate = total_messages_received - self.metrics.messages_received / time_diff
                else:
                    send_rate = receive_rate = message_send_rate = message_receive_rate = 0
                
                # Update history
                self._update_history('connections', total_connections)
                self._update_history('throughput', (send_rate, receive_rate))
                
                # Update metrics
                self.metrics.bytes_sent = total_bytes_sent
                self.metrics.bytes_received = total_bytes_received
                self.metrics.messages_sent = total_messages_sent
                self.metrics.messages_received = total_messages_received
                self.metrics.last_activity = current_time
                
                logger.info(
                    f"Network Metrics: Connections={total_connections}, "
                    f"Peers={total_peers}, Banned={banned_peers}, "
                    f"Send Rate={send_rate/1024:.1f}KB/s, Receive Rate={receive_rate/1024:.1f}KB/s, "
                    f"Messages Sent={message_send_rate:.1f}/s, Received={message_receive_rate:.1f}/s"
                )
                
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(300)
    
    async def manage_rate_limits(self):
        """Manage rate limiting"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        while network.running:
            try:
                current_time = time.time()
                
                # Reset rate limits periodically
                for connection_id, limit_data in list(self.rate_limits.items()):
                    if current_time - limit_data['last_reset'] >= 60:  # 1 minute
                        limit_data['message_count'] = 0
                        limit_data['bytes_sent'] = 0
                        limit_data['bytes_received'] = 0
                        limit_data['last_reset'] = current_time
                        limit_data['penalty_level'] = max(0, limit_data['penalty_level'] - 1)
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Rate limiter error: {e}")
                await asyncio.sleep(5)
    
    async def check_rate_limit(self, connection_id: str, message_size: int) -> bool:
        """Check if rate limit is exceeded"""
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = {
                'message_count': 0,
                'last_reset': time.time(),
                'bytes_sent': 0,
                'bytes_received': 0,
                'last_penalty': 0,
                'penalty_level': 0
            }
            
        limit_data = self.rate_limits[connection_id]
        
        # Apply penalty if needed
        penalty_factor = 1 + (limit_data['penalty_level'] * 0.5)  # 50% reduction per penalty level
        effective_limit = self.config.rate_limit_per_peer / penalty_factor
        
        # Check message count limit
        if limit_data['message_count'] >= effective_limit:
            # Apply penalty
            if time.time() - limit_data['last_penalty'] > 60:
                limit_data['penalty_level'] += 1
                limit_data['last_penalty'] = time.time()
                logger.warning(f"Rate limit penalty for {connection_id}: level {limit_data['penalty_level']}")
            return False
        
        # Check bandwidth limit (optional)
        bandwidth_limit = self.config.rate_limit_per_peer * 1024  # 1KB per message average
        if limit_data['bytes_received'] + message_size > bandwidth_limit:
            return False
        
        # Update counters
        limit_data['message_count'] += 1
        limit_data['bytes_received'] += message_size
        
        return True
    
    def update_sent_metrics(self, connection_id: str, message_size: int):
        """Update metrics for sent messages"""
        if connection_id in self.rate_limits:
            self.rate_limits[connection_id]['bytes_sent'] += message_size
    
    async def manage_bans(self):
        """Manage banned peers"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        while network.running:
            try:
                current_time = time.time()
                
                # Remove expired bans
                expired_bans = [
                    peer for peer, ban_until in network.peer_manager.banned_peers.items()
                    if ban_until <= current_time
                ]
                
                for peer in expired_bans:
                    del network.peer_manager.banned_peers[peer]
                    logger.info(f"Ban expired for peer: {peer}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Ban manager error: {e}")
                await asyncio.sleep(300)
    
    def _update_history(self, metric_type: str, value: Any):
        """Update metrics history"""
        if metric_type in self.metrics_history:
            self.metrics_history[metric_type].append((time.time(), value))
            # Keep history size limited
            if len(self.metrics_history[metric_type]) > self.history_size:
                self.metrics_history[metric_type].pop(0)
    
    def get_metrics_history(self, metric_type: str, limit: int = 100) -> List:
        """Get metrics history for a specific type"""
        if metric_type in self.metrics_history:
            return self.metrics_history[metric_type][-limit:]
        return []
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics snapshot"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        return {
            'connections': len(network.connection_manager.connections),
            'peers': len(network.peer_manager.peers),
            'banned_peers': len([p for p in network.peer_manager.banned_peers.values() if p > time.time()]),
            'total_bytes_sent': self.metrics.bytes_sent,
            'total_bytes_received': self.metrics.bytes_received,
            'total_messages_sent': self.metrics.messages_sent,
            'total_messages_received': self.metrics.messages_received,
            'timestamp': time.time()
        }
    
    async def generate_report(self) -> Dict:
        """Generate a comprehensive metrics report"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        connection_stats = network.connection_manager.get_connection_stats()
        peer_stats = network.peer_manager.get_peer_stats()
        server_stats = network.protocol_manager.get_server_stats()
        message_stats = network.message_manager.get_processing_stats()
        
        return {
            'node_id': network.config.node_id,
            'network': network.config.network_type.name,
            'uptime': time.time() - network.start_time,
            'timestamp': time.time(),
            'connections': connection_stats,
            'peers': peer_stats,
            'servers': server_stats,
            'messages': message_stats,
            'rate_limits': {
                'tracked_connections': len(self.rate_limits),
                'total_penalties': sum(data['penalty_level'] for data in self.rate_limits.values())
            }
        }