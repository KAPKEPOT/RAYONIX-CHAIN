import time
import logging
from typing import Dict, Deque
from collections import deque
from ..models.connection_metrics import ConnectionMetrics

logger = logging.getLogger("MetricsCollector")

class MetricsCollector:
    """Metrics collection and reporting"""
    
    def __init__(self):
        self.metrics = ConnectionMetrics()
        self.connection_metrics: Dict[str, ConnectionMetrics] = {}
    
    def update_connection_metrics(self, connection_id: str, bytes_sent: int = 0, 
                                bytes_received: int = 0, messages_sent: int = 0,
                                messages_received: int = 0, latency: float = None):
        """Update connection metrics"""
        if connection_id not in self.connection_metrics:
            self.connection_metrics[connection_id] = ConnectionMetrics()
        
        metrics = self.connection_metrics[connection_id]
        metrics.bytes_sent += bytes_sent
        metrics.bytes_received += bytes_received
        metrics.messages_sent += messages_sent
        metrics.messages_received += messages_received
        metrics.last_activity = time.time()
        
        if latency is not None:
            metrics.latency_history.append(latency)
        
        # Update global metrics
        self.metrics.bytes_sent += bytes_sent
        self.metrics.bytes_received += bytes_received
        self.metrics.messages_sent += messages_sent
        self.metrics.messages_received += messages_received
    
    def get_connection_metrics(self, connection_id: str) -> ConnectionMetrics:
        """Get connection metrics"""
        return self.connection_metrics.get(connection_id, ConnectionMetrics())
    
    def get_global_metrics(self) -> ConnectionMetrics:
        """Get global metrics"""
        return self.metrics
    
    def log_metrics(self, total_connections: int, total_peers: int, banned_peers: int):
        """Log metrics"""
        logger.info(
            f"Network Metrics: Connections={total_connections}, "
            f"Peers={total_peers}, Banned={banned_peers}, "
            f"Sent={self.metrics.bytes_sent} bytes, Received={self.metrics.bytes_received} bytes, "
            f"Messages Sent={self.metrics.messages_sent}, Received={self.metrics.messages_received}"
        )
    
    def remove_connection_metrics(self, connection_id: str):
        """Remove connection metrics"""
        if connection_id in self.connection_metrics:
            del self.connection_metrics[connection_id]