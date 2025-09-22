"""
Metrics collection for consensus system monitoring
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import logging
from collections import defaultdict
import threading

from prometheus_client import Counter, Gauge, Histogram, start_http_server
from prometheus_client.core import REGISTRY

from ..exceptions import ConsensusError

logger = logging.getLogger('consensus.metrics')

@dataclass
class ConsensusMetrics:
    """Consensus-related metrics"""
    # Counters
    blocks_proposed: Counter = field(default_factory=lambda: Counter(
        'consensus_blocks_proposed', 'Number of blocks proposed'
    ))
    blocks_committed: Counter = field(default_factory=lambda: Counter(
        'consensus_blocks_committed', 'Number of blocks committed'
    ))
    blocks_failed: Counter = field(default_factory=lambda: Counter(
        'consensus_blocks_failed', 'Number of blocks that failed to commit'
    ))
    votes_sent: Counter = field(default_factory=lambda: Counter(
        'consensus_votes_sent', 'Number of votes sent', ['vote_type']
    ))
    votes_received: Counter = field(default_factory=lambda: Counter(
        'consensus_votes_received', 'Number of votes received', ['vote_type']
    ))
    timeouts: Counter = field(default_factory=lambda: Counter(
        'consensus_timeouts', 'Number of timeouts', ['timeout_type']
    ))
    rounds: Counter = field(default_factory=lambda: Counter(
        'consensus_rounds', 'Number of rounds started', ['height']
    ))
    
    # Gauges
    current_height: Gauge = field(default_factory=lambda: Gauge(
        'consensus_current_height', 'Current block height'
    ))
    current_round: Gauge = field(default_factory=lambda: Gauge(
        'consensus_current_round', 'Current consensus round'
    ))
    active_validators: Gauge = field(default_factory=lambda: Gauge(
        'consensus_active_validators', 'Number of active validators'
    ))
    total_stake: Gauge = field(default_factory=lambda: Gauge(
        'consensus_total_stake', 'Total staked amount'
    ))
    voting_power: Gauge = field(default_factory=lambda: Gauge(
        'consensus_voting_power', 'Voting power of this validator'
    ))
    peer_count: Gauge = field(default_factory=lambda: Gauge(
        'consensus_peer_count', 'Number of connected peers'
    ))
    queue_size: Gauge = field(default_factory=lambda: Gauge(
        'consensus_queue_size', 'Size of message queues', ['queue_type']
    ))
    
    # Histograms
    block_time: Histogram = field(default_factory=lambda: Histogram(
        'consensus_block_time', 'Time to commit blocks', ['height']
    ))
    round_time: Histogram = field(default_factory=lambda: Histogram(
        'consensus_round_time', 'Time spent in rounds', ['round_type']
    ))
    message_latency: Histogram = field(default_factory=lambda: Histogram(
        'consensus_message_latency', 'Message processing latency', ['message_type']
    ))
    validation_time: Histogram = field(default_factory=lambda: Histogram(
        'consensus_validation_time', 'Time spent validating messages'
    ))

class MetricsCollector:
    """Metrics collection and export"""
    
    def __init__(self, port: int = 9090, namespace: str = "consensus"):
        self.port = port
        self.namespace = namespace
        self.metrics = ConsensusMetrics()
        self.custom_metrics: Dict[str, Counter] = {}
        self.labels: Dict[str, Dict[str, str]] = {}
        
        self.started = False
        self.lock = threading.RLock()
        
        # Internal state
        self._last_block_time = 0.0
        self._round_start_times: Dict[tuple, float] = {}
        self._message_timestamps: Dict[str, float] = {}
    
    def start(self) -> None:
        """Start metrics server"""
        try:
            start_http_server(self.port)
            self.started = True
            logger.info(f"Metrics server started on port {self.port}")
        except Exception as e:
            raise ConsensusError(f"Failed to start metrics server: {e}")
    
    def stop(self) -> None:
        """Stop metrics server"""
        # Prometheus doesn't provide a stop method, but we can mark as stopped
        self.started = False
        logger.info("Metrics server stopped")
    
    def record_block_proposed(self, height: int) -> None:
        """Record block proposal"""
        with self.lock:
            self.metrics.blocks_proposed.inc()
            self.metrics.current_height.set(height)
    
    def record_block_committed(self, height: int, block_time: float) -> None:
        """Record block commitment"""
        with self.lock:
            self.metrics.blocks_committed.inc()
            self.metrics.block_time.labels(height=str(height)).observe(block_time)
            self.metrics.current_height.set(height)
    
    def record_block_failed(self, height: int) -> None:
        """Record block failure"""
        with self.lock:
            self.metrics.blocks_failed.inc()
    
    def record_vote_sent(self, vote_type: str) -> None:
        """Record vote sent"""
        with self.lock:
            self.metrics.votes_sent.labels(vote_type=vote_type).inc()
    
    def record_vote_received(self, vote_type: str) -> None:
        """Record vote received"""
        with self.lock:
            self.metrics.votes_received.labels(vote_type=vote_type).inc()
    
    def record_timeout(self, timeout_type: str) -> None:
        """Record timeout"""
        with self.lock:
            self.metrics.timeouts.labels(timeout_type=timeout_type).inc()
    
    def record_round_start(self, height: int, round_num: int) -> None:
        """Record round start"""
        with self.lock:
            self.metrics.rounds.labels(height=str(height)).inc()
            self.metrics.current_round.set(round_num)
            self._round_start_times[(height, round_num)] = time.time()
    
    def record_round_end(self, height: int, round_num: int, round_type: str) -> None:
        """Record round end and duration"""
        with self.lock:
            start_time = self._round_start_times.pop((height, round_num), None)
            if start_time:
                duration = time.time() - start_time
                self.metrics.round_time.labels(round_type=round_type).observe(duration)
    
    def record_validator_update(self, active_count: int, total_stake: int, voting_power: int) -> None:
        """Record validator set update"""
        with self.lock:
            self.metrics.active_validators.set(active_count)
            self.metrics.total_stake.set(total_stake)
            self.metrics.voting_power.set(voting_power)
    
    def record_peer_count(self, count: int) -> None:
        """Record peer count"""
        with self.lock:
            self.metrics.peer_count.set(count)
    
    def record_queue_size(self, queue_type: str, size: int) -> None:
        """Record queue size"""
        with self.lock:
            self.metrics.queue_size.labels(queue_type=queue_type).set(size)
    
    def record_message_latency(self, message_type: str, latency: float) -> None:
        """Record message processing latency"""
        with self.lock:
            self.metrics.message_latency.labels(message_type=message_type).observe(latency)
    
    def record_validation_time(self, duration: float) -> None:
        """Record validation time"""
        with self.lock:
            self.metrics.validation_time.observe(duration)
    
    def start_message_tracking(self, message_id: str) -> None:
        """Start tracking message processing time"""
        with self.lock:
            self._message_timestamps[message_id] = time.time()
    
    def end_message_tracking(self, message_id: str, message_type: str) -> None:
        """End message tracking and record latency"""
        with self.lock:
            start_time = self._message_timestamps.pop(message_id, None)
            if start_time:
                latency = time.time() - start_time
                self.record_message_latency(message_type, latency)
    
    def create_counter(self, name: str, description: str, labels: List[str] = None) -> Counter:
        """Create custom counter metric"""
        with self.lock:
            if name in self.custom_metrics:
                return self.custom_metrics[name]
            
            counter = Counter(name, description, labels or [])
            self.custom_metrics[name] = counter
            return counter
    
    def create_gauge(self, name: str, description: str, labels: List[str] = None) -> Gauge:
        """Create custom gauge metric"""
        with self.lock:
            if name in self.custom_metrics:
                return self.custom_metrics[name]
            
            gauge = Gauge(name, description, labels or [])
            self.custom_metrics[name] = gauge
            return gauge
    
    def create_histogram(self, name: str, description: str, labels: List[str] = None) -> Histogram:
        """Create custom histogram metric"""
        with self.lock:
            if name in self.custom_metrics:
                return self.custom_metrics[name]
            
            histogram = Histogram(name, description, labels or [])
            self.custom_metrics[name] = histogram
            return histogram
    
    def set_labels(self, labels: Dict[str, str]) -> None:
        """Set common labels for all metrics"""
        with self.lock:
            self.labels = labels
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics values"""
        # This would collect current values from all metrics
        # Implementation depends on specific requirements
        return {}
    
    def reset_metrics(self) -> None:
        """Reset all metrics (for testing)"""
        with self.lock:
            # Note: Prometheus counters cannot be reset in production
            # This is mainly for testing purposes
            for metric in REGISTRY.collect():
                if hasattr(metric, 'clear'):
                    metric.clear()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check based on metrics"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'metrics': {
                'blocks_committed': self.metrics.blocks_committed._value.get(),
                'blocks_failed': self.metrics.blocks_failed._value.get(),
                'current_height': self.metrics.current_height._value.get(),
                'peer_count': self.metrics.peer_count._value.get()
            }
        }
        
        # Check for issues
        if health['metrics']['blocks_failed'] > 10:
            health['status'] = 'degraded'
            health['issues'] = ['high_block_failure_rate']
        
        if health['metrics']['peer_count'] == 0:
            health['status'] = 'unhealthy'
            health['issues'] = ['no_peers_connected']
        
        return health