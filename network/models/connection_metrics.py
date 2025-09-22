from dataclasses import dataclass, field
from typing import Deque
from collections import deque
import time

@dataclass
class ConnectionMetrics:
    """Connection performance metrics"""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_rate: float = 1.0
    message_rate: float = 0.0  # Messages per second
    bandwidth_rate: float = 0.0  # Bytes per second