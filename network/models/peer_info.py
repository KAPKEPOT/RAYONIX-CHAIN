from dataclasses import dataclass, field
from typing import List, Optional
import time
from network.config.network_types import ProtocolType, ConnectionState

@dataclass
class PeerInfo:
    """Peer information"""
    node_id: str
    address: str
    port: int
    protocol: ProtocolType
    version: str
    capabilities: List[str]
    last_seen: float = field(default_factory=time.time)
    connection_count: int = 0
    failed_attempts: int = 0
    reputation: int = 100
    latency: float = 0.0
    state: ConnectionState = ConnectionState.DISCONNECTED
    public_key: Optional[str] = None
    user_agent: str = ""
    services: int = 0
    last_attempt: float = 0.0
    next_attempt: float = 0.0
    banned_until: Optional[float] = None