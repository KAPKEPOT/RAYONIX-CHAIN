from dataclasses import dataclass, field
from typing import Any, Optional
import time
from network.config.network_types import MessageType

@dataclass
class NetworkMessage:
    """Network message structure"""
    message_id: str
    message_type: MessageType
    payload: Any
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time-to-live for gossip
    signature: Optional[str] = None
    source_node: Optional[str] = None
    destination_node: Optional[str] = None
    priority: int = 0  # 0=low, 1=normal, 2=high, 3=critical