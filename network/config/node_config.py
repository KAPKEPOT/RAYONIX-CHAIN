from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from network.config.network_types import NetworkType

@dataclass
class NodeConfig:
    """Network node configuration"""
    network_type: NetworkType = NetworkType.TESTNET
    listen_ip: str = "0.0.0.0"
    listen_port: int = 30303
    public_ip: Optional[str] = None
    public_port: Optional[int] = None
    max_connections: int = 50
    max_peers: int = 1000
    connection_timeout: int = 30
    message_timeout: int = 10
    ping_interval: int = 60
    bootstrap_nodes: List[str] = field(default_factory=list)
    enable_nat_traversal: bool = True
    enable_encryption: bool = True
    enable_compression: bool = True
    enable_dht: bool = True
    enable_gossip: bool = True
    enable_syncing: bool = True
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_per_peer: int = 1000  # messages per minute
    ban_threshold: int = -100  # Reputation score for auto-ban
    ban_duration: int = 3600  # 1 hour in seconds
    dht_bootstrap_nodes: List[Tuple[str, int]] = field(default_factory=list)
    dns_seeds: List[str] = field(default_factory=list)
    websocket_port: int = 30305
    http_port: int = 30306
 

    def __post_init__(self):
        """Set up aliases and defaults"""
        if self.public_ip is None:
            self.public_ip = self.listen_ip
        if self.public_port is None:
            self.public_port = self.listen_port
        # Alias for compatibility
        self.port = self.listen_port
        self.host = self.listen_ip
    
    def validate(self):
        """Validate configuration"""
        if not (0 < self.listen_port < 65536):
            raise ValueError("Invalid port number")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        if self.max_message_size <= 0:
            raise ValueError("Max message size must be positive")