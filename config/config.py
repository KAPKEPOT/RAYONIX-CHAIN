# config/config.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import os
from pathlib import Path

@dataclass
class NetworkConfig:
    """Network node configuration"""
    network_type: NetworkType = NetworkType.TESTNET
    listen_ip: str = "0.0.0.0"
    listen_port: int = 52555  
    host: str = field(default="0.0.0.0")  
    tcp_port: int = field(default=52555)  
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
    websocket_port: int = 53556
    http_port: int = 52557

    def __post_init__(self):
        """Set up aliases and defaults with enhanced compatibility"""
        # Set defaults for public IP/port
        if self.public_ip is None:
            self.public_ip = self.listen_ip
        if self.public_port is None:
            self.public_port = self.listen_port
            
        # Backward compatibility - map old parameter names
        if self.host == "0.0.0.0" and self.listen_ip != "0.0.0.0":
            self.host = self.listen_ip
        elif self.listen_ip == "0.0.0.0" and self.host != "0.0.0.0":
            self.listen_ip = self.host
            
        # TCP port is just an alias for listen_port
        if self.tcp_port == 52555:
            self.tcp_port = self.listen_port
            
        # Aliases for maximum compatibility
        self.port = self.listen_port
    
    def validate(self):
        """Validate configuration"""
        if not (0 < self.listen_port < 65536):
            raise ValueError("Invalid port number")
        if self.max_connections <= 0:
            raise ValueError("Max connections must be positive")
        if self.max_message_size <= 0:
            raise ValueError("Max message size must be positive")

@dataclass
class DatabaseConfig:
    """Database configuration - SINGLE source"""
    db_path: str = "./rayonix_data"
    db_engine: str = "plyvel"
    max_connections: int = 10
    connection_timeout: int = 30

@dataclass
class APIConfig:
    """API configuration - SINGLE source"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 52557
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_websockets: bool = True
    websocket_port: int = 52556

@dataclass
class ConsensusConfig:
    consensus_type: str = "pos"
    min_stake: int = 1000
    max_stake: int = 1000000
    block_time: int = 10
    difficulty_adjustment_interval: int = 2016
    reward_halving_interval: int = 210000
    epoch_blocks: int = 100
    timeout_propose: int = 3000
    timeout_prevote: int = 1000  
    timeout_precommit: int = 1000
    timeout_commit: int = 1000
    max_validators: int = 100
    min_stake_amount: int = 1000
    unbonding_period: int = 86400 * 21
    slashing_percentage: float = 0.01
    jail_duration: int = 86400 * 2
    security_level: str = "high"
    enable_slashing: bool = True
    enable_jailing: bool = True
    max_block_size: int = 4000000
    block_time_target: int = 30
    max_transactions_per_block: int = 1000
    block_reward: int = 50
    developer_fee_percent: float = 0.05
    foundation_address: str = 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'
    
@dataclass
class GasConfig:
    base_gas_price: int = 1000000000
    min_gas_price: int = 500000000
    max_gas_price: int = 10000000000
    adjustment_factor: float = 1.125
    target_utilization: float = 0.5

@dataclass
class LoggingConfig:
    """Logging configuration - SINGLE source"""
    level: str = "INFO"
    file: str = "rayonix_node.log"
    max_size: int = 10485760
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class NodeConfig:
    """MAIN configuration class - SINGLE SOURCE OF TRUTH"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, config_path: str) -> 'NodeConfig':
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if not config_file.exists():
            # Return defaults if file doesn't exist
            return cls()
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f) or {}
        
        return cls.from_dict(config_data)

    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'NodeConfig':
        """Create config from dictionary"""
        return cls(
            network=NetworkConfig(**config_data.get('network', {})),
            database=DatabaseConfig(**config_data.get('database', {})),
            api=APIConfig(**config_data.get('api', {})),
            consensus=ConsensusConfig(**config_data.get('consensus', {})),
            gas=GasConfig(**config_data.get('gas', {})),
            logging=LoggingConfig(**config_data.get('logging', {}))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        return {
            'network': self.network.__dict__,
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'consensus': self.consensus.__dict__,
            'gas': self.gas.__dict__,
            'logging': self.logging.__dict__
        }

    def save(self, config_path: str):
        """Save configuration to file"""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def validate(self):
        """Validate configuration"""
        # Port validation
        ports = [
            self.network.listen_port,
            self.network.websocket_port, 
            self.network.http_port
        ]
        
        if len(ports) != len(set(ports)):
            raise ValueError("Port conflicts detected - all ports must be unique")
        
        for port in ports:
            if not (1024 <= port <= 65535):
                raise ValueError(f"Invalid port: {port}")
        
        # Network validation
        if self.network.max_connections <= 0:
            raise ValueError("max_connections must be positive")
        
        # API validation
        if self.api.enabled and self.api.port != self.network.http_port:
            raise ValueError("API port must match network.http_port")
        
        if self.api.enable_websockets and self.api.websocket_port != self.network.websocket_port:
            raise ValueError("WebSocket port must match network.websocket_port")

# Global configuration instance
_config_instance: Optional[NodeConfig] = None

def init_config(config_path: Optional[str] = None) -> NodeConfig:
    """Initialize global configuration"""
    global _config_instance
    
    if _config_instance is None:
        if config_path and os.path.exists(config_path):
            _config_instance = NodeConfig.from_file(config_path)
        else:
            _config_instance = NodeConfig()
    
    return _config_instance

def get_config() -> NodeConfig:
    """Get global configuration"""
    if _config_instance is None:
        raise RuntimeError("Configuration not initialized. Call init_config() first.")
    return _config_instance