# config/config_manager.py E

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import yaml
import os
from pathlib import Path
from enum import Enum

class NetworkType(Enum):
    MAINNET = "mainnet"
    TESTNET = "testnet" 
    DEVNET = "devnet"
    REGTEST = "regtest"

class ProtocolType(Enum):
    TCP = "tcp"
    #UDP = "udp"
    WEBSOCKET = "websocket"
    HTTP = "http"

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READY = "ready"

@dataclass
class NetworkPreset:
    """Complete network configuration preset"""
    name: str
    network_id: int
    magic_bytes: bytes
    default_ports: Dict[str, int]  # tcp, websocket, http
    bootstrap_nodes: List[str]
    dns_seeds: List[str]
    genesis_hash: str
    checkpoint_blocks: Dict[int, str]  # block_height: block_hash

# NETWORK PRESETS - ALL NETWORK CONFIGURATIONS
NETWORK_PRESETS = {
    'mainnet': NetworkPreset(
        name='mainnet',
        network_id=1,
        magic_bytes=b'RAYX',
        default_ports={'tcp': 52555, 'websocket': 52556, 'http': 52557},
        bootstrap_nodes=[
            'mainnet-node1.rayonix.site:52555',
            'mainnet-node2.rayonix.site:52555',
        ],
        dns_seeds=['mainnet-seeds.rayonix.site'],
        genesis_hash='mainnet_genesis_hash_123',
        checkpoint_blocks={50000: 'block_hash_50000', 100000: 'block_hash_100000'}
    ),
    'testnet': NetworkPreset(
        name='testnet',
        network_id=2, 
        magic_bytes=b'RAYT',
        default_ports={'tcp': 52555, 'websocket': 52556, 'http': 52557},
        bootstrap_nodes=[
            'testnet-node1.rayonix.site:52555',
            'testnet-node2.rayonix.site:52555',
        ],
        dns_seeds=['testnet-seeds.rayonix.site'],
        genesis_hash='testnet_genesis_hash_456',
        checkpoint_blocks={10000: 'block_hash_10000', 50000: 'block_hash_50000'}
    ),
    'devnet': NetworkPreset(
        name='devnet',
        network_id=3,
        magic_bytes=b'RAYD',
        default_ports={'tcp': 52555, 'websocket': 52556, 'http': 52557},
        bootstrap_nodes=[
            'localhost:52555',
            '127.0.0.1:52555',
        ],
        dns_seeds=[],
        genesis_hash='devnet_genesis_hash_789',
        checkpoint_blocks={}
    )
}

@dataclass
class NetworkConfig:
    """Network configuration - NO HARDCODED VALUES"""
    network_type: str = "testnet"
    network_id: int = field(init=False)
    magic_bytes: bytes = field(init=False)
    listen_ip: str = "0.0.0.0"
    listen_port: int = field(init=False)
    websocket_port: int = field(init=False)
    http_port: int = field(init=False)
    max_connections: int = 50
    max_peers: int = 1000
    connection_timeout: int = 30
    message_timeout: int = 10
    ping_interval: int = 60
    enable_nat_traversal: bool = True
    enable_encryption: bool = True
    enable_compression: bool = True
    enable_dht: bool = False
    enable_gossip: bool = True
    bootstrap_nodes: List[str] = field(default_factory=list)
    dns_seeds: List[str] = field(default_factory=list)
    ban_threshold: int = -100
    ban_duration: int = 3600
    rate_limit_per_peer: int = 1000
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    enable_transaction_relay: bool = True
    
    def __post_init__(self):
        self.apply_network_preset(self.network_type)
    
    def apply_network_preset(self, network_type: str):
        """Apply all settings from network preset"""
        if network_type not in NETWORK_PRESETS:
            raise ValueError(f"Unknown network type: {network_type}")
        
        preset = NETWORK_PRESETS[network_type]
        self.network_id = preset.network_id
        self.magic_bytes = preset.magic_bytes
        self.listen_port = preset.default_ports['tcp']
        self.websocket_port = preset.default_ports['websocket']
        self.http_port = preset.default_ports['http']
        
        # Set defaults if not overridden
        if not self.bootstrap_nodes:
            self.bootstrap_nodes = preset.bootstrap_nodes.copy()
        if not self.dns_seeds:
            self.dns_seeds = preset.dns_seeds.copy()

@dataclass
class DatabaseConfig:
    """Database configuration - NO HARDCODED VALUES"""
    db_path: str = "./rayonix_data"
    db_engine: str = "plyvel"
    max_connections: int = 10
    connection_timeout: int = 30
    cache_size: int = 134217728  # 128MB
    max_open_files: int = 1000
    write_buffer_size: int = 67108864  # 64MB
    bloom_filter_bits: int = 10
    auto_compaction: bool = True
    compaction_interval: int = 3600
    backup_enabled: bool = True
    backup_interval: int = 86400
    backup_retention: int = 7
    enable_state_pruning: bool = True
    checkpoint_interval: int = 1000

@dataclass
class APIConfig:
    """API configuration - NO HARDCODED VALUES"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = field(init=False)
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    enable_websockets: bool = True
    websocket_port: int = field(init=False)
    max_connections: int = 100
    request_timeout: int = 30
    enable_metrics: bool = True
    metrics_port: int = 52558
    enable_profiling: bool = False
    profiling_port: int = 52559
    rate_limiting: bool = True
    rate_limit: int = 100
    authentication: bool = False
    auth_tokens: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # These will be set by ConfigManager to match network ports
        self.port = 52557  # Default, will be overridden
        self.websocket_port = 52556  # Default, will be overridden

@dataclass
class ConsensusConfig:
    """Consensus configuration - NO HARDCODED VALUES"""
    consensus_type: str = "pos"
    min_stake: int = 1000
    max_stake: int = 1000000
    block_time: int = 10
    difficulty_adjustment_interval: int = 2016
    reward_halving_interval: int = 210000
    block_reward: int = 50
    epoch_blocks: int = 100
    max_validators: int = 100
    validator_commission: float = 0.1
    unbonding_period: int = 172800
    min_delegation: int = 100
    stake_locktime: int = 86400
    slash_percentage: float = 0.01
    jail_duration: int = 3600
    max_block_size: int = 4000000
    max_transactions_per_block: int = 1000
    developer_fee_percent: float = 0.05
    foundation_address: str = "RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX"
    governance_enabled: bool = True
    proposal_deposit: int = 1000
    voting_period: int = 259200
    max_reorganization_depth: int = 100
    enable_auto_staking: bool = True

@dataclass
class GasConfig:
    """Gas configuration - NO HARDCODED VALUES"""
    base_gas_price: int = 1000000000
    min_gas_price: int = 500000000
    max_gas_price: int = 10000000000
    adjustment_factor: float = 1.125
    target_utilization: float = 0.5
    min_transaction_fee: int = 1

@dataclass
class LoggingConfig:
    """Logging configuration - NO HARDCODED VALUES"""
    level: str = "INFO"
    file: str = "rayonix_node.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    file_path: str = "./logs"
    console_enabled: bool = True
    enable_rotation: bool = True
    rotation_interval: int = 86400
    json_format: bool = False

@dataclass
class SecurityConfig:
    """Security configuration - NO HARDCODED VALUES"""
    enable_encryption: bool = True
    encryption_algorithm: str = "aes-256-gcm"
    key_derivation_iterations: int = 100000
    enable_ssl: bool = False
    ssl_min_version: str = "TLSv1.2"
    enable_firewall: bool = True
    rate_limiting: bool = True
    rate_limit: int = 1000
    session_timeout: int = 3600
    audit_logging: bool = True
    audit_retention: int = 365

@dataclass  
class PeerDiscoveryConfig:
    """Peer discovery configuration - NO HARDCODED VALUES"""
    dns_discovery_enabled: bool = True
    dht_discovery_enabled: bool = True
    gossip_discovery_enabled: bool = True
    bootstrap_discovery_enabled: bool = True
    discovery_interval: int = 300  # 5 minutes
    max_peers_to_return: int = 10
    peer_list_request_count: int = 3
    stale_peer_timeout: int = 3600  # 1 hour

@dataclass
class MessageHandlerConfig:
    """Message handler configuration - NO HARDCODED VALUES"""
    max_retry_attempts: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 10.0
    circuit_breaker_failures: int = 5
    broadcast_batch_size: int = 10
    broadcast_concurrency: int = 5
    broadcast_batch_delay: float = 0.05

@dataclass
class ConnectionManagerConfig:
    """Connection manager configuration - NO HARDCODED VALUES"""
    health_check_interval: int = 30
    connection_timeout: int = 30
    max_connection_age: int = 3600
    cleanup_interval: int = 60

@dataclass
class Config:
    """COMPLETE configuration - ALL HARDCODED VALUES MOVED HERE"""
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    peer_discovery: PeerDiscoveryConfig = field(default_factory=PeerDiscoveryConfig)
    message_handler: MessageHandlerConfig = field(default_factory=MessageHandlerConfig)
    connection_manager: ConnectionManagerConfig = field(default_factory=ConnectionManagerConfig)

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, 
                 encryption_key: Optional[str] = None, 
                 auto_reload: bool = False):
        self.config_path = config_path
        self.auto_reload = auto_reload
        self.config = Config()
        
        self._load_config()
        self._sync_network_ports()  # Ensure API ports match network ports
        
    def _load_config(self):
        """Load configuration from file"""
        if not self.config_path:
            return
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            return
        
        try:
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            self._update_config_from_dict(config_data)
        except Exception as e:
            print(f"Error loading config file: {e}")

    def _sync_network_ports(self):
        """Ensure API ports match network ports"""
        self.config.api.port = self.config.network.http_port
        self.config.api.websocket_port = self.config.network.websocket_port

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update config from dictionary"""
        sections = [
            'network', 'database', 'api', 'consensus', 'gas', 'logging',
            'security', 'peer_discovery', 'message_handler', 'connection_manager'
        ]
        
        for section in sections:
            if section in config_data:
                section_config = getattr(self.config, section)
                for key, value in config_data[section].items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
        
        # Re-sync ports after loading config
        self._sync_network_ports()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            parts = key.split('.')
            obj = self.config
            
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return default
            
            return obj
        except (AttributeError, KeyError):
            return default

    def set_network(self, network_type: str) -> bool:
        """Change network with one call"""
        if network_type not in NETWORK_PRESETS:
            return False
        
        self.config.network.network_type = network_type
        self.config.network.apply_network_preset(network_type)
        self._sync_network_ports()
        
        logger.info(f"Switched to {network_type} network")
        return True

    def get_network_preset(self) -> NetworkPreset:
        """Get current network preset"""
        return NETWORK_PRESETS[self.config.network.network_type]

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return {
            'network': self.config.network.__dict__,
            'database': self.config.database.__dict__,
            'api': self.config.api.__dict__,
            'consensus': self.config.consensus.__dict__,
            'gas': self.config.gas.__dict__,
            'logging': self.config.logging.__dict__,
            'security': self.config.security.__dict__,
            'peer_discovery': self.config.peer_discovery.__dict__,
            'message_handler': self.config.message_handler.__dict__,
            'connection_manager': self.config.connection_manager.__dict__
        }

def init_config(config_path: Optional[str] = None, 
                encryption_key: Optional[str] = None, 
                auto_reload: bool = False) -> ConfigManager:
    """Initialize configuration manager"""
    return ConfigManager(config_path, encryption_key, auto_reload)