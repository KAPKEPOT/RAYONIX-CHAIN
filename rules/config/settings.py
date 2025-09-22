"""
Configuration settings management for consensus system
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum

from ..exceptions import ConsensusError

logger = logging.getLogger('consensus.config')

class Environment(Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "./consensus_db"
    max_open_files: int = 1000
    write_buffer_size: int = 4 * 1024 * 1024
    read_buffer_size: int = 1 * 1024 * 1024
    block_cache_size: int = 8 * 1024 * 1024
    create_if_missing: bool = True
    error_if_exists: bool = False
    paranoid_checks: bool = False

@dataclass
class NetworkConfig:
    """Network configuration"""
    host: str = "0.0.0.0"
    port: int = 26656
    max_peers: int = 50
    peer_discovery: bool = True
    peer_discovery_interval: int = 30000
    connection_timeout: int = 10000
    message_timeout: int = 30000
    max_message_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class ConsensusConfig:
    """Consensus algorithm configuration"""
    timeout_propose: int = 3000
    timeout_prevote: int = 1000
    timeout_precommit: int = 1000
    timeout_commit: int = 5000
    max_rounds: int = 10
    round_retry_timeout: int = 2000
    min_stake: int = 1000
    max_validators: int = 100
    jail_duration: int = 3600
    slash_percentage: float = 0.01
    epoch_blocks: int = 100
    reward_distribution_interval: int = 3600

@dataclass
class CryptoConfig:
    """Cryptography configuration"""
    key_algorithm: str = "secp256k1"
    hash_algorithm: str = "sha256"
    signature_format: str = "der"
    key_directory: str = "./keys"
    key_rotation_interval: int = 86400  # 24 hours
    signature_verification: bool = True

@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enabled: bool = True
    port: int = 9090
    interval: int = 5000
    namespace: str = "consensus"
    endpoint: str = "/metrics"

@dataclass
class APIConfig:
    """API configuration"""
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 26657
    max_workers: int = 10
    timeout: int = 30
    cors_allowed_origins: list = field(default_factory=list)

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = 100 * 1024 * 1024  # 100MB
    backup_count: int = 5
    compress: bool = True

@dataclass
class Settings:
    """Complete application settings"""
    environment: Environment = Environment.DEVELOPMENT
    node_id: str = "default-node"
    data_dir: str = "./data"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization validation"""
        self._validate()
    
    def _validate(self) -> None:
        """Validate configuration values"""
        if self.consensus.min_stake <= 0:
            raise ValueError("min_stake must be positive")
        if not 0 <= self.consensus.slash_percentage <= 1:
            raise ValueError("slash_percentage must be between 0 and 1")
        if self.network.port <= 1024 or self.network.port > 65535:
            raise ValueError("port must be between 1025 and 65535")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """Load settings from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls.from_dict(config_data)
        except Exception as e:
            raise ConsensusError(f"Failed to load config: {e}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Settings':
        """Create settings from dictionary"""
        # Convert nested dictionaries to config objects
        database_config = DatabaseConfig(**config_dict.get('database', {}))
        network_config = NetworkConfig(**config_dict.get('network', {}))
        consensus_config = ConsensusConfig(**config_dict.get('consensus', {}))
        crypto_config = CryptoConfig(**config_dict.get('crypto', {}))
        metrics_config = MetricsConfig(**config_dict.get('metrics', {}))
        api_config = APIConfig(**config_dict.get('api', {}))
        logging_config = LoggingConfig(**config_dict.get('logging', {}))
        
        # Get environment
        env_str = config_dict.get('environment', 'development')
        try:
            environment = Environment(env_str.lower())
        except ValueError:
            environment = Environment.DEVELOPMENT
        
        return cls(
            environment=environment,
            node_id=config_dict.get('node_id', 'default-node'),
            data_dir=config_dict.get('data_dir', './data'),
            database=database_config,
            network=network_config,
            consensus=consensus_config,
            crypto=crypto_config,
            metrics=metrics_config,
            api=api_config,
            logging=logging_config
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            'environment': self.environment.value,
            'node_id': self.node_id,
            'data_dir': self.data_dir,
            'database': self.database.__dict__,
            'network': self.network.__dict__,
            'consensus': self.consensus.__dict__,
            'crypto': self.crypto.__dict__,
            'metrics': self.metrics.__dict__,
            'api': self.api.__dict__,
            'logging': self.logging.__dict__
        }
    
    def save_to_file(self, config_path: str) -> None:
        """Save settings to YAML file"""
        try:
            config_path = Path(config_path)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            
        except Exception as e:
            raise ConsensusError(f"Failed to save config: {e}")
    
    def get_database_path(self) -> str:
        """Get full database path"""
        return str(Path(self.data_dir) / self.database.path)
    
    def get_key_directory(self) -> str:
        """Get full key directory path"""
        return str(Path(self.data_dir) / self.crypto.key_directory)
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def apply_environment_overrides(self) -> None:
        """Apply environment-specific overrides"""
        if self.is_production():
            # Production settings
            self.logging.level = "INFO"
            self.consensus.timeout_propose = 2000
            self.consensus.timeout_prevote = 1000
            self.consensus.timeout_precommit = 1000
            self.network.peer_discovery = True
            
        elif self.is_development():
            # Development settings
            self.logging.level = "DEBUG"
            self.consensus.timeout_propose = 4000
            self.consensus.timeout_prevote = 2000
            self.consensus.timeout_precommit = 2000
            self.network.peer_discovery = False
            
        # Apply environment variables
        self._apply_env_vars()
    
    def _apply_env_vars(self) -> None:
        """Apply configuration from environment variables"""
        # Database settings
        if db_path := os.getenv('CONSENSUS_DB_PATH'):
            self.database.path = db_path
        
        # Network settings
        if host := os.getenv('CONSENSUS_NETWORK_HOST'):
            self.network.host = host
        if port := os.getenv('CONSENSUS_NETWORK_PORT'):
            self.network.port = int(port)
        
        # Consensus settings
        if min_stake := os.getenv('CONSENSUS_MIN_STAKE'):
            self.consensus.min_stake = int(min_stake)
        
        # Logging settings
        if log_level := os.getenv('CONSENSUS_LOG_LEVEL'):
            self.logging.level = log_level