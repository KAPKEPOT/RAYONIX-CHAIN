# consensus/utils/config/settings.py
import os
import json
import yaml
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger('ConsensusConfig')

class ConsensusMode(Enum):
    """Consensus operation modes"""
    PRODUCTION = "production"
    TESTNET = "testnet"
    DEVELOPMENT = "development"
    RECOVERY = "recovery"

class LogLevel(Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class NetworkConfig:
    """Network configuration"""
    host: str = "0.0.0.0"
    port: int = 26656
    external_host: str = ""
    external_port: int = 26656
    max_connections: int = 100
    peer_discovery_interval: int = 30
    connection_timeout: int = 10
    message_size_limit: int = 10 * 1024 * 1024  # 10MB
    enable_compression: bool = True
    enable_encryption: bool = True

@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: str = "./consensus_db"
    backup_enabled: bool = True
    backup_interval: int = 3600  # 1 hour
    max_backups: int = 24
    compression_enabled: bool = True
    cache_size: int = 128 * 1024 * 1024  # 128MB
    write_buffer_size: int = 64 * 1024 * 1024  # 64MB

@dataclass
class TimeoutConfig:
    """Timeout configuration (in seconds)"""
    propose_timeout: float = 3.0
    prevote_timeout: float = 1.0
    precommit_timeout: float = 1.0
    view_change_timeout: float = 10.0
    state_sync_timeout: float = 30.0
    block_time_target: float = 5.0
    max_timeout_multiplier: float = 10.0

@dataclass
class StakingConfig:
    """Staking configuration"""
    min_stake: int = 1000
    max_validators: int = 100
    epoch_blocks: int = 100
    jail_duration: int = 3600  # 1 hour in seconds
    slash_percentage: float = 0.01  # 1%
    unbonding_period: int = 86400 * 7  # 7 days
    commission_rate_min: float = 0.0
    commission_rate_max: float = 0.2  # 20%

@dataclass
class CryptoConfig:
    """Cryptographic configuration"""
    key_algorithm: str = "ecdsa"  # ecdsa, rsa
    curve: str = "secp256k1"  # secp256k1, p256
    hash_algorithm: str = "sha256"
    signature_scheme: str = "ecdsa-sha256"
    enable_hardware_security: bool = False
    key_rotation_interval: int = 86400 * 90  # 90 days

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration"""
    enable_metrics: bool = True
    metrics_port: int = 26660
    health_check_interval: int = 30
    performance_metrics_interval: int = 60
    enable_tracing: bool = False
    trace_sample_rate: float = 0.1  # 10%

@dataclass
class ConsensusConfig:
    """Main consensus configuration"""
    
    # Basic configuration
    mode: ConsensusMode = ConsensusMode.PRODUCTION
    node_id: str = ""
    validator_address: str = ""
    chain_id: str = "mainnet-001"
    genesis_time: str = ""
    
    # Module configurations
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    timeouts: TimeoutConfig = field(default_factory=TimeoutConfig)
    staking: StakingConfig = field(default_factory=StakingConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Advanced configuration
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    data_dir: str = "./.consensus"
    config_file: str = ""
    
    # Consensus parameters
    block_size_limit: int = 1024 * 1024  # 1MB
    max_tx_per_block: int = 1000
    max_block_gas: int = 10000000
    evidence_max_age: int = 100000  # blocks
    snapshot_interval: int = 10000  # blocks
    
    # Feature flags
    enable_state_sync: bool = True
    enable_block_sync: bool = True
    enable_validator_set_changes: bool = True
    enable_governance: bool = False
    enable_cross_chain: bool = False
    
    # Performance tuning
    concurrent_verification: bool = True
    parallel_block_processing: bool = True
    cache_validator_set: bool = True
    preload_blocks: int = 100
    
    # Recovery options
    auto_recover: bool = True
    recovery_timeout: int = 300  # 5 minutes
    state_sync_trust_height: int = 0
    
    def __post_init__(self):
        """Post-initialization setup"""
        if not self.node_id:
            self.node_id = self._generate_node_id()
        
        if not self.genesis_time:
            import time
            self.genesis_time = str(int(time.time()))
    
    def _generate_node_id(self) -> str:
        """Generate a unique node ID"""
        import hashlib
        import socket
        import uuid
        
        hostname = socket.gethostname()
        unique_id = str(uuid.uuid4())
        combined = f"{hostname}-{unique_id}-{int(time.time())}"
        
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ConsensusConfig':
        """
        Load configuration from file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            ConsensusConfig instance
        """
        try:
            if not os.path.exists(config_path):
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return cls()
            
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config_data = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
            
            return cls.from_dict(config_data)
            
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConsensusConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ConsensusConfig instance
        """
        try:
            # Extract basic configuration
            basic_config = {k: v for k, v in config_dict.items() 
                          if not isinstance(v, dict)}
            
            # Extract module configurations
            network_config = config_dict.get('network', {})
            database_config = config_dict.get('database', {})
            timeout_config = config_dict.get('timeouts', {})
            staking_config = config_dict.get('staking', {})
            crypto_config = config_dict.get('crypto', {})
            monitoring_config = config_dict.get('monitoring', {})
            
            # Convert string enums to actual enum values
            if 'mode' in basic_config:
                basic_config['mode'] = ConsensusMode(basic_config['mode'])
            
            if 'log_level' in basic_config:
                basic_config['log_level'] = LogLevel(basic_config['log_level'])
            
            # Create configuration instance
            config = cls(**basic_config)
            
            # Update module configurations
            config.network = NetworkConfig(**network_config)
            config.database = DatabaseConfig(**database_config)
            config.timeouts = TimeoutConfig(**timeout_config)
            config.staking = StakingConfig(**staking_config)
            config.crypto = CryptoConfig(**crypto_config)
            config.monitoring = MonitoringConfig(**monitoring_config)
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating config from dict: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            'mode': self.mode.value,
            'node_id': self.node_id,
            'validator_address': self.validator_address,
            'chain_id': self.chain_id,
            'genesis_time': self.genesis_time,
            'log_level': self.log_level.value,
            'log_format': self.log_format,
            'data_dir': self.data_dir,
            'block_size_limit': self.block_size_limit,
            'max_tx_per_block': self.max_tx_per_block,
            'max_block_gas': self.max_block_gas,
            'evidence_max_age': self.evidence_max_age,
            'snapshot_interval': self.snapshot_interval,
            'enable_state_sync': self.enable_state_sync,
            'enable_block_sync': self.enable_block_sync,
            'enable_validator_set_changes': self.enable_validator_set_changes,
            'enable_governance': self.enable_governance,
            'enable_cross_chain': self.enable_cross_chain,
            'concurrent_verification': self.concurrent_verification,
            'parallel_block_processing': self.parallel_block_processing,
            'cache_validator_set': self.cache_validator_set,
            'preload_blocks': self.preload_blocks,
            'auto_recover': self.auto_recover,
            'recovery_timeout': self.recovery_timeout,
            'state_sync_trust_height': self.state_sync_trust_height,
        }
        
        # Add module configurations
        config_dict['network'] = self._dataclass_to_dict(self.network)
        config_dict['database'] = self._dataclass_to_dict(self.database)
        config_dict['timeouts'] = self._dataclass_to_dict(self.timeouts)
        config_dict['staking'] = self._dataclass_to_dict(self.staking)
        config_dict['crypto'] = self._dataclass_to_dict(self.crypto)
        config_dict['monitoring'] = self._dataclass_to_dict(self.monitoring)
        
        return config_dict
    
    def _dataclass_to_dict(self, obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary"""
        if hasattr(obj, '__dataclass_fields__'):
            return {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        return {}
    
    def save_to_file(self, config_path: str) -> bool:
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
            
            config_dict = self.to_dict()
            
            with open(config_path, 'w') as f:
                if config_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                elif config_path.endswith(('.yaml', '.yml')):
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    # Default to JSON
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving configuration to {config_path}: {e}")
            return False
    
    def validate(self) -> List[str]:
        """
        Validate configuration
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic configuration
        if not self.chain_id:
            errors.append("chain_id is required")
        
        if self.mode == ConsensusMode.PRODUCTION and not self.validator_address:
            errors.append("validator_address is required for production mode")
        
        # Validate network configuration
        if self.network.port < 1024 or self.network.port > 65535:
            errors.append("Network port must be between 1024 and 65535")
        
        if self.network.max_connections <= 0:
            errors.append("max_connections must be positive")
        
        # Validate staking configuration
        if self.staking.min_stake <= 0:
            errors.append("min_stake must be positive")
        
        if self.staking.max_validators <= 0:
            errors.append("max_validators must be positive")
        
        if not (0 <= self.staking.slash_percentage <= 1):
            errors.append("slash_percentage must be between 0 and 1")
        
        # Validate timeout configuration
        if any(timeout <= 0 for timeout in [
            self.timeouts.propose_timeout,
            self.timeouts.prevote_timeout,
            self.timeouts.precommit_timeout
        ]):
            errors.append("All timeouts must be positive")
        
        # Validate crypto configuration
        if self.crypto.key_algorithm not in ['ecdsa', 'rsa']:
            errors.append("key_algorithm must be 'ecdsa' or 'rsa'")
        
        if self.crypto.curve not in ['secp256k1', 'p256']:
            errors.append("curve must be 'secp256k1' or 'p256'")
        
        return errors
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get effective configuration with all defaults filled in
        
        Returns:
            Complete configuration dictionary
        """
        # Create a default config to compare against
        default_config = ConsensusConfig().to_dict()
        current_config = self.to_dict()
        
        effective_config = {}
        
        # Merge configurations, preferring current values
        for key in set(default_config.keys()) | set(current_config.keys()):
            if key in current_config and current_config[key] is not None:
                effective_config[key] = current_config[key]
            else:
                effective_config[key] = default_config.get(key)
        
        return effective_config
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_dict = self.to_dict()
        
        # Mask sensitive information
        if config_dict.get('validator_address'):
            config_dict['validator_address'] = '***MASKED***'
        if config_dict.get('node_id'):
            config_dict['node_id'] = '***MASKED***'
        
        return json.dumps(config_dict, indent=2)

class ConfigManager:
    """Manager for configuration loading and validation"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> ConsensusConfig:
        """Load configuration from file or environment"""
        config = None
        
        # Try loading from specified file
        if self.config_path and os.path.exists(self.config_path):
            config = ConsensusConfig.from_file(self.config_path)
        
        # Try loading from environment variable
        if not config:
            env_config_path = os.getenv('CONSENSUS_CONFIG_PATH')
            if env_config_path and os.path.exists(env_config_path):
                config = ConsensusConfig.from_file(env_config_path)
        
        # Try loading from default locations
        if not config:
            default_paths = [
                './consensus_config.json',
                './consensus_config.yaml',
                './config/consensus.json',
                os.path.expanduser('~/.consensus/config.json')
            ]
            
            for path in default_paths:
                if os.path.exists(path):
                    config = ConsensusConfig.from