# config/config_manager.py - Configuration management

import os
import yaml
import json
import toml
from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum

class ConfigFormat(Enum):
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"

class ConfigSource(Enum):
    FILE = "file"
    ENV = "environment"
    DEFAULT = "default"

@dataclass
class NetworkConfig:
    network_type: str = "testnet"
    network_id: int = 1
    enabled: bool = True
    listen_ip: str = "0.0.0.0"
    listen_port: int = 9333
    max_connections: int = 50
    bootstrap_nodes: list = field(default_factory=list)
    enable_encryption: bool = True
    enable_compression: bool = True
    enable_dht: bool = False
    connection_timeout: int = 30
    message_timeout: int = 10
    websocket_port: int = 9335  # WebSocket port
    http_port: int = 9336    # HTTP port

@dataclass
class DatabaseConfig:
    db_path: str = "./rayonix_data"
    db_engine: str = "plyvel"
    connection_string: str = ""
    max_connections: int = 10
    connection_timeout: int = 30

@dataclass
class APIConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8545
    enable_cors: bool = True
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass
class ConsensusConfig:
    consensus_type: str = "pos"
    min_stake: int = 1000
    max_stake: int = 1000000
    block_time: int = 10
    difficulty_adjustment_interval: int = 2016
    reward_halving_interval: int = 210000

@dataclass
class GasConfig:
    base_gas_price: int = 1000000000
    min_gas_price: int = 500000000
    max_gas_price: int = 10000000000
    adjustment_factor: float = 1.125
    target_utilization: float = 0.5

@dataclass
class LoggingConfig:
    level: str = "INFO"
    file: str = "rayonix_node.log"
    max_size: int = 10485760  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

@dataclass
class Config:
    network: NetworkConfig = field(default_factory=NetworkConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    gas: GasConfig = field(default_factory=GasConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None, 
                 encryption_key: Optional[str] = None, 
                 auto_reload: bool = False):
        self.config_path = config_path
        self.encryption_key = encryption_key
        self.auto_reload = auto_reload
        self.config = Config()
        self.config_format = ConfigFormat.YAML
        self.config_source = ConfigSource.DEFAULT
        self._load_config()

    def _load_config(self):
        """Load configuration from file or use defaults"""
        if not self.config_path:
            # Use default configuration
            return
        
        config_file = Path(self.config_path)
        if not config_file.exists():
            # Create default config file
            self._save_config()
            return
        
        # Determine config format
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            self.config_format = ConfigFormat.YAML
        elif config_file.suffix.lower() == '.json':
            self.config_format = ConfigFormat.JSON
        elif config_file.suffix.lower() == '.toml':
            self.config_format = ConfigFormat.TOML
        
        # Load config from file
        try:
            with open(config_file, 'r') as f:
                if self.config_format == ConfigFormat.YAML:
                    config_data = yaml.safe_load(f)
                elif self.config_format == ConfigFormat.JSON:
                    config_data = json.load(f)
                elif self.config_format == ConfigFormat.TOML:
                    config_data = toml.load(f)
                
                # Update config with file data
                self._update_config_from_dict(config_data)
                self.config_source = ConfigSource.FILE
                
        except Exception as e:
            print(f"Error loading config file: {e}")
            # Fall back to default config

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update config from dictionary"""
        if not config_data:
            return
        
        # Update network config
        if 'network' in config_data:
            network_data = config_data['network']
            for key, value in network_data.items():
                if hasattr(self.config.network, key):
                    setattr(self.config.network, key, value)
        
        # Update database config
        if 'database' in config_data:
            db_data = config_data['database']
            for key, value in db_data.items():
                if hasattr(self.config.database, key):
                    setattr(self.config.database, key, value)
        
        # Update API config
        if 'api' in config_data:
            api_data = config_data['api']
            for key, value in api_data.items():
                if hasattr(self.config.api, key):
                    setattr(self.config.api, key, value)
        
        # Update consensus config
        if 'consensus' in config_data:
            consensus_data = config_data['consensus']
            for key, value in consensus_data.items():
                if hasattr(self.config.consensus, key):
                    setattr(self.config.consensus, key, value)
        
        # Update gas config
        if 'gas' in config_data:
            gas_data = config_data['gas']
            for key, value in gas_data.items():
                if hasattr(self.config.gas, key):
                    setattr(self.config.gas, key, value)
        
        # Update logging config
        if 'logging' in config_data:
            logging_data = config_data['logging']
            for key, value in logging_data.items():
                if hasattr(self.config.logging, key):
                    setattr(self.config.logging, key, value)

    def _save_config(self):
        """Save current configuration to file"""
        if not self.config_path:
            return
        
        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        try:
            with open(config_file, 'w') as f:
                if self.config_format == ConfigFormat.YAML:
                    yaml.dump(config_dict, f, default_flow_style=False)
                elif self.config_format == ConfigFormat.JSON:
                    json.dump(config_dict, f, indent=2)
                elif self.config_format == ConfigFormat.TOML:
                    toml.dump(config_dict, f)
        except Exception as e:
            print(f"Error saving config file: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'network': asdict(self.config.network),
            'database': asdict(self.config.database),
            'api': asdict(self.config.api),
            'consensus': asdict(self.config.consensus),
            'gas': asdict(self.config.gas),
            'logging': asdict(self.config.logging)
        }

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

    def set(self, key: str, value: Any) -> bool:
        """Set configuration value using dot notation"""
        try:
            parts = key.split('.')
            obj = self.config
            
            # Navigate to the parent object
            for part in parts[:-1]:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    return False
            
            # Set the value on the final object
            final_key = parts[-1]
            if hasattr(obj, final_key):
                setattr(obj, final_key, value)
                
                # Save config if auto_reload is enabled
                if self.auto_reload:
                    self._save_config()
                
                return True
            else:
                return False
        except (AttributeError, KeyError):
            return False

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration as dictionary"""
        return self.to_dict()

def init_config(config_path: Optional[str] = None, 
                encryption_key: Optional[str] = None, 
                auto_reload: bool = False) -> ConfigManager:
    """Initialize configuration manager"""
    return ConfigManager(config_path, encryption_key, auto_reload)