"""
Production configuration with safe defaults
"""
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ProductionBlockchainConfig:
    """Production-safe blockchain configuration"""
    network_type: str = "mainnet"
    data_dir: str = "./rayonix_data"
    port: int = 30303
    max_connections: int = 50
    block_time_target: int = 30
    max_block_size: int = 4000000
    min_transaction_fee: int = 1
    stake_minimum: int = 1000
    developer_fee_percent: float = 0.05
    enable_auto_staking: bool = True
    enable_transaction_relay: bool = True
    enable_state_pruning: bool = True
    max_reorganization_depth: int = 100
    checkpoint_interval: int = 1000
    genesis_premine: int = 1000000
    max_supply: int = 21000000
    block_reward: int = 50
    foundation_address: str = 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'
    genesis_description: str = 'Production RAYONIX blockchain'
    consensus_algorithm: str = 'pos'
    security_level: str = 'high'
    
    # Production-specific settings
    enable_safe_mode: bool = True
    max_retry_attempts: int = 3
    connection_timeout: int = 30
    request_timeout: int = 60
    enable_compression: bool = True
    enable_encryption: bool = True
    log_level: str = 'INFO'
    
    def to_dict(self) -> Dict[str, Any]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}