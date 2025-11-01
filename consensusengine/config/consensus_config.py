from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class ConsensusConfig:
    """Consensus engine configuration"""
    # Timeout parameters (these were missing)
    epoch_blocks: int = 100
    timeout_propose: int = 3000  # ms
    timeout_prevote: int = 1000  # ms
    timeout_precommit: int = 1000  # ms
    timeout_commit: int = 1000  # ms
    # Validator parameters
    max_validators: int = 100
    min_stake_amount: int = 1000
    unbonding_period: int = 86400 * 21  # 21 days in seconds
    slashing_percentage: float = 0.01  # 1% slashing for downtime
    jail_duration: int = 86400 * 2  # 2 days in seconds

    # Security parameters  
    security_level: str = "high"
    enable_slashing: bool = True
    enable_jailing: bool = True
    
    # Performance parameters
    max_block_size: int = 4000000
    block_time_target: int = 30
    max_transactions_per_block: int = 1000
    
    # Economic parameters
    block_reward: int = 50
    developer_fee_percent: float = 0.05
    foundation_address: str = 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX'
    
    # Additional consensus parameters that might be expected
    network_type: str = "testnet"
    data_dir: str = "./rayonix_data"
    enable_auto_staking: bool = True
    enable_transaction_relay: bool = True
    max_reorganization_depth: int = 100
    checkpoint_interval: int = 1000
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if self.min_stake_amount < 100:
            raise ValueError("Minimum stake amount must be at least 100")
        if self.slashing_percentage < 0 or self.slashing_percentage > 1:
            raise ValueError("Slashing percentage must be between 0 and 1")
        if self.max_validators <= 0:
            raise ValueError("Max validators must be positive")
    
    @classmethod
    def from_blockchain_config(cls, blockchain_config: Any, **overrides) -> 'ConsensusConfig':
        """Create ConsensusConfig from BlockchainConfig with sensible defaults"""
        config = cls()
        
        # Map relevant parameters from blockchain config
        if hasattr(blockchain_config, 'stake_minimum'):
            config.min_stake_amount = blockchain_config.stake_minimum
        if hasattr(blockchain_config, 'block_time_target'):
            config.block_time_target = blockchain_config.block_time_target
        if hasattr(blockchain_config, 'max_block_size'):
            config.max_block_size = blockchain_config.max_block_size
        if hasattr(blockchain_config, 'developer_fee_percent'):
            config.developer_fee_percent = blockchain_config.developer_fee_percent
        if hasattr(blockchain_config, 'foundation_address'):
            config.foundation_address = blockchain_config.foundation_address
        if hasattr(blockchain_config, 'network_type'):
            config.network_type = blockchain_config.network_type
        if hasattr(blockchain_config, 'data_dir'):
            config.data_dir = blockchain_config.data_dir
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config