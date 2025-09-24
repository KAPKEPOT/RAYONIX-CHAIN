from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ConsensusConfig:
    """Consensus engine configuration"""
    epoch_blocks: int = 100
    timeout_propose: int = 3000  # ms
    timeout_prevote: int = 1000  # ms
    timeout_precommit: int = 1000  # ms
    timeout_commit: int = 1000  # ms
    db_path: str = "./consensus_data"
    max_validators: int = 100
    min_stake_amount: int = 1000
    unbonding_period: int = 86400 * 21  # 21 days in seconds
    slashing_percentage: float = 0.01  # 1% slashing for downtime
    jail_duration: int = 86400 * 2  # 2 days in seconds
    
    @classmethod
    def from_blockchain_config(cls, blockchain_config: Any, **overrides) -> 'ConsensusConfig':
        """Create ConsensusConfig from BlockchainConfig with sensible defaults"""
        config = cls()
        
        # Map relevant parameters from blockchain config
        if hasattr(blockchain_config, 'stake_minimum'):
            config.min_stake_amount = blockchain_config.stake_minimum
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config