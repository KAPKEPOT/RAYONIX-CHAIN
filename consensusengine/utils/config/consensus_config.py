# consensus/utils/config/consensus_config.py
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
    
    def get(self, key: str, default=None):
        """Dictionary-like get method for compatibility"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'epoch_blocks': self.epoch_blocks,
            'timeout_propose': self.timeout_propose,
            'timeout_prevote': self.timeout_prevote,
            'timeout_precommit': self.timeout_precommit,
            'timeout_commit': self.timeout_commit,
            'db_path': self.db_path,
            'max_validators': self.max_validators,
            'min_stake_amount': self.min_stake_amount,
            'unbonding_period': self.unbonding_period,
            'slashing_percentage': self.slashing_percentage,
            'jail_duration': self.jail_duration
        }
    
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