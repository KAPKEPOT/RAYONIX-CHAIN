# utils/config/factory.py
from consensusengine.utils.config.settings import ConsensusConfig
from network.config.node_config import NodeConfig

class ConfigFactory:
    """Factory for creating properly configured components"""
    
    @staticmethod
    def create_consensus_config(**kwargs) -> ConsensusConfig:
        """Create consensus configuration with defaults"""
        defaults = {
            'epoch_blocks': 100,
            'timeout_propose': 3000,
            'timeout_prevote': 1000,
            'timeout_precommit': 1000,
            'timeout_commit': 1000,
            'db_path': './consensus_data',
            'max_validators': 100,
            'min_stake_amount': 1000,
            'unbonding_period': 86400 * 21,
            'slashing_percentage': 0.01,
            'jail_duration': 86400 * 2
        }
        defaults.update(kwargs)
        return ConsensusConfig(**defaults)
    
    @staticmethod
    def create_network_config(**kwargs) -> NodeConfig:
        """Create network configuration with defaults"""
        defaults = {
            'host': '0.0.0.0',
            'port': 8080,
            'max_connections': 100,
            'rate_limit_per_peer': 1000,
            'ban_threshold': -100,
            'ban_duration': 3600
        }
        defaults.update(kwargs)
        return NodeConfig(**defaults)