from typing import Dict, Any
from blockchain.core.rayonix_chain import BlockchainConfig
from blockchain.config.consensus_config import ConsensusConfig

class ConfigFactory:
    """Factory for creating compatible configurations"""
    
    @staticmethod
    def create_consensus_config(blockchain_config: BlockchainConfig, **overrides) -> ConsensusConfig:
        """Create a consensus config that's compatible with blockchain config"""
        return ConsensusConfig.from_blockchain_config(blockchain_config, **overrides)
    
    @staticmethod
    def create_compatible_configs(network_type: str = "mainnet", 
                                custom_config: Dict[str, Any] = None) -> tuple:
        """Create both blockchain and consensus configs that are guaranteed compatible"""
        # Create blockchain config
        blockchain_config = BlockchainConfig(network_type=network_type)
        if custom_config:
            for key, value in custom_config.items():
                if hasattr(blockchain_config, key):
                    setattr(blockchain_config, key, value)
        
        # Create compatible consensus config
        consensus_config = ConsensusConfig.from_blockchain_config(blockchain_config)
        
        return blockchain_config, consensus_config