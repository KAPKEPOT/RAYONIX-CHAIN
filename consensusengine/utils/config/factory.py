# utils/config/factory.py
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("ConfigFactory")

try:
    from consensusengine.utils.config.settings import ConsensusConfig
except ImportError as e:
    logger.error(f"Failed to import ConsensusConfig: {e}")
    # Fallback class
    class ConsensusConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    from network.config.node_config import NodeConfig
except ImportError as e:
    logger.error(f"Failed to import NodeConfig: {e}")
    # Fallback class
    class NodeConfig:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

class ConfigFactory:
    """Factory for creating properly configured components with robust error handling"""
    
    @staticmethod
    def _get_consensus_config_defaults() -> Dict[str, Any]:
        """Get consensus configuration defaults that match the actual class"""
        return {
            # Core consensus parameters
            'epoch_blocks': 100,
            'timeout_propose': 3000,
            'timeout_prevote': 1000,
            'timeout_precommit': 1000,
            'timeout_commit': 1000,
            
            # Database and storage
            'db_path': './consensus_data',
            
            # Validator parameters
            'max_validators': 100,
            'min_stake_amount': 1000,
            'unbonding_period': 86400 * 21,  # 21 days
            
            # Slashing parameters
            'slashing_percentage': 0.01,  # 1%
            'jail_duration': 86400 * 2,   # 2 days
            
            # Additional safety parameters
            'max_block_size': 10485760,  # 10MB
            'max_tx_per_block': 10000,
            'block_time': 5,  # 5 seconds
        }
    
    @staticmethod
    def _get_network_config_defaults() -> Dict[str, Any]:
        """Get network configuration defaults"""
        return {
            'host': '0.0.0.0',
            'port': 8080,
            'max_connections': 100,
            'rate_limit_per_peer': 1000,
            'ban_threshold': -100,
            'ban_duration': 3600,
            'enable_encryption': True,
            'peer_discovery_interval': 300,
            'connection_timeout': 30,
            'ping_interval': 60,
            'bootstrap_nodes': []
        }
    
    @staticmethod
    def create_consensus_config(**kwargs) -> ConsensusConfig:
        """Create consensus configuration with proper error handling"""
        defaults = ConfigFactory._get_consensus_config_defaults()
        config_params = {**defaults, **kwargs}
        
        try:
            # Try to create with all parameters first
            return ConsensusConfig(**config_params)
        except TypeError as e:
            logger.warning(f"ConsensusConfig creation failed: {e}. Trying fallback approach.")
            
            # Fallback: Only pass parameters that the class accepts
            try:
                import inspect
                init_signature = inspect.signature(ConsensusConfig.__init__)
                valid_params = {}
                
                for param_name in init_signature.parameters.keys():
                    if param_name != 'self' and param_name in config_params:
                        valid_params[param_name] = config_params[param_name]
                
                if not valid_params:
                    # Last resort: create empty config and set attributes manually
                    config = ConsensusConfig()
                    for key, value in config_params.items():
                        if hasattr(config, key) or not key.startswith('_'):
                            setattr(config, key, value)
                    return config
                
                return ConsensusConfig(**valid_params)
                
            except Exception as fallback_error:
                logger.error(f"Fallback config creation failed: {fallback_error}")
                # Ultimate fallback: create generic object
                config = type('ConsensusConfig', (), config_params)()
                logger.warning("Using generic consensus config object")
                return config
    
    @staticmethod
    def create_network_config(**kwargs) -> NodeConfig:
        """Create network configuration with proper error handling"""
        defaults = ConfigFactory._get_network_config_defaults()
        config_params = {**defaults, **kwargs}
        
        try:
            return NodeConfig(**config_params)
        except TypeError as e:
            logger.warning(f"NodeConfig creation failed: {e}. Using fallback.")
            
            # Fallback approach for network config
            try:
                config = NodeConfig()
                for key, value in config_params.items():
                    if hasattr(config, key) or not key.startswith('_'):
                        setattr(config, key, value)
                return config
            except Exception as fallback_error:
                logger.error(f"Network config fallback failed: {fallback_error}")
                config = type('NodeConfig', (), config_params)()
                logger.warning("Using generic network config object")
                return config
    
    @staticmethod
    def create_safe_consensus_config(**kwargs) -> ConsensusConfig:
        """Create consensus config with only safe, known parameters"""
        safe_params = {}
        known_params = ConfigFactory._get_consensus_config_defaults()
        
        for key, value in {**known_params, **kwargs}.items():
            # Only include parameters that are in our known defaults
            if key in known_params:
                safe_params[key] = value
        
        return ConfigFactory.create_consensus_config(**safe_params)