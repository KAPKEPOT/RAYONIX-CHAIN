import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Validate configuration compatibility between components"""
    
    @staticmethod
    def validate_blockchain_consensus_config(blockchain_config: Any, consensus_config: Any) -> bool:
        """Validate that blockchain and consensus configurations are compatible"""
        try:
            # Check stake minimum consistency
            if (hasattr(blockchain_config, 'stake_minimum') and 
                hasattr(consensus_config, 'min_stake_amount')):
                if blockchain_config.stake_minimum != consensus_config.min_stake_amount:
                    logger.warning(
                        f"Stake minimum mismatch: blockchain={blockchain_config.stake_minimum}, "
                        f"consensus={consensus_config.min_stake_amount}"
                    )
                    # Auto-correct to avoid conflicts
                    consensus_config.min_stake_amount = blockchain_config.stake_minimum
            
            # Validate epoch settings
            if hasattr(consensus_config, 'epoch_blocks'):
                if consensus_config.epoch_blocks <= 0:
                    raise ValueError("Epoch blocks must be positive")
            
            # Validate timeout settings
            timeouts = ['timeout_propose', 'timeout_prevote', 'timeout_precommit', 'timeout_commit']
            for timeout in timeouts:
                if hasattr(consensus_config, timeout):
                    if getattr(consensus_config, timeout) <= 0:
                        raise ValueError(f"{timeout} must be positive")
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False