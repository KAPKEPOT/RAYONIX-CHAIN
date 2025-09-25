# config/patch_config.py
def get_safe_genesis_config():
    """Return a safe genesis configuration that avoids validation errors"""
    return {
        'premine_amount': 1000000,  # 1 million coins
        'max_supply': 21000000,     # 21 million coins
        'foundation_address': 'RYXFOUNDATIONXXXXXXXXXXXXXXXXXXXXXX',
        'block_reward': 50,
        'network_id': 1,
        'difficulty': 1,
        'block_time_target': 30,
        'developer_fee_percent': 0.05,
        'consensus_algorithm': 'pos'
    }

# Use this when initializing your blockchain
from config.patch_config import get_safe_genesis_config

class RayonixNode:
    def initialize_components(self):
        safe_config = get_safe_genesis_config()
        self.rayonix_chain = RayonixBlockchain(
            network_type=self.network_type,
            data_dir=self.data_dir,
            config=safe_config
        )