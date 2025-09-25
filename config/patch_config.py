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

