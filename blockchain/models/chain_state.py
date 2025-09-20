# blockchain/models/chain_state.py
from dataclasses import dataclass, field
from typing import Dict, List, Any
import time

@dataclass
class ChainState:
    total_supply: int
    circulating_supply: int
    staking_rewards_distributed: int
    foundation_funds: int
    active_validators: int
    total_stake: int
    average_block_time: float
    current_difficulty: int
    last_block_time: float
    network_hashrate: float = 0.0
    mempool_size: int = 0
    connected_peers: int = 0
    sync_status: Dict[str, Any] = field(default_factory=dict)
    governance_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chain state to dictionary"""
        return {
            'total_supply': self.total_supply,
            'circulating_supply': self.circulating_supply,
            'staking_rewards_distributed': self.staking_rewards_distributed,
            'foundation_funds': self.foundation_funds,
            'active_validators': self.active_validators,
            'total_stake': self.total_stake,
            'average_block_time': self.average_block_time,
            'current_difficulty': self.current_difficulty,
            'last_block_time': self.last_block_time,
            'network_hashrate': self.network_hashrate,
            'mempool_size': self.mempool_size,
            'connected_peers': self.connected_peers,
            'sync_status': self.sync_status,
            'governance_params': self.governance_params,
            'timestamp': time.time()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainState':
        """Create chain state from dictionary"""
        return cls(
            total_supply=data['total_supply'],
            circulating_supply=data['circulating_supply'],
            staking_rewards_distributed=data['staking_rewards_distributed'],
            foundation_funds=data['foundation_funds'],
            active_validators=data['active_validators'],
            total_stake=data['total_stake'],
            average_block_time=data['average_block_time'],
            current_difficulty=data['current_difficulty'],
            last_block_time=data['last_block_time'],
            network_hashrate=data.get('network_hashrate', 0.0),
            mempool_size=data.get('mempool_size', 0),
            connected_peers=data.get('connected_peers', 0),
            sync_status=data.get('sync_status', {}),
            governance_params=data.get('governance_params', {})
        )
    
    def calculate_state_hash(self) -> str:
        """Calculate hash of chain state for integrity checking"""
        import hashlib
        import json
        
        state_data = self.to_dict()
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()