# blockchain/models/fork.py
from dataclasses import dataclass, field
from typing import Dict, List, Any
import time

@dataclass
class ForkResolution:
    common_ancestor_height: int
    common_ancestor_hash: str
    old_chain_length: int
    new_chain_length: int
    chainwork_difference: int
    resolution_time: float
    blocks_rolled_back: int
    blocks_applied: int
    rolled_back_hashes: List[str] = field(default_factory=list)
    applied_hashes: List[str] = field(default_factory=list)
    resolution_strategy: str = "chainwork"
    validator_consensus: Dict[str, bool] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fork resolution to dictionary"""
        return {
            'common_ancestor_height': self.common_ancestor_height,
            'common_ancestor_hash': self.common_ancestor_hash,
            'old_chain_length': self.old_chain_length,
            'new_chain_length': self.new_chain_length,
            'chainwork_difference': self.chainwork_difference,
            'resolution_time': self.resolution_time,
            'blocks_rolled_back': self.blocks_rolled_back,
            'blocks_applied': self.blocks_applied,
            'rolled_back_hashes': self.rolled_back_hashes,
            'applied_hashes': self.applied_hashes,
            'resolution_strategy': self.resolution_strategy,
            'validator_consensus': self.validator_consensus,
            'timestamp': time.time()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ForkResolution':
        """Create fork resolution from dictionary"""
        return cls(
            common_ancestor_height=data['common_ancestor_height'],
            common_ancestor_hash=data['common_ancestor_hash'],
            old_chain_length=data['old_chain_length'],
            new_chain_length=data['new_chain_length'],
            chainwork_difference=data['chainwork_difference'],
            resolution_time=data['resolution_time'],
            blocks_rolled_back=data['blocks_rolled_back'],
            blocks_applied=data['blocks_applied'],
            rolled_back_hashes=data.get('rolled_back_hashes', []),
            applied_hashes=data.get('applied_hashes', []),
            resolution_strategy=data.get('resolution_strategy', 'chainwork'),
            validator_consensus=data.get('validator_consensus', {})
        )
    
    def was_successful(self) -> bool:
        """Check if fork resolution was successful"""
        return (self.blocks_rolled_back > 0 or self.blocks_applied > 0) and self.resolution_time > 0
    
    def get_efficiency_ratio(self) -> float:
        """Calculate resolution efficiency ratio"""
        if self.resolution_time <= 0:
            return 0.0
        total_blocks = self.blocks_rolled_back + self.blocks_applied
        return total_blocks / self.resolution_time if total_blocks > 0 else 0.0