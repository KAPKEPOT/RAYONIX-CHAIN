# blockchain/forks/fork_resolution.py
import time
from typing import Dict, List, Any, Optional
from enum import Enum, auto

class ForkResolutionStrategy(Enum):
    CHAINWORK = auto()
    VALIDATOR_VOTING = auto()
    TIMESTAMP = auto()
    HYBRID = auto()

class ForkResolver:
    """Handles different fork resolution strategies"""
    
    def __init__(self, config: Dict[str, Any], consensus_manager: Any):
        self.config = config
        self.consensus_manager = consensus_manager
        self.strategy_weights = {
            ForkResolutionStrategy.CHAINWORK: 0.6,
            ForkResolutionStrategy.VALIDATOR_VOTING: 0.3,
            ForkResolutionStrategy.TIMESTAMP: 0.1
        }
    
    def resolve_fork(self, chain_a: List[Any], chain_b: List[Any], 
                    common_ancestor: Any) -> Dict[str, Any]:
        """Resolve fork using configured strategy"""
        strategies = self._get_available_strategies()
        results = {}
        
        for strategy in strategies:
            if strategy == ForkResolutionStrategy.CHAINWORK:
                results['chainwork'] = self._resolve_by_chainwork(chain_a, chain_b)
            elif strategy == ForkResolutionStrategy.VALIDATOR_VOTING:
                results['voting'] = self._resolve_by_voting(chain_a, chain_b)
            elif strategy == ForkResolutionStrategy.TIMESTAMP:
                results['timestamp'] = self._resolve_by_timestamp(chain_a, chain_b)
        
        # Combine results using strategy weights
        final_result = self._combine_results(results)
        return final_result
    
    def _get_available_strategies(self) -> List[ForkResolutionStrategy]:
        """Get available resolution strategies based on configuration"""
        strategies = []
        
        if self.config.get('fork_resolution_chainwork', True):
            strategies.append(ForkResolutionStrategy.CHAINWORK)
        
        if self.config.get('fork_resolution_voting', False) and hasattr(self.consensus_manager, 'get_validator_votes'):
            strategies.append(ForkResolutionStrategy.VALIDATOR_VOTING)
        
        if self.config.get('fork_resolution_timestamp', True):
            strategies.append(ForkResolutionStrategy.TIMESTAMP)
        
        return strategies
    
    def _resolve_by_chainwork(self, chain_a: List[Any], chain_b: List[Any]) -> Dict[str, Any]:
        """Resolve fork by comparing chainwork"""
        chainwork_a = sum(block.chainwork for block in chain_a)
        chainwork_b = sum(block.chainwork for block in chain_b)
        
        return {
            'preferred_chain': 'a' if chainwork_a > chainwork_b else 'b',
            'confidence': abs(chainwork_a - chainwork_b) / max(chainwork_a, chainwork_b, 1),
            'chainwork_a': chainwork_a,
            'chainwork_b': chainwork_b
        }
    
    def _resolve_by_voting(self, chain_a: List[Any], chain_b: List[Any]) -> Dict[str, Any]:
        """Resolve fork by validator voting"""
        try:
            # Get validator votes for each chain
            votes_a = self.consensus_manager.get_validator_votes(chain_a[-1].hash if chain_a else None)
            votes_b = self.consensus_manager.get_validator_votes(chain_b[-1].hash if chain_b else None)
            
            total_votes = len(votes_a) + len(votes_b)
            if total_votes == 0:
                return {'preferred_chain': None, 'confidence': 0.0}
            
            # Calculate voting power for each chain
            voting_power_a = sum(vote.power for vote in votes_a)
            voting_power_b = sum(vote.power for vote in votes_b)
            
            return {
                'preferred_chain': 'a' if voting_power_a > voting_power_b else 'b',
                'confidence': abs(voting_power_a - voting_power_b) / max(voting_power_a, voting_power_b, 1),
                'voting_power_a': voting_power_a,
                'voting_power_b': voting_power_b
            }
            
        except Exception as e:
            # Fallback if voting is not available
            return {'preferred_chain': None, 'confidence': 0.0}
    
    def _resolve_by_timestamp(self, chain_a: List[Any], chain_b: List[Any]) -> Dict[str, Any]:
        """Resolve fork by comparing timestamps"""
        if not chain_a or not chain_b:
            return {'preferred_chain': None, 'confidence': 0.0}
        
        # Use median timestamp to reduce outlier impact
        timestamps_a = [block.header.timestamp for block in chain_a]
        timestamps_b = [block.header.timestamp for block in chain_b]
        
        median_a = sorted(timestamps_a)[len(timestamps_a) // 2]
        median_b = sorted(timestamps_b)[len(timestamps_b) // 2]
        
        # Prefer chain with more recent median timestamp
        return {
            'preferred_chain': 'a' if median_a > median_b else 'b',
            'confidence': abs(median_a - median_b) / max(median_a, median_b, 1),
            'median_timestamp_a': median_a,
            'median_timestamp_b': median_b
        }
    
    def _combine_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple resolution strategies"""
        if not results:
            return {'preferred_chain': None, 'confidence': 0.0}
        
        # Count votes from each strategy
        votes = {'a': 0.0, 'b': 0.0}
        total_confidence = 0.0
        
        for strategy_name, result in results.items():
            if result['preferred_chain'] and result['confidence'] > 0:
                weight = self.strategy_weights.get(
                    ForkResolutionStrategy[strategy_name.upper()], 0.0
                )
                votes[result['preferred_chain']] += result['confidence'] * weight
                total_confidence += result['confidence'] * weight
        
        if total_confidence == 0:
            return {'preferred_chain': None, 'confidence': 0.0}
        
        # Determine winner
        if votes['a'] > votes['b']:
            preferred_chain = 'a'
            confidence = votes['a'] / total_confidence
        elif votes['b'] > votes['a']:
            preferred_chain = 'b'
            confidence = votes['b'] / total_confidence
        else:
            preferred_chain = None
            confidence = 0.0
        
        return {
            'preferred_chain': preferred_chain,
            'confidence': confidence,
            'strategy_results': results
        }
    
    def update_strategy_weights(self, new_weights: Dict[ForkResolutionStrategy, float]):
        """Update strategy weights based on performance"""
        # Normalize weights to sum to 1.0
        total = sum(new_weights.values())
        if total > 0:
            self.strategy_weights = {k: v/total for k, v in new_weights.items()}
    
    def evaluate_strategy_performance(self, fork_history: List[Any]) -> Dict[ForkResolutionStrategy, float]:
        """Evaluate performance of resolution strategies"""
        performance = {strategy: 0.0 for strategy in self.strategy_weights.keys()}
        successful_resolutions = 0
        
        for fork in fork_history:
            if not hasattr(fork, 'was_successful') or not fork.was_successful():
                continue
            
            successful_resolutions += 1
            
            # Evaluate each strategy's performance on this fork
            # This would require storing strategy results in fork history
            
        # Normalize performance scores
        if successful_resolutions > 0:
            for strategy in performance:
                performance[strategy] /= successful_resolutions
        
        return performance