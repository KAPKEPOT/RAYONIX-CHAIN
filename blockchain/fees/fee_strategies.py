# blockchain/fees/fee_strategies.py
import time
import statistics
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

class FeeStrategy(ABC):
    """Abstract base class for fee estimation strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    @abstractmethod
    def estimate_fee(self, mempool_stats: Dict[str, Any], 
                    historical_data: List[Dict[str, Any]],
                    confirmation_target: int) -> int:
        """Estimate fee for given confirmation target"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass

class ConservativeFeeStrategy(FeeStrategy):
    """Conservative fee estimation - prioritizes reliability over speed"""
    
    def estimate_fee(self, mempool_stats: Dict[str, Any], 
                    historical_data: List[Dict[str, Any]],
                    confirmation_target: int) -> int:
        base_fee = self.config['min_transaction_fee']
        
        if not historical_data:
            return base_fee
        
        # Use 90th percentile of historical fees for safety
        recent_fees = [item['medium_fee'] for item in historical_data[-100:]]
        if recent_fees:
            conservative_fee = statistics.quantiles(recent_fees, n=10)[8]  # 90th percentile
            return max(base_fee, int(conservative_fee))
        
        return base_fee
    
    def get_strategy_name(self) -> str:
        return "conservative"

class AggressiveFeeStrategy(FeeStrategy):
    """Aggressive fee estimation - prioritizes speed over cost"""
    
    def estimate_fee(self, mempool_stats: Dict[str, Any], 
                    historical_data: List[Dict[str, Any]],
                    confirmation_target: int) -> int:
        base_fee = self.config['min_transaction_fee']
        
        # Use current mempool conditions aggressively
        capacity_usage = mempool_stats.get('capacity_usage', 0)
        avg_fee_rate = mempool_stats.get('average_fee_rate', 0)
        
        aggression_factor = 1.0 + (capacity_usage * 2.0)  # More congestion = more aggressive
        estimated_fee = max(base_fee, int(avg_fee_rate * aggression_factor))
        
        return estimated_fee
    
    def get_strategy_name(self) -> str:
        return "aggressive"

class AdaptiveFeeStrategy(FeeStrategy):
    """Adaptive fee estimation - balances cost and speed based on conditions"""
    
    def estimate_fee(self, mempool_stats: Dict[str, Any], 
                    historical_data: List[Dict[str, Any]],
                    confirmation_target: int) -> int:
        base_fee = self.config['min_transaction_fee']
        
        if not historical_data or not mempool_stats:
            return base_fee
        
        # Calculate based on both historical and current conditions
        recent_fees = [item['medium_fee'] for item in historical_data[-50:]]
        historical_avg = statistics.mean(recent_fees) if recent_fees else base_fee
        
        current_fee_rate = mempool_stats.get('average_fee_rate', historical_avg)
        capacity_usage = mempool_stats.get('capacity_usage', 0.5)
        
        # Weight current vs historical based on congestion
        current_weight = min(1.0, capacity_usage * 1.5)  # More congestion = more weight to current
        historical_weight = 1.0 - current_weight
        
        estimated_fee = (current_fee_rate * current_weight) + (historical_avg * historical_weight)
        
        # Adjust for confirmation target
        target_factor = max(1.0, 6.0 / confirmation_target)
        estimated_fee *= target_factor
        
        return max(base_fee, int(estimated_fee))
    
    def get_strategy_name(self) -> str:
        return "adaptive"

class MachineLearningFeeStrategy(FeeStrategy):
    """Machine learning-based fee estimation (placeholder implementation)"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize machine learning model (placeholder)"""
        # This would load or train a ML model for fee prediction
        # For now, this is a placeholder
        pass
    
    def estimate_fee(self, mempool_stats: Dict[str, Any], 
                    historical_data: List[Dict[str, Any]],
                    confirmation_target: int) -> int:
        base_fee = self.config['min_transaction_fee']
        
        # Placeholder ML-based estimation
        # In a real implementation, this would use features like:
        # - Mempool size and composition
        # - Historical fee patterns
        # - Time of day, day of week
        # - Network activity metrics
        
        if self.model:
            # Would use actual model prediction here
            predicted_fee = base_fee * 2  # Placeholder
        else:
            # Fallback to adaptive strategy
            adaptive = AdaptiveFeeStrategy(self.config)
            predicted_fee = adaptive.estimate_fee(mempool_stats, historical_data, confirmation_target)
        
        return max(base_fee, int(predicted_fee))
    
    def get_strategy_name(self) -> str:
        return "machine_learning"

class FeeStrategyFactory:
    """Factory for creating fee estimation strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {
            'conservative': ConservativeFeeStrategy(config),
            'aggressive': AggressiveFeeStrategy(config),
            'adaptive': AdaptiveFeeStrategy(config),
            'ml': MachineLearningFeeStrategy(config)
        }
    
    def get_strategy(self, strategy_name: str) -> FeeStrategy:
        """Get fee strategy by name"""
        return self.strategies.get(strategy_name, self.strategies['adaptive'])
    
    def register_strategy(self, strategy_name: str, strategy: FeeStrategy):
        """Register a custom fee strategy"""
        self.strategies[strategy_name] = strategy
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())
    
    def create_composite_strategy(self, strategy_weights: Dict[str, float]) -> FeeStrategy:
        """Create a composite strategy that combines multiple strategies"""
        class CompositeFeeStrategy(FeeStrategy):
            def __init__(self, strategies, weights, config):
                super().__init__(config)
                self.strategies = strategies
                self.weights = weights
            
            def estimate_fee(self, mempool_stats, historical_data, confirmation_target):
                fees = []
                weights = []
                
                for strategy_name, weight in self.weights.items():
                    if weight > 0 and strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        fee = strategy.estimate_fee(mempool_stats, historical_data, confirmation_target)
                        fees.append(fee)
                        weights.append(weight)
                
                if not fees:
                    return self.config['min_transaction_fee']
                
                # Weighted average of strategy estimates
                weighted_sum = sum(fee * weight for fee, weight in zip(fees, weights))
                total_weight = sum(weights)
                
                return int(weighted_sum / total_weight)
            
            def get_strategy_name(self):
                return "composite"
        
        return CompositeFeeStrategy(self.strategies, strategy_weights, self.config)