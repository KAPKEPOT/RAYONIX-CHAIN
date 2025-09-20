# blockchain/utils/gas_management.py
import time
from typing import Dict, List, Any, Deque
from collections import deque

class GasPriceManager:
    """Manages gas price calculations and adjustments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_gas_price = config.get('base_gas_price', 1000000000)
        self.min_gas_price = config.get('min_gas_price', 500000000)
        self.max_gas_price = config.get('max_gas_price', 10000000000)
        self.adjustment_factor = config.get('adjustment_factor', 1.125)
        self.target_utilization = config.get('target_utilization', 0.5)
        self.gas_price = self.base_gas_price
        self.gas_price_history: Deque = deque(maxlen=1000)
        self.update_errors = 0
    
    def update_gas_price(self, mempool_size: int, block_utilization: float) -> int:
        """Update gas price based on network conditions"""
        try:
            # Calculate new gas price based on utilization
            utilization = min(1.0, max(0.0, block_utilization))
            
            if utilization > self.target_utilization:
                # Increase gas price when utilization is high
                adjustment = self.adjustment_factor
                new_price = int(self.gas_price * adjustment)
            else:
                # Decrease gas price when utilization is low
                adjustment = 1.0 / self.adjustment_factor
                new_price = int(self.gas_price * adjustment)
            
            # Apply min/max bounds
            new_price = max(self.min_gas_price, min(new_price, self.max_gas_price))
            
            # Update with smoothing to avoid drastic changes
            smoothing = self.config.get('smoothing_factor', 0.8)
            self.gas_price = int(smoothing * self.gas_price + (1 - smoothing) * new_price)
            
            # Record in history
            self.gas_price_history.append({
                'timestamp': time.time(),
                'gas_price': self.gas_price,
                'mempool_size': mempool_size,
                'utilization': utilization,
                'adjustment': adjustment
            })
            
            self.update_errors = 0
            return self.gas_price
            
        except Exception as e:
            self.update_errors += 1
            # If too many errors, reset to base price
            if self.update_errors > 10:
                self.gas_price = self.base_gas_price
            return self.gas_price
    
    def get_gas_estimate(self, computation_units: int, priority: str = 'medium') -> int:
        """Estimate gas cost for transaction"""
        priority_multipliers = {
            'low': 0.8,
            'medium': 1.0,
            'high': 1.2,
            'urgent': 1.5
        }
        
        multiplier = priority_multipliers.get(priority, 1.0)
        estimated_gas = computation_units * self.gas_price * multiplier
        
        # Round to significant figures
        return int(round(estimated_gas, -int(len(str(int(estimated_gas))) - 2)))
    
    def get_gas_price_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get gas price history for specified period"""
        cutoff = time.time() - (hours * 3600)
        return [entry for entry in self.gas_price_history if entry['timestamp'] >= cutoff]
    
    def calculate_utilization(self, block: Any) -> float:
        """Calculate block utilization percentage"""
        if not hasattr(block, 'gas_used') or not hasattr(block, 'gas_limit'):
            return 0.0
        
        if block.gas_limit == 0:
            return 0.0
        
        return block.gas_used / block.gas_limit
    
    def get_recommended_priority(self, confirmation_target: int) -> str:
        """Get recommended priority level for confirmation target"""
        # Analyze recent blocks to determine appropriate priority
        recent_utilization = self._get_recent_utilization()
        
        if confirmation_target <= 1:
            return 'urgent'
        elif confirmation_target <= 3:
            return 'high'
        elif confirmation_target <= 6:
            if recent_utilization > 0.8:
                return 'high'
            else:
                return 'medium'
        else:
            if recent_utilization > 0.9:
                return 'medium'
            else:
                return 'low'
    
    def _get_recent_utilization(self) -> float:
        """Get recent block utilization average"""
        if not self.gas_price_history:
            return 0.5  # Default assumption
        
        # Get recent utilization values (last 10 entries)
        recent_entries = list(self.gas_price_history)[-10:]
        if not recent_entries:
            return 0.5
        
        utilizations = [entry['utilization'] for entry in recent_entries]
        return sum(utilizations) / len(utilizations)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gas price statistics"""
        if not self.gas_price_history:
            return {
                'current_gas_price': self.gas_price,
                'average_gas_price': self.gas_price,
                'min_gas_price': self.gas_price,
                'max_gas_price': self.gas_price,
                'update_errors': self.update_errors
            }
        
        prices = [entry['gas_price'] for entry in self.gas_price_history]
        utilizations = [entry['utilization'] for entry in self.gas_price_history]
        
        return {
            'current_gas_price': self.gas_price,
            'average_gas_price': sum(prices) / len(prices),
            'min_gas_price': min(prices),
            'max_gas_price': max(prices),
            'average_utilization': sum(utilizations) / len(utilizations) if utilizations else 0,
            'update_errors': self.update_errors,
            'history_size': len(self.gas_price_history)
        }