# blockchain/fees/fee_estimator.py
import time
import statistics
from typing import Dict, List, Any, Deque, Optional
from collections import deque
import logging

from blockchain.models.transaction_results import FeeEstimate

logger = logging.getLogger(__name__)

class FeeEstimator:
    """Dynamic fee estimation based on network conditions"""
    
    def __init__(self, state_manager: Any, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.config = config
        self.fee_history: Deque = deque(maxlen=1000)
        self.mempool_stats: Deque = deque(maxlen=100)
        self.last_estimate = FeeEstimate(0, 0, 0, 0, time.time(), 0.0, 0)
        self.historical_data: List[Dict[str, Any]] = []
        self.update_interval = config.get('fee_estimation_interval', 30)
        self.last_update = 0
    
    def estimate_fee(self, strategy: str = 'medium', confirmation_target: int = 6) -> int:
        """Estimate transaction fee based on strategy and confirmation target"""
        current_stats = self._get_current_mempool_stats()
        historical_stats = self._get_historical_stats()
        
        # Update estimate if needed
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self._update_fee_estimate()
            self.last_update = current_time
        
        if strategy == 'low':
            return self._calculate_low_fee(current_stats, historical_stats, confirmation_target)
        elif strategy == 'medium':
            return self._calculate_medium_fee(current_stats, historical_stats, confirmation_target)
        elif strategy == 'high':
            return self._calculate_high_fee(current_stats, historical_stats, confirmation_target)
        elif strategy == 'urgent':
            return self._calculate_urgent_fee(current_stats, historical_stats, confirmation_target)
        else:
            return self.config['min_transaction_fee']
    
    def _update_fee_estimate(self):
        """Update fee estimate based on current conditions"""
        try:
            mempool_stats = self._get_current_mempool_stats()
            historical_stats = self._get_historical_stats()
            
            low_fee = self._calculate_low_fee(mempool_stats, historical_stats, 12)
            medium_fee = self._calculate_medium_fee(mempool_stats, historical_stats, 6)
            high_fee = self._calculate_high_fee(mempool_stats, historical_stats, 3)
            urgent_fee = self._calculate_urgent_fee(mempool_stats, historical_stats, 1)
            
            confidence = self._calculate_confidence(mempool_stats, historical_stats)
            mempool_size = mempool_stats.get('transaction_count', 0)
            
            self.last_estimate = FeeEstimate(
                low=low_fee,
                medium=medium_fee,
                high=high_fee,
                urgent=urgent_fee,
                timestamp=time.time(),
                confidence=confidence,
                mempool_size=mempool_size
            )
            
            # Store historical data
            self.historical_data.append({
                'timestamp': time.time(),
                'low_fee': low_fee,
                'medium_fee': medium_fee,
                'high_fee': high_fee,
                'urgent_fee': urgent_fee,
                'confidence': confidence,
                'mempool_size': mempool_size
            })
            
            # Keep only recent historical data
            if len(self.historical_data) > 1000:
                self.historical_data = self.historical_data[-1000:]
                
        except Exception as e:
            logger.error(f"Failed to update fee estimate: {e}")
    
    def _get_current_mempool_stats(self) -> Dict[str, Any]:
        """Get current mempool statistics"""
        try:
            # This would query the actual mempool state
            mempool = self.state_manager.transaction_manager.mempool
            if hasattr(mempool, 'get_stats'):
                return mempool.get_stats()
            
            # Fallback implementation
            fee_rates = []
            sizes = []
            
            for tx_hash, (tx, timestamp, fee_rate) in mempool.items():
                fee_rates.append(fee_rate)
                sizes.append(len(tx.to_bytes()))
            
            return {
                'transaction_count': len(mempool),
                'total_size_bytes': sum(sizes) if sizes else 0,
                'average_fee_rate': statistics.mean(fee_rates) if fee_rates else 0,
                'min_fee_rate': min(fee_rates) if fee_rates else 0,
                'max_fee_rate': max(fee_rates) if fee_rates else 0,
                'capacity_usage': len(mempool) / self.config.get('max_mempool_size', 10000)
            }
        except Exception as e:
            logger.error(f"Error getting mempool stats: {e}")
            return {
                'transaction_count': 0,
                'total_size_bytes': 0,
                'average_fee_rate': 0,
                'capacity_usage': 0
            }
    
    def _get_historical_stats(self) -> Dict[str, Any]:
        """Get historical fee statistics"""
        if not self.historical_data:
            return {
                'average_fees': [],
                'congestion_patterns': {},
                'seasonal_trends': {}
            }
        
        # Calculate moving averages
        window = min(100, len(self.historical_data))
        recent_data = self.historical_data[-window:]
        
        avg_low = statistics.mean(item['low_fee'] for item in recent_data)
        avg_medium = statistics.mean(item['medium_fee'] for item in recent_data)
        avg_high = statistics.mean(item['high_fee'] for item in recent_data)
        avg_urgent = statistics.mean(item['urgent_fee'] for item in recent_data)
        
        return {
            'average_fees': {
                'low': avg_low,
                'medium': avg_medium,
                'high': avg_high,
                'urgent': avg_urgent
            },
            'congestion_patterns': self._analyze_congestion_patterns(),
            'seasonal_trends': self._analyze_seasonal_trends()
        }
    
    def _calculate_low_fee(self, current_stats: Dict, historical_stats: Dict, 
                          confirmation_target: int) -> int:
        """Calculate low priority fee"""
        base_fee = self.config['min_transaction_fee']
        
        # Add congestion adjustment
        congestion_factor = self._get_congestion_factor(current_stats, confirmation_target)
        adjusted_fee = base_fee * congestion_factor
        
        # Apply historical smoothing
        historical_low = historical_stats['average_fees'].get('low', base_fee)
        smoothed_fee = (adjusted_fee * 0.7) + (historical_low * 0.3)
        
        return max(base_fee, int(smoothed_fee))
    
    def _calculate_medium_fee(self, current_stats: Dict, historical_stats: Dict,
                             confirmation_target: int) -> int:
        """Calculate medium priority fee"""
        base_fee = self.config['min_transaction_fee'] * 2
        
        congestion_factor = self._get_congestion_factor(current_stats, confirmation_target)
        adjusted_fee = base_fee * congestion_factor
        
        historical_medium = historical_stats['average_fees'].get('medium', base_fee)
        smoothed_fee = (adjusted_fee * 0.6) + (historical_medium * 0.4)
        
        return max(base_fee, int(smoothed_fee))
    
    def _calculate_high_fee(self, current_stats: Dict, historical_stats: Dict,
                           confirmation_target: int) -> int:
        """Calculate high priority fee"""
        base_fee = self.config['min_transaction_fee'] * 5
        
        congestion_factor = self._get_congestion_factor(current_stats, confirmation_target)
        adjusted_fee = base_fee * congestion_factor
        
        historical_high = historical_stats['average_fees'].get('high', base_fee)
        smoothed_fee = (adjusted_fee * 0.5) + (historical_high * 0.5)
        
        return max(base_fee, int(smoothed_fee))
    
    def _calculate_urgent_fee(self, current_stats: Dict, historical_stats: Dict,
                             confirmation_target: int) -> int:
        """Calculate urgent priority fee"""
        base_fee = self.config['min_transaction_fee'] * 10
        
        congestion_factor = self._get_congestion_factor(current_stats, confirmation_target)
        adjusted_fee = base_fee * congestion_factor
        
        historical_urgent = historical_stats['average_fees'].get('urgent', base_fee)
        smoothed_fee = (adjusted_fee * 0.4) + (historical_urgent * 0.6)
        
        return max(base_fee, int(smoothed_fee))
    
    def _get_congestion_factor(self, current_stats: Dict, confirmation_target: int) -> float:
        """Calculate congestion factor based on current mempool state"""
        capacity_usage = current_stats.get('capacity_usage', 0)
        avg_fee_rate = current_stats.get('average_fee_rate', 0)
        min_fee_rate = current_stats.get('min_fee_rate', 0)
        
        # Base congestion factor
        congestion = max(1.0, capacity_usage / 0.5)  # 1.0 at 50% capacity
        
        # Adjust for confirmation target
        target_factor = max(1.0, 6.0 / confirmation_target)  # More urgent = higher fee
        
        # Adjust for fee rate distribution
        if avg_fee_rate > 0 and min_fee_rate > 0:
            fee_spread = avg_fee_rate / min_fee_rate
            spread_factor = min(2.0, fee_spread)  # Cap at 2.0
        else:
            spread_factor = 1.0
        
        return congestion * target_factor * spread_factor
    
    def _calculate_confidence(self, current_stats: Dict, historical_stats: Dict) -> float:
        """Calculate confidence level for fee estimate"""
        # Base confidence on data quality
        data_points = len(self.historical_data)
        data_confidence = min(1.0, data_points / 100.0)  # 100% confidence at 100 data points
        
        # Confidence based on stability
        stability_confidence = self._calculate_stability_confidence()
        
        # Confidence based on mempool health
        mempool_health = 1.0 - min(1.0, current_stats.get('capacity_usage', 0) / 0.8)  # 80% capacity = 0% health confidence
        
        return (data_confidence * 0.4) + (stability_confidence * 0.4) + (mempool_health * 0.2)
    
    def _calculate_stability_confidence(self) -> float:
        """Calculate confidence based on fee stability"""
        if len(self.historical_data) < 10:
            return 0.5  # Medium confidence with little data
        
        # Calculate fee volatility
        recent_fees = [item['medium_fee'] for item in self.historical_data[-10:]]
        if len(recent_fees) > 1:
            volatility = statistics.stdev(recent_fees) / statistics.mean(recent_fees)
            stability = 1.0 - min(1.0, volatility)  # 0 volatility = 1.0 stability
            return max(0.1, stability)  # At least 10% confidence
        else:
            return 0.5
    
    def _analyze_congestion_patterns(self) -> Dict[str, Any]:
        """Analyze historical congestion patterns"""
        if len(self.historical_data) < 24:  # Need at least 24 data points
            return {}
        
        # Analyze daily patterns
        hourly_patterns = {}
        for hour in range(24):
            hour_data = [item for item in self.historical_data 
                        if time.localtime(item['timestamp']).tm_hour == hour]
            if hour_data:
                avg_fee = statistics.mean(item['medium_fee'] for item in hour_data)
                hourly_patterns[hour] = avg_fee
        
        return {'hourly_patterns': hourly_patterns}
    
    def _analyze_seasonal_trends(self) -> Dict[str, Any]:
        """Analyze seasonal trends in fee patterns"""
        # This would implement more sophisticated seasonal analysis
        # For now, return empty result
        return {}
    
    def update_fee_history(self, block: Any):
        """Update fee history with new block data"""
        try:
            block_fees = sum(tx.fee for tx in block.transactions if hasattr(tx, 'fee') and tx.fee > 0)
            transaction_count = len(block.transactions)
            average_fee = block_fees / transaction_count if transaction_count > 0 else 0
            
            self.fee_history.append({
                'height': block.header.height,
                'timestamp': block.header.timestamp,
                'average_fee': average_fee,
                'total_fees': block_fees,
                'transaction_count': transaction_count,
                'block_size': block.size
            })
            
        except Exception as e:
            logger.error(f"Error updating fee history: {e}")
    
    def get_fee_estimate(self) -> FeeEstimate:
        """Get current fee estimate"""
        return self.last_estimate
    
    def get_historical_estimates(self, hours: int = 24) -> List[FeeEstimate]:
        """Get historical fee estimates for specified period"""
        cutoff = time.time() - (hours * 3600)
        return [item for item in self.historical_data if item['timestamp'] >= cutoff]