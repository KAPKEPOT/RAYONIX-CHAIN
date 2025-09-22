# smart_contract/security/behavioral_analyzer.py
import time
import logging
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque

logger = logging.getLogger("SmartContract.BehavioralAnalyzer")

@dataclass
class BehavioralConfig:
    """Configuration for behavioral analysis"""
    history_size: int = 1000
    anomaly_threshold: float = 3.0  # Standard deviations
    min_samples: int = 10
    update_interval: int = 300  # 5 minutes
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        'execution_time': 0.3,
        'gas_used': 0.25,
        'memory_used': 0.2,
        'storage_operations': 0.15,
        'call_frequency': 0.1
    })

class BehavioralAnalyzer:
    """Advanced behavioral analysis system for anomaly detection"""
    
    def __init__(self, config: Optional[BehavioralConfig] = None):
        self.config = config or BehavioralConfig()
        
        # Behavioral history storage
        self.history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self.config.history_size))
        )
        
        # Statistical models
        self.models: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        
        # Anomaly detection state
        self.anomaly_scores: Dict[str, List[float]] = defaultdict(list)
        self.last_update = time.time()
        
        logger.info("BehavioralAnalyzer initialized")
    
    def analyze(self, contract_id: str, operation: str, metrics: Dict[str, Any]) -> Tuple[bool, float, List[str]]:
        """
        Analyze behavior for anomalies
        Returns: (is_normal, confidence, anomalies)
        """
        try:
            # Update history
            self._update_history(contract_id, operation, metrics)
            
            # Check if we have enough data
            if len(self.history[contract_id][operation]) < self.config.min_samples:
                return True, 0.0, []
            
            # Calculate anomaly score
            anomaly_score, anomalies = self._calculate_anomaly_score(contract_id, operation, metrics)
            
            # Update anomaly scores
            self.anomaly_scores[contract_id].append(anomaly_score)
            if len(self.anomaly_scores[contract_id]) > 1000:
                self.anomaly_scores[contract_id] = self.anomaly_scores[contract_id][-500:]
            
            # Determine if normal based on threshold
            is_normal = anomaly_score <= self.config.anomaly_threshold
            confidence = min(1.0, 1.0 - (anomaly_score / (self.config.anomaly_threshold * 2)))
            
            # Update models periodically
            current_time = time.time()
            if current_time - self.last_update >= self.config.update_interval:
                self._update_models()
                self.last_update = current_time
            
            return is_normal, confidence, anomalies
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed for {contract_id}.{operation}: {e}")
            return True, 0.0, []  # Allow on error
    
    def _update_history(self, contract_id: str, operation: str, metrics: Dict[str, Any]) -> None:
        """Update behavioral history"""
        timestamp = time.time()
        entry = {
            'timestamp': timestamp,
            'metrics': metrics.copy()
        }
        self.history[contract_id][operation].append(entry)
    
    def _calculate_anomaly_score(self, contract_id: str, operation: str, current_metrics: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Calculate anomaly score for current behavior"""
        history = list(self.history[contract_id][operation])
        anomalies = []
        total_score = 0.0
        total_weight = 0.0
        
        # Get historical data for each metric
        for metric_name, weight in self.config.feature_weights.items():
            if metric_name in current_metrics:
                historical_values = [
                    entry['metrics'].get(metric_name, 0) 
                    for entry in history
                    if metric_name in entry['metrics']
                ]
                
                if len(historical_values) >= self.config.min_samples:
                    current_value = current_metrics[metric_name]
                    anomaly_score = self._calculate_metric_anomaly(
                        historical_values, current_value, metric_name
                    )
                    
                    if anomaly_score > self.config.anomaly_threshold:
                        anomalies.append(f"{metric_name}_anomaly")
                    
                    total_score += anomaly_score * weight
                    total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            normalized_score = total_score / total_weight
        else:
            normalized_score = 0.0
        
        return normalized_score, anomalies
    
    def _calculate_metric_anomaly(self, historical_values: List[float], current_value: float, metric_name: str) -> float:
        """Calculate anomaly score for a single metric"""
        try:
            if not historical_values:
                return 0.0
            
            # Calculate statistics
            mean = statistics.mean(historical_values)
            stdev = statistics.stdev(historical_values) if len(historical_values) > 1 else 0.0
            
            # Handle zero standard deviation
            if stdev == 0:
                # If all historical values are the same, any difference is anomalous
                return abs(current_value - mean) / (mean + 1e-9)  # Avoid division by zero
            
            # Calculate z-score
            z_score = abs(current_value - mean) / stdev
            
            # Apply metric-specific adjustments
            if metric_name in ['execution_time', 'gas_used', 'memory_used']:
                # These metrics often have right-skewed distributions
                z_score = self._adjust_for_skewness(z_score, historical_values, current_value)
            
            return z_score
            
        except Exception as e:
            logger.error(f"Anomaly calculation failed for {metric_name}: {e}")
            return 0.0
    
    def _adjust_for_skewness(self, z_score: float, historical_values: List[float], current_value: float) -> float:
        """Adjust z-score for skewed distributions"""
        # Use logarithmic transformation for right-skewed data
        if current_value > 0 and all(v > 0 for v in historical_values):
            log_values = [np.log(v) for v in historical_values]
            log_current = np.log(current_value)
            log_mean = statistics.mean(log_values)
            log_stdev = statistics.stdev(log_values) if len(log_values) > 1 else 0.0
            
            if log_stdev > 0:
                log_z_score = abs(log_current - log_mean) / log_stdev
                # Use the maximum of original and log z-score
                return max(z_score, log_z_score)
        
        return z_score
    
    def _update_models(self) -> None:
        """Update statistical models based on recent history"""
        for contract_id in self.history:
            for operation in self.history[contract_id]:
                history = list(self.history[contract_id][operation])
                if len(history) >= self.config.min_samples:
                    self.models[contract_id][operation] = self._create_model(history)
    
    def _create_model(self, history: List[Dict]) -> Dict[str, Any]:
        """Create a statistical model from history"""
        model = {}
        
        # Extract all metrics from history
        all_metrics = set()
        for entry in history:
            all_metrics.update(entry['metrics'].keys())
        
        # Calculate statistics for each metric
        for metric in all_metrics:
            values = [entry['metrics'].get(metric, 0) for entry in history]
            if values:
                model[metric] = {
                    'mean': statistics.mean(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return model
    
    def get_behavioral_profile(self, contract_id: str, operation: str) -> Optional[Dict[str, Any]]:
        """Get behavioral profile for a contract operation"""
        if contract_id in self.models and operation in self.models[contract_id]:
            return self.models[contract_id][operation].copy()
        return None
    
    def get_anomaly_history(self, contract_id: str, window: int = 3600) -> List[float]:
        """Get anomaly scores for a contract within time window"""
        current_time = time.time()
        # This would filter by timestamp in a real implementation
        return self.anomaly_scores.get(contract_id, [])[-100:]  # Return last 100 scores
    
    def reset_behavior(self, contract_id: str, operation: Optional[str] = None) -> None:
        """Reset behavioral history for a contract or operation"""
        if operation:
            if contract_id in self.history and operation in self.history[contract_id]:
                self.history[contract_id][operation].clear()
        else:
            if contract_id in self.history:
                self.history[contract_id].clear()
        
        logger.info(f"Behavioral history reset for {contract_id}" + (f".{operation}" if operation else ""))
    
    def cleanup(self) -> None:
        """Clean up old data"""
        current_time = time.time()
        for contract_id in list(self.history.keys()):
            for operation in list(self.history[contract_id].keys()):
                # Remove old entries
                self.history[contract_id][operation] = deque(
                    [entry for entry in self.history[contract_id][operation] 
                     if current_time - entry['timestamp'] <= 86400],  # 24 hours
                    maxlen=self.config.history_size
                )
                
                # Remove empty operations
                if not self.history[contract_id][operation]:
                    del self.history[contract_id][operation]
            
            # Remove empty contracts
            if not self.history[contract_id]:
                del self.history[contract_id]
        
        logger.debug("BehavioralAnalyzer cleanup completed")