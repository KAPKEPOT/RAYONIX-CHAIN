# smart_contract/core/gas_system/gas_optimizer.py
import logging
import statistics
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("SmartContract.GasOptimizer")

@dataclass
class OptimizationProfile:
    """Profile for gas optimization strategies"""
    memory_optimization: bool = True
    computation_optimization: bool = True
    storage_optimization: bool = True
    caching_enabled: bool = True
    precomputation_enabled: bool = True
    batch_operations: bool = True
    
    # Thresholds for optimization
    memory_threshold_bytes: int = 1024 * 1024  # 1MB
    computation_threshold_ops: int = 1000
    storage_threshold_ops: int = 100
    cache_size_threshold: int = 1000

class GasOptimizer:
    """Advanced gas optimization system with machine learning capabilities"""
    
    def __init__(self, profile: Optional[OptimizationProfile] = None):
        self.profile = profile or OptimizationProfile()
        self.optimization_cache: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self.pattern_detection: Dict[str, Any] = {}
        
        # Machine learning features (would integrate with actual ML models)
        self.feature_weights = {
            'memory_usage': 0.3,
            'computation_intensity': 0.4,
            'storage_operations': 0.2,
            'network_latency': 0.1
        }
        
        logger.info("GasOptimizer initialized with advanced optimization capabilities")
    
    def analyze_contract(self, contract_bytecode: bytes, historical_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Analyze contract bytecode for optimization opportunities"""
        analysis = {
            'optimization_opportunities': [],
            'estimated_savings': 0,
            'risk_level': 'low',
            'recommended_actions': []
        }
        
        try:
            # Pattern detection for common gas-intensive operations
            patterns = self._detect_patterns(contract_bytecode)
            
            # Memory optimization analysis
            if self.profile.memory_optimization:
                memory_analysis = self._analyze_memory_usage(patterns)
                if memory_analysis['savings_potential'] > 0:
                    analysis['optimization_opportunities'].append({
                        'type': 'memory',
                        'savings_potential': memory_analysis['savings_potential'],
                        'recommendations': memory_analysis['recommendations']
                    })
            
            # Computation optimization analysis
            if self.profile.computation_optimization:
                comp_analysis = self._analyze_computation(patterns)
                if comp_analysis['savings_potential'] > 0:
                    analysis['optimization_opportunities'].append({
                        'type': 'computation',
                        'savings_potential': comp_analysis['savings_potential'],
                        'recommendations': comp_analysis['recommendations']
                    })
            
            # Storage optimization analysis
            if self.profile.storage_optimization:
                storage_analysis = self._analyze_storage(patterns)
                if storage_analysis['savings_potential'] > 0:
                    analysis['optimization_opportunities'].append({
                        'type': 'storage',
                        'savings_potential': storage_analysis['savings_potential'],
                        'recommendations': storage_analysis['recommendations']
                    })
            
            # Calculate total estimated savings
            analysis['estimated_savings'] = sum(
                opp['savings_potential'] for opp in analysis['optimization_opportunities']
            )
            
            # Risk assessment
            analysis['risk_level'] = self._assess_risk(patterns)
            
            # Generate recommendations
            analysis['recommended_actions'] = self._generate_recommendations(analysis['optimization_opportunities'])
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _detect_patterns(self, bytecode: bytes) -> Dict[str, Any]:
        """Detect patterns in contract bytecode"""
        # This would use advanced pattern recognition algorithms
        # For now, return mock patterns
        return {
            'memory_patterns': {
                'excessive_allocation': True,
                'inefficient_copying': False,
                'memory_leak_pattern': False
            },
            'computation_patterns': {
                'expensive_loops': True,
                'redundant_calculations': True,
                'inefficient_algorithms': False
            },
            'storage_patterns': {
                'frequent_small_writes': True,
                'inefficient_data_layout': False,
                'unnecessary_storage': True
            }
        }
    
    def _analyze_memory_usage(self, patterns: Dict) -> Dict[str, Any]:
        """Analyze memory usage patterns"""
        savings = 0
        recommendations = []
        
        if patterns['memory_patterns']['excessive_allocation']:
            savings += 15000
            recommendations.append("Use memory pools for frequent allocations")
        
        if patterns['memory_patterns']['inefficient_copying']:
            savings += 8000
            recommendations.append("Optimize memory copying using bulk operations")
        
        return {
            'savings_potential': savings,
            'recommendations': recommendations
        }
    
    def _analyze_computation(self, patterns: Dict) -> Dict[str, Any]:
        """Analyze computation patterns"""
        savings = 0
        recommendations = []
        
        if patterns['computation_patterns']['expensive_loops']:
            savings += 25000
            recommendations.append("Optimize loop structures and reduce iterations")
        
        if patterns['computation_patterns']['redundant_calculations']:
            savings += 18000
            recommendations.append("Cache results of expensive computations")
        
        if patterns['computation_patterns']['inefficient_algorithms']:
            savings += 35000
            recommendations.append("Replace with more efficient algorithms")
        
        return {
            'savings_potential': savings,
            'recommendations': recommendations
        }
    
    def _analyze_storage(self, patterns: Dict) -> Dict[str, Any]:
        """Analyze storage patterns"""
        savings = 0
        recommendations = []
        
        if patterns['storage_patterns']['frequent_small_writes']:
            savings += 42000
            recommendations.append("Batch storage operations together")
        
        if patterns['storage_patterns']['inefficient_data_layout']:
            savings += 28000
            recommendations.append("Optimize data layout for storage efficiency")
        
        if patterns['storage_patterns']['unnecessary_storage']:
            savings += 15000
            recommendations.append("Remove unnecessary storage operations")
        
        return {
            'savings_potential': savings,
            'recommendations': recommendations
        }
    
    def _assess_risk(self, patterns: Dict) -> str:
        """Assess optimization risk level"""
        risk_score = 0
        
        if patterns['memory_patterns']['memory_leak_pattern']:
            risk_score += 30
        
        if patterns['computation_patterns']['expensive_loops']:
            risk_score += 20
        
        if patterns['storage_patterns']['frequent_small_writes']:
            risk_score += 10
        
        if risk_score >= 50:
            return 'high'
        elif risk_score >= 20:
            return 'medium'
        else:
            return 'low'
    
    def _generate_recommendations(self, opportunities: List[Dict]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        for opp in opportunities:
            if opp['savings_potential'] > 10000:  # Significant savings
                recommendations.extend(opp['recommendations'])
        
        return list(set(recommendations))  # Remove duplicates
    
    def optimize_execution_plan(self, execution_plan: Dict, current_metrics: Dict) -> Dict:
        """Optimize execution plan based on current metrics"""
        optimized_plan = execution_plan.copy()
        
        # Memory optimization
        if (self.profile.memory_optimization and 
            current_metrics.get('memory_usage', 0) > self.profile.memory_threshold_bytes):
            optimized_plan = self._apply_memory_optimization(optimized_plan)
        
        # Computation optimization
        if (self.profile.computation_optimization and 
            current_metrics.get('computation_intensity', 0) > self.profile.computation_threshold_ops):
            optimized_plan = self._apply_computation_optimization(optimized_plan)
        
        # Storage optimization
        if (self.profile.storage_optimization and 
            current_metrics.get('storage_operations', 0) > self.profile.storage_threshold_ops):
            optimized_plan = self._apply_storage_optimization(optimized_plan)
        
        return optimized_plan
    
    def _apply_memory_optimization(self, plan: Dict) -> Dict:
        """Apply memory optimization strategies"""
        # Implementation would include:
        # - Memory pooling
        # - Bulk memory operations
        # - Cache-friendly data structures
        return plan
    
    def _apply_computation_optimization(self, plan: Dict) -> Dict:
        """Apply computation optimization strategies"""
        # Implementation would include:
        # - Loop unrolling
        # - Precomputation
        # - Algorithm optimization
        return plan
    
    def _apply_storage_optimization(self, plan: Dict) -> Dict:
        """Apply storage optimization strategies"""
        # Implementation would include:
        # - Batch storage operations
        # - Efficient data encoding
        # - Storage layout optimization
        return plan
    
    def update_performance_metrics(self, metrics: Dict) -> None:
        """Update performance metrics for ML training"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.performance_metrics[key].append(value)
                
                # Keep only recent data for analysis
                if len(self.performance_metrics[key]) > 1000:
                    self.performance_metrics[key] = self.performance_metrics[key][-1000:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                stats[f'{key}_mean'] = statistics.mean(values)
                stats[f'{key}_median'] = statistics.median(values)
                stats[f'{key}_std'] = statistics.stdev(values) if len(values) > 1 else 0
                stats[f'{key}_min'] = min(values)
                stats[f'{key}_max'] = max(values)
                stats[f'{key}_count'] = len(values)
        
        return stats
    
    def predict_optimal_gas_limit(self, contract_id: str, function_name: str, 
                                args: Dict) -> int:
        """Predict optimal gas limit for a function call"""
        # This would use machine learning models trained on historical data
        # For now, return a conservative estimate
        base_estimate = 1000000  # 1 million gas
        
        # Adjust based on function complexity
        complexity_factors = {
            'complex_math': 1.5,
            'storage_heavy': 2.0,
            'memory_intensive': 1.8,
            'network_calls': 2.5
        }
        
        # Simple heuristic-based prediction
        adjustment = 1.0
        if 'transfer' in function_name.lower():
            adjustment *= 1.2
        if 'calculate' in function_name.lower():
            adjustment *= 1.5
        if 'batch' in function_name.lower():
            adjustment *= 1.8
        
        return int(base_estimate * adjustment)
    
    def cleanup(self) -> None:
        """Cleanup optimization resources"""
        self.optimization_cache.clear()
        self.performance_metrics.clear()
        logger.info("GasOptimizer cleanup completed")