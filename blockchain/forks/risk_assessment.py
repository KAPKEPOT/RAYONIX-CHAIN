# blockchain/forks/risk_assessment.py
import time
from typing import Dict, List, Any, Optional
import statistics

class ForkRiskAssessor:
    """Assesses and predicts fork risks"""
    
    def __init__(self, config: Dict[str, Any], network_manager: Any, consensus_manager: Any):
        self.config = config
        self.network_manager = network_manager
        self.consensus_manager = consensus_manager
        self.risk_history: List[Dict[str, Any]] = []
    
    def assess_current_risk(self) -> Dict[str, Any]:
        """Assess current fork risk"""
        network_health = self._assess_network_health()
        consensus_health = self._assess_consensus_health()
        historical_risk = self._analyze_historical_risk()
        
        # Combine risk factors
        overall_risk = (
            network_health['risk_score'] * 0.4 +
            consensus_health['risk_score'] * 0.4 +
            historical_risk['risk_score'] * 0.2
        )
        
        risk_assessment = {
            'overall_risk': overall_risk,
            'risk_level': self._get_risk_level(overall_risk),
            'timestamp': time.time(),
            'network_health': network_health,
            'consensus_health': consensus_health,
            'historical_risk': historical_risk,
            'recommendations': self._generate_recommendations(overall_risk, network_health, consensus_health)
        }
        
        # Store in history
        self.risk_history.append(risk_assessment)
        if len(self.risk_history) > 1000:
            self.risk_history = self.risk_history[-1000:]
        
        return risk_assessment
    
    def _assess_network_health(self) -> Dict[str, Any]:
        """Assess network health factors that contribute to fork risk"""
        try:
            # Get network metrics
            peer_count = self.network_manager.get_peer_count()
            latency_stats = self.network_manager.get_latency_stats()
            bandwidth_stats = self.network_manager.get_bandwidth_stats()
            
            # Calculate risk factors
            risk_factors = {
                'low_peer_count': max(0, 1 - (peer_count / 10)),  # <10 peers = higher risk
                'high_latency': min(1.0, latency_stats.get('average', 0) / 1000),  # >1s latency = high risk
                'low_bandwidth': min(1.0, 10 / bandwidth_stats.get('average', 10)),  # <10 Mbps = higher risk
                'connectivity_issues': self.network_manager.get_connectivity_issues()
            }
            
            # Calculate overall risk score
            risk_score = (
                risk_factors['low_peer_count'] * 0.3 +
                risk_factors['high_latency'] * 0.3 +
                risk_factors['low_bandwidth'] * 0.2 +
                risk_factors['connectivity_issues'] * 0.2
            )
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'metrics': {
                    'peer_count': peer_count,
                    'average_latency': latency_stats.get('average', 0),
                    'average_bandwidth': bandwidth_stats.get('average', 0)
                }
            }
            
        except Exception as e:
            # If network metrics are unavailable, assume moderate risk
            return {
                'risk_score': 0.5,
                'risk_factors': {'network_unavailable': 1.0},
                'metrics': {}
            }
    
    def _assess_consensus_health(self) -> Dict[str, Any]:
        """Assess consensus health factors"""
        try:
            validator_stats = self.consensus_manager.get_validator_stats()
            block_production_stats = self.consensus_manager.get_block_production_stats()
            
            risk_factors = {
                'low_validator_participation': max(0, 1 - validator_stats.get('participation_rate', 1.0)),
                'high_orphan_rate': min(1.0, block_production_stats.get('orphan_rate', 0) * 10),
                'validator_concentration': validator_stats.get('concentration_risk', 0),
                'consensus_instability': self.consensus_manager.get_instability_metric()
            }
            
            risk_score = (
                risk_factors['low_validator_participation'] * 0.4 +
                risk_factors['high_orphan_rate'] * 0.3 +
                risk_factors['validator_concentration'] * 0.2 +
                risk_factors['consensus_instability'] * 0.1
            )
            
            return {
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'metrics': {
                    'validator_count': validator_stats.get('count', 0),
                    'participation_rate': validator_stats.get('participation_rate', 0),
                    'orphan_rate': block_production_stats.get('orphan_rate', 0)
                }
            }
            
        except Exception as e:
            return {
                'risk_score': 0.5,
                'risk_factors': {'consensus_unavailable': 1.0},
                'metrics': {}
            }
    
    def _analyze_historical_risk(self) -> Dict[str, Any]:
        """Analyze historical fork risk patterns"""
        if not self.risk_history:
            return {'risk_score': 0.5, 'trend': 'stable'}
        
        # Analyze recent risk history
        recent_risks = [assessment['overall_risk'] for assessment in self.risk_history[-24:]]  # Last 24 assessments
        
        if len(recent_risks) < 2:
            return {'risk_score': 0.5, 'trend': 'stable'}
        
        # Calculate trend
        trend = self._calculate_trend(recent_risks)
        volatility = statistics.stdev(recent_risks) if len(recent_risks) > 1 else 0
        
        # Higher volatility increases risk
        risk_score = min(1.0, statistics.mean(recent_risks) + (volatility * 2))
        
        return {
            'risk_score': risk_score,
            'trend': trend,
            'volatility': volatility,
            'average_risk': statistics.mean(recent_risks)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend of values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        try:
            slope = statistics.linear_regression(x, values).slope
        except:
            return 'stable'
        
        if slope > 0.01:
            return 'increasing'
        elif slope < -0.01:
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_risk_level(self, risk_score: float) -> str:
        """Convert risk score to risk level"""
        if risk_score < 0.2:
            return 'low'
        elif risk_score < 0.4:
            return 'moderate'
        elif risk_score < 0.6:
            return 'elevated'
        elif risk_score < 0.8:
            return 'high'
        else:
            return 'critical'
    
    def _generate_recommendations(self, overall_risk: float, 
                                network_health: Dict[str, Any],
                                consensus_health: Dict[str, Any]) -> List[str]:
        """Generate risk mitigation recommendations"""
        recommendations = []
        
        if overall_risk > 0.6:
            recommendations.append("Consider increasing block time temporarily")
            recommendations.append("Monitor network connectivity closely")
        
        if network_health['risk_score'] > 0.5:
            if network_health['risk_factors'].get('low_peer_count', 0) > 0.5:
                recommendations.append("Add more peer connections")
            if network_health['risk_factors'].get('high_latency', 0) > 0.5:
                recommendations.append("Check network infrastructure")
        
        if consensus_health['risk_score'] > 0.5:
            if consensus_health['risk_factors'].get('low_validator_participation', 0) > 0.5:
                recommendations.append("Investigate validator performance issues")
            if consensus_health['risk_factors'].get('high_orphan_rate', 0) > 0.5:
                recommendations.append("Review block propagation efficiency")
        
        if not recommendations:
            recommendations.append("No immediate action needed")
        
        return recommendations
    
    def predict_fork_probability(self, time_horizon: int = 3600) -> float:
        """Predict probability of fork within time horizon"""
        current_risk = self.assess_current_risk()
        base_probability = current_risk['overall_risk'] * 0.5  # Base conversion factor
        
        # Adjust based on time horizon (longer horizon = higher probability)
        time_factor = min(1.0, time_horizon / 86400)  # Cap at 1.0 for 24 hours
        
        return min(0.95, base_probability + (time_factor * 0.3))
    
    def get_risk_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get risk assessment history"""
        cutoff = time.time() - (hours * 3600)
        return [assessment for assessment in self.risk_history if assessment['timestamp'] >= cutoff]