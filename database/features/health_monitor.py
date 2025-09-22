import time
import threading
from typing import Dict, Any
import logging
from dataclasses import dataclass

from database.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

@dataclass
class HealthStatus:
    status: str  # 'healthy', 'degraded', 'unhealthy'
    message: str
    timestamp: float
    metrics: Dict[str, Any]

class DatabaseHealthMonitor:
    """Database health monitoring and alerting system"""
    
    def __init__(self, db, check_interval: int = 60):
        self.db = db
        self.check_interval = check_interval
        self.status = HealthStatus('healthy', 'Database is healthy', time.time(), {})
        self.monitoring = False
        self.thread = None
    
    def start_monitoring(self):
        """Start health monitoring"""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self):
        """Health monitoring loop"""
        while self.monitoring:
            try:
                self._check_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _check_health(self):
        """Perform health check"""
        try:
            metrics = self._collect_metrics()
            status = self._evaluate_health(metrics)
            
            self.status = HealthStatus(
                status['level'],
                status['message'],
                time.time(),
                metrics
            )
            
            if status['level'] != 'healthy':
                logger.warning(f"Database health issue: {status['message']}")
                
        except Exception as e:
            self.status = HealthStatus(
                'unhealthy',
                f'Health check failed: {e}',
                time.time(),
                {}
            )
            logger.error(f"Health check failed: {e}")
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect database metrics"""
        stats = self.db.get_stats()
        
        return {
            'operation_stats': stats,
            'cache_utilization': len(self.db.cache) / self.db.config.max_cache_size,
            'index_count': len(self.db.indexes),
            'database_size': self.db._get_database_size(),
            'uptime': stats.get('uptime_seconds', 0),
            'error_rate': self._calculate_error_rate(stats)
        }
    
    def _calculate_error_rate(self, stats: Dict[str, Any]) -> float:
        """Calculate error rate from statistics"""
        total_operations = (
            stats.get('put_operations', 0) +
            stats.get('get_operations', 0) +
            stats.get('delete_operations', 0) +
            stats.get('batch_operations', 0)
        )
        
        total_errors = (
            stats.get('put_errors', 0) +
            stats.get('get_errors', 0) +
            stats.get('delete_errors', 0) +
            stats.get('batch_errors', 0)
        )
        
        if total_operations == 0:
            return 0.0
        
        return total_errors / total_operations
    
    def _evaluate_health(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Evaluate health based on metrics"""
        error_rate = metrics.get('error_rate', 0)
        cache_utilization = metrics.get('cache_utilization', 0)
        
        if error_rate > 0.1:  # 10% error rate
            return {
                'level': 'unhealthy',
                'message': f'High error rate: {error_rate:.2%}'
            }
        elif error_rate > 0.01:  # 1% error rate
            return {
                'level': 'degraded',
                'message': f'Moderate error rate: {error_rate:.2%}'
            }
        elif cache_utilization > 0.9:  # 90% cache full
            return {
                'level': 'degraded',
                'message': f'Cache nearly full: {cache_utilization:.2%}'
            }
        else:
            return {
                'level': 'healthy',
                'message': 'All systems operational'
            }
    
    def get_status(self) -> HealthStatus:
        """Get current health status"""
        return self.status
    
    def add_custom_check(self, check_func):
        """Add custom health check function"""
        # Implementation would store custom checks and run them during health evaluation
        pass