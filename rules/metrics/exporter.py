"""
Metrics export and integration with external monitoring systems
"""

import time
import json
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import requests
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from .collector import MetricsCollector
from ..exceptions import ConsensusError

logger = logging.getLogger('consensus.metrics')

@dataclass
class ExportConfig:
    """Metrics export configuration"""
    enabled: bool = True
    interval: int = 300  # 5 minutes
    endpoints: List[str] = field(default_factory=list)
    format: str = "prometheus"  # prometheus, json, influxdb
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5

class MetricsExporter:
    """Metrics export to external systems"""
    
    def __init__(self, collector: MetricsCollector, config: ExportConfig):
        self.collector = collector
        self.config = config
        self.session = requests.Session()
        self.exporters = {
            "prometheus": self._export_prometheus,
            "json": self._export_json,
            "influxdb": self._export_influxdb
        }
        
        self.running = False
        self.last_export = 0.0
    
    def start(self) -> None:
        """Start metrics export service"""
        if not self.config.enabled:
            logger.info("Metrics export is disabled")
            return
        
        self.running = True
        logger.info("Metrics exporter started")
        
        # Start background export thread
        import threading
        thread = threading.Thread(target=self._export_loop, daemon=True)
        thread.start()
    
    def stop(self) -> None:
        """Stop metrics export service"""
        self.running = False
        logger.info("Metrics exporter stopped")
    
    def _export_loop(self) -> None:
        """Background export loop"""
        while self.running:
            try:
                current_time = time.time()
                if current_time - self.last_export >= self.config.interval:
                    self.export_metrics()
                    self.last_export = current_time
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Export loop error: {e}")
                time.sleep(60)
    
    def export_metrics(self) -> None:
        """Export metrics to all configured endpoints"""
        if not self.config.endpoints:
            return
        
        for endpoint in self.config.endpoints:
            for attempt in range(self.config.retry_attempts):
                try:
                    exporter = self.exporters.get(self.config.format)
                    if exporter:
                        exporter(endpoint)
                    break
                except Exception as e:
                    if attempt == self.config.retry_attempts - 1:
                        logger.error(f"Failed to export metrics to {endpoint}: {e}")
                    else:
                        logger.warning(f"Export attempt {attempt + 1} failed: {e}")
                        time.sleep(self.config.retry_delay)
    
    def _export_prometheus(self, endpoint: str) -> None:
        """Export metrics in Prometheus format"""
        metrics_data = generate_latest()
        
        response = self.session.post(
            endpoint,
            data=metrics_data,
            headers={'Content-Type': CONTENT_TYPE_LATEST},
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        logger.debug(f"Exported Prometheus metrics to {endpoint}")
    
    def _export_json(self, endpoint: str) -> None:
        """Export metrics in JSON format"""
        # Collect metrics summary
        metrics_summary = self.collector.get_metrics_summary()
        
        response = self.session.post(
            endpoint,
            json=metrics_summary,
            headers={'Content-Type': 'application/json'},
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        logger.debug(f"Exported JSON metrics to {endpoint}")
    
    def _export_influxdb(self, endpoint: str) -> None:
        """Export metrics in InfluxDB line protocol format"""
        # This would convert metrics to InfluxDB format
        # Implementation depends on specific requirements
        influx_data = self._convert_to_influxdb()
        
        response = self.session.post(
            endpoint,
            data=influx_data,
            headers={'Content-Type': 'application/octet-stream'},
            timeout=self.config.timeout
        )
        
        response.raise_for_status()
        logger.debug(f"Exported InfluxDB metrics to {endpoint}")
    
    def _convert_to_influxdb(self) -> str:
        """Convert metrics to InfluxDB line protocol"""
        # Placeholder implementation
        # In production, this would properly format all metrics
        lines = []
        timestamp = int(time.time() * 1e9)  # nanoseconds
        
        # Add sample metrics
        lines.append(f"consensus_blocks_committed value={self.collector.metrics.blocks_committed._value.get()} {timestamp}")
        lines.append(f"consensus_current_height value={self.collector.metrics.current_height._value.get()} {timestamp}")
        
        return "\n".join(lines)
    
    def export_now(self) -> None:
        """Trigger immediate metrics export"""
        self.export_metrics()
    
    def add_endpoint(self, endpoint: str) -> None:
        """Add new export endpoint"""
        if endpoint not in self.config.endpoints:
            self.config.endpoints.append(endpoint)
            logger.info(f"Added metrics export endpoint: {endpoint}")
    
    def remove_endpoint(self, endpoint: str) -> None:
        """Remove export endpoint"""
        if endpoint in self.config.endpoints:
            self.config.endpoints.remove(endpoint)
            logger.info(f"Removed metrics export endpoint: {endpoint}")
    
    def get_endpoints(self) -> List[str]:
        """Get list of configured endpoints"""
        return self.config.endpoints.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """Check export service health"""
        return {
            'running': self.running,
            'endpoints_count': len(self.config.endpoints),
            'last_export': self.last_export,
            'next_export_in': max(0, self.config.interval - (time.time() - self.last_export))
        }

def create_prometheus_handler(collector: MetricsCollector):
    """Create Prometheus metrics HTTP handler"""
    def handler(environ, start_response):
        try:
            output = generate_latest()
            status = '200 OK'
            headers = [('Content-type', CONTENT_TYPE_LATEST),
                      ('Content-Length', str(len(output)))]
            start_response(status, headers)
            return [output]
        except Exception as e:
            logger.error(f"Prometheus handler error: {e}")
            status = '500 Internal Server Error'
            headers = [('Content-type', 'text/plain')]
            start_response(status, headers)
            return [b'Internal Server Error']
    return handler