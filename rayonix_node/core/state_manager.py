# core/state_manager.py
import time
import os
import sys
import threading
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, system metrics will be limited")

#logger = logging.getLogger("rayonix_node.state_manager")

class NodeStateManager:
    
    def __init__(self):
        self.sync_state = {
            'syncing': False,
            'current_block': 0,
            'target_block': 0,
            'peers_connected': 0,
            'last_sync_time': 0,
            'sync_speed': 0,
            'sync_progress': 0,
            'sync_start_time': 0,
            'estimated_time_remaining': 0
        }
        
        self.node_state = {
            'start_time': time.time(),
            'uptime': 0,
            'blocks_processed': 0,
            'transactions_processed': 0,
            'memory_usage': 0,
            'cpu_usage': 0,
            'disk_usage': 0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
            'last_update': time.time()
        }
        
        # Performance tracking
        self.performance_metrics = {
            'blocks_per_second': 0,
            'transactions_per_second': 0,
            'average_block_time': 0,
            'peak_memory_usage': 0
        }
        
        # Historical data for trend analysis
        self.history = {
            'memory_usage': [],
            'cpu_usage': [],
            'sync_progress': [],
            'max_history_size': 1000
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._last_cpu_time = time.time()
        self._last_cpu_measurement = 0.0
        
        # Initialize system monitoring
        self._initialize_system_monitoring()
    
    def _initialize_system_monitoring(self):
        """Initialize system monitoring baseline"""
        try:
            if PSUTIL_AVAILABLE:
                # Get initial process handle
                self._process = psutil.Process(os.getpid())
                # Get initial network I/O
                net_io = psutil.net_io_counters()
                self.node_state['network_io']['bytes_sent'] = net_io.bytes_sent
                self.node_state['network_io']['bytes_recv'] = net_io.bytes_recv
            else:
                self._process = None
                logger.warning("System monitoring limited - install psutil for detailed metrics")
        except Exception as e:
            logger.warning(f"Failed to initialize system monitoring: {e}")
            self._process = None
    
    def get_uptime(self) -> str:
        """Get node uptime as human-readable string with high precision"""
        with self._lock:
            uptime_seconds = time.time() - self.node_state['start_time']
            return self._format_uptime_detailed(uptime_seconds)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB with fallback mechanisms"""
        with self._lock:
            try:
                if PSUTIL_AVAILABLE and self._process:
                    memory_info = self._process.memory_info()
                    usage_mb = memory_info.rss / 1024 / 1024
                    
                    # Update peak memory usage
                    if usage_mb > self.performance_metrics['peak_memory_usage']:
                        self.performance_metrics['peak_memory_usage'] = usage_mb
                    
                    # Store in history for trend analysis
                    self._add_to_history('memory_usage', usage_mb)
                    
                    return round(usage_mb, 2)
                
                # Fallback: use resource module on Unix-like systems
                elif hasattr(resource, 'getrusage'):
                    import resource
                    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    if sys.platform == "darwin":  # macOS
                        return round(usage / 1024 / 1024, 2)  # bytes to MB
                    else:  # Linux
                        return round(usage / 1024, 2)  # KB to MB
                
                else:
                    # Final fallback
                    return 0.0
                    
            except Exception as e:
                logger.debug(f"Memory usage measurement failed: {e}")
                return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage with process-specific measurement"""
        with self._lock:
            try:
                if PSUTIL_AVAILABLE and self._process:
                    # Get process CPU percentage
                    cpu_percent = self._process.cpu_percent(interval=0.1)
                    
                    # Store in history for trend analysis
                    self._add_to_history('cpu_usage', cpu_percent)
                    
                    return round(cpu_percent, 2)
                else:
                    # Simple time-based estimation as fallback
                    current_time = time.time()
                    time_diff = current_time - self._last_cpu_time
                    
                    if time_diff > 1.0:  # Update at most once per second
                        # Simple estimation - can be enhanced based on actual work
                        estimated_usage = min(100.0, self._last_cpu_measurement * 0.9)
                        self._last_cpu_time = current_time
                        self._last_cpu_measurement = estimated_usage
                        return round(estimated_usage, 2)
                    else:
                        return round(self._last_cpu_measurement, 2)
                        
            except Exception as e:
                logger.debug(f"CPU usage measurement failed: {e}")
                return 0.0
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information"""
        try:
            if PSUTIL_AVAILABLE:
                # Get disk usage of current working directory
                disk_usage = psutil.disk_usage('.')
                return {
                    'total_gb': round(disk_usage.total / 1024 / 1024 / 1024, 2),
                    'used_gb': round(disk_usage.used / 1024 / 1024 / 1024, 2),
                    'free_gb': round(disk_usage.free / 1024 / 1024 / 1024, 2),
                    'percent_used': round(disk_usage.percent, 2)
                }
            else:
                return {
                    'total_gb': 0,
                    'used_gb': 0,
                    'free_gb': 0,
                    'percent_used': 0
                }
        except Exception as e:
            logger.debug(f"Disk usage measurement failed: {e}")
            return {
                'total_gb': 0,
                'used_gb': 0,
                'free_gb': 0,
                'percent_used': 0
            }
    
    def get_network_io(self) -> Dict[str, int]:
        """Get network I/O statistics"""
        with self._lock:
            try:
                if PSUTIL_AVAILABLE:
                    net_io = psutil.net_io_counters()
                    current_state = {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv
                    }
                    
                    # Calculate deltas if we have previous state
                    if hasattr(self, '_last_network_io'):
                        delta_sent = current_state['bytes_sent'] - self._last_network_io['bytes_sent']
                        delta_recv = current_state['bytes_recv'] - self._last_network_io['bytes_recv']
                        current_state.update({
                            'bytes_sent_delta': delta_sent,
                            'bytes_recv_delta': delta_recv
                        })
                    
                    self._last_network_io = current_state.copy()
                    return current_state
                else:
                    return self.node_state['network_io'].copy()
                    
            except Exception as e:
                logger.debug(f"Network I/O measurement failed: {e}")
                return self.node_state['network_io'].copy()
    
    def _add_to_history(self, metric: str, value: float):
        """Add metric value to history for trend analysis"""
        if metric not in self.history:
            self.history[metric] = []
        
        self.history[metric].append({
            'timestamp': time.time(),
            'value': value
        })
        
        # Trim history if it gets too large
        if len(self.history[metric]) > self.history['max_history_size']:
            self.history[metric] = self.history[metric][-self.history['max_history_size']:]
    
    def _format_uptime_detailed(self, seconds: float) -> str:
        """Format uptime with high precision and readability"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{int(hours)}h {int(minutes)}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            return f"{int(days)}d {int(hours)}h"
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        with self._lock:
            return {
                'current': {
                    'memory_usage_mb': self.get_memory_usage(),
                    'cpu_usage_percent': self.get_cpu_usage(),
                    'uptime': self.get_uptime(),
                    'disk_usage': self.get_disk_usage(),
                    'network_io': self.get_network_io()
                },
                'historical': {
                    'peak_memory_usage_mb': self.performance_metrics['peak_memory_usage'],
                    'average_cpu_usage': self._calculate_average('cpu_usage'),
                    'average_memory_usage': self._calculate_average('memory_usage')
                },
                'node_specific': self.performance_metrics.copy()
            }
    
    def _calculate_average(self, metric: str) -> float:
        """Calculate average of historical metric values"""
        if metric not in self.history or not self.history[metric]:
            return 0.0
        
        values = [entry['value'] for entry in self.history[metric]]
        return round(sum(values) / len(values), 2)
    
    # Existing methods with enhancements
    def update_sync_state(self, **kwargs):
        """Update synchronization state with enhanced metrics"""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.sync_state:
                    self.sync_state[key] = value
            
            # Track sync start time
            if kwargs.get('syncing') and not self.sync_state['syncing']:
                self.sync_state['sync_start_time'] = time.time()
            elif not kwargs.get('syncing') and self.sync_state['syncing']:
                self.sync_state['sync_start_time'] = 0
            
            # Calculate sync progress if we have target and current block
            if (self.sync_state['target_block'] > 0 and 
                self.sync_state['current_block'] > 0):
                progress = (self.sync_state['current_block'] / 
                           self.sync_state['target_block']) * 100
                self.sync_state['sync_progress'] = min(100.0, round(progress, 2))
                
                # Store in history
                self._add_to_history('sync_progress', self.sync_state['sync_progress'])
            
            # Calculate sync speed and ETA
            current_time = time.time()
            if (self.sync_state['last_sync_time'] > 0 and
                self.sync_state['current_block'] > 0):
                time_diff = current_time - self.sync_state['last_sync_time']
                if time_diff > 0:
                    blocks_diff = (self.sync_state['current_block'] - 
                                  self.sync_state.get('last_block_count', 0))
                    self.sync_state['sync_speed'] = blocks_diff / time_diff
                    
                    # Calculate estimated time remaining
                    if self.sync_state['sync_speed'] > 0:
                        blocks_remaining = self.sync_state['target_block'] - self.sync_state['current_block']
                        self.sync_state['estimated_time_remaining'] = blocks_remaining / self.sync_state['sync_speed']
            
            self.sync_state['last_sync_time'] = current_time
            self.sync_state['last_block_count'] = self.sync_state['current_block']
    
    def update_node_state(self, **kwargs):
        """Update node state with automatic metrics collection"""
        with self._lock:
            for key, value in kwargs.items():
                if key in self.node_state:
                    self.node_state[key] = value
            
            # Always update these metrics
            self.node_state['uptime'] = time.time() - self.node_state['start_time']
            self.node_state['memory_usage'] = self.get_memory_usage()
            self.node_state['cpu_usage'] = self.get_cpu_usage()
            self.node_state['last_update'] = time.time()
            
            # Update performance metrics
            if 'blocks_processed' in kwargs:
                self._update_performance_metrics()
    
    def _update_performance_metrics(self):
        """Update derived performance metrics"""
        # This can be enhanced to calculate blocks/sec, tx/sec etc.
        current_time = time.time()
        time_diff = current_time - self.node_state.get('last_metric_update', current_time)
        
        if time_diff > 0:
            blocks_diff = self.node_state['blocks_processed'] - self.node_state.get('last_blocks_processed', 0)
            self.performance_metrics['blocks_per_second'] = blocks_diff / time_diff
            
            tx_diff = self.node_state['transactions_processed'] - self.node_state.get('last_transactions_processed', 0)
            self.performance_metrics['transactions_per_second'] = tx_diff / time_diff
        
        self.node_state['last_metric_update'] = current_time
        self.node_state['last_blocks_processed'] = self.node_state['blocks_processed']
        self.node_state['last_transactions_processed'] = self.node_state['transactions_processed']
    
    def get_sync_state(self) -> Dict[str, Any]:
        """Get current synchronization state"""
        with self._lock:
            return self.sync_state.copy()
    
    def get_node_state(self) -> Dict[str, Any]:
        """Get current node state"""
        with self._lock:
            return self.node_state.copy()
    
    def is_synced(self) -> bool:
        """Check if node is fully synchronized"""
        with self._lock:
            return (not self.sync_state['syncing'] and 
                    self.sync_state['current_block'] >= self.sync_state['target_block'])
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary for API responses"""
        with self._lock:
            return {
                'sync_state': self.get_sync_state(),
                'node_state': self.get_node_state(),
                'performance_metrics': self.get_performance_metrics(),
                'is_synced': self.is_synced(),
                'timestamp': time.time(),
                'system_health': self._get_system_health_status()
            }
    
    def _get_system_health_status(self) -> Dict[str, Any]:
        """Get system health status with warnings"""
        health_status = {
            'status': 'healthy',
            'warnings': [],
            'metrics_available': PSUTIL_AVAILABLE
        }
        
        # Check memory usage
        memory_usage = self.get_memory_usage()
        if memory_usage > 1024:  # Warning if over 1GB
            health_status['warnings'].append(f'High memory usage: {memory_usage}MB')
        
        # Check CPU usage
        cpu_usage = self.get_cpu_usage()
        if cpu_usage > 80:  # Warning if over 80%
            health_status['warnings'].append(f'High CPU usage: {cpu_usage}%')
        
        if health_status['warnings']:
            health_status['status'] = 'degraded'
        
        return health_status