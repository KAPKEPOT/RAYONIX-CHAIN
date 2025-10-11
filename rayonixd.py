#!/usr/bin/env python3
"""
RAYONIX Blockchain Daemon - Production Ready
Pure background service with enterprise-grade features
"""

import argparse
import asyncio
import signal
import sys
import logging
import os
import time
import psutil
import gc
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

from rayonix_node.core.node import RayonixNode
from rayonix_node.utils.helpers2 import configure_logging, setup_pid_file, remove_pid_file
from rayonix_node.utils.daemonize import daemonize_process

logger = logging.getLogger("rayonix_daemon")

@dataclass
class DaemonConfig:
    """Daemon configuration container"""
    config_path: Optional[str] = None
    network: str = "testnet"
    port: Optional[int] = None
    api_port: Optional[int] = None
    no_api: bool = False
    no_network: bool = False
    data_dir: str = "./rayonix_data"
    encryption_key: Optional[str] = None
    daemon: bool = False
    log_level: str = "INFO"
    log_file: Optional[str] = None
    max_memory: int = 0  # 0 = unlimited
    health_check_interval: int = 300  # seconds
    graceful_shutdown_timeout: int = 30  # seconds

class ResourceMonitor:
    """Monitor system resources and enforce limits"""
    
    def __init__(self, max_memory: int = 0):
        self.max_memory = max_memory
        self.process = psutil.Process()
        self.memory_warnings = 0
        self.last_gc_run = time.time()
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage and enforce limits"""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        stats = {
            'memory_mb': memory_mb,
            'memory_percent': self.process.memory_percent(),
            'cpu_percent': self.process.cpu_percent(),
            'threads': self.process.num_threads(),
            'max_memory': self.max_memory
        }
        
        # Enforce memory limit
        if self.max_memory > 0 and memory_mb > self.max_memory:
            logger.warning(f"Memory usage {memory_mb:.1f}MB exceeds limit {self.max_memory}MB")
            self.memory_warnings += 1
            
            # Force garbage collection
            gc.collect()
            self.last_gc_run = time.time()
            
            # If we've had multiple warnings, consider more aggressive measures
            if self.memory_warnings > 3:
                logger.error("Multiple memory limit violations - may need restart")
                
        return stats
    
    def should_perform_gc(self) -> bool:
        """Determine if garbage collection should be run"""
        current_time = time.time()
        return current_time - self.last_gc_run > 3600  # Run GC every hour

class HealthMonitor:
    """Monitor node health and performance"""
    
    def __init__(self, node: 'RayonixDaemon'):
        self.node = node
        self.health_checks = 0
        self.consecutive_failures = 0
        self.last_successful_check = time.time()
        
    async def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        self.health_checks += 1
        health_status = {
            'timestamp': time.time(),
            'checks_performed': self.health_checks,
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # Check node state
            if self.node.node and self.node.node.running:
                health_status['components']['node'] = {
                    'status': 'healthy',
                    'running': True,
                    'sync_state': self.node.node.state_manager.get_sync_state()
                }
            else:
                health_status['components']['node'] = {
                    'status': 'unhealthy', 
                    'running': False,
                    'error': 'Node not running'
                }
                health_status['overall_status'] = 'degraded'
            
            # Check API server
            if self.node.node and self.node.node.api_server:
                health_status['components']['api'] = {
                    'status': 'healthy',
                    'enabled': True
                }
            else:
                health_status['components']['api'] = {
                    'status': 'disabled',
                    'enabled': False
                }
            
            # Check network
            if (self.node.node and self.node.node.network and 
            self.node.node.config_manager.get('network.enabled', True)):
                try:
                    peers = await self.node.node.network.get_peers()
                    health_status['components']['network'] = {
                        'status': 'healthy',
                        'peers_connected': len(peers),
                        'enabled': True
                    }
                except Exception as e:
                    health_status['components']['network'] = {
                        'status': 'unhealthy',
                        'error': str(e),
                        'enabled': True
                    }
                    health_status['overall_status'] = 'degraded'
            else:
                health_status['components']['network'] = {
                    'status': 'disabled',
                    'enabled': False
                }
            
            # Update failure tracking
            if health_status['overall_status'] == 'healthy':
                self.consecutive_failures = 0
                self.last_successful_check = time.time()
            else:
                self.consecutive_failures += 1
                
            # Check if we have too many consecutive failures
            if self.consecutive_failures > 5:
                health_status['overall_status'] = 'critical'
                logger.error(f"Critical health status after {self.consecutive_failures} consecutive failures")
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            self.consecutive_failures += 1
            
        return health_status

class SignalHandler:
    """Handle system signals for graceful shutdown"""
    
    def __init__(self, daemon: 'RayonixDaemon'):
        self.daemon = daemon
        self.original_handlers = {}
        self.shutdown_initiated = False
        
    def setup_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signals = [signal.SIGINT, signal.SIGTERM]
        
        for sig in signals:
            self.original_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, self._signal_handler)
        
        # Ignore SIGHUP to survive terminal disconnects
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        
        logger.info("Signal handlers configured")
    
    def restore_handlers(self):
        """Restore original signal handlers"""
        for sig, handler in self.original_handlers.items():
            if handler:
                signal.signal(sig, handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        if self.shutdown_initiated:
            logger.warning(f"Signal {signum} received during shutdown - forcing exit")
            sys.exit(1)
            
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_initiated = True
        
        # Schedule shutdown in event loop
        asyncio.create_task(self.daemon.initiate_shutdown())

class PerformanceMonitor:
    """Monitor and log performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'start_time': time.time(),
            'blocks_processed': 0,
            'transactions_processed': 0,
            'api_requests': 0,
            'network_messages': 0
        }
        self.last_report_time = time.time()
        
    def increment_counter(self, metric: str, amount: int = 1):
        """Increment performance counter"""
        if metric in self.metrics:
            self.metrics[metric] += amount
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        current_time = time.time()
        uptime = current_time - self.metrics['start_time']
        
        metrics = self.metrics.copy()
        metrics.update({
            'uptime_seconds': uptime,
            'uptime_human': self._format_uptime(uptime),
            'blocks_per_hour': self._calculate_rate('blocks_processed', uptime),
            'transactions_per_hour': self._calculate_rate('transactions_processed', uptime),
            'api_requests_per_hour': self._calculate_rate('api_requests', uptime)
        })
        
        return metrics
    
    def _calculate_rate(self, metric: str, uptime: float) -> float:
        """Calculate hourly rate for a metric"""
        if uptime > 0:
            return (self.metrics[metric] / uptime) * 3600
        return 0.0
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime as human readable string"""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m"

class RayonixDaemon:
    """Production daemon service with comprehensive management"""
    
    def __init__(self):
        self.node: Optional[RayonixNode] = None
        self.running = False
        self.shutdown_initiated = False
        self.pid_file: Optional[str] = None
        self.config: Optional[DaemonConfig] = None
        self.resource_monitor: Optional[ResourceMonitor] = None
        self.health_monitor: Optional[HealthMonitor] = None
        self.signal_handler: Optional[SignalHandler] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.background_tasks: List[asyncio.Task] = []
        self.thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rayonixd")
        
    async def initialize(self, args) -> bool:
        """Initialize daemon with comprehensive setup"""
        try:
            logger.info("Initializing RAYONIX Daemon...")
            
            # Parse and validate configuration
            self.config = self._parse_args(args)
            
            # Handle daemonization if requested
            if self.config.daemon:
                if not await self._daemonize():
                    return False
            
            # Setup PID file for process management
            self.pid_file = setup_pid_file('rayonixd', self.config.data_dir)
            
            # Initialize monitoring components
            self.resource_monitor = ResourceMonitor(self.config.max_memory)
            self.performance_monitor = PerformanceMonitor()
            
            # Create and initialize node
            self.node = RayonixNode()
            if not await self._initialize_node():
                return False
            
            # Initialize health monitoring
            self.health_monitor = HealthMonitor(self)
            
            # Setup signal handling
            self.signal_handler = SignalHandler(self)
            self.signal_handler.setup_handlers()
            
            # Validate critical services
            if not await self._validate_services():
                return False
            
            logger.info("RAYONIX Daemon initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Daemon initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_args(self, args) -> DaemonConfig:
        """Parse command line arguments into configuration"""
        return DaemonConfig(
            config_path=args.config,
            network=args.network or "testnet",
            port=args.port,
            api_port=args.api_port,
            no_api=args.no_api,
            no_network=args.no_network,
            data_dir=args.data_dir or "./rayonix_data",
            encryption_key=args.encryption_key,
            daemon=args.daemon,
            log_level=args.log_level,
            log_file=args.log_file,
            max_memory=args.max_memory,
            health_check_interval=args.health_check_interval,
            graceful_shutdown_timeout=args.graceful_shutdown_timeout
        )
    
    async def _daemonize(self) -> bool:
        """Daemonize the process with proper error handling"""
        try:
            logger.info("Daemonizing process...")
            
            # Ensure proper file permissions
            os.umask(0o022)
            
            # Perform daemonization
            daemonize_process()
            
            # Reconfigure logging for daemon mode
            configure_logging(
                level=self.config.log_level,
                log_file=self.config.log_file,
                component='daemon'
            )
            
            logger.info("Process daemonized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to daemonize: {e}")
            return False
    
    async def _initialize_node(self) -> bool:
        """Initialize the blockchain node with configuration"""
        try:
            logger.info("Initializing RAYONIX Node with production configuration...")
            
            # Create node with proper dependency injection
            self.node = RayonixNode()
            
            # Initialize with comprehensive error handling
            success = await self.node.initialize_components(
                self.config.config_path,
                self.config.encryption_key
            )
            if not success:
            	logger.error("Node component initialization failed")
            	return False
            	
            # Apply production configuration overrides
            await self._apply_production_config()
            
            # Validate all critical components
            if not await self._validate_components():
            	logger.error("Production component validation failed")
            	return False
            	
            logger.info("RAYONIX Node initialized successfully for production")
            return True
            
        except Exception as e:
        	logger.error(f"Production node initialization failed: {e}")
        	import traceback
        	traceback.print_exc()
        	return False
        	
    async def _apply_production_config(self):
    	# Ensure API is enabled for CLI communication
    	if self.config.no_api:
    		logger.warning("API disabled - CLI will not function")
    		
    	# Set production timeouts
    	if self.node.config_manager:
    		self.node.config_manager.set('api.timeout', 30)
    		self.node.config_manager.set('network.connection_timeout', 60)
    		
    async def _validate_components(self) -> bool:
    	components = [
    	    ('config_manager', self.node.config_manager),
    	    ('rayonix_chain', self.node.rayonix_chain),
    	    ('state_manager', self.node.state_manager)
    	]
    	
    	for name, component in components:
    		if not component:
    			logger.error(f"Production component missing: {name}")
    			return False
    			
    	# Validate API server if enabled
    	if (self.node.config_manager.get('api.enabled', True) and not self.node.api_server):
    		logger.error("API server required but not initialized")
    		return False
    	return True
  
    async def _apply_config_overrides(self):
        """Apply command line configuration overrides"""
        if self.config.network:
            self.node.config_manager.set('network.network_type', self.config.network)
        
        if self.config.port:
            self.node.config_manager.set('network.listen_port', self.config.port)
        
        if self.config.api_port:
            self.node.config_manager.set('api.port', self.config.api_port)
        
        if self.config.no_api:
            self.node.config_manager.set('api.enabled', False)
        
        if self.config.no_network:
            self.node.config_manager.set('network.enabled', False)
        
        if self.config.data_dir:
            self.node.config_manager.set('database.db_path', self.config.data_dir)
    
    async def _verify_components(self) -> bool:
        """Verify critical components are properly initialized"""
        components = [
            ('blockchain', self.node.rayonix_chain),
            ('config', self.node.config_manager)
        ]
        
        for name, component in components:
            if not component:
                logger.error(f"Critical component not initialized: {name}")
                return False
        
        # Verify network if enabled
        if (self.node.config_manager.get('network.enabled', True) and 
            not self.node.network):
            logger.error("Network component not initialized but network is enabled")
            return False
        
        # Verify API server if enabled
        if (self.node.config_manager.get('api.enabled', True) and 
            not self.node.api_server):
            logger.error("API server not initialized but API is enabled")
            return False
        
        logger.info("All critical components verified successfully")
        return True
    
    async def _validate_services(self) -> bool:
        """Validate that required services are accessible"""
        try:
            # Validate data directory
            data_dir = Path(self.config.data_dir)
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
            
            # Check write permissions
            test_file = data_dir / '.write_test'
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                logger.error(f"Data directory not writable: {e}")
                return False
            
            logger.info("Service validation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Service validation failed: {e}")
            return False
    
    async def start(self) -> bool:
        """Start the daemon service with comprehensive monitoring"""
        if self.running:
            logger.warning("Daemon already running")
            return False
        
        try:
            logger.info("Starting RAYONIX Daemon service...")
            
            # Start the node
            if not await self.node.start():
                logger.error("Failed to start node")
                return False
            
            self.running = True
            
            # Start background monitoring tasks
            self._start_background_tasks()
            
            logger.info("RAYONIX Daemon service started successfully")
            
            # Main service loop
            await self._run_service_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start daemon: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        # Resource monitoring
        resource_task = asyncio.create_task(self._resource_monitoring_loop())
        self.background_tasks.append(resource_task)
        
        # Health monitoring
        health_task = asyncio.create_task(self._health_monitoring_loop())
        self.background_tasks.append(health_task)
        
        # Performance logging
        perf_task = asyncio.create_task(self._performance_logging_loop())
        self.background_tasks.append(perf_task)
        
        logger.info(f"Started {len(self.background_tasks)} background tasks")
    
    async def _resource_monitoring_loop(self):
        """Monitor system resources and perform maintenance"""
        while self.running and not self.shutdown_initiated:
            try:
                # Check memory usage
                stats = self.resource_monitor.check_memory_usage()
                
                # Log resource usage periodically
                if hasattr(self, '_last_resource_log'):
                    if time.time() - self._last_resource_log > 3600:  # Log every hour
                        logger.info(
                            f"Resource usage: Memory={stats['memory_mb']:.1f}MB, "
                            f"CPU={stats['cpu_percent']:.1f}%, Threads={stats['threads']}"
                        )
                        self._last_resource_log = time.time()
                else:
                    self._last_resource_log = time.time()
                
                # Perform garbage collection if needed
                if self.resource_monitor.should_perform_gc():
                    gc.collect()
                    self.resource_monitor.last_gc_run = time.time()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _health_monitoring_loop(self):
        """Continuous health monitoring"""
        while self.running and not self.shutdown_initiated:
            try:
                health_status = await self.health_monitor.perform_health_check()
                
                # Log health status changes
                if (hasattr(self, '_last_health_status') and 
                    self._last_health_status != health_status['overall_status']):
                    logger.info(f"Health status changed: {self._last_health_status} -> {health_status['overall_status']}")
                
                self._last_health_status = health_status['overall_status']
                
                # Take action based on health status
                if health_status['overall_status'] == 'critical':
                    logger.error("Critical health status detected - considering restart")
                    # In a production environment, this might trigger an automatic restart
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _performance_logging_loop(self):
        """Log performance metrics periodically"""
        while self.running and not self.shutdown_initiated:
            try:
                metrics = self.performance_monitor.get_metrics()
                
                # Log performance every 30 minutes
                if hasattr(self, '_last_perf_log'):
                    if time.time() - self._last_perf_log > 1800:
                        logger.info(
                            f"Performance: Uptime={metrics['uptime_human']}, "
                            f"Blocks={metrics['blocks_processed']}, "
                            f"Txs={metrics['transactions_processed']}, "
                            f"API={metrics['api_requests']}"
                        )
                        self._last_perf_log = time.time()
                else:
                    self._last_perf_log = time.time()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance logging error: {e}")
                await asyncio.sleep(60)
    
    async def _run_service_loop(self):
        """Main service loop - wait for shutdown signal"""
        startup_time = time.time()
        logger.info("Daemon service loop started")
        
        try:
            while self.running and not self.shutdown_initiated:
                # Brief sleep to prevent busy waiting
                await asyncio.sleep(1)
                
                # Log startup completion
                if not hasattr(self, '_startup_logged') and time.time() - startup_time > 10:
                    logger.info("Daemon startup completed - service is now fully operational")
                    self._startup_logged = True
                
        except asyncio.CancelledError:
            logger.info("Service loop cancelled")
        except Exception as e:
            logger.error(f"Service loop error: {e}")
        finally:
            await self._shutdown()
    
    async def initiate_shutdown(self):
        """Initiate graceful shutdown process"""
        if self.shutdown_initiated:
            return
            
        self.shutdown_initiated = True
        logger.info("Initiating graceful shutdown...")
        
        # Set a timeout for shutdown completion
        try:
            await asyncio.wait_for(self._shutdown(), timeout=self.config.graceful_shutdown_timeout)
        except asyncio.TimeoutError:
            logger.error("Shutdown timeout exceeded - forcing exit")
            sys.exit(1)
    
    async def _shutdown(self):
        """Perform comprehensive shutdown"""
        if not self.running:
            return
        
        logger.info("Beginning comprehensive shutdown...")
        self.running = False
        self.shutdown_initiated = True
        
        try:
            # Cancel all background tasks
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete cancellation
            if self.background_tasks:
                await asyncio.wait(self.background_tasks, timeout=10.0)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True, timeout=5.0)
            
            # Stop node components
            if self.node:
                await self.node.stop()
            
            # Restore signal handlers
            if self.signal_handler:
                self.signal_handler.restore_handlers()
            
            # Cleanup PID file
            if self.pid_file:
                remove_pid_file(self.pid_file)
            
            # Log final performance metrics
            if self.performance_monitor:
                metrics = self.performance_monitor.get_metrics()
                logger.info(
                    f"Final metrics: Uptime={metrics['uptime_human']}, "
                    f"Blocks={metrics['blocks_processed']}, Txs={metrics['transactions_processed']}"
                )
            
            logger.info("RAYONIX Daemon shutdown completed gracefully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Ensure process exits
            sys.exit(0)
            
    def check_status():
    	"""Check if daemon is running"""
    	from rayonix_node.utils.helpers2 import PIDManager
    	pid_file = './rayonix_data/run/rayonixd.pid'
    	
    	try:
    		if PIDManager._is_process_running_from_file(pid_file):
    			print("✅ rayonixd is running")
    			
    			return True
    		else:
    			print("❌ rayonixd is not running")
    			# Clean up stale PID file
    			import os
    			if os.path.exists(pid_file):
    				os.remove(pid_file)
    			return False
    	except Exception as e:
    		print(f"❌ Error checking status: {e}")
    		return False
   
def parse_arguments():
    """Parse command line arguments for production daemon"""
    parser = argparse.ArgumentParser(
        description='RAYONIX Blockchain Daemon - Production Ready',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core configuration
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--network', '-n', choices=['mainnet', 'testnet', 'regtest'], 
                       default='testnet', help='Network type')
    parser.add_argument('--port', '-p', type=int, help='P2P network port')
    parser.add_argument('--api-port', type=int, help='API server port')
    parser.add_argument('--no-api', action='store_true', help='Disable API server')
    parser.add_argument('--no-network', action='store_true', help='Disable P2P network')
    parser.add_argument('--data-dir', help='Data directory path', default='./rayonix_data')
    parser.add_argument('--encryption-key', help='Configuration encryption key')
    
    # Daemon operation
    parser.add_argument('--daemon', '-d', action='store_true', 
                       help='Run as background daemon')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Log level')
    parser.add_argument('--log-file', help='Log file path')
    
    # Resource management
    parser.add_argument('--max-memory', type=int, default=0,
                       help='Maximum memory usage in MB (0 = unlimited)')
    parser.add_argument('--health-check-interval', type=int, default=300,
                       help='Health check interval in seconds')
    parser.add_argument('--graceful-shutdown-timeout', type=int, default=30,
                       help='Graceful shutdown timeout in seconds')
    
    return parser.parse_args()

async def main():
    """Main daemon entry point with comprehensive error handling"""
    args = parse_arguments()
    
    # Configure logging first
    configure_logging(
        level=args.log_level,
        log_file=args.log_file,
        component='daemon'
    )
    
    # Log startup information
    logger.info("RAYONIX Daemon starting...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create and run daemon
    daemon = RayonixDaemon()
    
    try:
        if not await daemon.initialize(args):
            logger.error("Daemon initialization failed")
            sys.exit(1)
        
        if not await daemon.start():
            logger.error("Daemon startup failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
        await daemon.initiate_shutdown()
    except Exception as e:
        logger.critical(f"Unhandled exception in daemon: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        # Set up event loop policy for better performance
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nDaemon startup interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal daemon error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)