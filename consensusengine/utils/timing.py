# consensus/utils/timing.py
import time
import threading
import asyncio
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('TimingManager')

class TimeoutType(Enum):
    """Types of timeouts in consensus protocol"""
    PROPOSE = auto()
    PREVOTE = auto()
    PRECOMMIT = auto()
    VIEW_CHANGE = auto()
    STATE_SYNC = auto()
    PEER_DISCOVERY = auto()
    HEALTH_CHECK = auto()

@dataclass
class TimeoutConfig:
    """Configuration for timeout parameters"""
    propose_timeout: float = 3.0  # seconds
    prevote_timeout: float = 1.0
    precommit_timeout: float = 1.0
    view_change_timeout: float = 10.0
    state_sync_timeout: float = 30.0
    peer_discovery_interval: float = 30.0
    health_check_interval: float = 60.0
    max_timeout_multiplier: float = 10.0  # Maximum timeout backoff multiplier

@dataclass
class TimeoutEvent:
    """Timeout event structure"""
    timeout_id: str
    timeout_type: TimeoutType
    callback: Callable
    scheduled_time: float
    expiration_time: float
    data: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3

class TimeoutManager:
    """Production-ready timeout management for consensus protocol"""
    
    def __init__(self, config: TimeoutConfig = None):
        self.config = config or TimeoutConfig()
        self.timeouts: Dict[str, TimeoutEvent] = {}
        self.scheduled_timers: Dict[str, threading.Timer] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="timeout")
        
        # Statistics
        self.stats = {
            'timeouts_triggered': 0,
            'timeouts_cancelled': 0,
            'total_wait_time': 0.0,
            'average_timeout_delay': 0.0
        }
        
        # Adaptive timeout adjustment
        self.adaptive_multipliers: Dict[TimeoutType, float] = {
            timeout_type: 1.0 for timeout_type in TimeoutType
        }
        
        # Start background monitoring
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_timeouts, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Timeout manager initialized")
    
    def set_timeout(self, height: int, round: int, step: Any, timeout: float, 
                   callback: Callable, data: Dict[str, Any] = None, 
                   max_retries: int = 0) -> str:
        """
        Set a timeout for consensus steps
        
        Args:
            height: Block height
            round: Consensus round
            step: Consensus step (PROPOSE, PREVOTE, PRECOMMIT)
            timeout: Timeout duration in seconds
            callback: Callback function to execute on timeout
            data: Additional data to pass to callback
            max_retries: Maximum number of retries (0 for no retry)
            
        Returns:
            Timeout ID for cancellation
        """
        with self.lock:
            try:
                # Convert step to TimeoutType
                timeout_type = self._step_to_timeout_type(step)
                
                # Apply adaptive multiplier
                adaptive_timeout = timeout * self.adaptive_multipliers.get(timeout_type, 1.0)
                adaptive_timeout = min(adaptive_timeout, timeout * self.config.max_timeout_multiplier)
                
                # Generate unique timeout ID
                timeout_id = f"{height}_{round}_{timeout_type.name}_{int(time.time() * 1000)}"
                
                current_time = time.time()
                expiration_time = current_time + adaptive_timeout
                
                timeout_event = TimeoutEvent(
                    timeout_id=timeout_id,
                    timeout_type=timeout_type,
                    callback=callback,
                    scheduled_time=current_time,
                    expiration_time=expiration_time,
                    data=data or {},
                    max_retries=max_retries
                )
                
                # Schedule the timeout
                self._schedule_timeout(timeout_event)
                
                self.timeouts[timeout_id] = timeout_event
                
                logger.debug(f"Scheduled timeout {timeout_id} for {adaptive_timeout:.2f}s "
                           f"(base: {timeout:.2f}s, multiplier: {self.adaptive_multipliers[timeout_type]:.2f})")
                
                return timeout_id
                
            except Exception as e:
                logger.error(f"Error setting timeout: {e}")
                return ""
    
    def _step_to_timeout_type(self, step: Any) -> TimeoutType:
        """Convert consensus step to timeout type"""
        step_name = step.name if hasattr(step, 'name') else str(step)
        
        if 'PROPOSE' in step_name:
            return TimeoutType.PROPOSE
        elif 'PREVOTE' in step_name:
            return TimeoutType.PREVOTE
        elif 'PRECOMMIT' in step_name:
            return TimeoutType.PRECOMMIT
        elif 'VIEW_CHANGE' in step_name:
            return TimeoutType.VIEW_CHANGE
        else:
            return TimeoutType.PROPOSE  # Default
    
    def _schedule_timeout(self, timeout_event: TimeoutEvent):
        """Schedule a timeout using threading.Timer"""
        try:
            delay = timeout_event.expiration_time - time.time()
            if delay <= 0:
                # Execute immediately if already expired
                self._execute_timeout(timeout_event)
                return
            
            def timeout_handler():
                try:
                    self._execute_timeout(timeout_event)
                except Exception as e:
                    logger.error(f"Error in timeout handler: {e}")
            
            timer = threading.Timer(delay, timeout_handler)
            timer.daemon = True
            timer.start()
            
            self.scheduled_timers[timeout_event.timeout_id] = timer
            
        except Exception as e:
            logger.error(f"Error scheduling timeout: {e}")
    
    def _execute_timeout(self, timeout_event: TimeoutEvent):
        """Execute timeout callback"""
        with self.lock:
            try:
                # Calculate actual delay
                actual_delay = time.time() - timeout_event.scheduled_time
                self.stats['total_wait_time'] += actual_delay
                self.stats['timeouts_triggered'] += 1
                
                # Update average delay
                total_triggered = self.stats['timeouts_triggered']
                self.stats['average_timeout_delay'] = (
                    (self.stats['average_timeout_delay'] * (total_triggered - 1) + actual_delay) 
                    / total_triggered
                )
                
                # Execute callback in thread pool
                future = self.executor.submit(self._safe_execute_callback, timeout_event)
                
                # Check if we need to retry
                if timeout_event.retry_count < timeout_event.max_retries:
                    self._schedule_retry(timeout_event)
                else:
                    # Clean up completed timeout
                    self._cleanup_timeout(timeout_event.timeout_id)
                
                logger.debug(f"Executed timeout {timeout_event.timeout_id} "
                           f"after {actual_delay:.2f}s (scheduled: {timeout_event.expiration_time - timeout_event.scheduled_time:.2f}s)")
                
            except Exception as e:
                logger.error(f"Error executing timeout: {e}")
    
    def _safe_execute_callback(self, timeout_event: TimeoutEvent):
        """Safely execute timeout callback with error handling"""
        try:
            # Prepare callback data
            callback_data = {
                'timeout_id': timeout_event.timeout_id,
                'timeout_type': timeout_event.timeout_type,
                'retry_count': timeout_event.retry_count,
                'scheduled_time': timeout_event.scheduled_time,
                'actual_delay': time.time() - timeout_event.scheduled_time,
                **timeout_event.data
            }
            
            # Execute callback
            timeout_event.callback(callback_data)
            
        except Exception as e:
            logger.error(f"Error in timeout callback for {timeout_event.timeout_id}: {e}")
            
            # Adjust adaptive multiplier for this timeout type
            self._adjust_adaptive_multiplier(timeout_event.timeout_type, increase=True)
    
    def _schedule_retry(self, timeout_event: TimeoutEvent):
        """Schedule a retry for the timeout"""
        try:
            timeout_event.retry_count += 1
            
            # Exponential backoff for retries
            backoff_factor = 2 ** timeout_event.retry_count
            retry_delay = min(
                (timeout_event.expiration_time - timeout_event.scheduled_time) * backoff_factor,
                self.config.max_timeout_multiplier * (timeout_event.expiration_time - timeout_event.scheduled_time)
            )
            
            # Update expiration time for retry
            timeout_event.expiration_time = time.time() + retry_delay
            
            # Reschedule
            self._schedule_timeout(timeout_event)
            
            logger.debug(f"Scheduled retry {timeout_event.retry_count} for timeout {timeout_event.timeout_id} "
                       f"in {retry_delay:.2f}s")
            
        except Exception as e:
            logger.error(f"Error scheduling retry: {e}")
    
    def cancel_timeout(self, timeout_id: str) -> bool:
        """
        Cancel a scheduled timeout
        
        Args:
            timeout_id: ID of timeout to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        with self.lock:
            try:
                if timeout_id not in self.timeouts:
                    return False
                
                # Cancel the timer
                timer = self.scheduled_timers.get(timeout_id)
                if timer:
                    timer.cancel()
                
                # Clean up
                self._cleanup_timeout(timeout_id)
                
                self.stats['timeouts_cancelled'] += 1
                logger.debug(f"Cancelled timeout {timeout_id}")
                
                return True
                
            except Exception as e:
                logger.error(f"Error cancelling timeout {timeout_id}: {e}")
                return False
    
    def cancel_timeouts_for_height_round(self, height: int, round: int) -> int:
        """
        Cancel all timeouts for a specific height and round
        
        Args:
            height: Block height
            round: Consensus round
            
        Returns:
            Number of timeouts cancelled
        """
        with self.lock:
            cancelled_count = 0
            prefix = f"{height}_{round}_"
            
            for timeout_id in list(self.timeouts.keys()):
                if timeout_id.startswith(prefix):
                    if self.cancel_timeout(timeout_id):
                        cancelled_count += 1
            
            logger.debug(f"Cancelled {cancelled_count} timeouts for height {height}, round {round}")
            return cancelled_count
    
    def _cleanup_timeout(self, timeout_id: str):
        """Clean up timeout resources"""
        try:
            # Remove from tracking dictionaries
            self.timeouts.pop(timeout_id, None)
            self.scheduled_timers.pop(timeout_id, None)
            
        except Exception as e:
            logger.error(f"Error cleaning up timeout {timeout_id}: {e}")
    
    def _adjust_adaptive_multiplier(self, timeout_type: TimeoutType, increase: bool = False):
        """Adjust adaptive timeout multiplier based on performance"""
        try:
            current_multiplier = self.adaptive_multipliers.get(timeout_type, 1.0)
            
            if increase:
                # Increase multiplier when timeouts fail
                new_multiplier = min(current_multiplier * 1.1, self.config.max_timeout_multiplier)
            else:
                # Gradually decrease multiplier when things are working well
                new_multiplier = max(current_multiplier * 0.99, 1.0)
            
            self.adaptive_multipliers[timeout_type] = new_multiplier
            
            logger.debug(f"Adjusted {timeout_type.name} timeout multiplier: {current_multiplier:.2f} -> {new_multiplier:.2f}")
            
        except Exception as e:
            logger.error(f"Error adjusting adaptive multiplier: {e}")
    
    def _monitor_timeouts(self):
        """Background thread to monitor and manage timeouts"""
        while self._running:
            try:
                self._check_expired_timeouts()
                self._cleanup_stale_timeouts()
                self._log_timeout_statistics()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in timeout monitor: {e}")
                time.sleep(10)  # Longer delay on error
    
    def _check_expired_timeouts(self):
        """Check for timeouts that should have triggered but didn't"""
        with self.lock:
            current_time = time.time()
            expired_timeouts = []
            
            for timeout_id, timeout_event in self.timeouts.items():
                if current_time > timeout_event.expiration_time + 1.0:  # 1 second grace period
                    expired_timeouts.append(timeout_event)
            
            for timeout_event in expired_timeouts:
                logger.warning(f"Timeout {timeout_event.timeout_id} expired but wasn't triggered")
                self._execute_timeout(timeout_event)
    
    def _cleanup_stale_timeouts(self):
        """Clean up timeouts that are too old"""
        with self.lock:
            current_time = time.time()
            stale_threshold = 3600  # 1 hour
            
            stale_timeouts = [
                timeout_id for timeout_id, timeout_event in self.timeouts.items()
                if current_time - timeout_event.scheduled_time > stale_threshold
            ]
            
            for timeout_id in stale_timeouts:
                logger.debug(f"Cleaning up stale timeout: {timeout_id}")
                self._cleanup_timeout(timeout_id)
    
    def _log_timeout_statistics(self):
        """Log timeout statistics periodically"""
        if self.stats['timeouts_triggered'] % 100 == 0 and self.stats['timeouts_triggered'] > 0:
            logger.info(
                f"Timeout statistics: "
                f"triggered={self.stats['timeouts_triggered']}, "
                f"cancelled={self.stats['timeouts_cancelled']}, "
                f"avg_delay={self.stats['average_timeout_delay']:.2f}s"
            )
    
    def get_timeout_statistics(self) -> Dict[str, Any]:
        """Get current timeout statistics"""
        with self.lock:
            return {
                'total_triggered': self.stats['timeouts_triggered'],
                'total_cancelled': self.stats['timeouts_cancelled'],
                'average_delay': self.stats['average_timeout_delay'],
                'active_timeouts': len(self.timeouts),
                'scheduled_timers': len(self.scheduled_timers),
                'adaptive_multipliers': {k.name: v for k, v in self.adaptive_multipliers.items()}
            }
    
    def set_adaptive_multiplier(self, timeout_type: TimeoutType, multiplier: float):
        """Manually set adaptive multiplier for a timeout type"""
        with self.lock:
            if 0.1 <= multiplier <= self.config.max_timeout_multiplier:
                self.adaptive_multipliers[timeout_type] = multiplier
                logger.info(f"Set {timeout_type.name} timeout multiplier to {multiplier:.2f}")
            else:
                logger.warning(f"Invalid multiplier {multiplier:.2f} for {timeout_type.name}")
    
    def reset_statistics(self):
        """Reset timeout statistics"""
        with self.lock:
            self.stats = {
                'timeouts_triggered': 0,
                'timeouts_cancelled': 0,
                'total_wait_time': 0.0,
                'average_timeout_delay': 0.0
            }
            logger.info("Timeout statistics reset")
    
    def shutdown(self):
        """Shutdown timeout manager"""
        with self.lock:
            self._running = False
            
            # Cancel all pending timeouts
            for timeout_id in list(self.timeouts.keys()):
                self.cancel_timeout(timeout_id)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Timeout manager shutdown complete")

class PeriodicTaskManager:
    """Manager for periodic background tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self.lock = threading.RLock()
        self._running = True
        
    def register_task(self, task_id: str, interval: float, task_func: Callable, 
                     initial_delay: float = 0, data: Dict = None) -> bool:
        """
        Register a periodic task
        
        Args:
            task_id: Unique task identifier
            interval: Execution interval in seconds
            task_func: Task function to execute
            initial_delay: Initial delay before first execution
            data: Additional data for task function
            
        Returns:
            True if registered successfully
        """
        with self.lock:
            if task_id in self.tasks:
                logger.warning(f"Task {task_id} already registered")
                return False
            
            self.tasks[task_id] = {
                'interval': interval,
                'task_func': task_func,
                'data': data or {},
                'last_execution': 0,
                'execution_count': 0,
                'enabled': True
            }
            
            # Schedule initial execution
            threading.Timer(initial_delay, self._execute_task, args=[task_id]).start()
            
            logger.debug(f"Registered periodic task {task_id} with interval {interval}s")
            return True
    
    def _execute_task(self, task_id: str):
        """Execute a periodic task"""
        with self.lock:
            if task_id not in self.tasks or not self.tasks[task_id]['enabled']:
                return
            
            task = self.tasks[task_id]
            
            try:
                # Execute task function
                task['task_func'](task['data'])
                task['last_execution'] = time.time()
                task['execution_count'] += 1
                
            except Exception as e:
                logger.error(f"Error executing periodic task {task_id}: {e}")
            
            finally:
                # Schedule next execution if still enabled
                if task['enabled'] and self._running:
                    threading.Timer(task['interval'], self._execute_task, args=[task_id]).start()
    
    def unregister_task(self, task_id: str) -> bool:
        """Unregister a periodic task"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id]['enabled'] = False
            del self.tasks[task_id]
            
            logger.debug(f"Unregistered periodic task {task_id}")
            return True
    
    def pause_task(self, task_id: str) -> bool:
        """Pause a periodic task"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id]['enabled'] = False
            logger.debug(f"Paused periodic task {task_id}")
            return True
    
    def resume_task(self, task_id: str) -> bool:
        """Resume a paused periodic task"""
        with self.lock:
            if task_id not in self.tasks:
                return False
            
            self.tasks[task_id]['enabled'] = True
            # Schedule immediate execution
            threading.Timer(0.1, self._execute_task, args=[task_id]).start()
            
            logger.debug(f"Resumed periodic task {task_id}")
            return True
    
    def get_task_statistics(self) -> Dict[str, Any]:
        """Get statistics for all tasks"""
        with self.lock:
            stats = {}
            for task_id, task in self.tasks.items():
                stats[task_id] = {
                    'interval': task['interval'],
                    'last_execution': task['last_execution'],
                    'execution_count': task['execution_count'],
                    'enabled': task['enabled']
                }
            return stats
    
    def shutdown(self):
        """Shutdown task manager"""
        with self.lock:
            self._running = False
            for task_id in self.tasks:
                self.tasks[task_id]['enabled'] = False
            logger.info("Periodic task manager shutdown")