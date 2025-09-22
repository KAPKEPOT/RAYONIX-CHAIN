import threading
import time
from typing import List, Dict, Any
import logging
from dataclasses import dataclass

from ..utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

@dataclass
class BackgroundTask:
    name: str
    interval: int
    last_run: float = 0
    enabled: bool = True

class BackgroundTaskService:
    """Background task service for database maintenance"""
    
    def __init__(self, db):
        self.db = db
        self.tasks: Dict[str, BackgroundTask] = {}
        self.thread = None
        self.running = False
    
    def start(self):
        """Start background task service"""
        self.running = True
        self._register_default_tasks()
        self.thread = threading.Thread(target=self._task_loop, daemon=True)
        self.thread.start()
        logger.info("Background task service started")
    
    def stop(self):
        """Stop background task service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("Background task service stopped")
    
    def register_task(self, name: str, interval: int, task_func, enabled: bool = True):
        """Register a new background task"""
        self.tasks[name] = BackgroundTask(name, interval)
        setattr(self, f"_task_{name}", task_func)
    
    def _register_default_tasks(self):
        """Register default maintenance tasks"""
        self.register_task("ttl_cleanup", 300, self._ttl_cleanup_task)
        self.register_task("cache_cleanup", 60, self._cache_cleanup_task)
        self.register_task("index_maintenance", 3600, self._index_maintenance_task)
        self.register_task("stats_report", 300, self._stats_report_task)
    
    def _task_loop(self):
        """Main task execution loop"""
        while self.running:
            try:
                current_time = time.time()
                
                for task_name, task in self.tasks.items():
                    if task.enabled and current_time - task.last_run >= task.interval:
                        try:
                            task_func = getattr(self, f"_task_{task_name}")
                            task_func()
                            task.last_run = current_time
                        except Exception as e:
                            logger.error(f"Background task {task_name} failed: {e}")
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Background task loop error: {e}")
                time.sleep(5)
    
    def _ttl_cleanup_task(self):
        """Cleanup expired entries based on TTL"""
        try:
            current_time = time.time()
            expired_count = 0
            
            # Iterate through timestamp index to find expired entries
            if 'timestamp' in self.db.indexes:
                # This would use the timestamp index to find expired entries
                # For now, we'll use a simple iteration approach
                for key, value, metadata in self.db.iterate(include_metadata=True):
                    if metadata and self.db._is_expired(metadata):
                        self.db.delete(key, update_indexes=True)
                        expired_count += 1
            
            if expired_count > 0:
                logger.info(f"TTL cleanup removed {expired_count} expired entries")
                
        except Exception as e:
            logger.error(f"TTL cleanup task failed: {e}")
    
    def _cache_cleanup_task(self):
        """Cleanup expired cache entries"""
        try:
            current_time = time.time()
            expired_count = 0
            
            with self.db.locks['cache']:
                keys_to_remove = []
                for key, (value, timestamp) in self.db.cache.items():
                    if current_time - timestamp > self.db.config.cache_ttl:
                        keys_to_remove.append(key)
                        expired_count += 1
                
                for key in keys_to_remove:
                    del self.db.cache[key]
            
            if expired_count > 0:
                logger.debug(f"Cache cleanup removed {expired_count} expired entries")
                
        except Exception as e:
            logger.error(f"Cache cleanup task failed: {e}")
    
    def _index_maintenance_task(self):
        """Perform index maintenance"""
        try:
            # Rebuild fragmented indexes
            for index_name, index in self.db.indexes.items():
                if hasattr(index, 'needs_maintenance') and index.needs_maintenance():
                    logger.info(f"Rebuilding index {index_name}")
                    if hasattr(index, 'rebuild'):
                        index.rebuild()
            
            # Update index statistics
            self.db.stats.index_count = len(self.db.indexes)
            
        except Exception as e:
            logger.error(f"Index maintenance task failed: {e}")
    
    def _stats_report_task(self):
        """Report database statistics"""
        try:
            stats = self.db.get_stats()
            logger.info(f"Database stats: {stats}")
            
            # Check for potential issues
            if stats.get('error_rate', 0) > 0.05:
                logger.warning(f"High error rate detected: {stats['error_rate']:.2%}")
            
        except Exception as e:
            logger.error(f"Stats report task failed: {e}")