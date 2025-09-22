from typing import Dict, List, Optional
import threading
import time
import logging
from pathlib import Path

from database.utils.exceptions import DatabaseError

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database instance management and connection pooling"""
    
    _instances: Dict[str, Any] = {}
    _lock = threading.RLock()
    
    def __init__(self, max_connections: int = 10, connection_timeout: int = 300):
        self.max_connections = max_connections
        self.connection_timeout = connection_timeout
        self.cleanup_thread = None
        self.running = False
    
    @classmethod
    def get_database(cls, db_path: str, config: Optional[Dict] = None) -> Any:
        """Get or create database instance"""
        with cls._lock:
            if db_path not in cls._instances:
                from ..core.database import AdvancedDatabase
                from ..utils.types import DatabaseConfig
                
                db_config = DatabaseConfig(**(config or {}))
                cls._instances[db_path] = AdvancedDatabase(db_path, db_config)
            
            return cls._instances[db_path]
    
    @classmethod
    def close_database(cls, db_path: str) -> bool:
        """Close database instance"""
        with cls._lock:
            if db_path in cls._instances:
                try:
                    cls._instances[db_path].close()
                    del cls._instances[db_path]
                    return True
                except Exception as e:
                    logger.error(f"Failed to close database {db_path}: {e}")
                    return False
            return False
    
    @classmethod
    def close_all(cls) -> bool:
        """Close all database instances"""
        with cls._lock:
            success = True
            for db_path in list(cls._instances.keys()):
                if not cls.close_database(db_path):
                    success = False
            return success
    
    def start_cleanup(self):
        """Start connection cleanup thread"""
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        logger.info("Database cleanup started")
    
    def stop_cleanup(self):
        """Stop connection cleanup"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        logger.info("Database cleanup stopped")
    
    def _cleanup_loop(self):
        """Cleanup idle connections"""
        while self.running:
            try:
                self._cleanup_idle_connections()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                time.sleep(60)
    
    def _cleanup_idle_connections(self):
        """Cleanup connections that have been idle too long"""
        with self._lock:
            current_time = time.time()
            for db_path, db_instance in list(self._instances.items()):
                # Check if connection is idle (simplified)
                stats = db_instance.get_stats()
                last_activity = current_time - stats.get('uptime_seconds', 0)
                
                if last_activity > self.connection_timeout:
                    logger.info(f"Closing idle database connection: {db_path}")
                    self.close_database(db_path)
    
    def list_databases(self) -> List[str]:
        """List all managed databases"""
        with self._lock:
            return list(self._instances.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics"""
        with self._lock:
            return {
                'total_connections': len(self._instances),
                'database_paths': list(self._instances.keys()),
                'max_connections': self.max_connections,
                'connection_timeout': self.connection_timeout
            }