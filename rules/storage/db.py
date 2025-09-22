"""
Database utilities and connection management
"""

import plyvel
from typing import Optional
import logging
from pathlib import Path

from ..exceptions import StorageError

logger = logging.getLogger('consensus.storage')

class DBManager:
    """Database connection manager"""
    
    _instance = None
    
    def __new__(cls, db_path: Optional[str] = None, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(db_path, **kwargs)
        return cls._instance
    
    def _initialize(self, db_path: str, **kwargs):
        self.db_path = Path(db_path) if db_path else Path("./consensus_db")
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.connections = {}
        self.default_options = {
            'create_if_missing': True,
            'max_open_files': 1000,
            'write_buffer_size': 4 * 1024 * 1024,
            'read_buffer_size': 1 * 1024 * 1024,
            **kwargs
        }
    
    def get_connection(self, name: str, **options) -> plyvel.DB:
        """Get database connection by name"""
        if name not in self.connections:
            db_path = self.db_path / name
            db_path.mkdir(exist_ok=True)
            
            connection_options = {**self.default_options, **options}
            self.connections[name] = plyvel.DB(str(db_path), **connection_options)
            
            logger.info(f"Database connection established: {name}")
        
        return self.connections[name]
    
    def close_connection(self, name: str) -> None:
        """Close specific database connection"""
        if name in self.connections:
            self.connections[name].close()
            del self.connections[name]
            logger.info(f"Database connection closed: {name}")
    
    def close_all(self) -> None:
        """Close all database connections"""
        for name, connection in self.connections.items():
            connection.close()
            logger.info(f"Database connection closed: {name}")
        self.connections.clear()
    
    def get_stats(self, name: str) -> Dict:
        """Get database statistics"""
        if name in self.connections:
            db = self.connections[name]
            # LevelDB doesn't provide extensive stats directly
            # This would need implementation based on specific requirements
            return {}
        return {}