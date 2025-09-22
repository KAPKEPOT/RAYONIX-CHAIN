from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class IConnectionManager(ABC):
    """Interface for connection management"""
    
    @abstractmethod
    async def start(self):
        """Start the connection manager"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the connection manager"""
        pass
    
    @abstractmethod
    async def connect_to_peer(self, address: str, port: int, protocol: str) -> Optional[str]:
        """Connect to a peer"""
        pass
    
    @abstractmethod
    async def close_connection(self, connection_id: str):
        """Close a connection"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if network has active connections"""
        pass
    
    @abstractmethod
    def get_connection(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get connection details"""
        pass