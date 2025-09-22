from abc import ABC, abstractmethod
from typing import Any

class IProtocolHandler(ABC):
    """Interface for protocol handlers"""
    
    @abstractmethod
    async def start_server(self):
        """Start the protocol server"""
        pass
    
    @abstractmethod
    async def stop_server(self):
        """Stop the protocol server"""
        pass
    
    @abstractmethod
    async def send_message(self, connection_id: str, message: Any) -> bool:
        """Send message via this protocol"""
        pass
    
    @abstractmethod
    async def handle_connection(self, *args, **kwargs):
        """Handle incoming connection"""
        pass