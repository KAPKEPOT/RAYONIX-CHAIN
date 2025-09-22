from abc import ABC, abstractmethod
from typing import Callable, Any

class IMessageProcessor(ABC):
    """Interface for message processing"""
    
    @abstractmethod
    async def process_message(self, connection_id: str, message: Any):
        """Process incoming message"""
        pass
    
    @abstractmethod
    def register_handler(self, message_type: Any, handler: Callable):
        """Register message handler"""
        pass
    
    @abstractmethod
    def unregister_handler(self, message_type: Any, handler: Callable):
        """Unregister message handler"""
        pass
    
    @abstractmethod
    async def broadcast_message(self, message: Any, exclude: list = None):
        """Broadcast message to all connected peers"""
        pass