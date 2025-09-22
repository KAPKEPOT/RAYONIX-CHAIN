from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List

class DatabaseInterface(ABC):
    """Abstract base interface for database operations"""
    
    @abstractmethod
    def put(self, key: Any, value: Any, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def get(self, key: Any, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def delete(self, key: Any, **kwargs) -> bool:
        pass
    
    @abstractmethod
    def iterate(self, **kwargs) -> Any:
        pass
    
    @abstractmethod
    def close(self) -> None:
        pass

class IndexInterface(ABC):
    """Abstract base interface for index operations"""
    
    @abstractmethod
    def update(self, key: Any, value: Any, update_data: Any) -> None:
        pass
    
    @abstractmethod
    def remove(self, key: Any, removal_data: Any) -> None:
        pass
    
    @abstractmethod
    def query(self, query: Any, **kwargs) -> List[Any]:
        pass

class SerializerInterface(ABC):
    """Abstract base interface for serialization"""
    
    @abstractmethod
    def serialize(self, value: Any) -> bytes:
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        pass

class CompressionInterface(ABC):
    """Abstract base interface for compression"""
    
    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        pass
    
    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        pass

class EncryptionInterface(ABC):
    """Abstract base interface for encryption"""
    
    @abstractmethod
    def encrypt(self, data: bytes) -> bytes:
        pass
    
    @abstractmethod
    def decrypt(self, data: bytes) -> bytes:
        pass