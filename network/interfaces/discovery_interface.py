from abc import ABC, abstractmethod
from typing import List, Dict

class IPeerDiscovery(ABC):
    """Interface for peer discovery mechanisms"""
    
    @abstractmethod
    async def discover_peers(self) -> List[Dict]:
        """Discover new peers"""
        pass
    
    @abstractmethod
    async def bootstrap_network(self):
        """Bootstrap to the network"""
        pass
    
    @abstractmethod
    async def maintain_peer_list(self):
        """Maintain and update peer list"""
        pass
    
    @abstractmethod
    async def request_peer_lists(self):
        """Request peer lists from connected peers"""
        pass