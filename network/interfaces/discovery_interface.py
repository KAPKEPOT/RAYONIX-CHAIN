# network/interfaces/discovery_interface.py
import abc
from typing import List, Dict, Any
from network.models.peer_info import PeerInfo

class IPeerDiscovery(abc.ABC):
    """Interface for peer discovery implementations"""
    
    @abc.abstractmethod
    async def discover_peers(self) -> List[Dict]:
        """Discover new peers"""
        pass
    
    @abc.abstractmethod
    async def bootstrap_network(self):
        """Bootstrap the network"""
        pass
    
    @abc.abstractmethod
    async def maintain_peer_list(self):
        """Maintain peer list"""
        pass
    
    @abc.abstractmethod
    async def request_peer_lists(self):
        """Request peer lists from connected peers"""
        pass
    
    @abc.abstractmethod
    def add_peer(self, peer_info: PeerInfo):
        """Add a peer"""
        pass