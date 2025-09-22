import time
from typing import Dict, Set

class BanManager:
    """Ban management implementation"""
    
    def __init__(self, ban_threshold: int = -100, ban_duration: int = 3600):
        self.ban_threshold = ban_threshold
        self.ban_duration = ban_duration
        self.banned_peers: Dict[str, float] = {}
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
    
    async def is_peer_banned(self, address: str) -> bool:
        """Check if peer is banned"""
        if address in self.whitelist:
            return False
            
        if address in self.blacklist:
            return True
            
        if address in self.banned_peers:
            if self.banned_peers[address] > time.time():
                return True
            else:
                # Ban expired
                del self.banned_peers[address]
        return False
    
    async def ban_peer(self, address: str, duration: int = None):
        """Ban a peer"""
        if duration is None:
            duration = self.ban_duration
        self.banned_peers[address] = time.time() + duration
    
    async def unban_peer(self, address: str):
        """Unban a peer"""
        if address in self.banned_peers:
            del self.banned_peers[address]
    
    async def add_to_whitelist(self, address: str):
        """Add peer to whitelist"""
        self.whitelist.add(address)
        if address in self.banned_peers:
            del self.banned_peers[address]
    
    async def add_to_blacklist(self, address: str):
        """Add peer to blacklist"""
        self.blacklist.add(address)
        self.banned_peers[address] = float('inf')
    
    async def cleanup_expired_bans(self):
        """Clean up expired bans"""
        current_time = time.time()
        expired_bans = [
            peer for peer, ban_until in self.banned_peers.items()
            if ban_until <= current_time and peer not in self.blacklist
        ]
        
        for peer in expired_bans:
            del self.banned_peers[peer]