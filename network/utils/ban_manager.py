import time
import ipaddress
from typing import Dict, Set, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger("BanManager")

@dataclass
class BanRecord:
    address: str
    banned_until: float
    reason: str
    severity: int  # 1=low, 2=medium, 3=high
    created_at: float

class BanManager:  
    def __init__(self, ban_threshold: int = -100, ban_duration: int = 3600):
        self.ban_threshold = ban_threshold
        self.ban_duration = ban_duration
        self.banned_peers: Dict[str, BanRecord] = {}
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.auto_ban_enabled: bool = True
    
    async def is_peer_banned(self, address: str) -> bool:
        """Check if peer is banned with enhanced validation"""
        # Validate IP address format
        try:
            ipaddress.ip_address(address)
        except ValueError:
            logger.warning(f"Invalid IP address format: {address}")
            return True  # Ban invalid addresses
        
        if address in self.whitelist:
            return False
            
        if address in self.blacklist:
            return True
            
        if address in self.banned_peers:
            record = self.banned_peers[address]
            if record.banned_until > time.time():
                return True
            else:
                # Ban expired, remove it
                del self.banned_peers[address]
                logger.info(f"Ban expired for {address}")
        return False
    
    async def ban_peer(self, address: str, duration: int = None, reason: str = "Unknown", severity: int = 1):
        """Enhanced ban with reason and severity tracking"""
        if duration is None:
            duration = self.ban_duration
            
        ban_record = BanRecord(
            address=address,
            banned_until=time.time() + duration,
            reason=reason,
            severity=severity,
            created_at=time.time()
        )
        
        self.banned_peers[address] = ban_record
        logger.warning(f"Banned peer {address} for {duration}s: {reason} (severity: {severity})")
    
    async def unban_peer(self, address: str):
        """Unban a peer"""
        if address in self.banned_peers:
            del self.banned_peers[address]
            logger.info(f"Unbanned peer: {address}")
    
    async def get_ban_stats(self) -> Dict[str, Any]:
        """Get ban statistics"""
        current_time = time.time()
        active_bans = [
            record for record in self.banned_peers.values() 
            if record.banned_until > current_time
        ]
        
        return {
            'total_banned': len(active_bans),
            'whitelisted': len(self.whitelist),
            'blacklisted': len(self.blacklist),
            'by_severity': {
                1: len([r for r in active_bans if r.severity == 1]),
                2: len([r for r in active_bans if r.severity == 2]),
                3: len([r for r in active_bans if r.severity == 3]),
            }
        }
    
    async def add_to_whitelist(self, address: str):
        """Add peer to whitelist"""
        self.whitelist.add(address)
        if address in self.banned_peers:
            del self.banned_peers[address]
        logger.info(f"Added {address} to whitelist")
    
    async def add_to_blacklist(self, address: str):
        """Add peer to blacklist"""
        self.blacklist.add(address)
        # Permanent ban
        await self.ban_peer(address, duration=365*24*3600, reason="Blacklisted", severity=3)
        logger.warning(f"Added {address} to blacklist")
    
    async def cleanup_expired_bans(self):
        """Clean up expired bans"""
        current_time = time.time()
        expired_bans = [
            address for address, record in self.banned_peers.items()
            if record.banned_until <= current_time and address not in self.blacklist
        ]
        
        for address in expired_bans:
            del self.banned_peers[address]
        
        if expired_bans:
            logger.info(f"Cleaned up {len(expired_bans)} expired bans")