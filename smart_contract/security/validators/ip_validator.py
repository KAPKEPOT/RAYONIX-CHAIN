# smart_contract/security/validators/ip_validator.py
import re
import logging
import ipaddress
from typing import Optional, Set
from dataclasses import dataclass

logger = logging.getLogger("SmartContract.IPValidator")

@dataclass
class IPValidatorConfig:
    """Configuration for IP validation"""
    allow_private: bool = False
    allow_reserved: bool = False
    allow_multicast: bool = False
    blacklist: Set[str] = None
    whitelist: Set[str] = None

class IPValidator:
    """Advanced IP address validation with network range checking"""
    
    def __init__(self, config: Optional[IPValidatorConfig] = None):
        self.config = config or IPValidatorConfig()
        if self.config.blacklist is None:
            self.config.blacklist = set()
        if self.config.whitelist is None:
            self.config.whitelist = set()
        
        # Common malicious IP ranges (simplified)
        self.malicious_ranges = {
            ipaddress.ip_network('192.0.2.0/24'),  # TEST-NET-1
            ipaddress.ip_network('198.51.100.0/24'),  # TEST-NET-2
            ipaddress.ip_network('203.0.113.0/24'),  # TEST-NET-3
            ipaddress.ip_network('169.254.0.0/16'),  # Link-local
            ipaddress.ip_network('127.0.0.0/8'),     # Loopback
        }
        
        logger.info("IPValidator initialized")
    
    def validate(self, ip_address: str) -> bool:
        """Validate an IP address"""
        try:
            # Basic format validation
            if not self._validate_format(ip_address):
                return False
            
            # Parse IP address
            ip = ipaddress.ip_address(ip_address)
            
            # Check whitelist first
            if str(ip) in self.config.whitelist:
                return True
            
            # Check blacklist
            if str(ip) in self.config.blacklist:
                logger.warning(f"IP address blacklisted: {ip_address}")
                return False
            
            # Check for malicious ranges
            if any(ip in network for network in self.malicious_ranges):
                logger.warning(f"IP address in malicious range: {ip_address}")
                return False
            
            # Check IP type restrictions
            if not self.config.allow_private and ip.is_private:
                logger.warning(f"Private IP address not allowed: {ip_address}")
                return False
            
            if not self.config.allow_reserved and ip.is_reserved:
                logger.warning(f"Reserved IP address not allowed: {ip_address}")
                return False
            
            if not self.config.allow_multicast and ip.is_multicast:
                logger.warning(f"Multicast IP address not allowed: {ip_address}")
                return False
            
            # Check if IP is global (public)
            if not ip.is_global:
                logger.warning(f"Non-global IP address: {ip_address}")
                return False
            
            return True
            
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip_address}")
            return False
        except Exception as e:
            logger.error(f"IP validation failed for {ip_address}: {e}")
            return False
    
    def _validate_format(self, ip_address: str) -> bool:
        """Validate IP address format"""
        # IPv4 pattern
        ipv4_pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'
        
        # IPv6 pattern (simplified)
        ipv6_pattern = r'^[0-9a-fA-F:]+$'
        
        if re.match(ipv4_pattern, ip_address):
            # Validate IPv4 octets
            octets = ip_address.split('.')
            for octet in octets:
                num = int(octet)
                if num < 0 or num > 255:
                    return False
            return True
        
        elif re.match(ipv6_pattern, ip_address):
            # Basic IPv6 validation
            # In production, this would use more comprehensive validation
            return True
        
        return False
    
    def add_to_blacklist(self, ip_address: str) -> bool:
        """Add an IP address to the blacklist"""
        try:
            ip = ipaddress.ip_address(ip_address)
            self.config.blacklist.add(str(ip))
            logger.info(f"Added to blacklist: {ip_address}")
            return True
        except ValueError:
            logger.error(f"Invalid IP address for blacklist: {ip_address}")
            return False
    
    def remove_from_blacklist(self, ip_address: str) -> bool:
        """Remove an IP address from the blacklist"""
        if ip_address in self.config.blacklist:
            self.config.blacklist.remove(ip_address)
            logger.info(f"Removed from blacklist: {ip_address}")
            return True
        return False
    
    def add_to_whitelist(self, ip_address: str) -> bool:
        """Add an IP address to the whitelist"""
        try:
            ip = ipaddress.ip_address(ip_address)
            self.config.whitelist.add(str(ip))
            logger.info(f"Added to whitelist: {ip_address}")
            return True
        except ValueError:
            logger.error(f"Invalid IP address for whitelist: {ip_address}")
            return False
    
    def remove_from_whitelist(self, ip_address: str) -> bool:
        """Remove an IP address from the whitelist"""
        if ip_address in self.config.whitelist:
            self.config.whitelist.remove(ip_address)
            logger.info(f"Removed from whitelist: {ip_address}")
            return True
        return False
    
    def add_malicious_range(self, network: str) -> bool:
        """Add a network range to malicious ranges"""
        try:
            net = ipaddress.ip_network(network)
            self.malicious_ranges.add(net)
            logger.info(f"Added malicious range: {network}")
            return True
        except ValueError:
            logger.error(f"Invalid network range: {network}")
            return False
    
    def remove_malicious_range(self, network: str) -> bool:
        """Remove a network range from malicious ranges"""
        try:
            net = ipaddress.ip_network(network)
            if net in self.malicious_ranges:
                self.malicious_ranges.remove(net)
                logger.info(f"Removed malicious range: {network}")
                return True
        except ValueError:
            pass
        return False
    
    def is_private(self, ip_address: str) -> bool:
        """Check if an IP address is private"""
        try:
            ip = ipaddress.ip_address(ip_address)
            return ip.is_private
        except ValueError:
            return False
    
    def is_global(self, ip_address: str) -> bool:
        """Check if an IP address is global (public)"""
        try:
            ip = ipaddress.ip_address(ip_address)
            return ip.is_global
        except ValueError:
            return False
    
    def get_validation_stats(self) -> dict:
        """Get validation statistics"""
        return {
            'blacklist_size': len(self.config.blacklist),
            'whitelist_size': len(self.config.whitelist),
            'malicious_ranges': len(self.malicious_ranges),
            'allow_private': self.config.allow_private,
            'allow_reserved': self.config.allow_reserved,
            'allow_multicast': self.config.allow_multicast
        }