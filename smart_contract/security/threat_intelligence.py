# smart_contract/security/threat_intelligence.py
import time
import logging
import aiohttp
import asyncio
from typing import Dict, Any, List, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger("SmartContract.ThreatIntelligence")

@dataclass
class ThreatIntelConfig:
    """Configuration for threat intelligence system"""
    update_interval: int = 300  # 5 minutes
    max_blacklist_size: int = 10000
    feed_urls: List[str] = field(default_factory=lambda: [
        "https://api.threatintel.example.com/blacklist",
        "https://malware-api.example.com/ips",
        "https://blockchain-security.example.com/threats"
    ])
    cache_duration: int = 3600  # 1 hour
    timeout: int = 30  # seconds

class ThreatIntelligenceFeed:
    """Advanced threat intelligence system with multiple feed integration"""
    
    def __init__(self, config: Optional[ThreatIntelConfig] = None):
        self.config = config or ThreatIntelConfig()
        
        # Threat data storage
        self.blacklisted_addresses: Set[str] = set()
        self.malicious_ips: Set[str] = set()
        self.suspicious_domains: Set[str] = set()
        self.known_threats: Dict[str, Dict] = {}
        
        # Cache for API responses
        self.cache: Dict[str, Dict] = {}
        self.last_update: Dict[str, float] = {}
        
        # Threat level calculation
        self.threat_level: int = 0  # 0-100 scale
        self.threat_metrics: Dict[str, Any] = {
            'last_updated': 0,
            'blacklist_count': 0,
            'ip_count': 0,
            'domain_count': 0,
            'update_errors': 0
        }
        
        # HTTP session for API calls
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("ThreatIntelligenceFeed initialized")
    
    async def update(self) -> None:
        """Update threat intelligence from all feeds"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        
        try:
            # Update from all feeds in parallel
            tasks = []
            for url in self.config.feed_urls:
                tasks.append(self._update_from_feed(url))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_updates = 0
            for result in results:
                if not isinstance(result, Exception):
                    successful_updates += 1
            
            # Update threat level
            self._update_threat_level(successful_updates, len(self.config.feed_urls))
            
            # Cleanup old cache entries
            self._cleanup_cache()
            
            logger.info(f"Threat intelligence update completed. Success: {successful_updates}/{len(self.config.feed_urls)}")
            
        except Exception as e:
            logger.error(f"Threat intelligence update failed: {e}")
            self.threat_metrics['update_errors'] += 1
    
    async def _update_from_feed(self, url: str) -> bool:
        """Update from a specific threat intelligence feed"""
        cache_key = f"feed_{hash(url)}"
        current_time = time.time()
        
        # Check cache first
        if (cache_key in self.cache and 
            current_time - self.last_update.get(cache_key, 0) < self.config.cache_duration):
            cached_data = self.cache[cache_key]
            self._process_feed_data(url, cached_data)
            return True
        
        try:
            # Fetch from API
            async with self.session.get(url, timeout=self.config.timeout) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Cache the response
                    self.cache[cache_key] = data
                    self.last_update[cache_key] = current_time
                    
                    # Process the data
                    self._process_feed_data(url, data)
                    
                    return True
                else:
                    logger.warning(f"Feed {url} returned status {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.warning(f"Feed {url} timeout after {self.config.timeout}s")
            return False
        except Exception as e:
            logger.error(f"Failed to update from feed {url}: {e}")
            return False
    
    def _process_feed_data(self, url: str, data: Dict) -> None:
        """Process data from a threat intelligence feed"""
        try:
            # Extract blacklisted addresses
            if 'blacklisted_addresses' in data:
                addresses = data['blacklisted_addresses']
                if isinstance(addresses, list):
                    self.blacklisted_addresses.update(addresses[:self.config.max_blacklist_size])
            
            # Extract malicious IPs
            if 'malicious_ips' in data:
                ips = data['malicious_ips']
                if isinstance(ips, list):
                    self.malicious_ips.update(ips)
            
            # Extract suspicious domains
            if 'suspicious_domains' in data:
                domains = data['suspicious_domains']
                if isinstance(domains, list):
                    self.suspicious_domains.update(domains)
            
            # Extract known threats
            if 'known_threats' in data:
                threats = data['known_threats']
                if isinstance(threats, dict):
                    self.known_threats.update(threats)
            
            logger.debug(f"Processed data from {url}. Blacklist: {len(self.blacklisted_addresses)} addresses")
            
        except Exception as e:
            logger.error(f"Failed to process data from {url}: {e}")
    
    def _update_threat_level(self, successful_updates: int, total_feeds: int) -> None:
        """Update overall threat level"""
        # Base threat level based on blacklist size
        base_level = min(100, len(self.blacklisted_addresses) / 100)
        
        # Adjust based on update success rate
        success_rate = successful_updates / total_feeds if total_feeds > 0 else 0
        success_factor = 1.0 - (success_rate * 0.3)  # Lower success rate increases threat level
        
        # Adjust based on recent errors
        error_factor = 1.0 + (min(self.threat_metrics['update_errors'], 10) * 0.05)
        
        # Calculate final threat level
        self.threat_level = min(100, int(base_level * success_factor * error_factor))
        
        # Update metrics
        self.threat_metrics.update({
            'last_updated': time.time(),
            'blacklist_count': len(self.blacklisted_addresses),
            'ip_count': len(self.malicious_ips),
            'domain_count': len(self.suspicious_domains),
            'success_rate': success_rate,
            'threat_level': self.threat_level
        })
    
    def _cleanup_cache(self) -> None:
        """Clean up old cache entries"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamp in self.last_update.items():
            if current_time - timestamp > self.config.cache_duration * 2:  # 2x cache duration
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            if key in self.last_update:
                del self.last_update[key]
    
    def is_address_blacklisted(self, address: str) -> bool:
        """Check if an address is blacklisted"""
        return address in self.blacklisted_addresses
    
    def is_ip_malicious(self, ip_address: str) -> bool:
        """Check if an IP address is malicious"""
        return ip_address in self.malicious_ips
    
    def is_domain_suspicious(self, domain: str) -> bool:
        """Check if a domain is suspicious"""
        return domain in self.suspicious_domains
    
    def get_threat_info(self, identifier: str) -> Optional[Dict]:
        """Get detailed threat information"""
        return self.known_threats.get(identifier)
    
    def get_blacklisted_addresses(self) -> Set[str]:
        """Get all blacklisted addresses"""
        return self.blacklisted_addresses.copy()
    
    def get_current_threat_level(self) -> int:
        """Get current threat level (0-100)"""
        return self.threat_level
    
    def get_threat_metrics(self) -> Dict[str, Any]:
        """Get threat intelligence metrics"""
        return self.threat_metrics.copy()
    
    def add_custom_blacklist(self, addresses: List[str]) -> None:
        """Add custom addresses to blacklist"""
        self.blacklisted_addresses.update(addresses)
        # Ensure we don't exceed max size
        if len(self.blacklisted_addresses) > self.config.max_blacklist_size:
            # Convert to list, trim, and convert back to set
            addresses_list = list(self.blacklisted_addresses)
            self.blacklisted_addresses = set(addresses_list[-self.config.max_blacklist_size:])
        
        logger.info(f"Added {len(addresses)} addresses to custom blacklist")
    
    def remove_from_blacklist(self, address: str) -> bool:
        """Remove an address from blacklist"""
        if address in self.blacklisted_addresses:
            self.blacklisted_addresses.remove(address)
            logger.info(f"Removed {address} from blacklist")
            return True
        return False
    
    async def close(self) -> None:
        """Close resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.session:
                asyncio.create_task(self.close())
        except:
            pass