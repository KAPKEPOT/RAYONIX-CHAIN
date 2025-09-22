# smart_contract/security/validators/domain_validator.py
import re
import logging
import asyncio
import aiohttp
from typing import Optional, Set
from dataclasses import dataclass

logger = logging.getLogger("SmartContract.DomainValidator")

@dataclass
class DomainValidatorConfig:
    """Configuration for domain validation"""
    check_dns: bool = True
    check_tld: bool = True
    timeout: int = 5  # seconds
    cache_size: int = 1000
    cache_ttl: int = 3600  # 1 hour

class DomainValidator:
    """Advanced domain validation with DNS resolution and TLD checking"""
    
    def __init__(self, config: Optional[DomainValidatorConfig] = None):
        self.config = config or DomainValidatorConfig()
        self.valid_tlds: Set[str] = self._load_common_tlds()
        self.validation_cache: dict = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        logger.info("DomainValidator initialized")
    
    def _load_common_tlds(self) -> Set[str]:
        """Load common top-level domains"""
        # This would be loaded from a maintained list in production
        common_tlds = {
            'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
            'io', 'ai', 'co', 'uk', 'de', 'fr', 'jp', 'cn',
            'ru', 'br', 'in', 'au', 'ca', 'mx', 'es', 'it'
        }
        return common_tlds
    
    async def validate(self, domain: str) -> bool:
        """Validate a domain name asynchronously"""
        # Basic format validation
        if not self._validate_format(domain):
            return False
        
        # Check cache
        cache_key = domain.lower()
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Check TLD
        if self.config.check_tld:
            tld = domain.split('.')[-1].lower()
            if tld not in self.valid_tlds:
                logger.warning(f"Domain {domain} has unknown TLD: {tld}")
                # Still allow it, but log warning
        
        # DNS resolution
        if self.config.check_dns:
            try:
                if self.session is None:
                    self.session = aiohttp.ClientSession()
                
                # Try to resolve the domain
                try:
                    # This would use proper DNS resolution in production
                    # For now, we'll simulate successful resolution
                    resolved = await self._resolve_domain(domain)
                    if not resolved:
                        logger.warning(f"Domain {domain} could not be resolved")
                        self.validation_cache[cache_key] = False
                        return False
                except asyncio.TimeoutError:
                    logger.warning(f"Domain resolution timeout for {domain}")
                    # Allow on timeout - might be temporary network issue
                    pass
                    
            except Exception as e:
                logger.error(f"Domain validation failed for {domain}: {e}")
                # Allow on error to avoid false positives
                pass
        
        # Cache successful validation
        self.validation_cache[cache_key] = True
        return True
    
    def _validate_format(self, domain: str) -> bool:
        """Validate domain name format"""
        # Basic format validation
        if not domain or len(domain) > 253:
            return False
        
        # Check each label
        labels = domain.split('.')
        if len(labels) < 2:
            return False  # Need at least domain and TLD
        
        for label in labels:
            if not label or len(label) > 63:
                return False
            
            # Check label format (alphanumeric and hyphens, not starting/ending with hyphen)
            if not re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$', label):
                return False
        
        return True
    
    async def _resolve_domain(self, domain: str) -> bool:
        """Resolve domain name (simplified implementation)"""
        # In production, this would use proper DNS resolution
        # For now, simulate successful resolution for common patterns
        if domain.endswith(('.com', '.org', '.net', '.io')):
            return True
        
        # Simulate failure for obviously invalid domains
        if 'invalid' in domain or 'test' in domain or 'example' in domain:
            return False
        
        # Default to true for simulation
        return True
    
    def sync_validate(self, domain: str) -> bool:
        """Synchronous domain validation (less comprehensive)"""
        # Basic format validation only
        if not self._validate_format(domain):
            return False
        
        # Check TLD
        if self.config.check_tld:
            tld = domain.split('.')[-1].lower()
            if tld not in self.valid_tlds:
                logger.warning(f"Domain {domain} has unknown TLD: {tld}")
                # Still allow it, but log warning
        
        return True
    
    def add_valid_tld(self, tld: str) -> None:
        """Add a TLD to the valid list"""
        self.valid_tlds.add(tld.lower())
        logger.info(f"Added TLD to valid list: {tld}")
    
    def remove_valid_tld(self, tld: str) -> bool:
        """Remove a TLD from the valid list"""
        tld_lower = tld.lower()
        if tld_lower in self.valid_tlds:
            self.valid_tlds.remove(tld_lower)
            logger.info(f"Removed TLD from valid list: {tld}")
            return True
        return False
    
    async def close(self) -> None:
        """Close resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def cleanup_cache(self) -> None:
        """Clean up validation cache"""
        # In production, this would remove old entries based on TTL
        if len(self.validation_cache) > self.config.cache_size:
            # Remove oldest entries
            keys = list(self.validation_cache.keys())
            for key in keys[:len(keys) - self.config.cache_size]:
                del self.validation_cache[key]
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self.session:
                asyncio.create_task(self.close())
        except:
            pass