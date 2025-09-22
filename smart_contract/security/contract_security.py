# smart_contract/security/contract_security.py
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque

from ..security.behavioral_analyzer import BehavioralAnalyzer
from ..security.threat_intelligence import ThreatIntelligenceFeed
from ..security.validators.input_validator import InputValidator
from ..security.validators.domain_validator import DomainValidator
from ..security.validators.ip_validator import IPValidator
from ....exceptions.security_errors import (
    SecurityViolationError, RateLimitExceededError, BlacklistedAddressError, InputValidationError
)

logger = logging.getLogger("SmartContract.Security")

@dataclass
class SecurityConfig:
    """Configuration for contract security system"""
    max_execution_time: int = 30  # seconds
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    max_storage_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    max_gas_per_call: int = 10 * 1000 * 1000  # 10 million gas
    
    # Rate limiting
    rate_limit_window: int = 60  # seconds
    max_calls_per_window: int = 1000
    max_storage_ops_per_window: int = 500
    
    # Threat intelligence
    threat_intelligence_update_interval: int = 300  # 5 minutes
    max_blacklisted_addresses: int = 10000
    
    # Behavioral analysis
    behavioral_analysis_enabled: bool = True
    anomaly_detection_threshold: float = 3.0  # Standard deviations

class ContractSecurity:
    """Comprehensive security system for smart contract execution"""
    
    def __init__(self, config: Optional[SecurityConfig] = None):
        self.config = config or SecurityConfig()
        
        # Core security components
        self.threat_intelligence = ThreatIntelligenceFeed()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.input_validator = InputValidator()
        self.domain_validator = DomainValidator()
        self.ip_validator = IPValidator()
        
        # Security state
        self.blacklisted_addresses: Set[str] = set()
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.security_events: List[Dict] = []
        self.threat_level: int = 0  # 0-100 scale
        
        # Start background tasks
        self.running = True
        self._start_background_tasks()
        
        logger.info("ContractSecurity system initialized")
    
    def _start_background_tasks(self) -> None:
        """Start background security monitoring tasks"""
        # Start threat intelligence updates
        self.threat_update_task = asyncio.create_task(self._update_threat_intelligence())
        
        # Start security cleanup
        self.cleanup_task = asyncio.create_task(self._periodic_cleanup())
    
    async def _update_threat_intelligence(self) -> None:
        """Periodically update threat intelligence data"""
        while self.running:
            try:
                await self.threat_intelligence.update()
                self.blacklisted_addresses = self.threat_intelligence.get_blacklisted_addresses()
                self.threat_level = self.threat_intelligence.get_current_threat_level()
                
                logger.debug(f"Threat intelligence updated. Threat level: {self.threat_level}")
                await asyncio.sleep(self.config.threat_intelligence_update_interval)
                
            except Exception as e:
                logger.error(f"Threat intelligence update failed: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of security data"""
        while self.running:
            try:
                self._cleanup_old_rate_limits()
                self._cleanup_old_security_events()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
            except Exception as e:
                logger.error(f"Security cleanup failed: {e}")
                await asyncio.sleep(60)
    
    def _cleanup_old_rate_limits(self) -> None:
        """Clean up old rate limit entries"""
        current_time = time.time()
        for key in list(self.rate_limits.keys()):
            # Remove entries older than 2x window size
            self.rate_limits[key] = deque(
                [t for t in self.rate_limits[key] if current_time - t <= self.config.rate_limit_window * 2],
                maxlen=1000
            )
    
    def _cleanup_old_security_events(self) -> None:
        """Clean up old security events"""
        current_time = time.time()
        # Keep only events from last 24 hours
        self.security_events = [
            event for event in self.security_events 
            if current_time - event['timestamp'] <= 86400
        ]
        # Limit to 10000 events
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
    
    def check_rate_limit(self, address: str, operation: str, cost: int = 1) -> bool:
        """Check if an operation is rate limited"""
        key = f"{address}_{operation}"
        current_time = time.time()
        
        # Remove old entries
        window_start = current_time - self.config.rate_limit_window
        while self.rate_limits[key] and self.rate_limits[key][0] < window_start:
            self.rate_limits[key].popleft()
        
        # Check if over limit
        if len(self.rate_limits[key]) + cost > self.config.max_calls_per_window:
            self._log_security_event(
                "rate_limit_exceeded",
                f"Rate limit exceeded for {address} operation {operation}",
                {'address': address, 'operation': operation, 'current_count': len(self.rate_limits[key]), 'limit': self.config.max_calls_per_window}
            )
            return False
        
        # Add new entry
        for _ in range(cost):
            self.rate_limits[key].append(current_time)
        
        return True
    
    def is_blacklisted(self, address: str) -> bool:
        """Check if an address is blacklisted"""
        return address in self.blacklisted_addresses
    
    def validate_input(self, input_data: Any, context: Optional[Dict] = None) -> tuple[bool, Optional[str]]:
        """Validate input data with context-aware validation"""
        try:
            return self.input_validator.validate(input_data, context or {})
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, str(e)
    
    def check_resource_limits(self, execution_time: float, memory_used: int, 
                            gas_used: int, storage_size: int) -> tuple[bool, Optional[str]]:
        """Check if resource usage is within limits"""
        violations = []
        
        if execution_time > self.config.max_execution_time:
            violations.append(f"Execution time exceeded: {execution_time}s > {self.config.max_execution_time}s")
        
        if memory_used > self.config.max_memory_usage:
            violations.append(f"Memory usage exceeded: {memory_used} bytes > {self.config.max_memory_usage} bytes")
        
        if gas_used > self.config.max_gas_per_call:
            violations.append(f"Gas usage exceeded: {gas_used} > {self.config.max_gas_per_call}")
        
        if storage_size > self.config.max_storage_size:
            violations.append(f"Storage size exceeded: {storage_size} bytes > {self.config.max_storage_size} bytes")
        
        if violations:
            error_msg = "; ".join(violations)
            self._log_security_event(
                "resource_limit_exceeded",
                error_msg,
                {
                    'execution_time': execution_time,
                    'memory_used': memory_used,
                    'gas_used': gas_used,
                    'storage_size': storage_size
                }
            )
            return False, error_msg
        
        return True, None
    
    def analyze_behavior(self, contract_id: str, operation: str, metrics: Dict) -> bool:
        """Analyze behavior for anomalies"""
        if not self.config.behavioral_analysis_enabled:
            return True
        
        try:
            is_normal, confidence, anomalies = self.behavioral_analyzer.analyze(
                contract_id, operation, metrics
            )
            
            if not is_normal:
                self._log_security_event(
                    "behavioral_anomaly",
                    f"Behavioral anomaly detected in {contract_id} operation {operation}",
                    {
                        'contract_id': contract_id,
                        'operation': operation,
                        'confidence': confidence,
                        'anomalies': anomalies,
                        'metrics': metrics
                    }
                )
            
            return is_normal
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return True  # Allow execution on analysis failure
    
    def validate_domain(self, domain: str) -> bool:
        """Validate a domain name"""
        return self.domain_validator.validate(domain)
    
    def validate_ip(self, ip_address: str) -> bool:
        """Validate an IP address"""
        return self.ip_validator.validate(ip_address)
    
    def _log_security_event(self, event_type: str, message: str, details: Dict) -> None:
        """Log a security event"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'message': message,
            'details': details,
            'threat_level': self.threat_level
        }
        self.security_events.append(event)
        logger.warning(f"Security event: {event_type} - {message}")
    
    def get_security_events(self, since: Optional[float] = None, 
                          event_type: Optional[str] = None) -> List[Dict]:
        """Get security events with optional filtering"""
        events = self.security_events
        
        if since is not None:
            events = [e for e in events if e['timestamp'] >= since]
        
        if event_type is not None:
            events = [e for e in events if e['type'] == event_type]
        
        return events
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security system statistics"""
        return {
            'blacklisted_addresses': len(self.blacklisted_addresses),
            'security_events_count': len(self.security_events),
            'threat_level': self.threat_level,
            'rate_limit_keys': len(self.rate_limits),
            'behavioral_analysis_enabled': self.config.behavioral_analysis_enabled,
            'max_execution_time': self.config.max_execution_time,
            'max_memory_usage': self.config.max_memory_usage,
            'max_gas_per_call': self.config.max_gas_per_call,
            'rate_limit_window': self.config.rate_limit_window,
            'max_calls_per_window': self.config.max_calls_per_window
        }
    
    def stop(self) -> None:
        """Stop the security system"""
        self.running = False
        if hasattr(self, 'threat_update_task'):
            self.threat_update_task.cancel()
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
        logger.info("ContractSecurity system stopped")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.stop()
        except:
            pass