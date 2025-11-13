import time
import asyncio
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("RateLimiter")

@dataclass
class RateLimitData:
    """Data structure for tracking rate limits per connection"""
    incoming_message_count: int = 0
    outgoing_message_count: int = 0
    incoming_bytes: int = 0
    outgoing_bytes: int = 0
    connection_attempts: int = 0
    last_reset: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    limited_until: Optional[float] = None

class RateLimiter:
    def __init__(self, config_manager):
        """
        Initialize with ConfigManager instance
        All configuration comes from config_manager, no hardcoded defaults
        """
        if not config_manager or not hasattr(config_manager, 'config'):
            raise ValueError("ConfigManager is required for RateLimiter")
        
        self.config_manager = config_manager
        self.rate_limits: Dict[str, RateLimitData] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Initialize stats
        self.total_checked = 0
        self.total_limited = 0
    
    def _get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value from ConfigManager"""
        return self.config_manager.get(key, default)
    
    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled"""
        return self._get_config_value('network.rate_limiting', True)
    
    @property
    def messages_per_minute(self) -> int:
        """Get messages per minute limit"""
        return self._get_config_value('network.rate_limit_per_peer', 1000)
    
    @property
    def bandwidth_per_minute(self) -> int:
        """Get bandwidth per minute limit"""
        return self._get_config_value('network.bandwidth_limit_per_peer', 1024 * 1024)  # 1MB
    
    @property
    def connection_attempts_per_minute(self) -> int:
        """Get connection attempts per minute limit"""
        return self._get_config_value('peer_discovery.connection_attempts_per_minute', 10)
    
    @property
    def burst_capacity(self) -> int:
        """Get burst capacity"""
        return self._get_config_value('message_handler.burst_capacity', 100)
    
    async def start(self):
        """Start the rate limiter background tasks"""
        if self.enabled:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Rate limiter started with config from ConfigManager")
    
    async def stop(self):
        """Stop the rate limiter"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Rate limiter stopped")
    
    async def check_rate_limit(self, connection_id: str, message_size: int = 0) -> bool:
        """
        Check incoming rate limit for a connection
        Returns True if allowed, False if rate limited
        """
        return await self._check_limit(connection_id, message_size, 'incoming')
    
    async def check_outgoing_rate_limit(self, connection_id: str, message_size: int = 0) -> bool:
        """
        Check outgoing rate limit for a connection
        Returns True if allowed, False if rate limited
        """
        return await self._check_limit(connection_id, message_size, 'outgoing')
    
    async def check_connection_rate_limit(self, address: str) -> bool:
        """
        Check connection attempt rate limit for an IP address
        Returns True if allowed, False if rate limited
        """
        if not self.enabled:
            return True
        
        async with self._lock:
            self.total_checked += 1
            
            # Check if currently limited
            if address in self.rate_limits:
                limit_data = self.rate_limits[address]
                if limit_data.limited_until and time.time() < limit_data.limited_until:
                    self.total_limited += 1
                    return False
            
            # Reset counters if needed
            limit_data = self._get_or_create_limit_data(address)
            self._reset_if_needed(limit_data)
            
            # Check connection attempt limit
            if limit_data.connection_attempts >= self.connection_attempts_per_minute:
                # Apply temporary limit
                limit_data.limited_until = time.time() + 60  # 1 minute limit
                self.total_limited += 1
                logger.warning(f"Connection rate limit exceeded for {address}")
                return False
            
            # Update counters
            limit_data.connection_attempts += 1
            limit_data.last_activity = time.time()
            
            return True
    
    async def _check_limit(self, connection_id: str, message_size: int, direction: str) -> bool:
        """Internal method to check rate limits"""
        if not self.enabled:
            return True
        
        async with self._lock:
            self.total_checked += 1
            
            # Check if currently limited
            if connection_id in self.rate_limits:
                limit_data = self.rate_limits[connection_id]
                if limit_data.limited_until and time.time() < limit_data.limited_until:
                    self.total_limited += 1
                    return False
            
            # Reset counters if needed
            limit_data = self._get_or_create_limit_data(connection_id)
            self._reset_if_needed(limit_data)
            
            # Check message count limit with burst capacity
            max_messages = self.messages_per_minute + self.burst_capacity
            
            if direction == 'incoming':
                current_count = limit_data.incoming_message_count
                if current_count >= max_messages:
                    return self._apply_limit(connection_id, limit_data, "incoming message count")
                limit_data.incoming_message_count += 1
            else:  # outgoing
                current_count = limit_data.outgoing_message_count
                if current_count >= max_messages:
                    return self._apply_limit(connection_id, limit_data, "outgoing message count")
                limit_data.outgoing_message_count += 1
            
            # Check bandwidth limit
            max_bandwidth = self.bandwidth_per_minute
            
            if direction == 'incoming':
                current_bandwidth = limit_data.incoming_bytes
                if current_bandwidth + message_size > max_bandwidth:
                    return self._apply_limit(connection_id, limit_data, "incoming bandwidth")
                limit_data.incoming_bytes += message_size
            else:  # outgoing
                current_bandwidth = limit_data.outgoing_bytes
                if current_bandwidth + message_size > max_bandwidth:
                    return self._apply_limit(connection_id, limit_data, "outgoing bandwidth")
                limit_data.outgoing_bytes += message_size
            
            limit_data.last_activity = time.time()
            return True
    
    def _get_or_create_limit_data(self, identifier: str) -> RateLimitData:
        """Get or create rate limit data for an identifier"""
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = RateLimitData()
        return self.rate_limits[identifier]
    
    def _reset_if_needed(self, limit_data: RateLimitData):
        """Reset counters if the time window has passed"""
        current_time = time.time()
        if current_time - limit_data.last_reset >= 60:  # 1 minute window
            limit_data.incoming_message_count = 0
            limit_data.outgoing_message_count = 0
            limit_data.incoming_bytes = 0
            limit_data.outgoing_bytes = 0
            limit_data.connection_attempts = 0
            limit_data.last_reset = current_time
            limit_data.limited_until = None
    
    def _apply_limit(self, connection_id: str, limit_data: RateLimitData, limit_type: str) -> bool:
        """Apply rate limit and return False"""
        # Set limited until time (30 seconds penalty)
        limit_data.limited_until = time.time() + 30
        self.total_limited += 1
        
        logger.warning(
            f"Rate limit exceeded for {connection_id}: {limit_type} limit. "
            f"Limited for 30 seconds."
        )
        return False
    
    async def remove_connection(self, connection_id: str):
        """Remove connection from rate limiting"""
        async with self._lock:
            if connection_id in self.rate_limits:
                del self.rate_limits[connection_id]
                logger.debug(f"Removed rate limit tracking for {connection_id}")
    
    async def get_connection_stats(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get rate limit statistics for a connection"""
        if connection_id not in self.rate_limits:
            return None
        
        limit_data = self.rate_limits[connection_id]
        return {
            'incoming_message_count': limit_data.incoming_message_count,
            'outgoing_message_count': limit_data.outgoing_message_count,
            'incoming_bytes': limit_data.incoming_bytes,
            'outgoing_bytes': limit_data.outgoing_bytes,
            'connection_attempts': limit_data.connection_attempts,
            'last_reset': limit_data.last_reset,
            'last_activity': limit_data.last_activity,
            'limited_until': limit_data.limited_until,
            'is_limited': limit_data.limited_until and time.time() < limit_data.limited_until
        }
    
    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        return {
            'total_checked': self.total_checked,
            'total_limited': self.total_limited,
            'current_active_limits': len(self.rate_limits),
            'limit_percentage': (
                (self.total_limited / self.total_checked * 100) 
                if self.total_checked > 0 else 0
            ),
            'enabled': self.enabled,
            'messages_per_minute': self.messages_per_minute,
            'bandwidth_per_minute': self.bandwidth_per_minute,
            'connection_attempts_per_minute': self.connection_attempts_per_minute,
            'burst_capacity': self.burst_capacity
        }
    
    async def reset_connection_limits(self, connection_id: str):
        """Reset rate limits for a specific connection"""
        async with self._lock:
            if connection_id in self.rate_limits:
                self.rate_limits[connection_id] = RateLimitData()
                logger.info(f"Reset rate limits for {connection_id}")
    
    async def _cleanup_loop(self):
        """Background task to cleanup old rate limit entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_old_entries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_old_entries(self):
        """Clean up rate limit entries for inactive connections"""
        async with self._lock:
            current_time = time.time()
            inactive_threshold = self._get_config_value('connection_manager.cleanup_interval', 3600)  # 1 hour default
            
            to_remove = []
            for connection_id, limit_data in self.rate_limits.items():
                if current_time - limit_data.last_activity > inactive_threshold:
                    to_remove.append(connection_id)
            
            for connection_id in to_remove:
                del self.rate_limits[connection_id]
            
            if to_remove:
                logger.debug(f"Cleaned up {len(to_remove)} inactive rate limit entries")
    
    def is_connection_limited(self, connection_id: str) -> bool:
        """Check if a connection is currently rate limited"""
        if connection_id not in self.rate_limits:
            return False
        
        limit_data = self.rate_limits[connection_id]
        return limit_data.limited_until and time.time() < limit_data.limited_until