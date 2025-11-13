import time
import asyncio
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger("RateLimiter")

class RateLimitType(Enum):
    """Types of rate limits"""
    INCOMING_MESSAGES = "incoming_messages"
    OUTGOING_MESSAGES = "outgoing_messages"
    INCOMING_BANDWIDTH = "incoming_bandwidth"
    OUTGOING_BANDWIDTH = "outgoing_bandwidth"
    CONNECTION_RATE = "connection_rate"

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    messages_per_minute: int = 1000
    bandwidth_per_minute: int = 1024 * 1024  # 1MB per minute
    connection_attempts_per_minute: int = 10
    burst_capacity: int = 100  # Allow bursts up to this many messages
    enabled: bool = True

@dataclass
class RateLimitStats:
    """Statistics for rate limiting"""
    total_limited: int = 0
    total_checked: int = 0
    current_active_limits: int = 0

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
    def __init__(self, config: RateLimitConfig = None):
        if config is None:
        	raise ValueError("RateLimitConfig is required")
        self.config = config
        self.rate_limits: Dict[str, RateLimitData] = {}
        self.global_stats = RateLimitStats()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the rate limiter background tasks"""
        if self.config.enabled:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("Rate limiter started")
    
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
        return await self._check_limit(
            connection_id, 
            message_size, 
            RateLimitType.INCOMING_MESSAGES,
            RateLimitType.INCOMING_BANDWIDTH
        )
    
    async def check_outgoing_rate_limit(self, connection_id: str, message_size: int = 0) -> bool:
        """
        Check outgoing rate limit for a connection
        Returns True if allowed, False if rate limited
        """
        return await self._check_limit(
            connection_id,
            message_size,
            RateLimitType.OUTGOING_MESSAGES,
            RateLimitType.OUTGOING_BANDWIDTH
        )
    
    async def check_connection_rate_limit(self, address: str) -> bool:
        """
        Check connection attempt rate limit for an IP address
        Returns True if allowed, False if rate limited
        """
        if not self.config.enabled:
            return True
        
        async with self._lock:
            self.global_stats.total_checked += 1
            
            # Check if currently limited
            if address in self.rate_limits:
                limit_data = self.rate_limits[address]
                if limit_data.limited_until and time.time() < limit_data.limited_until:
                    self.global_stats.total_limited += 1
                    return False
            
            # Reset counters if needed
            limit_data = self._get_or_create_limit_data(address)
            self._reset_if_needed(limit_data)
            
            # Check connection attempt limit
            if limit_data.connection_attempts >= self.config.connection_attempts_per_minute:
                # Apply temporary limit
                limit_data.limited_until = time.time() + 60  # 1 minute limit
                self.global_stats.total_limited += 1
                logger.warning(f"Connection rate limit exceeded for {address}")
                return False
            
            # Update counters
            limit_data.connection_attempts += 1
            limit_data.last_activity = time.time()
            
            return True
    
    async def _check_limit(self, connection_id: str, message_size: int, 
                          message_limit_type: RateLimitType, 
                          bandwidth_limit_type: RateLimitType) -> bool:
        """Internal method to check rate limits"""
        if not self.config.enabled:
            return True
        
        async with self._lock:
            self.global_stats.total_checked += 1
            
            # Check if currently limited
            if connection_id in self.rate_limits:
                limit_data = self.rate_limits[connection_id]
                if limit_data.limited_until and time.time() < limit_data.limited_until:
                    self.global_stats.total_limited += 1
                    return False
            
            # Reset counters if needed
            limit_data = self._get_or_create_limit_data(connection_id)
            self._reset_if_needed(limit_data)
            
            # Check message count limit with burst capacity
            max_messages = self.config.messages_per_minute + self.config.burst_capacity
            
            if message_limit_type == RateLimitType.INCOMING_MESSAGES:
                current_count = limit_data.incoming_message_count
                if current_count >= max_messages:
                    return self._apply_limit(connection_id, limit_data, "message count")
                limit_data.incoming_message_count += 1
            else:  # OUTGOING_MESSAGES
                current_count = limit_data.outgoing_message_count
                if current_count >= max_messages:
                    return self._apply_limit(connection_id, limit_data, "message count")
                limit_data.outgoing_message_count += 1
            
            # Check bandwidth limit
            max_bandwidth = self.config.bandwidth_per_minute
            
            if bandwidth_limit_type == RateLimitType.INCOMING_BANDWIDTH:
                current_bandwidth = limit_data.incoming_bytes
                if current_bandwidth + message_size > max_bandwidth:
                    return self._apply_limit(connection_id, limit_data, "bandwidth")
                limit_data.incoming_bytes += message_size
            else:  # OUTGOING_BANDWIDTH
                current_bandwidth = limit_data.outgoing_bytes
                if current_bandwidth + message_size > max_bandwidth:
                    return self._apply_limit(connection_id, limit_data, "bandwidth")
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
        self.global_stats.total_limited += 1
        
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
            'total_checked': self.global_stats.total_checked,
            'total_limited': self.global_stats.total_limited,
            'current_active_limits': len(self.rate_limits),
            'limit_percentage': (
                (self.global_stats.total_limited / self.global_stats.total_checked * 100) 
                if self.global_stats.total_checked > 0 else 0
            ),
            'enabled': self.config.enabled
        }
    
    async def reset_connection_limits(self, connection_id: str):
        """Reset rate limits for a specific connection"""
        async with self._lock:
            if connection_id in self.rate_limits:
                self.rate_limits[connection_id] = RateLimitData()
                logger.info(f"Reset rate limits for {connection_id}")
    
    async def set_custom_limits(self, connection_id: str, 
                              messages_per_minute: Optional[int] = None,
                              bandwidth_per_minute: Optional[int] = None):
        """Set custom rate limits for a specific connection"""
        # This can be extended to support per-connection custom limits
        logger.debug(f"Custom limits requested for {connection_id}")
        # Implementation would store custom limits in a separate dictionary
    
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
            inactive_threshold = 3600  # 1 hour
            
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