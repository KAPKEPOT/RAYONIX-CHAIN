import time
from typing import Dict, Any

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, rate_limit_per_peer: int = 1000):
        self.rate_limit_per_peer = rate_limit_per_peer
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
    
    async def check_rate_limit(self, connection_id: str, message_size: int) -> bool:
        """Check if rate limit is exceeded"""
        if connection_id not in self.rate_limits:
            self.rate_limits[connection_id] = {
                'message_count': 0,
                'last_reset': time.time(),
                'bytes_sent': 0,
                'bytes_received': 0
            }
            return True
            
        limit_data = self.rate_limits[connection_id]
        
        # Reset if needed
        current_time = time.time()
        if current_time - limit_data['last_reset'] >= 60:
            limit_data['message_count'] = 0
            limit_data['bytes_sent'] = 0
            limit_data['bytes_received'] = 0
            limit_data['last_reset'] = current_time
        
        # Check message count limit
        if limit_data['message_count'] >= self.rate_limit_per_peer:
            return False
        
        # Check bandwidth limit
        bandwidth_limit = self.rate_limit_per_peer * 1024
        if limit_data['bytes_received'] + message_size > bandwidth_limit:
            return False
        
        # Update counters
        limit_data['message_count'] += 1
        limit_data['bytes_received'] += message_size
        
        return True
    
    def remove_connection(self, connection_id: str):
        """Remove connection from rate limiting"""
        if connection_id in self.rate_limits:
            del self.rate_limits[connection_id]