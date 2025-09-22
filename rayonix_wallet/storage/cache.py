import time
import threading
from typing import Dict, Any, Optional
from collections import OrderedDict

class CacheManager:
    """In-memory caching layer for performance optimization"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry['expires'] > time.time():
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return entry['value']
                else:
                    # Remove expired entry
                    del self._cache[key]
            
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache"""
        with self._lock:
            if key in self._cache:
                # Remove existing entry to update position
                del self._cache[key]
            
            # Remove oldest entry if cache is full
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            expires = time.time() + (ttl or self.default_ttl)
            self._cache[key] = {
                'value': value,
                'expires': expires,
                'created': time.time()
            }
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear entire cache"""
        with self._lock:
            self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            
            return {
                'size': len(self._cache),
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'max_size': self.max_size,
                'default_ttl': self.default_ttl
            }
    
    def cleanup(self) -> int:
        """Remove expired entries and return count removed"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry['expires'] <= current_time
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            return len(expired_keys)
    
    def keys(self) -> list:
        """Get all cache keys"""
        with self._lock:
            return list(self._cache.keys())
    
    def contains(self, key: str) -> bool:
        """Check if key exists in cache (even if expired)"""
        with self._lock:
            return key in self._cache