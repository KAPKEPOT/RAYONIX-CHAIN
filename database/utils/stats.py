from dataclasses import dataclass, field
from typing import Dict, Any
import time

@dataclass
class DatabaseStats:
    put_operations: int = 0
    get_operations: int = 0
    delete_operations: int = 0
    batch_operations: int = 0
    iterate_operations: int = 0
    index_queries: int = 0
    put_errors: int = 0
    get_errors: int = 0
    delete_errors: int = 0
    batch_errors: int = 0
    iterate_errors: int = 0
    index_errors: int = 0
    cache_hits: int = 0
    misses: int = 0
    bytes_written: int = 0
    bytes_read: int = 0
    start_time: float = field(default_factory=time.time)
    
    def get_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary"""
        uptime = time.time() - self.start_time
        return {
            'put_operations': self.put_operations,
            'get_operations': self.get_operations,
            'delete_operations': self.delete_operations,
            'batch_operations': self.batch_operations,
            'iterate_operations': self.iterate_operations,
            'index_queries': self.index_queries,
            'put_errors': self.put_errors,
            'get_errors': self.get_errors,
            'delete_errors': self.delete_errors,
            'batch_errors': self.batch_errors,
            'iterate_errors': self.iterate_errors,
            'index_errors': self.index_errors,
            'cache_hits': self.cache_hits,
            'misses': self.misses,
            'bytes_written': self.bytes_written,
            'bytes_read': self.bytes_read,
            'uptime_seconds': uptime,
            'operations_per_second': (self.put_operations + self.get_operations + 
                                    self.delete_operations) / uptime if uptime > 0 else 0
        }
    
    def reset(self):
        """Reset all statistics"""
        self.__init__()