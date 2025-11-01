# core/state_manager.py - Sync state and node state management

import time
from typing import Dict, Any

class NodeStateManager:
    """Manages node state including synchronization status"""
    
    def __init__(self):
        self.sync_state = {
            'syncing': False,
            'current_block': 0,
            'target_block': 0,
            'peers_connected': 0,
            'last_sync_time': 0,
            'sync_speed': 0,
            'sync_progress': 0
        }
        
        self.node_state = {
            'start_time': time.time(),
            'uptime': 0,
            'blocks_processed': 0,
            'transactions_processed': 0,
            'memory_usage': 0,
            'cpu_usage': 0
        }
    
    def update_sync_state(self, **kwargs):
        """Update synchronization state"""
        for key, value in kwargs.items():
            if key in self.sync_state:
                self.sync_state[key] = value
        
        # Calculate sync progress if we have target and current block
        if (self.sync_state['target_block'] > 0 and 
            self.sync_state['current_block'] > 0):
            progress = (self.sync_state['current_block'] / 
                       self.sync_state['target_block']) * 100
            self.sync_state['sync_progress'] = min(100, progress)
        
        # Calculate sync speed
        current_time = time.time()
        if (self.sync_state['last_sync_time'] > 0 and
            self.sync_state['current_block'] > 0):
            time_diff = current_time - self.sync_state['last_sync_time']
            if time_diff > 0:
                blocks_diff = (self.sync_state['current_block'] - 
                              self.sync_state.get('last_block_count', 0))
                self.sync_state['sync_speed'] = blocks_diff / time_diff
        
        self.sync_state['last_sync_time'] = current_time
        self.sync_state['last_block_count'] = self.sync_state['current_block']
    
    def update_node_state(self, **kwargs):
        """Update node state"""
        for key, value in kwargs.items():
            if key in self.node_state:
                self.node_state[key] = value
        
        # Update uptime
        self.node_state['uptime'] = time.time() - self.node_state['start_time']
    
    def get_sync_state(self) -> Dict[str, Any]:
        """Get current synchronization state"""
        return self.sync_state.copy()
    
    def get_node_state(self) -> Dict[str, Any]:
        """Get current node state"""
        return self.node_state.copy()
    
    def is_synced(self) -> bool:
        """Check if node is fully synchronized"""
        return (not self.sync_state['syncing'] and 
                self.sync_state['current_block'] >= self.sync_state['target_block'])
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive state summary"""
        return {
            'sync_state': self.get_sync_state(),
            'node_state': self.get_node_state(),
            'is_synced': self.is_synced(),
            'timestamp': time.time()
        }
        
#class NodeStateManager:
    # ... your existing code ...
    
    def get_uptime(self) -> str:
        """Get node uptime as human-readable string"""
        uptime_seconds = time.time() - self.node_state['start_time']
        return self._format_uptime(uptime_seconds)
    
    def get_memory_usage(self) -> float:
        """Get memory usage in MB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0
    
    def _format_uptime(self, seconds: int) -> str:
        """Format uptime in seconds to human readable format"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s"
        elif seconds < 86400:
            return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
        else:
            return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"        