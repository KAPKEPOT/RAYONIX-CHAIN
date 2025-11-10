# rayonix_node/cli/base_commands/base_command.py

from typing import List, Dict, Any
from datetime import datetime


class BaseCommand:
    """Base class for all commands to avoid circular imports"""
    
    def __init__(self, rpc_client):
        self.client = rpc_client
    
    def execute(self, args: List[str]) -> str:
        """Execute the command - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute method")
    
    def _format_rpc_error(self, error: Exception) -> str:
        """Format RPC errors consistently"""
        error_msg = str(error)
        if "404" in error_msg or "endpoint" in error_msg:
            return "❌ Command not supported by node. Ensure node supports this operation."
        elif "401" in error_msg or "authentication" in error_msg:
            return "❌ Authentication failed. Check your API key."
        elif "connection" in error_msg.lower():
            return "❌ Cannot connect to node. Ensure rayonixd is running."
        else:
            return f"❌ Error: {error_msg}"
    
    def _format_uptime(self, uptime_value) -> str:
        """Format uptime in seconds to human readable format"""
        if isinstance(uptime_value, str) and any(x in uptime_value for x in ['s', 'm', 'h', 'd']):
            return uptime_value
        
        try:
            seconds = int(uptime_value)
            if seconds < 60:
                return f"{seconds}s"
            elif seconds < 3600:
                return f"{seconds // 60}m {seconds % 60}s"
            elif seconds < 86400:
                return f"{seconds // 3600}h {(seconds % 3600) // 60}m"
            else:
                return f"{seconds // 86400}d {(seconds % 86400) // 3600}h"
        except (ValueError, TypeError):
            return str(uptime_value)
    
    def _format_bytes(self, bytes_count: int) -> str:
        """Format bytes to human readable format"""
        if bytes_count < 1024:
            return f"{bytes_count} B"
        elif bytes_count < 1024 ** 2:
            return f"{bytes_count / 1024:.1f} KB"
        elif bytes_count < 1024 ** 3:
            return f"{bytes_count / (1024 ** 2):.1f} MB"
        else:
            return f"{bytes_count / (1024 ** 3):.1f} GB"
    
    def _format_timestamp(self, timestamp: int) -> str:
        """Format timestamp to human readable date"""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return "Unknown"