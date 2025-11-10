# rayonix_node/cli/base_commands/network_commands.py

from typing import List, Dict, Any
from .lrayonix_node.cli.command_handler import CommandHandler


class NetworkCommands:
    """Network and peer management commands"""
    
    def __init__(self, command_handler: CommandHandler):
        self.handler = command_handler
        self.client = command_handler.client
    
    def execute_peers(self, args: List[str]) -> str:
        """Show connected peers"""
        try:
            peers = self.client.get_peers()
            if not peers:
                return "🌐 No peers connected"
            
            response = f"🌐 CONNECTED PEERS ({len(peers)} total)\n"
            response += "────────────────────────────────\n"
            for i, peer in enumerate(peers):
                address = peer.get('address', 'Unknown')
                peer_id = peer.get('id', 'Unknown')[:16] + '...' if len(peer.get('id', '')) > 16 else peer.get('id', 'Unknown')
                version = peer.get('version', 'Unknown')
                response += f"{i+1}. {address} (v{version})\n"
                response += f"    ID: {peer_id}\n"
            
            return response
        except Exception as e:
            return self.handler._format_rpc_error(e)
    
    def execute_network(self, args: List[str]) -> str:
        """Network statistics"""
        try:
            stats = self.client.get_network_stats()
            return f"🌐 NETWORK STATISTICS\n" \
                   f"────────────────────────────────\n" \
                   f"Connected Peers:    {stats.get('peers_connected', 0)}\n" \
                   f"Messages Sent:      {stats.get('messages_sent', 0):,}\n" \
                   f"Messages Received:  {stats.get('messages_received', 0):,}\n" \
                   f"Bytes Sent:         {self.handler._format_bytes(stats.get('bytes_sent', 0))}\n" \
                   f"Bytes Received:     {self.handler._format_bytes(stats.get('bytes_received', 0))}\n" \
                   f"Uptime:             {self.handler._format_uptime(stats.get('uptime', 0))}"
        except Exception as e:
            return self.handler._format_rpc_error(e)