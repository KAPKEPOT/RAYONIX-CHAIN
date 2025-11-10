# rayonix_node/cli/base_commands/node_commands.py

from typing import List, Dict, Any
from rayonix_node.cli.base_commands.base_command import BaseCommand


class NodeCommands(BaseCommand):
    """Node management and information commands"""
    
    def execute_info(self, args: List[str]) -> str:
        """Show detailed node information"""
        try:
            info = self.client.get_detailed_info()
            return f"""🏠 NODE INFORMATION
────────────────────────────────
Version:        {info.get('version', 'Unknown')}
Network:        {info.get('network', 'Unknown').upper()}
Block Height:   {info.get('block_height', 0):,}
Connected Peers: {info.get('peers_connected', 0)}
Sync Status:    {info.get('sync_status', 'Unknown')}
Consensus:      {info.get('consensus', 'Unknown').upper()}
Uptime:         {self._format_uptime(info.get('uptime', 0))}
Memory Usage:   {info.get('memory_usage', 0)} MB
Wallet Loaded:  {info.get('wallet_loaded', False)}
API Enabled:    {info.get('api_enabled', False)}"""
        except Exception as e:
            return self._format_rpc_error(e)
    
    def execute_status(self, args: List[str]) -> str:
        """Show node status"""
        try:
            status = self.client.get_node_status()
            node_info = self.client.get_detailed_info()
            
            status_text = "📊 NODE STATUS\n"
            status_text += "────────────────────────────────\n"
            status_text += f"Status:         {status.get('status', 'Unknown')}\n"
            status_text += f"Block Height:   {node_info.get('block_height', 0):,}\n"
            status_text += f"Network:        {node_info.get('network', 'Unknown').upper()}\n"
            status_text += f"Peers:          {node_info.get('peers_connected', 0)}\n"
            status_text += f"Sync Progress:  {status.get('sync_progress', 0)}%\n"
            status_text += f"Uptime:         {self._format_uptime(node_info.get('uptime', 0))}\n"
            
            # Add wallet balance if available
            try:
                balance = self.client.get_balance()
                status_text += f"Wallet Balance: {balance:,.6f} RYX\n"
            except:
                status_text += "Wallet Balance: Not available\n"
            
            return status_text
        except Exception as e:
            return self._format_rpc_error(e)
    
    def execute_config(self, args: List[str]) -> str:
        """Show configuration information"""
        try:
            config = self.client.get_config()
            if args:
                # Show specific config key
                key = args[0]
                keys = key.split('.')
                value = config
                for k in keys:
                    if isinstance(value, dict):
                        value = value.get(k, {})
                    else:
                        value = "Not found"
                        break
                return f"⚙️  CONFIG: {key}\n────────────────────────────────\nValue: {value}"
            else:
                # Show all config sections
                response = "⚙️  CONFIGURATION SECTIONS\n"
                response += "────────────────────────────────\n"
                for section, settings in config.items():
                    response += f"{section}:\n"
                    for key, value in settings.items():
                        response += f"  {key}: {value}\n"
                    response += "\n"
                return response
        except Exception as e:
            return self._format_rpc_error(e)