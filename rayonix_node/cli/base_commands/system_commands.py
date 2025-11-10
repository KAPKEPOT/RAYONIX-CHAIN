# rayonix_node/cli/base_commands/system_commands.py

from typing import List, Dict, Any
from rayonix_node.cli.command_handler import CommandHandler


class SystemCommands:
    """System utility and maintenance commands"""
    
    def __init__(self, command_handler: CommandHandler):
        self.handler = command_handler
        self.client = command_handler.client
    
    def execute_stats(self, args: List[str]) -> str:
        """Show CLI statistics"""
        try:
            metrics = self.client.get_performance_metrics()
            hit_rate = (metrics['cache_hits'] / metrics['requests_made'] * 100) if metrics['requests_made'] > 0 else 0
            
            return f"📈 CLI PERFORMANCE METRICS\n" \
                   f"────────────────────────────────\n" \
                   f"Total Requests:      {metrics['requests_made']:,}\n" \
                   f"Cache Hits:          {metrics['cache_hits']:,}\n" \
                   f"Average Response:    {metrics['average_response_time']:.3f}s\n" \
                   f"Cache Hit Rate:      {hit_rate:.1f}%"
        except:
            return "📈 Performance metrics not available"
    
    def execute_generate_api_key(self, args: List[str]) -> str:
        """Generate a strong API key"""
        from rayonix_node.utils.api_key_manager import APIKeyManager
        
        length = int(args[0]) if args and args[0].isdigit() else 128    
        key = APIKeyManager.generate_strong_api_key(length)
        
        response = "🔐 GENERATED STRONG API KEY\n"
        response += "=" * 60 + "\n"
        response += key + "\n"
        response += "=" * 60 + "\n\n"
        response += "⚠️  SECURITY INSTRUCTIONS:\n"
        response += "────────────────────────────────\n"
        response += "1. Set this key in your node configuration:\n"
        response += "   api.auth_key = \"YOUR_KEY_HERE\"\n\n"
        response += "2. Use with CLI commands:\n"
        response += "   rayonix-cli --api-key \"KEY\" wallet-info\n"
        response += "   OR: export RAYONIX_API_KEY=\"KEY\"\n"
        response += "   rayonix-cli --api-key-env wallet-info\n\n"
        response += "3. Store securely - this key cannot be recovered!\n"
        response += "4. Never commit to version control or share\n"
        
        return response