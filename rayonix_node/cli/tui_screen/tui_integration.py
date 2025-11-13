"""
Integration between TUI and existing RPC system
"""

import asyncio
from typing import Dict, Any, List
from datetime import datetime

class TUIIntegration:
    """Bridge between TUI and existing RPC system"""
    
    def __init__(self, rpc_client):
        self.client = rpc_client
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for dashboard"""
        try:
            # Node status
            node_status = self.client.get_node_status()
            
            # Wallet data
            wallet_info = self.client.get_wallet_info()
            balance_info = self.client.get_wallet_detailed_balance()
            
            # Staking data
            staking_info = self.client.get_staking_info()
            
            # Network data
            network_stats = self.client.get_network_stats()
            
            return {
                'node': node_status,
                'wallet': {
                    **wallet_info,
                    **balance_info
                },
                'staking': staking_info,
                'network': network_stats,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def execute_wallet_action(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute wallet actions from TUI"""
        try:
            if action == "send":
                return await self.send_transaction(**kwargs)
            elif action == "receive":
                return await self.generate_address(**kwargs)
            elif action == "backup":
                return await self.backup_wallet(**kwargs)
            else:
                return {'error': f'Unknown action: {action}'}
        except Exception as e:
            return {'error': str(e)}
    
    async def send_transaction(self, to_address: str, amount: float, fee: float = 0.001) -> Dict[str, Any]:
        """Send transaction from TUI"""
        try:
            result = self.client.send_transaction(to_address, amount, fee)
            return {'success': True, 'tx_hash': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_address(self) -> Dict[str, Any]:
        """Generate new address"""
        try:
            address = self.client.get_new_address()
            return {'success': True, 'address': address}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def backup_wallet(self, file_path: str) -> Dict[str, Any]:
        """Backup wallet"""
        try:
            result = self.client.backup_wallet(file_path)
            return {'success': True, 'backup_file': result.get('backup_file')}
        except Exception as e:
            return {'success': False, 'error': str(e)}