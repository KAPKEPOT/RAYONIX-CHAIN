"""
Bridge between modern TUI and existing command handler
"""

import asyncio
from typing import Dict, Any, List
from rayonix_node.cli.command_handler import CommandHandler

class TUICommandBridge:
    """Bridge modern TUI actions to existing command handler"""
    
    def __init__(self, command_handler: CommandHandler):
        self.handler = command_handler
        self.client = command_handler.client
    
    async def get_wallet_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive wallet data for TUI dashboard"""
        try:
            wallet_info = self.client.get_wallet_info()
            balance_info = self.client.get_wallet_detailed_balance()
            addresses = self.client.get_wallet_addresses()
            transactions = self.client.get_transaction_history(10)
            
            return {
                'total_balance': balance_info.get('total', 0),
                'available_balance': balance_info.get('available', 0),
                'staked_balance': balance_info.get('staked', 0),
                'pending_balance': balance_info.get('pending', 0),
                'address_count': len(addresses),
                'primary_address': addresses[0] if addresses else None,
                'recent_transactions': transactions[:5],
                'wallet_type': wallet_info.get('type', 'unknown'),
                'encrypted': wallet_info.get('encrypted', False),
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def send_funds(self, to_address: str, amount: float, fee: float = 0.001) -> Dict[str, Any]:
        """Send funds through TUI"""
        try:
            result = self.handler.execute_command('send', [to_address, str(amount), str(fee)])
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def get_staking_data(self) -> Dict[str, Any]:
        """Get staking data for TUI"""
        try:
            staking_info = self.client.get_staking_info()
            validators = self.client.get_validators()
            
            return {
                'enabled': staking_info.get('enabled', False),
                'active': staking_info.get('staking', False),
                'total_staked': staking_info.get('total_staked', 0),
                'rewards': staking_info.get('expected_rewards', 0),
                'validator_status': staking_info.get('validator_status', 'inactive'),
                'validator_count': len(validators),
                'my_validator_rank': self._find_my_validator_rank(validators),
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _find_my_validator_rank(self, validators: List[Dict]) -> int:
        """Find user's validator rank"""
        # Implementation depends on how validators are identified
        return 0