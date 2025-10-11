# cli/command_handler.py - Command parsing and execution for RPC client

import json
from typing import Dict, List, Any

class CommandHandler:
    """Handles CLI command parsing and execution via RPC"""
    
    def __init__(self, rpc_client):
        self.client = rpc_client
        self.commands = self._setup_commands()
    
    def _setup_commands(self) -> Dict[str, Dict]:
        """Setup available commands with metadata"""
        return {
            'help': {
                'function': self.cmd_help,
                'description': 'Show help information',
                'usage': 'help [command]'
            },
            'info': {
                'function': self.cmd_info,
                'description': 'Show node information',
                'usage': 'info'
            },
            'status': {
                'function': self.cmd_status,
                'description': 'Show node status',
                'usage': 'status'
            },
            'balance': {
                'function': self.cmd_balance,
                'description': 'Show wallet balance',
                'usage': 'balance [address]'
            },
            'send': {
                'function': self.cmd_send,
                'description': 'Send coins to address',
                'usage': 'send <address> <amount> [fee]'
            },
            'address': {
                'function': self.cmd_address,
                'description': 'Generate new address',
                'usage': 'address'
            },
            'peers': {
                'function': self.cmd_peers,
                'description': 'Show connected peers',
                'usage': 'peers'
            },
            'block': {
                'function': self.cmd_block,
                'description': 'Show block information',
                'usage': 'block <hash_or_height>'
            },
            'transaction': {
                'function': self.cmd_transaction,
                'description': 'Show transaction information',
                'usage': 'transaction <hash>'
            },
            'mempool': {
                'function': self.cmd_mempool,
                'description': 'Show mempool information',
                'usage': 'mempool'
            },
            'stake': {
                'function': self.cmd_stake,
                'description': 'Show staking information',
                'usage': 'stake'
            }
        }
    
    def execute_command(self, command: str, args: List[str] = None) -> str:
        """Execute a command and return the result"""
        if args is None:
            args = []
            
        try:
            if command in self.commands:
                return self.commands[command]['function'](args)
            else:
                return f"Unknown command: {command}. Type 'help' for available commands."
        except Exception as e:
            return f"Error executing command: {e}"
    
    def cmd_help(self, args: List[str]) -> str:
        """Help command implementation"""
        if args:
            cmd_name = args[0].lower()
            if cmd_name in self.commands:
                cmd_info = self.commands[cmd_name]
                return f"{cmd_name}: {cmd_info['description']}\nUsage: {cmd_info['usage']}"
            else:
                return f"Unknown command: {cmd_name}"
        else:
            help_text = "Available commands:\n"
            for cmd_name, cmd_info in sorted(self.commands.items()):
                help_text += f"  {cmd_name:<12} - {cmd_info['description']}\n"
            help_text += "\nType 'help <command>' for detailed help."
            return help_text
    
    def cmd_info(self, args: List[str]) -> str:
        """Info command implementation"""
        try:
            info = self.client.get_info()
            return f"""Node Information:
Version: {info.get('version', 'Unknown')}
Protocol: {info.get('protocolversion', 'Unknown')}
Blocks: {info.get('blocks', 0)}
Connections: {info.get('connections', 0)}
Difficulty: {info.get('difficulty', 0)}
Network: {'Testnet' if info.get('testnet') else 'Mainnet'}
Balance: {info.get('balance', 0)} RAY
"""
        except Exception as e:
            return f"Error getting node info: {e}"
    
    def cmd_status(self, args: List[str]) -> str:
        """Status command implementation"""
        try:
            status = self.client.get_node_status()
            sync_state = status['sync_state']
            node_state = status['node_state']
            
            status_text = f"RAYONIX Node Status\n"
            status_text += f"==================\n"
            status_text += f"Block Height: {sync_state['current_block']}\n"
            status_text += f"Sync Progress: {sync_state['sync_progress']:.2f}%\n"
            status_text += f"Peers: {sync_state['peers_connected']}\n"
            status_text += f"Uptime: {node_state['uptime']:.0f} seconds\n"
            status_text += f"Memory Usage: {node_state['memory_usage']} MB\n"
            
            # Get wallet balance
            try:
                balance = self.client.get_balance()
                status_text += f"Wallet Balance: {balance} RAY\n"
            except:
                status_text += "Wallet Balance: Unavailable\n"
            
            return status_text
        except Exception as e:
            return f"Error getting status: {e}"
    
    def cmd_balance(self, args: List[str]) -> str:
        """Balance command implementation"""
        try:
            address = args[0] if args else None
            balance = self.client.get_balance(address)
            if address:
                return f"Balance for {address}: {balance} RAY"
            else:
                return f"Wallet Balance: {balance} RAY"
        except Exception as e:
            return f"Error getting balance: {e}"
    
    def cmd_send(self, args: List[str]) -> str:
        """Send command implementation"""
        if len(args) < 2:
            return "Usage: send <address> <amount> [fee]"
        
        try:
            address = args[0]
            amount = float(args[1])
            fee = float(args[2]) if len(args) > 2 else 0.0
            
            tx_hash = self.client.send_transaction(address, amount, fee)
            return f"Transaction sent successfully\nTX Hash: {tx_hash}"
        except Exception as e:
            return f"Error sending transaction: {e}"
    
    def cmd_address(self, args: List[str]) -> str:
        """Address command implementation"""
        try:
            address = self.client.get_new_address()
            return f"New Address: {address}"
        except Exception as e:
            return f"Error generating address: {e}"
    
    def cmd_peers(self, args: List[str]) -> str:
        """Peers command implementation"""
        try:
            peers = self.client.get_peers()
            if not peers:
                return "No peers connected"
            
            peers_text = "Connected Peers:\n"
            for i, peer in enumerate(peers):
                address = peer.get('address', 'Unknown')
                peer_id = peer.get('id', 'Unknown')[:16] + '...' if len(peer.get('id', '')) > 16 else peer.get('id', 'Unknown')
                peers_text += f"{i+1}. {address} (ID: {peer_id})\n"
            
            return peers_text
        except Exception as e:
            return f"Error getting peers: {e}"
    
    def cmd_block(self, args: List[str]) -> str:
        """Block command implementation"""
        if not args:
            return "Usage: block <hash_or_height>"
        
        try:
            block = self.client.get_block(args[0])
            return json.dumps(block, indent=2)
        except Exception as e:
            return f"Error getting block: {e}"
    
    def cmd_transaction(self, args: List[str]) -> str:
        """Transaction command implementation"""
        if not args:
            return "Usage: transaction <hash>"
        
        try:
            transaction = self.client.get_transaction(args[0])
            return json.dumps(transaction, indent=2)
        except Exception as e:
            return f"Error getting transaction: {e}"
    
    def cmd_mempool(self, args: List[str]) -> str:
        """Mempool command implementation"""
        try:
            status = self.client.get_blockchain_status()
            return f"Mempool Size: {status['mempool_size']} transactions"
        except Exception as e:
            return f"Error getting mempool info: {e}"
    
    def cmd_stake(self, args: List[str]) -> str:
        """Stake command implementation"""
        try:
            # This would require a staking info endpoint in the API
            return "Staking information requires API endpoint implementation"
        except Exception as e:
            return f"Error getting staking info: {e}"