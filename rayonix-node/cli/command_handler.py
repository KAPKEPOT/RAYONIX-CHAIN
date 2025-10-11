# cli/command_handler.py - Command parsing and execution

import cmd
import shlex
import asyncio
from typing import Dict, List, Callable, Optional

class CommandHandler:
    """Handles CLI command parsing and execution"""
    
    def __init__(self, node):
        self.node = node
        self.commands = self._setup_commands()
    
    def _setup_commands(self) -> Dict[str, Dict]:
        """Setup available commands with metadata"""
        return {
            'help': {
                'function': self.cmd_help,
                'description': 'Show help information',
                'usage': 'help [command]'
            },
            'status': {
                'function': self.cmd_status,
                'description': 'Show node status',
                'usage': 'status'
            },
            'balance': {
                'function': self.cmd_balance,
                'description': 'Show wallet balance',
                'usage': 'balance'
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
            'connect': {
                'function': self.cmd_connect,
                'description': 'Connect to peer',
                'usage': 'connect <ip>:<port>'
            },
            'disconnect': {
                'function': self.cmd_disconnect,
                'description': 'Disconnect from peer',
                'usage': 'disconnect <peer_id>'
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
            },
            'stop': {
                'function': self.cmd_stop,
                'description': 'Stop the node',
                'usage': 'stop'
            },
            'restart': {
                'function': self.cmd_restart,
                'description': 'Restart the node',
                'usage': 'restart'
            },
            'config': {
                'function': self.cmd_config,
                'description': 'Show or modify configuration',
                'usage': 'config [get|set] <key> [value]'
            }
        }
    
    async def execute_command(self, command_line: str) -> str:
        """Execute a command and return the result"""
        try:
            parts = shlex.split(command_line)
            if not parts:
                return ""
            
            command = parts[0].lower()
            args = parts[1:]
            
            if command in self.commands:
                return await self.commands[command]['function'](args)
            else:
                return f"Unknown command: {command}. Type 'help' for available commands."
        except Exception as e:
            return f"Error executing command: {e}"
    
    async def cmd_help(self, args: List[str]) -> str:
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
    
    async def cmd_status(self, args: List[str]) -> str:
        """Status command implementation"""
        try:
            state = self.node.state_manager.get_state_summary()
            sync_state = state['sync_state']
            node_state = state['node_state']
            
            status_text = f"RAYONIX Node Status\n"
            status_text += f"==================\n"
            status_text += f"Network: {self.node.config_manager.get('network.network_type')}\n"
            status_text += f"Block Height: {sync_state['current_block']}\n"
            status_text += f"Sync Progress: {sync_state['sync_progress']:.2f}%\n"
            status_text += f"Peers: {sync_state['peers_connected']}\n"
            status_text += f"Uptime: {node_state['uptime']:.0f} seconds\n"
            status_text += f"Memory Usage: {node_state['memory_usage']} MB\n"
            
            if self.node.wallet:
                balance = self.node.wallet.get_balance()
                status_text += f"Wallet Balance: {balance} RAY\n"
            
            return status_text
        except Exception as e:
            return f"Error getting status: {e}"
    
    async def cmd_balance(self, args: List[str]) -> str:
        """Balance command implementation"""
        if not self.node.wallet:
            return "Wallet not available"
        
        try:
            balance = self.node.wallet.get_balance()
            return f"Wallet Balance: {balance} RAY"
        except Exception as e:
            return f"Error getting balance: {e}"
    
    async def cmd_send(self, args: List[str]) -> str:
        """Send command implementation"""
        if not self.node.wallet:
            return "Wallet not available"
        
        if len(args) < 2:
            return "Usage: send <address> <amount> [fee]"
        
        try:
            address = args[0]
            amount = float(args[1])
            fee = float(args[2]) if len(args) > 2 else 0.0
            
            tx_hash = self.node.wallet.send(address, amount, fee)
            if tx_hash:
                return f"Transaction sent successfully\nTX Hash: {tx_hash}"
            else:
                return "Failed to send transaction"
        except Exception as e:
            return f"Error sending transaction: {e}"
    
    async def cmd_address(self, args: List[str]) -> str:
        """Address command implementation"""
        if not self.node.wallet:
            return "Wallet not available"
        
        try:
            address = self.node.wallet.get_new_address()
            return f"New Address: {address}"
        except Exception as e:
            return f"Error generating address: {e}"
    
    async def cmd_peers(self, args: List[str]) -> str:
        """Peers command implementation"""
        if not self.node.network:
            return "Network not available"
        
        try:
            peers = await self.node.network.get_peers()
            if not peers:
                return "No peers connected"
            
            peers_text = "Connected Peers:\n"
            for i, peer in enumerate(peers):
                peers_text += f"{i+1}. {peer['address']} (ID: {peer['id']})\n"
            
            return peers_text
        except Exception as e:
            return f"Error getting peers: {e}"
    
    async def cmd_connect(self, args: List[str]) -> str:
        """Connect command implementation"""
        if not self.node.network:
            return "Network not available"
        
        if not args:
            return "Usage: connect <ip>:<port>"
        
        try:
            address = args[0]
            if ':' not in address:
                address += f":{self.node.config_manager.get('network.listen_port')}"
            
            success = await self.node.network.connect_to_peer(address)
            if success:
                return f"Connected to {address}"
            else:
                return f"Failed to connect to {address}"
        except Exception as e:
            return f"Error connecting to peer: {e}"
    
    async def cmd_disconnect(self, args: List[str]) -> str:
        """Disconnect command implementation"""
        if not self.node.network:
            return "Network not available"
        
        if not args:
            return "Usage: disconnect <peer_id>"
        
        try:
            peer_id = args[0]
            success = await self.node.network.disconnect_peer(peer_id)
            if success:
                return f"Disconnected peer {peer_id}"
            else:
                return f"Failed to disconnect peer {peer_id}"
        except Exception as e:
            return f"Error disconnecting peer: {e}"
    
    async def cmd_block(self, args: List[str]) -> str:
        """Block command implementation"""
        if not args:
            return "Usage: block <hash_or_height>"
        
        try:
            block_id = args[0]
            if block_id.isdigit():
                block = self.node.rayonix_coin.get_block_by_height(int(block_id))
            else:
                block = self.node.rayonix_coin.get_block_by_hash(block_id)
            
            if not block:
                return "Block not found"
            
            import json
            return json.dumps(block, indent=2)
        except Exception as e:
            return f"Error getting block: {e}"
    
    async def cmd_transaction(self, args: List[str]) -> str:
        """Transaction command implementation"""
        if not args:
            return "Usage: transaction <hash>"
        
        try:
            tx_hash = args[0]
            transaction = self.node.rayonix_coin.get_transaction(tx_hash)
            
            if not transaction:
                return "Transaction not found"
            
            import json
            return json.dumps(transaction, indent=2)
        except Exception as e:
            return f"Error getting transaction: {e}"
    
    async def cmd_mempool(self, args: List[str]) -> str:
        """Mempool command implementation"""
        try:
            mempool = self.node.rayonix_coin.mempool
            count = len(mempool)
            
            if count == 0:
                return "Mempool is empty"
            
            total_size = sum(len(str(tx)) for tx in mempool.values())
            
            mempool_text = f"Mempool Information:\n"
            mempool_text += f"Transactions: {count}\n"
            mempool_text += f"Total Size: {total_size} bytes\n"
            
            # Show first few transactions
            if count > 0:
                mempool_text += "\nFirst 5 transactions:\n"
                for i, tx_hash in enumerate(list(mempool.keys())[:5]):
                    tx = mempool[tx_hash]
                    mempool_text += f"{i+1}. {tx_hash[:16]}... ({len(str(tx))} bytes)\n"
            
            return mempool_text
        except Exception as e:
            return f"Error getting mempool info: {e}"
    
    async def cmd_stake(self, args: List[str]) -> str:
        """Stake command implementation"""
        if not self.node.wallet:
            return "Wallet not available"
        
        try:
            staking_info = self.node.wallet.get_staking_info()
            if not staking_info:
                return "Staking not available"
            
            stake_text = "Staking Information:\n"
            stake_text += f"Staking Enabled: {staking_info.get('enabled', False)}\n"
            stake_text += f"Staking Balance: {staking_info.get('staking_balance', 0)} RAY\n"
            stake_text += f"Expected Reward: {staking_info.get('expected_reward', 0)} RAY\n"
            stake_text += f"Last Stake Time: {staking_info.get('last_stake_time', 'Never')}\n"
            
            return stake_text
        except Exception as e:
            return f"Error getting staking info: {e}"
    
    async def cmd_stop(self, args: List[str]) -> str:
        """Stop command implementation"""
        try:
            await self.node.stop()
            return "Node stopped"
        except Exception as e:
            return f"Error stopping node: {e}"
    
    async def cmd_restart(self, args: List[str]) -> str:
        """Restart command implementation"""
        try:
            await self.node.stop()
            await asyncio.sleep(2)  # Brief delay before restart
            await self.node.start()
            return "Node restarted"
        except Exception as e:
            return f"Error restarting node: {e}"
    
    async def cmd_config(self, args: List[str]) -> str:
        """Config command implementation"""
        if not args:
            # Show all config
            config = self.node.config_manager.get_all()
            import json
            return json.dumps(config, indent=2)
        
        action = args[0].lower()
        
        if action == 'get' and len(args) > 1:
            # Get config value
            key = args[1]
            value = self.node.config_manager.get(key)
            return f"{key} = {value}"
        
        elif action == 'set' and len(args) > 2:
            # Set config value
            key = args[1]
            value = args[2]
            
            # Try to convert value to appropriate type
            try:
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit():
                    value = float(value)
            except:
                pass
            
            success = self.node.config_manager.set(key, value)
            if success:
                return f"Set {key} = {value}"
            else:
                return f"Failed to set {key}"
        
        else:
            return "Usage: config [get|set] <key> [value]"

def setup_cli_handlers(node) -> CommandHandler:
    """Setup CLI command handlers"""
    return CommandHandler(node)