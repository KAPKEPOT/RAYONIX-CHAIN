# rayonix_node/cli/command_handler.py -command handler
import json
import getpass
from typing import Dict, List, Any
from datetime import datetime
from rayonix_node.cli.wallet_commands.create_wallet import CreateWalletCommand
from rayonix_node.cli.wallet_commands.load_wallet import LoadWalletCommand
from rayonix_node.cli.base_commands.api_commands import APICommands
from rayonix_node.cli.base_commands.node_commands import NodeCommands
from rayonix_node.cli.base_commands.blockchain_commands import BlockchainCommands
from rayonix_node.cli.base_commands.network_commands import NetworkCommands
from rayonix_node.cli.base_commands.system_commands import SystemCommands

class CommandHandler:
    """ command handler"""
    
    def __init__(self, rpc_client):
        self.client = rpc_client
        self.commands = self._setup_commands()
        
        # Initialize command modules
        self.create_wallet_cmd = CreateWalletCommand(self.client)
        self.load_wallet_cmd = LoadWalletCommand(self.client)
        self.node_cmd = NodeCommands(self.client)
        self.blockchain_cmd = BlockchainCommands(self.client)
        self.network_cmd = NetworkCommands(self.client)
        self.system_cmd = SystemCommands(self.client)
        self.api_cmd = APICommands(self.client)
    
    def _setup_commands(self) -> Dict[str, Dict]:
        """Setup commands beyond main.py"""
        return {
            'help': {
                'function': self.cmd_help,
                'description': 'Show help information',
                'usage': 'help [command]',
                'category': 'Basic'
            },
            'info': {
                'function': self.cmd_detailed_info,
                'description': 'Show detailed node information',
                'usage': 'info',
                'category': 'Node'
            },
            'status': {
                'function': self.cmd_status,
                'description': 'Show node status',
                'usage': 'status',
                'category': 'Node'
            },
            # Wallet Management Commands
            'create-wallet': {
                'function': self.cmd_create_wallet,
                'description': 'Create a new wallet',
                'usage': 'create-wallet',
                'category': 'Wallet'
            },
            'load-wallet': {
                'function': self.cmd_load_wallet,
                'description': 'Load wallet from mnemonic phrase',
                'usage': 'load-wallet',
                'category': 'Wallet'
            },
            'import-wallet': {
                'function': self.cmd_import_wallet,
                'description': 'Import wallet from backup file',
                'usage': 'import-wallet <filename> [password]',
                'category': 'Wallet'
            },
            'backup-wallet': {
                'function': self.cmd_backup_wallet,
                'description': 'Backup wallet to file',
                'usage': 'backup-wallet [filename]',
                'category': 'Wallet'
            },
            'wallet-info': {
                'function': self.cmd_wallet_info,
                'description': 'Show detailed wallet information',
                'usage': 'wallet-info',
                'category': 'Wallet'
            },
            'list-addresses': {
                'function': self.cmd_list_addresses,
                'description': 'List all wallet addresses',
                'usage': 'list-addresses',
                'category': 'Wallet'
            },
            'balance': {
                'function': self.cmd_detailed_balance,
                'description': 'Show detailed wallet balance',
                'usage': 'balance [address]',
                'category': 'Wallet'
            },
            'send': {
                'function': self.cmd_send,
                'description': 'Send coins to address ',
                'usage': 'send <address> <amount> [fee]',
                'category': 'Wallet'
            },
            'address': {
                'function': self.cmd_address,
                'description': 'Generate new address',
                'usage': 'address',
                'category': 'Wallet'
            },
            # Network Commands
            'peers': {
                'function': self.cmd_peers,
                'description': 'Show connected peers with details',
                'usage': 'peers',
                'category': 'Network'
            },
            'network': {
                'function': self.cmd_network_stats,
                'description': 'Show network statistics',
                'usage': 'network',
                'category': 'Network'
            },
            # Blockchain Commands
            'blockchain-info': {
                'function': self.cmd_blockchain_info,
                'description': 'Show detailed blockchain information',
                'usage': 'blockchain-info',
                'category': 'Blockchain'
            },
            'block': {
                'function': self.cmd_block,
                'description': 'Show block information',
                'usage': 'block <hash_or_height>',
                'category': 'Blockchain'
            },
            'transaction': {
                'function': self.cmd_transaction,
                'description': 'Show transaction information',
                'usage': 'transaction <hash>',
                'category': 'Blockchain'
            },
            'mempool': {
                'function': self.cmd_mempool,
                'description': 'Show mempool information',
                'usage': 'mempool',
                'category': 'Blockchain'
            },
            'sync-status': {
                'function': self.cmd_sync_status,
                'description': 'Show synchronization status',
                'usage': 'sync-status',
                'category': 'Blockchain'
            },
            'history': {
                'function': self.cmd_history,
                'description': 'Show transaction history',
                'usage': 'history [count]',
                'category': 'Blockchain'
            },
            # Advanced Features
            'staking': {
                'function': self.cmd_staking,
                'description': 'Show staking information',
                'usage': 'staking',
                'category': 'Advanced'
            },
            'stake': {
                'function': self.cmd_stake,
                'description': 'Stake tokens for validation',
                'usage': 'stake <amount>',
                'category': 'Advanced'
            },
            'validator-info': {
                'function': self.cmd_validator_info,
                'description': 'Show validator information',
                'usage': 'validator-info',
                'category': 'Advanced'
            },
            'contracts': {
                'function': self.cmd_contracts,
                'description': 'List smart contracts',
                'usage': 'contracts',
                'category': 'Advanced'
            },
            'deploy-contract': {
                'function': self.cmd_deploy_contract,
                'description': 'Deploy smart contract',
                'usage': 'deploy-contract <code>',
                'category': 'Advanced'
            },
            'call-contract': {
                'function': self.cmd_call_contract,
                'description': 'Call contract function',
                'usage': 'call-contract <address> <function> [args...]',
                'category': 'Advanced'
            },
            # System Commands
            'config': {
                'function': self.cmd_config,
                'description': 'Show configuration information',
                'usage': 'config [key]',
                'category': 'System'
            },
            'stats': {
                'function': self.cmd_stats,
                'description': 'Show CLI statistics',
                'usage': 'stats',
                'category': 'System'
            },
            'generate-api-key': {
                'function': self.cmd_generate_api_key,
                'description': 'Generate API KEY',
                'usage': 'generate-api-key',
                'category': 'API System'              
            },
            'validate-api-key': {
                'function': self.cmd_validate_api_key,
                'description': 'Validate API key strength',
                'usage': 'validate-api-key <key>',
                'category': 'API'
            },
            'api-key-info': {
                'function': self.cmd_api_key_info,
                'description': 'Show API key authentication status',
                'usage': 'api-key-info',
                'category': 'API'
            },
            'setup-api-auth': {
                'function': self.cmd_setup_api_auth,
                'description': 'Interactive API authentication setup',
                'usage': 'setup-api-auth',
                'category': 'API'
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

    # Basic Commands
    def cmd_help(self, args: List[str]) -> str:
        """help with categories"""
        if args:
            cmd_name = args[0].lower()
            if cmd_name in self.commands:
                cmd_info = self.commands[cmd_name]
                return f"{cmd_name}: {cmd_info['description']}\nUsage: {cmd_info['usage']}\nCategory: {cmd_info['category']}"
            else:
                return f"Unknown command: {cmd_name}"
        else:
            help_text = "RAYONIX CLI - Available Commands by Category\n"
            help_text += "=" * 50 + "\n\n"
            
            categories = {}
            for cmd_name, cmd_info in self.commands.items():
                category = cmd_info['category']
                if category not in categories:
                    categories[category] = []
                categories[category].append((cmd_name, cmd_info))
            
            for category in sorted(categories.keys()):
                help_text += f"【{category}】\n"
                for cmd_name, cmd_info in sorted(categories[category]):
                    help_text += f"  {cmd_name:<20} - {cmd_info['description']}\n"
                help_text += "\n"
            
            help_text += "Type 'help <command>' for detailed usage information."
            return help_text
    
    def cmd_detailed_info(self, args: List[str]) -> str:
        """Detailed node information"""
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
            return f"❌ Error getting node info: {e}"
    
    def cmd_status(self, args: List[str]) -> str:
        """ node status"""
        return self.node_cmd.execute_status(args)

    # Wallet Management Commands
    def cmd_create_wallet(self, args: List[str]) -> str:
        """Delegate to CreateWalletCommand module"""
        return self.create_wallet_cmd.execute(args)
    
    def cmd_load_wallet(self, args: List[str]) -> str:
        """Delegate to LoadWalletCommand module"""
        return self.load_wallet_cmd.execute(args)
        
    def cmd_import_wallet(self, args: List[str]) -> str:
        """Import wallet from backup file"""
        if not args:
            return "❌ Usage: import-wallet <filename> [password]"
        
        try:
            file_path = args[0]
            password = getpass.getpass("🔐 Enter wallet password: ") if len(args) < 2 else args[1]
            
            result = self.client.import_wallet(file_path, password)
            
            return f"✅ WALLET IMPORTED SUCCESSFULLY\n" \
                   f"────────────────────────────────\n" \
                   f"File:        {file_path}\n" \
                   f"Addresses:   {len(result.get('addresses', []))}\n" \
                   f"Balance:     {result.get('balance', 0):,.6f} RYX"
        except Exception as e:
            return f"❌ Error importing wallet: {e}"
    
    def cmd_backup_wallet(self, args: List[str]) -> str:
        """Backup wallet to file"""
        filename = args[0] if args else f"rayonix_wallet_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dat"
        
        try:
            result = self.client.backup_wallet(filename)
            return f"✅ WALLET BACKUP CREATED\n" \
                   f"────────────────────────────────\n" \
                   f"Backup File: {result.get('backup_file', filename)}\n" \
                   f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                   f"Store this file in a secure location!"
        except Exception as e:
            return f"❌ Error backing up wallet: {e}"
    
    def cmd_wallet_info(self, args: List[str]) -> str:
        """Show detailed wallet information"""
        try:
            wallet_info = self.client.get_wallet_info()
            addresses = self.client.get_wallet_addresses()
            balance_info = self.client.get_wallet_detailed_balance()
            
            response = "💼 WALLET INFORMATION\n"
            response += "────────────────────────────────\n"
            response += f"Type:           {wallet_info.get('type', 'Unknown')}\n"
            response += f"Address Count:  {len(addresses)}\n"
            response += f"Total Balance:  {balance_info.get('total', 0):,.6f} RYX\n"
            response += f"Available:      {balance_info.get('available', 0):,.6f} RYX\n"
            response += f"Pending:        {balance_info.get('pending', 0):,.6f} RYX\n"
            response += f"Staked:         {balance_info.get('staked', 0):,.6f} RYX\n"
            response += f"Encrypted:      {wallet_info.get('encrypted', False)}\n"
            response += f"Backup Created: {wallet_info.get('backup_created', False)}\n"
            
            # Show first few addresses
            if addresses:
                response += f"\n📫 Addresses (showing first 5):\n"
                for i, address in enumerate(addresses[:5]):
                    try:
                        balance = self.client.get_balance(address)
                        response += f"  {i+1}. {address} - {balance:,.6f} RYX\n"
                    except:
                        response += f"  {i+1}. {address} - Balance unavailable\n"
                if len(addresses) > 5:
                    response += f"  ... and {len(addresses) - 5} more addresses\n"
            else:
                response += f"\n📫 No addresses in wallet\n"
            
            return response
        except Exception as e:
            return f"❌ Error getting wallet info: {e}"
    
    def cmd_list_addresses(self, args: List[str]) -> str:
        """List all wallet addresses"""
        try:
            addresses = self.client.get_wallet_addresses()
            if not addresses:
                return "📫 No addresses in wallet"
            
            response = f"📫 WALLET ADDRESSES ({len(addresses)} total)\n"
            response += "────────────────────────────────\n"
            for i, address in enumerate(addresses):
                # Get balance for each address
                try:
                    balance = self.client.get_balance(address)
                    response += f"{i+1:2d}. {address} - {balance:,.6f} RYX\n"
                except:
                    response += f"{i+1:2d}. {address} - Balance unavailable\n"
            
            return response
        except Exception as e:
            return f"❌ Error listing addresses: {e}"
    
    def cmd_detailed_balance(self, args: List[str]) -> str:
        """Detailed balance information"""
        try:
            address = args[0] if args else None
            if address:
                balance = self.client.get_balance(address)
                return f"💰 BALANCE FOR {address}\n" \
                       f"────────────────────────────────\n" \
                       f"Balance: {balance:,.6f} RYX"
            else:
                # Get detailed balance breakdown
                detailed_balance = self.client.get_wallet_detailed_balance()
                return f"💰 WALLET BALANCE DETAILS\n" \
                       f"────────────────────────────────\n" \
                       f"Total:     {detailed_balance.get('total', 0):,.6f} RYX\n" \
                       f"Available: {detailed_balance.get('available', 0):,.6f} RYX\n" \
                       f"Pending:   {detailed_balance.get('pending', 0):,.6f} RYX\n" \
                       f"Staked:    {detailed_balance.get('staked', 0):,.6f} RYX"
        except Exception as e:
            return f"❌ Error getting balance: {e}"
    
    def cmd_send(self, args: List[str]) -> str:
        """ send with fee estimation"""
        if len(args) < 2:
            return "❌ Usage: send <address> <amount> [fee]"
        
        try:
            address = args[0]
            amount = float(args[1])
            fee = float(args[2]) if len(args) > 2 else 0.0
            
            # Validate amount
            if amount <= 0:
                return "❌ Error: Amount must be positive"
            
            # Show confirmation for large amounts
            if amount > 1000:
                response = f"⚠️  LARGE TRANSACTION WARNING\n"
                response += f"────────────────────────────────\n"
                response += f"Amount: {amount:,.6f} RYX\n"
                response += f"To:     {address}\n"
                response += f"Fee:    {fee:,.6f} RYX\n"
                response += f"Total:  {amount + fee:,.6f} RYX\n"
                response += f"\nAre you sure you want to proceed? (yes/NO): "
                print(response)
                confirm = input().strip().lower()
                if confirm not in ['yes', 'y']:
                    return "❌ Transaction cancelled by user"
            
            tx_hash = self.client.send_transaction(address, amount, fee)
            return f"✅ TRANSACTION SENT SUCCESSFULLY\n" \
                   f"────────────────────────────────\n" \
                   f"Amount: {amount:,.6f} RYX\n" \
                   f"To:     {address}\n" \
                   f"Fee:    {fee:,.6f} RYX\n" \
                   f"TX Hash: {tx_hash}"
        except Exception as e:
            return f"❌ Error sending transaction: {e}"
    
    def cmd_address(self, args: List[str]) -> str:
        """Generate new address"""
        try:
            address = self.client.get_new_address()
            return f"📍 NEW ADDRESS GENERATED\n" \
                   f"────────────────────────────────\n" \
                   f"Address: {address}"
        except Exception as e:
            return f"❌ Error generating address: {e}"

    # Network Commands
    def cmd_peers(self, args: List[str]) -> str:
        return self.network_cmd.execute_peers(args)
    
    def cmd_network_stats(self, args: List[str]) -> str:
        return self.network_cmd.execute_network(args)

    # Blockchain Commands
    def cmd_blockchain_info(self, args: List[str]) -> str:
        return self.blockchain_cmd.execute_blockchain_info(args)
    
    def cmd_sync_status(self, args: List[str]) -> str:
        """Show synchronization status"""
        try:
            status = self.client.get_node_status()
            sync_state = status.get('sync_state', {})
            
            response = "⚡ SYNCHRONIZATION STATUS\n"
            response += "────────────────────────────────\n"
            response += f"Syncing:          {sync_state.get('syncing', False)}\n"
            response += f"Current Block:    {sync_state.get('current_block', 0):,}\n"
            response += f"Target Block:     {sync_state.get('target_block', 0):,}\n"
            response += f"Progress:         {sync_state.get('sync_progress', 0)}%\n"
            response += f"Connected Peers:  {sync_state.get('peers_connected', 0)}\n"
            
            if sync_state.get('syncing', False):
                blocks_remaining = sync_state.get('target_block', 0) - sync_state.get('current_block', 0)
                response += f"Blocks Remaining: {blocks_remaining:,}\n"
            
            return response
        except Exception as e:
            return f"❌ Error getting sync status: {e}"
    
    def cmd_block(self, args: List[str]) -> str:
        return self.blockchain_cmd.execute_block(args)
    
    def cmd_transaction(self, args: List[str]) -> str:
        return self.blockchain_cmd.execute_transaction(args)
    
    def cmd_mempool(self, args: List[str]) -> str:
        """Show mempool information"""
        try:
            status = self.client.get_blockchain_status()
            mempool_size = status.get('mempool_size', 0)
            
            if mempool_size == 0:
                return "📋 MEMPOOL IS EMPTY"
            else:
                return f"📋 MEMPOOL INFORMATION\n" \
                       f"────────────────────────────────\n" \
                       f"Transactions: {mempool_size:,}\n" \
                       f"Status:       Active"
        except Exception as e:
            return f"❌ Error getting mempool info: {e}"
    
    def cmd_history(self, args: List[str]) -> str:
        return self.blockchain_cmd.execute_history(args)

    # Features
    def cmd_staking(self, args: List[str]) -> str:
        """Staking information"""
        try:
            staking_info = self.client.get_staking_info()
            return f"🎯 STAKING INFORMATION\n" \
                   f"────────────────────────────────\n" \
                   f"Staking Enabled:   {staking_info.get('enabled', False)}\n" \
                   f"Total Staked:      {staking_info.get('total_staked', 0):,.6f} RYX\n" \
                   f"Validator Status:  {staking_info.get('validator_status', 'Unknown')}\n" \
                   f"Expected Rewards:  {staking_info.get('expected_rewards', 0):,.6f} RYX\n" \
                   f"Staking Power:     {staking_info.get('staking_power', 0)}"
        except Exception as e:
            return f"❌ Error getting staking info: {e}"
    
    def cmd_stake(self, args: List[str]) -> str:
        """Stake tokens for validation"""
        if not args:
            return "❌ Usage: stake <amount>"
        
        try:
            amount = float(args[0])
            if amount <= 0:
                return "❌ Error: Amount must be positive"
            
            result = self.client.stake_tokens(amount)
            return f"✅ TOKENS STAKED SUCCESSFULLY\n" \
                   f"────────────────────────────────\n" \
                   f"Staked Amount:    {result.get('staked_amount', amount):,.6f} RYX\n" \
                   f"Validator Address: {result.get('validator_address', 'Unknown')}\n" \
                   f"Staking Power:    +{result.get('staking_power', 0)}"
        except Exception as e:
            return f"❌ Error staking tokens: {e}"
    
    def cmd_validator_info(self, args: List[str]) -> str:
        """Show validator information"""
        try:
            validators = self.client.get_validators()
            staking_info = self.client.get_staking_info()
            
            response = "👑 VALIDATOR INFORMATION\n"
            response += "────────────────────────────────\n"
            response += f"Total Validators:   {len(validators)}\n"
            response += f"Your Staked Amount: {staking_info.get('total_staked', 0):,.6f} RYX\n"
            response += f"Validator Status:   {staking_info.get('validator_status', 'Unknown')}\n"
            
            if validators:
                response += f"\n🏆 TOP VALIDATORS:\n"
                for i, validator in enumerate(validators[:5]):
                    address = validator.get('address', 'Unknown')[:20] + '...'
                    stake = validator.get('stake', 0)
                    power = validator.get('power', 0)
                    response += f"{i+1}. {address}\n"
                    response += f"    Stake: {stake:,.6f} RYX | Power: {power}\n"
            
            return response
        except Exception as e:
            return f"❌ Error getting validator info: {e}"
    
    def cmd_contracts(self, args: List[str]) -> str:
        """List smart contracts"""
        try:
            contracts = self.client.get_smart_contracts()
            if not contracts:
                return "📄 No contracts deployed"
            
            response = f"📄 DEPLOYED SMART CONTRACTS ({len(contracts)} total)\n"
            response += "────────────────────────────────\n"
            for i, contract in enumerate(contracts):
                address = contract.get('address', 'Unknown')
                name = contract.get('name', 'Unnamed Contract')
                balance = contract.get('balance', 0)
                response += f"{i+1}. {name}\n"
                response += f"    Address: {address}\n"
                response += f"    Balance: {balance:,.6f} RYX\n"
            
            return response
        except Exception as e:
            return f"❌ Error getting contracts: {e}"
    
    def cmd_deploy_contract(self, args: List[str]) -> str:
        """Deploy smart contract"""
        if not args:
            return "❌ Usage: deploy-contract <code>"
        
        try:
            code = args[0]
            contract_address = self.client.deploy_contract(code)
            return f"✅ CONTRACT DEPLOYED SUCCESSFULLY\n" \
                   f"────────────────────────────────\n" \
                   f"Contract Address: {contract_address}\n" \
                   f"Code Length:      {len(code)} characters\n" \
                   f"Status:           Deployed and active"
        except Exception as e:
            return f"❌ Error deploying contract: {e}"
    
    def cmd_call_contract(self, args: List[str]) -> str:
        """Call contract function"""
        if len(args) < 2:
            return "❌ Usage: call-contract <address> <function> [args...]"
        
        try:
            address = args[0]
            function = args[1]
            call_args = args[2:] if len(args) > 2 else []
            
            result = self.client.call_contract(address, function, call_args)
            return f"✅ CONTRACT CALL EXECUTED\n" \
                   f"────────────────────────────────\n" \
                   f"Contract: {address}\n" \
                   f"Function: {function}\n" \
                   f"Arguments: {call_args}\n" \
                   f"Result: {result}"
        except Exception as e:
            return f"❌ Error calling contract: {e}"

    # System Commands
    def cmd_config(self, args: List[str]) -> str:
        """Show configuration information"""
        return self.node_cmd.execute_config(args)
    
    def cmd_stats(self, args: List[str]) -> str:
        return self.system_cmd.execute_stats(args)
        
    def cmd_generate_api_key(self, args: List[str]) -> str:
    	"""Generate a strong API key"""
    	return self.system_cmd.execute_generate_api_key(args)
    
    def cmd_validate_api_key(self, args: List[str]) -> str:
    	"""Validate API key strength"""
    	from rayonix_node.utils.api_key_manager import validate_api_key
    	
    	if not args:
    		return "❌ Usage: validate-api-key <api_key>"
    	api_key = args[0]
    	is_valid, reason = validate_api_key(api_key)
    	
    	if is_valid:
    		return f"✅ API KEY VALIDATION PASSED\n────────────────────────────────\n{reason}"
    	else:
    		return f"❌ API KEY VALIDATION FAILED\n────────────────────────────────\n{reason}\n\nUse 'generate-api-key' to create a strong key."
    
    def cmd_api_key_info(self, args: List[str]) -> str:
    	"""Show current API key authentication status"""
    	return self.api_cmd.execute_api_key_info(args)
    
    def cmd_setup_api_auth(self, args: List[str]) -> str:
    	return self.api_cmd.execute_setup_api_auth(args)
    
    def cmd_validate_api_key(self, args: List[str]) -> str:
    	return self.api_cmd.execute_validate_api_key(args)

    # Utility Methods
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
    	
    def _format_uptime(self, uptime_value) -> str:
        """Format uptime in seconds to human readable format"""
        # If it's already a formatted string, return it as-is
        if isinstance(uptime_value, str) and any(x in uptime_value for x in ['s', 'm', 'h', 'd']):
        	return uptime_value
        
        # If it's a number, use the original formatting logic
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
        	# If we can't convert it, return as string
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
