# rayonix_node/cli/interactive.py - interactive mode implementation

import cmd
import os
import time
from typing import Optional
from datetime import datetime
from rayonix_node.cli.history_manager import HistoryManager

class RayonixInteractiveCLI(cmd.Cmd):
    """Interactive CLI """
    
    def __init__(self, rpc_client, history_manager=None):
        super().__init__()
        self.client = rpc_client
        self.history_manager = history_manager
        self.prompt = "rayonix> "
        self.intro = "RAYONIX Blockchain CLI\nType 'help' for available commands"
        
        from rayonix_node.cli.command_handler import CommandHandler
        self.command_handler = CommandHandler(rpc_client)
        
        # Advanced features
        self.start_time = time.time()
        self.command_count = 0
        self.last_command = None
        
    def precmd(self, line):
        """Process command before execution"""
        self.command_count += 1
        self.last_command = line
        if self.history_manager:
            self.history_manager.add_to_history(line)
        return line
    
    def default(self, line):
        """Handle commands """
        parts = line.strip().split()
        if not parts:
            return
            
        command = parts[0]
        args = parts[1:]
        
        result = self.command_handler.execute_command(command, args)
        if result:
            print(result)
    
    # Enhanced basic commands
    def do_help(self, arg):
        """Show help information"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('help', args)
        print(result)
    
    def do_info(self, arg):
        """Show detailed node information"""
        result = self.command_handler.execute_command('info')
        print(result)
    
    def do_status(self, arg):
        """Show  node status"""
        result = self.command_handler.execute_command('status')
        print(result)
    
    # Wallet Management Commands
    def do_create_wallet(self, arg):
        """Create a new wallet with advanced options
        Usage: create-wallet [type] [password]
        Examples:
          create-wallet
          create-wallet hd
          create-wallet hd mypassword123
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('create-wallet', args)
        print(result)
    
    def do_load_wallet(self, arg):
        """Load wallet from mnemonic phrase
        Usage: load-wallet <mnemonic_phrase> [password]
        Examples:
          load-wallet "word1 word2 ... word12"
          load-wallet "word1 word2 ... word12" mypassword
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('load-wallet', args)
        print(result)
    
    def do_import_wallet(self, arg):
        """Import wallet from backup file
        Usage: import-wallet <filename> [password]
        Examples:
          import-wallet wallet_backup.dat
          import-wallet backup.json mypassword
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('import-wallet', args)
        print(result)
    
    def do_backup_wallet(self, arg):
        """Backup wallet to file
        Usage: backup-wallet [filename]
        Examples:
          backup-wallet
          backup-wallet my_wallet_backup.dat
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('backup-wallet', args)
        print(result)
    
    def do_wallet_info(self, arg):
        """Show detailed wallet information"""
        result = self.command_handler.execute_command('wallet-info')
        print(result)
    
    def do_list_addresses(self, arg):
        """List all wallet addresses with balances"""
        result = self.command_handler.execute_command('list-addresses')
        print(result)
    
    # Enhanced wallet operations
    def do_balance(self, arg):
        """Show detailed wallet balance
        Usage: balance [address]
        Examples:
          balance
          balance RXyAbc123...
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('balance', args)
        print(result)
    
    def do_send(self, arg):
        """Send coins to address
        Usage: send <address> <amount> [fee]
        Examples:
          send RXyAbc123... 10.5
          send RXyAbc123... 25.0 0.001
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('send', args)
        print(result)
    
    def do_address(self, arg):
        """Generate new address"""
        result = self.command_handler.execute_command('address')
        print(result)
    
    # Network commands
    def do_peers(self, arg):
        """Show connected peers with details"""
        result = self.command_handler.execute_command('peers')
        print(result)
    
    def do_network(self, arg):
        """Show network statistics"""
        result = self.command_handler.execute_command('network')
        print(result)
    
    # Blockchain commands
    def do_blockchain_info(self, arg):
        """Show detailed blockchain information"""
        result = self.command_handler.execute_command('blockchain-info')
        print(result)
    
    def do_sync_status(self, arg):
        """Show synchronization status"""
        result = self.command_handler.execute_command('sync-status')
        print(result)
    
    def do_block(self, arg):
        """Show block information
        Usage: block <height_or_hash>
        Examples:
          block 1500
          block 0a1b2c3d...
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('block', args)
        print(result)
    
    def do_transaction(self, arg):
        """Show transaction information
        Usage: transaction <hash>
        Examples:
          transaction a1b2c3d4e5f6...
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('transaction', args)
        print(result)
    
    def do_mempool(self, arg):
        """Show mempool information"""
        result = self.command_handler.execute_command('mempool')
        print(result)
    
    # Advanced features 
    def do_staking(self, arg):
        """Show staking information"""
        result = self.command_handler.execute_command('staking')
        print(result)
    
    def do_stake(self, arg):
        """Stake tokens for validation
        Usage: stake <amount>
        Examples:
          stake 1000
          stake 500.5
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('stake', args)
        print(result)
    
    def do_validator_info(self, arg):
        """Show validator information"""
        result = self.command_handler.execute_command('validator-info')
        print(result)
    
    def do_contracts(self, arg):
        """List smart contracts"""
        result = self.command_handler.execute_command('contracts')
        print(result)
    
    def do_deploy_contract(self, arg):
        """Deploy smart contract
        Usage: deploy-contract <code>
        Examples:
          deploy-contract "contract Code {...}"
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('deploy-contract', args)
        print(result)
    
    def do_call_contract(self, arg):
        """Call contract function
        Usage: call-contract <address> <function> [args...]
        Examples:
          call-contract RXyContract123 getBalance
          call-contract RXyContract123 transfer RXyRecipient456 10.5
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('call-contract', args)
        print(result)
    
    def do_history(self, arg):
        """Show transaction history
        Usage: history [count]
        Examples:
          history
          history 25
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('history', args)
        print(result)
    
    # System and monitoring commands
    def do_config(self, arg):
        """Show configuration information
        Usage: config [key]
        Examples:
          config
          config network.port
        """
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('config', args)
        print(result)
    
    def do_stats(self, arg):
        """Show CLI and node statistics"""
        # Show CLI statistics
        uptime = time.time() - self.start_time
        print("CLI Statistics:")
        print(f"  Commands executed: {self.command_count}")
        print(f"  Uptime: {uptime:.1f} seconds")
        print(f"  Average commands/minute: {self.command_count / (uptime / 60):.1f}")
        if self.last_command:
            print(f"  Last command: {self.last_command}")
        
        # Show client performance metrics
        try:
            metrics = self.client.get_performance_metrics()
            print(f"\nRPC Performance:")
            print(f"  Total Requests: {metrics['requests_made']}")
            print(f"  Cache Hits: {metrics['cache_hits']}")
            print(f"  Average Response Time: {metrics['average_response_time']:.3f}s")
            if metrics['requests_made'] > 0:
                hit_rate = (metrics['cache_hits'] / metrics['requests_made'] * 100)
                print(f"  Cache Hit Rate: {hit_rate:.1f}%")
        except Exception as e:
            print(f"  Performance metrics: {e}")
        
        # Show node stats
        try:
            node_info = self.client.get_detailed_info()
            print(f"\nNode Statistics:")
            print(f"  Block Height: {node_info.get('block_height', 0)}")
            print(f"  Connected Peers: {node_info.get('peers_connected', 0)}")
            print(f"  Uptime: {node_info.get('uptime', 0)} seconds")
        except Exception as e:
            print(f"  Node stats: {e}")
            
    def do_generate_api_key(self, arg):
    	"""Generate a strong API key for authentication"""
    	args = arg.split() if arg else []
    	result = self.command_handler.execute_command('generate-api-key', args)
    	print(result)
    
    def do_validate_api_key(self, arg):
    	"""Validate API key strength"""
    	args = arg.split() if arg else []
    	result = self.command_handler.execute_command('validate-api-key', args)
    	print(result)
    
    def do_api_key_info(self, arg):
    	"""Show current API key authentication status"""
    	result = self.command_handler.execute_command('api-key-info', [])
    	print(result)
    	
    def do_clear(self, arg):
        """Clear the screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
        print("RAYONIX CLI - Screen cleared")
    
    def do_repeat(self, arg):
        """Repeat the last command"""
        if self.last_command:
            print(f"Repeating: {self.last_command}")
            self.onecmd(self.last_command)
        else:
            print("No previous command to repeat")
    
    def do_history_list(self, arg):
        """Show command history"""
        if self.history_manager:
            history = self.history_manager.get_history()
            if not history:
                print("No command history")
                return
            
            print("Command History (last 20 commands):")
            for i, cmd in enumerate(history[-20:], 1):
                print(f"  {i:2d}. {cmd}")
        else:
            print("History manager not available")
    
    def do_search_history(self, arg):
        """Search command history
        Usage: search-history <term>
        Examples:
          search-history send
          search-history balance
        """
        if not arg:
            print("Usage: search-history <search_term>")
            return
        
        if self.history_manager:
            results = self.history_manager.search_history(arg)
            if not results:
                print(f"No commands found containing '{arg}'")
                return
            
            print(f"Commands containing '{arg}':")
            for i, cmd in enumerate(results, 1):
                print(f"  {i:2d}. {cmd}")
        else:
            print("History manager not available")
    
    def do_clear_history(self, arg):
        """Clear command history"""
        if self.history_manager:
            self.history_manager.clear_history()
            print("Command history cleared")
        else:
            print("History manager not available")
    
    # Exit commands
    def do_exit(self, arg):
        """Exit the CLI"""
        print("Exiting RAYONIX CLI...")
        print("Thank you for using RAYONIX Blockchain!")
        return True
    
    def do_quit(self, arg):
        """Exit the CLI"""
        return self.do_exit(arg)
    
    def do_EOF(self, arg):
        """Handle Ctrl-D"""
        print()
        return self.do_exit(arg)
    
    def emptyline(self):
        """Do nothing on empty line"""
        pass
    
    def postcmd(self, stop, line):
        """Called after each command execution"""
        # Add a small separator for better readability
        if not stop:
            print("-" * 60)
        return stop
    
    # command completion
    def complete_send(self, text, line, begidx, endidx):
        """Auto-completion for send command"""
        options = ['--fee']
        return [opt for opt in options if opt.startswith(text)]
    
    def complete_block(self, text, line, begidx, endidx):
        """Auto-completion for block command"""
        # Could implement block height/hash completion here
        return []
    
    def complete_config(self, text, line, begidx, endidx):
        """Auto-completion for config command"""
        # Common config keys for completion
        config_keys = [
            'network.port', 'network.host', 'api.port', 'api.enabled',
            'database.path', 'consensus.type', 'wallet.encryption'
        ]
        return [key for key in config_keys if key.startswith(text)]

def run_interactive_mode(rpc_client, data_dir: str):
    """Run  interactive CLI mode"""
    history_file = os.path.join(data_dir, '.rayonix_history')
    history_manager = HistoryManager(history_file)
    history_manager.load_history()
    
    # Create and run CLI
    cli = RayonixInteractiveCLI(rpc_client, history_manager)
    
    try:
        print("\n" + "="*70)
        print("RAYONIX BLOCKCHAIN CLI")     
        print("Connected to daemon via RPC")
        print("="*70)
        
        # Show quick status on startup
        try:
            status = rpc_client.get_node_status()
            print(f"Node Status: {status.get('status', 'Unknown')}")
            print(f"Block Height: {status.get('block_height', 0)}")
            print(f"Connected Peers: {status.get('peers_connected', 0)}")
        except Exception as e:
            print(f"Status check: {e}")
        
        print("\nType 'help' for available commands")
        print("Type 'exit' or 'quit' to exit")
        print("="*70)
        
        cli.cmdloop()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n CLI error: {e}")
    finally:
        # Save history
        history_manager.save_history()
        print("Command history saved")