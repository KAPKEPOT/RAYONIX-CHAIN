# cli/interactive.py - Interactive mode implementation for RPC client

import cmd
import os
from typing import Optional
from rayonix_node.cli.history_manager import HistoryManager

class RayonixInteractiveCLI(cmd.Cmd):
    """Interactive CLI for RAYONIX blockchain node using RPC client"""
    
    def __init__(self, rpc_client, history_manager: Optional[HistoryManager] = None):
        super().__init__()
        self.client = rpc_client
        self.history_manager = history_manager or HistoryManager()
        self.prompt = "rayonix> "
        self.intro = "RAYONIX Blockchain Node CLI (RPC Mode)\nType 'help' for available commands"
        
        # Setup command handler
        from rayonix_node.cli.command_handler import CommandHandler
        self.command_handler = CommandHandler(rpc_client)
        
        # Load command history
        self.history_manager.load_history()
    
    def precmd(self, line):
        """Process command before execution"""
        self.history_manager.add_to_history(line)
        return line
    
    def default(self, line):
        """Handle commands"""
        parts = line.strip().split()
        if not parts:
            return
            
        command = parts[0]
        args = parts[1:]
        
        result = self.command_handler.execute_command(command, args)
        if result:
            print(result)
    
    def do_help(self, arg):
        """Show help information"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('help', args)
        print(result)
    
    def do_info(self, arg):
        """Show node information"""
        result = self.command_handler.execute_command('info')
        print(result)
    
    def do_status(self, arg):
        """Show node status"""
        result = self.command_handler.execute_command('status')
        print(result)
    
    def do_balance(self, arg):
        """Show wallet balance"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('balance', args)
        print(result)
    
    def do_send(self, arg):
        """Send coins to address"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('send', args)
        print(result)
    
    def do_address(self, arg):
        """Generate new address"""
        result = self.command_handler.execute_command('address')
        print(result)
    
    def do_peers(self, arg):
        """Show connected peers"""
        result = self.command_handler.execute_command('peers')
        print(result)
    
    def do_block(self, arg):
        """Show block information"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('block', args)
        print(result)
    
    def do_transaction(self, arg):
        """Show transaction information"""
        args = arg.split() if arg else []
        result = self.command_handler.execute_command('transaction', args)
        print(result)
    
    def do_mempool(self, arg):
        """Show mempool information"""
        result = self.command_handler.execute_command('mempool')
        print(result)
    
    def do_stake(self, arg):
        """Show staking information"""
        result = self.command_handler.execute_command('stake')
        print(result)
    
    def do_exit(self, arg):
        """Exit the CLI"""
        print("Exiting CLI...")
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

def run_interactive_mode(rpc_client, data_dir: str):
    """Run interactive CLI mode with RPC client"""
    history_file = os.path.join(data_dir, '.rayonix_cli_history')
    history_manager = HistoryManager(history_file)
    
    # Create and run CLI
    cli = RayonixInteractiveCLI(rpc_client, history_manager)
    
    try:
        print("\n" + "="*50)
        print("RAYONIX BLOCKCHAIN NODE - INTERACTIVE MODE")
        print("Connected to daemon via RPC")
        print("Type 'help' for available commands")
        print("="*50)
        
        cli.cmdloop()
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"CLI error: {e}")
    finally:
        # Save history
        history_manager.save_history()