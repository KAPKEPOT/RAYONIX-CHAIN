# cli/interactive.py - Interactive mode implementation

import cmd
import shlex
import asyncio
import threading
import readline
import os
from typing import Optional

from cli.history_manager import HistoryManager

class RayonixInteractiveCLI(cmd.Cmd):
    """Interactive CLI for RAYONIX blockchain node"""
    
    def __init__(self, node, history_manager: Optional[HistoryManager] = None):
        super().__init__()
        self.node = node
        self.history_manager = history_manager or HistoryManager()
        self.prompt = "rayonix> "
        self.intro = "RAYONIX Blockchain Node CLI\nType 'help' for available commands"
        
        # Setup command handler
        from cli.command_handler import CommandHandler
        self.command_handler = CommandHandler(node)
        
        # Load command history
        self.history_manager.load_history()
    
    def precmd(self, line):
        """Process command before execution"""
        self.history_manager.add_to_history(line)
        return line
    
    def default(self, line):
        """Handle unknown commands"""
        # Run async commands in the event loop
        result = asyncio.run(self._execute_command(line))
        if result:
            print(result)
    
    async def _execute_command(self, command_line):
        """Execute command and return result"""
        try:
            result = await self.command_handler.execute_command(command_line)
            return result
        except Exception as e:
            return f"Error: {e}"
    
    def do_help(self, arg):
        """Show help information"""
        result = asyncio.run(self._execute_command(f"help {arg}"))
        if result:
            print(result)
    
    def do_status(self, arg):
        """Show node status"""
        result = asyncio.run(self._execute_command("status"))
        if result:
            print(result)
    
    def do_balance(self, arg):
        """Show wallet balance"""
        result = asyncio.run(self._execute_command("balance"))
        if result:
            print(result)
    
    def do_send(self, arg):
        """Send coins to address"""
        result = asyncio.run(self._execute_command(f"send {arg}"))
        if result:
            print(result)
    
    def do_address(self, arg):
        """Generate new address"""
        result = asyncio.run(self._execute_command("address"))
        if result:
            print(result)
    
    def do_peers(self, arg):
        """Show connected peers"""
        result = asyncio.run(self._execute_command("peers"))
        if result:
            print(result)
    
    def do_connect(self, arg):
        """Connect to peer"""
        result = asyncio.run(self._execute_command(f"connect {arg}"))
        if result:
            print(result)
    
    def do_disconnect(self, arg):
        """Disconnect from peer"""
        result = asyncio.run(self._execute_command(f"disconnect {arg}"))
        if result:
            print(result)
    
    def do_block(self, arg):
        """Show block information"""
        result = asyncio.run(self._execute_command(f"block {arg}"))
        if result:
            print(result)
    
    def do_transaction(self, arg):
        """Show transaction information"""
        result = asyncio.run(self._execute_command(f"transaction {arg}"))
        if result:
            print(result)
    
    def do_mempool(self, arg):
        """Show mempool information"""
        result = asyncio.run(self._execute_command("mempool"))
        if result:
            print(result)
    
    def do_stake(self, arg):
        """Show staking information"""
        result = asyncio.run(self._execute_command("stake"))
        if result:
            print(result)
    
    def do_stop(self, arg):
        """Stop the node"""
        result = asyncio.run(self._execute_command("stop"))
        if result:
            print(result)
        return True  # Exit after stop
    
    def do_restart(self, arg):
        """Restart the node"""
        result = asyncio.run(self._execute_command("restart"))
        if result:
            print(result)
    
    def do_config(self, arg):
        """Show or modify configuration"""
        result = asyncio.run(self._execute_command(f"config {arg}"))
        if result:
            print(result)
    
    def do_exit(self, arg):
        """Exit the CLI"""
        print("Exiting...")
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

async def run_interactive_mode(node):
    """Run interactive CLI mode"""
    # Create history manager
    data_dir = node.config_manager.get('database.db_path', './rayonix_data')
    history_file = os.path.join(data_dir, '.cli_history')
    history_manager = HistoryManager(history_file)
    
    # Create and run CLI - THIS BLOCKS UNTIL USER EXITS
    cli = RayonixInteractiveCLI(node, history_manager)
    
    try:
        # Run CLI in main thread - this will block and show the prompt
        print("\n" + "="*50)
        print("RAYONIX BLOCKCHAIN NODE - INTERACTIVE MODE")
        print("="*50)
        cli.cmdloop()
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"CLI error: {e}")
    finally:
        # Save history
        history_manager.save_history()
        print("Interactive session ended")