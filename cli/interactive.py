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
        asyncio.create_task(self._execute_command(line))
    
    async def _execute_command(self, command_line):
        """Execute command and print result"""
        try:
            result = await self.command_handler.execute_command(command_line)
            if result:
                print(result)
        except Exception as e:
            print(f"Error: {e}")
    
    def do_help(self, arg):
        """Show help information"""
        asyncio.create_task(self._execute_command(f"help {arg}"))
    
    def do_status(self, arg):
        """Show node status"""
        asyncio.create_task(self._execute_command("status"))
    
    def do_balance(self, arg):
        """Show wallet balance"""
        asyncio.create_task(self._execute_command("balance"))
    
    def do_send(self, arg):
        """Send coins to address"""
        asyncio.create_task(self._execute_command(f"send {arg}"))
    
    def do_address(self, arg):
        """Generate new address"""
        asyncio.create_task(self._execute_command("address"))
    
    def do_peers(self, arg):
        """Show connected peers"""
        asyncio.create_task(self._execute_command("peers"))
    
    def do_connect(self, arg):
        """Connect to peer"""
        asyncio.create_task(self._execute_command(f"connect {arg}"))
    
    def do_disconnect(self, arg):
        """Disconnect from peer"""
        asyncio.create_task(self._execute_command(f"disconnect {arg}"))
    
    def do_block(self, arg):
        """Show block information"""
        asyncio.create_task(self._execute_command(f"block {arg}"))
    
    def do_transaction(self, arg):
        """Show transaction information"""
        asyncio.create_task(self._execute_command(f"transaction {arg}"))
    
    def do_mempool(self, arg):
        """Show mempool information"""
        asyncio.create_task(self._execute_command("mempool"))
    
    def do_stake(self, arg):
        """Show staking information"""
        asyncio.create_task(self._execute_command("stake"))
    
    def do_stop(self, arg):
        """Stop the node"""
        asyncio.create_task(self._execute_command("stop"))
    
    def do_restart(self, arg):
        """Restart the node"""
        asyncio.create_task(self._execute_command("restart"))
    
    def do_config(self, arg):
        """Show or modify configuration"""
        asyncio.create_task(self._execute_command(f"config {arg}"))
    
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
    
    # Create and run CLI
    cli = RayonixInteractiveCLI(node, history_manager)
    
    try:
        # Run CLI in a separate thread to not block the event loop
        def run_cli():
            cli.cmdloop()
        
        # Start CLI in thread
        cli_thread = threading.Thread(target=run_cli, daemon=True)
        cli_thread.start()
        
        # Keep the main event loop running
        while node.running and cli_thread.is_alive():
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        # Save history
        history_manager.save_history()