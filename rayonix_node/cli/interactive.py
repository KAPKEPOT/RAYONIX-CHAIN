# cli/interactive.py - PRODUCTION READY

import cmd
import asyncio
import threading
import readline
import os
from typing import Optional
from rayonix_node.cli.history_manager import HistoryManager

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
        
        # Main event loop for async operations
        self.main_loop = asyncio.get_event_loop()
    
    def precmd(self, line):
        """Process command before execution"""
        self.history_manager.add_to_history(line)
        return line
    
    def _execute_command_sync(self, command_line: str) -> str:
        """Execute command synchronously in main event loop"""
        try:
            # Run the async command in the main event loop
            future = asyncio.run_coroutine_threadsafe(
                self.command_handler.execute_command(command_line),
                self.main_loop
            )
            return future.result(timeout=10)  # 10 second timeout
        except Exception as e:
            return f"Error: {e}"
    
    def do_help(self, arg):
        """Show help information"""
        result = self._execute_command_sync(f"help {arg}")
        print(result)
    
    def do_status(self, arg):
        """Show node status"""
        result = self._execute_command_sync("status")
        print(result)
    
    def do_balance(self, arg):
        """Show wallet balance"""
        result = self._execute_command_sync("balance")
        print(result)
    
    def do_send(self, arg):
        """Send coins to address"""
        result = self._execute_command_sync(f"send {arg}")
        print(result)
    
    def do_address(self, arg):
        """Generate new address"""
        result = self._execute_command_sync("address")
        print(result)
    
    def do_peers(self, arg):
        """Show connected peers"""
        result = self._execute_command_sync("peers")
        print(result)
    
    def do_connect(self, arg):
        """Connect to peer"""
        result = self._execute_command_sync(f"connect {arg}")
        print(result)
    
    def do_disconnect(self, arg):
        """Disconnect from peer"""
        result = self._execute_command_sync(f"disconnect {arg}")
        print(result)
    
    def do_block(self, arg):
        """Show block information"""
        result = self._execute_command_sync(f"block {arg}")
        print(result)
    
    def do_transaction(self, arg):
        """Show transaction information"""
        result = self._execute_command_sync(f"transaction {arg}")
        print(result)
    
    def do_mempool(self, arg):
        """Show mempool information"""
        result = self._execute_command_sync("mempool")
        print(result)
    
    def do_stake(self, arg):
        """Show staking information"""
        result = self._execute_command_sync("stake")
        print(result)
    
    def do_stop(self, arg):
        """Stop the node"""
        result = self._execute_command_sync("stop")
        print(result)
        return True
    
    def do_restart(self, arg):
        """Restart the node"""
        result = self._execute_command_sync("restart")
        print(result)
    
    def do_config(self, arg):
        """Show or modify configuration"""
        result = self._execute_command_sync(f"config {arg}")
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

async def run_interactive_mode(node):
    """Run interactive CLI mode - PRODUCTION READY"""
    # Create history manager
    data_dir = node.config_manager.get('database.db_path', './rayonix_data')
    history_file = os.path.join(data_dir, '.cli_history')
    history_manager = HistoryManager(history_file)
    
    # Create CLI
    cli = RayonixInteractiveCLI(node, history_manager)
    
    def run_cli():
        """Run the CLI in a separate thread"""
        try:
            print("\n" + "="*50)
            print("RAYONIX BLOCKCHAIN NODE - INTERACTIVE MODE")
            print("Type 'help' for available commands")
            print("="*50)
            cli.cmdloop()
        except Exception as e:
            print(f"CLI error: {e}")
    
    # Start CLI in a separate thread
    cli_thread = threading.Thread(target=run_cli, daemon=True)
    cli_thread.start()
    
    try:
        # Keep the main async loop running while CLI is active
        while node.running and cli_thread.is_alive():
            await asyncio.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    finally:
        # Save history
        history_manager.save_history()