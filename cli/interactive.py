# cli/interactive.py - Interactive mode implementation

import cmd
import shlex
import asyncio
import readline
import os
from typing import Optional
import threading

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
        
        # Event for stopping the CLI
        self.should_exit = asyncio.Event()
    
    def precmd(self, line):
        """Process command before execution"""
        self.history_manager.add_to_history(line)
        return line
    
    async def execute_command_async(self, command_line: str) -> str:
        """Execute command asynchronously"""
        try:
            return await self.command_handler.execute_command(command_line)
        except Exception as e:
            return f"Error: {e}"
    
    def default(self, line):
        """Handle unknown commands"""
        # For production, we run commands in a thread-safe way
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(line), 
            asyncio.get_event_loop()
        ).result(timeout=30)  # 30 second timeout
        
        if result:
            print(result)
    
    def do_help(self, arg):
        """Show help information"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"help {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_status(self, arg):
        """Show node status"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("status"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_balance(self, arg):
        """Show wallet balance"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("balance"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_send(self, arg):
        """Send coins to address"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"send {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=30)
        if result:
            print(result)
    
    def do_address(self, arg):
        """Generate new address"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("address"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_peers(self, arg):
        """Show connected peers"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("peers"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_connect(self, arg):
        """Connect to peer"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"connect {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=15)
        if result:
            print(result)
    
    def do_disconnect(self, arg):
        """Disconnect from peer"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"disconnect {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_block(self, arg):
        """Show block information"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"block {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_transaction(self, arg):
        """Show transaction information"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"transaction {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_mempool(self, arg):
        """Show mempool information"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("mempool"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_stake(self, arg):
        """Show staking information"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("stake"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_stop(self, arg):
        """Stop the node"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("stop"), 
            asyncio.get_event_loop()
        ).result(timeout=30)
        if result:
            print(result)
        self.should_exit.set()
        return True
    
    def do_restart(self, arg):
        """Restart the node"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async("restart"), 
            asyncio.get_event_loop()
        ).result(timeout=30)
        if result:
            print(result)
    
    def do_config(self, arg):
        """Show or modify configuration"""
        result = asyncio.run_coroutine_threadsafe(
            self.execute_command_async(f"config {arg}"), 
            asyncio.get_event_loop()
        ).result(timeout=10)
        if result:
            print(result)
    
    def do_exit(self, arg):
        """Exit the CLI"""
        print("Exiting CLI...")
        self.should_exit.set()
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
    """Run interactive CLI mode - Production ready"""
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
        finally:
            cli.should_exit.set()
    
    # Start CLI in a separate thread
    cli_thread = threading.Thread(target=run_cli, daemon=True)
    cli_thread.start()
    
    try:
        # Keep the main async loop running while CLI is active
        while node.running and not cli.should_exit.is_set():
            await asyncio.sleep(0.5)
            
        # If we get here, CLI has exited but node might still be running
        if node.running and cli.should_exit.is_set():
            print("\nCLI session ended. Node continues running in background.")
            print("Use Ctrl+C to stop the node completely.")
            
    except KeyboardInterrupt:
        print("\nReceived shutdown signal...")
    except Exception as e:
        print(f"Interactive mode error: {e}")
    finally:
        # Save history
        history_manager.save_history()
        
        # If CLI thread is still alive, we need to stop it
        if cli_thread.is_alive():
            cli.should_exit.set()