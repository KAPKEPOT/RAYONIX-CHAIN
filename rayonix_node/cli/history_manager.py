# rayonix_node/cli/history_manager.py - Command history management

import os
import readline
from typing import List

class HistoryManager:
    """Manages CLI command history"""
    
    def __init__(self, history_file: str = ".rayonix_history"):
        self.history_file = history_file
        self.history: List[str] = []
        self.max_history_size = 1000
    
    def load_history(self):
        """Load command history from file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.history = [line.strip() for line in f.readlines() if line.strip()]
                
                # Set readline history
                readline.clear_history()
                for command in self.history:
                    readline.add_history(command)
                
                return True
        except Exception as e:
            print(f"Error loading history: {e}")
        return False
    
    def save_history(self):
        """Save command history to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            with open(self.history_file, 'w') as f:
                for command in self.history[-self.max_history_size:]:
                    f.write(command + '\n')
            
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
        return False
    
    def add_to_history(self, command: str):
        """Add command to history"""
        if command and command not in self.history:
            self.history.append(command)
            readline.add_history(command)
    
    def get_history(self) -> List[str]:
        """Get command history"""
        return self.history.copy()
    
    def clear_history(self):
        """Clear command history"""
        self.history.clear()
        readline.clear_history()
        if os.path.exists(self.history_file):
            os.remove(self.history_file)
    
    def search_history(self, search_term: str) -> List[str]:
        """Search command history"""
        return [cmd for cmd in self.history if search_term.lower() in cmd.lower()]