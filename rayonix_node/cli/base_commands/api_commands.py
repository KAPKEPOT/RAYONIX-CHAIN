# rayonix_node/cli/base_commands/api_commands.py

import getpass
import os
from typing import List, Dict, Any
from pathlib import Path
from rayonix_node.cli.base_commands.base_command import BaseCommand


class APICommands:
    """API key management and authentication commands"""
    
    def execute_api_key_info(self, args: List[str]) -> str:
        """Show current API key authentication status"""
        try:
            # Test authentication by making a simple API call
            info = self.client.get_detailed_info()
            
            response = "🔐 API KEY AUTHENTICATION STATUS\n"
            response += "────────────────────────────────\n"
            response += "Status: ✅ Authenticated\n"
            response += f"Node: {info.get('network', 'Unknown')}\n"
            response += f"Block Height: {info.get('block_height', 0):,}\n"
            response += f"API Enabled: {info.get('api_enabled', False)}\n"
            
            # Show API key source if available
            if hasattr(self.client, 'api_key') and self.client.api_key:
                masked_key = self.client.api_key[:8] + "..." + self.client.api_key[-4:] if len(self.client.api_key) > 12 else "***"
                response += f"API Key: {masked_key}\n"
            
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "authentication" in error_msg.lower():
                return "🔐 API KEY AUTHENTICATION STATUS\n────────────────────────────────\nStatus: ❌ Authentication Failed\nError: Invalid or missing API key"
            elif "connection" in error_msg.lower():
                return "🔐 API KEY AUTHENTICATION STATUS\n────────────────────────────────\nStatus: ❌ Connection Failed\nError: Cannot connect to node"
            else:
                return f"🔐 API KEY AUTHENTICATION STATUS\n────────────────────────────────\nStatus: ⚠️  Unknown\nError: {error_msg}"
    
    def execute_generate_api_key(self, args: List[str]) -> str:
        """Generate a strong API key with interactive options"""
        try:
            print("\n🔐 API KEY GENERATION WIZARD")
            print("────────────────────────────────")
            
            # Get key length
            length = self._prompt_key_length()
            if not length:
                return "❌ API key generation cancelled."
            
            # Get key name/description
            key_name = self._prompt_key_name()
            if key_name is None:  # Explicit None means cancelled
                return "❌ API key generation cancelled."
            
            # Generate the key
            from rayonix_node.utils.api_key_manager import APIKeyManager
            key = APIKeyManager.generate_strong_api_key(length)
            
            return self._format_key_generation_result(key, key_name, length)
            
        except KeyboardInterrupt:
            return "\n❌ API key generation cancelled by user."
        except Exception as e:
            return f"❌ Error generating API key: {e}"
    
    def execute_validate_api_key(self, args: List[str]) -> str:
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
    
    def execute_setup_api_auth(self, args: List[str]) -> str:
        """Interactive API authentication setup wizard"""
        try:
            print("\n🔐 API AUTHENTICATION SETUP WIZARD")
            print("────────────────────────────────")
            
            # Step 1: Check current node API status
            print("🔄 Checking node API status...")
            try:
                node_info = self.client.get_detailed_info()
                api_enabled = node_info.get('api_enabled', False)
                
                if not api_enabled:
                    return "❌ Node API is not enabled. Start node with --api-enabled flag."
                    
            except Exception as e:
                return f"❌ Cannot connect to node: {e}"
            
            # Step 2: Choose setup method
            method = self._prompt_setup_method()
            if not method:
                return "❌ Setup cancelled."
            
            if method == "generate":
                return self._setup_with_new_key()
            elif method == "existing":
                return self._setup_with_existing_key()
            else:
                return self._setup_environment_key()
                
        except KeyboardInterrupt:
            return "\n❌ Setup cancelled by user."
        except Exception as e:
            return f"❌ Setup failed: {e}"
    
    def _prompt_key_length(self) -> int:
        """Prompt for API key length"""
        print("\n📏 SELECT API KEY STRENGTH:")
        print("1. Standard (64 characters) 🔒")
        print("   • Good for most use cases")
        print("   • Balanced security and usability")
        
        print("\n2. Strong (128 characters) 🔒🔒")
        print("   • High security")
        print("   • Recommended for production")
        
        print("\n3. Maximum (256 characters) 🔒🔒🔒")
        print("   • Maximum security")
        print("   • For critical systems")
        
        while True:
            try:
                choice = input("\nChoose strength (1-3) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice == '1':
                    return 64
                elif choice == '2':
                    return 128
                elif choice == '3':
                    return 256
                else:
                    print("❌ Please enter 1, 2, 3, or 'q' to quit.")
            except KeyboardInterrupt:
                return None
    
    def _prompt_key_name(self) -> str:
        """Prompt for optional key name/description"""
        print("\n🏷️  KEY DESCRIPTION (OPTIONAL):")
        print("• Helps identify the key's purpose")
        print("• Press Enter to skip")
        print("• Type 'cancel' to quit\n")
        
        try:
            name = input("Key description: ").strip()
            if name.lower() == 'cancel':
                return None
            return name or "Unnamed API Key"
        except KeyboardInterrupt:
            return None
    
    def _prompt_setup_method(self) -> str:
        """Prompt for setup method"""
        print("\n🛠️  SELECT SETUP METHOD:")
        print("1. Generate new API key 🔑")
        print("   • Create a new secure key")
        print("   • Configure node automatically")
        
        print("\n2. Use existing API key 📝")
        print("   • You already have an API key")
        print("   • Configure CLI to use it")
        
        print("\n3. Environment variable 🌐")
        print("   • Set RAYONIX_API_KEY environment variable")
        print("   • Most secure method")
        
        while True:
            try:
                choice = input("\nChoose method (1-3) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice == '1':
                    return "generate"
                elif choice == '2':
                    return "existing"
                elif choice == '3':
                    return "environment"
                else:
                    print("❌ Please enter 1, 2, 3, or 'q' to quit.")
            except KeyboardInterrupt:
                return None
    
    def _setup_with_new_key(self) -> str:
        """Setup with newly generated key"""
        from rayonix_node.utils.api_key_manager import APIKeyManager
        
        # Generate key
        key = APIKeyManager.generate_strong_api_key(128)
        
        response = "\n✅ NEW API KEY GENERATED\n"
        response += "────────────────────────────────\n"
        response += f"API Key: {key}\n\n"
        
        response += "📋 NODE CONFIGURATION:\n"
        response += "────────────────────────────────\n"
        response += "Add this to your node configuration file:\n\n"
        response += "api:\n"
        response += "  enabled: true\n"
        response += f"  auth_key: \"{key}\"\n\n"
        
        response += "🔧 CLI USAGE:\n"
        response += "────────────────────────────────\n"
        response += "Use with CLI commands:\n"
        response += f'rayonix-cli --api-key "{key}" wallet-info\n\n'
        
        response += "🌐 ENVIRONMENT VARIABLE:\n"
        response += "────────────────────────────────\n"
        response += "Or set environment variable:\n"
        response += f'export RAYONIX_API_KEY="{key}"\n'
        response += "rayonix-cli --api-key-env wallet-info\n\n"
        
        response += "⚠️  SECURITY NOTES:\n"
        response += "• Store this key securely - it cannot be recovered!\n"
        response += "• Never commit to version control\n"
        response += "• Restart node after updating configuration\n"
        
        return response
    
    def _setup_with_existing_key(self) -> str:
        """Setup with existing API key"""
        print("\n🔑 ENTER EXISTING API KEY:")
        print("• Paste your existing API key")
        print("• Type 'cancel' to quit\n")
        
        try:
            api_key = getpass.getpass("API Key: ")
            if api_key.strip().lower() == 'cancel':
                return "❌ Setup cancelled."
            
            # Validate the key
            from rayonix_node.utils.api_key_manager import validate_api_key
            is_valid, reason = validate_api_key(api_key)
            
            if not is_valid:
                return f"❌ Invalid API key: {reason}"
            
            response = "\n✅ API KEY CONFIGURED\n"
            response += "────────────────────────────────\n"
            response += "🔧 CLI USAGE:\n"
            response += f'rayonix-cli --api-key "{api_key}" wallet-info\n\n'
            
            response += "📁 CONFIGURATION FILE:\n"
            response += "Create ~/.rayonix/cli_config with:\n"
            response += f'api_key = "{api_key}"\n\n'
            
            response += "🌐 ENVIRONMENT VARIABLE:\n"
            response += f'export RAYONIX_API_KEY="{api_key}"\n'
            response += "rayonix-cli --api-key-env wallet-info\n"
            
            return response
            
        except KeyboardInterrupt:
            return "❌ Setup cancelled by user."
    
    def _setup_environment_key(self) -> str:
        """Setup with environment variable"""
        response = "\n🌐 ENVIRONMENT VARIABLE SETUP\n"
        response += "────────────────────────────────\n"
        response += "Add to your shell profile (~/.bashrc, ~/.zshrc, etc.):\n\n"
        response += "export RAYONIX_API_KEY=\"your-api-key-here\"\n\n"
        
        response += "Then reload your shell:\n"
        response += "source ~/.bashrc  # or restart terminal\n\n"
        
        response += "Usage with CLI:\n"
        response += "rayonix-cli --api-key-env wallet-info\n\n"
        
        response += "This is the most secure method as the key\n"
        response += "is never stored in files or command history.\n"
        
        return response
    
    def _format_key_generation_result(self, key: str, key_name: str, length: int) -> str:
        """Format API key generation result"""
        response = "\n✅ API KEY GENERATED SUCCESSFULLY\n"
        response += "=" * 70 + "\n"
        response += f"Name: {key_name}\n"
        response += f"Length: {length} characters\n"
        response += f"Key: {key}\n"
        response += "=" * 70 + "\n\n"
        
        response += "🔧 USAGE INSTRUCTIONS:\n"
        response += "────────────────────────────────\n"
        
        response += "1. NODE CONFIGURATION:\n"
        response += "Add to your node config file:\n"
        response += "api:\n"
        response += "  enabled: true\n"
        response += f"  auth_key: \"{key}\"\n\n"
        
        response += "2. COMMAND LINE USAGE:\n"
        response += f'rayonix-cli --api-key "{key}" wallet-info\n\n'
        
        response += "3. ENVIRONMENT VARIABLE:\n"
        response += f'export RAYONIX_API_KEY="{key}"\n'
        response += "rayonix-cli --api-key-env wallet-info\n\n"
        
        response += "4. CONFIGURATION FILE:\n"
        response += "Create ~/.rayonix/cli_config:\n"
        response += f'api_key = "{key}"\n\n'
        
        response += "⚠️  SECURITY WARNINGS:\n"
        response += "────────────────────────────────\n"
        response += "• Store this key securely - it cannot be recovered!\n"
        response += "• Never commit to version control\n"
        response += "• Restart node after updating configuration\n"
        response += "• Environment variable method is most secure\n"
        
        return response