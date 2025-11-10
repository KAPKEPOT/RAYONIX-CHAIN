# rayonix_node/cli/wallet_commands/create_wallet.py

import getpass
from typing import Optional, Dict, Any
from rayonix_node.cli.wallet_commands.base_wallet_command import BaseWalletCommand


class CreateWalletCommand(BaseWalletCommand):
    def execute(self, args: list) -> str:
        """Execute the wallet creation wizard"""
        try:
            print("\n🔄 WALLET CREATION WIZARD")
            print("────────────────────────────────")
            
            # Step 1: Get wallet type
            wallet_type = self._prompt_wallet_type()
            if not wallet_type:
                return "❌ Wallet creation cancelled."
            
            # Step 2: Get security level
            mnemonic_length = self._prompt_security_level()
            if not mnemonic_length:
                return "❌ Wallet creation cancelled."
            
            # Step 3: Get password
            password = self._prompt_creation_password()
            if password is None:
                return "❌ Wallet creation cancelled."
            
            # Step 4: Confirm and create
            if self._confirm_wallet_creation(wallet_type, mnemonic_length):
                return self._execute_wallet_creation(wallet_type, mnemonic_length, password)
            else:
                return "❌ Wallet creation cancelled."
                
        except KeyboardInterrupt:
            return "\n❌ Wallet creation cancelled by user."
        except Exception as e:
            return f"❌ Error during wallet creation: {e}"
    
    def _prompt_wallet_type(self) -> Optional[str]:
        """Prompt user for wallet type with explanations"""
        print("\n📁 SELECT WALLET TYPE:")
        print("1. HD Wallet (Recommended) 📊")
        print("   • Hierarchical Deterministic")
        print("   • Generates unlimited addresses from single seed")
        print("   • Best security and backup")
        print("   • Supports multiple cryptocurrencies")
        
        print("\n2. Legacy Wallet 🔐")
        print("   • Single key wallet")
        print("   • Simple but limited")
        print("   • One address per wallet")
        print("   • Good for testing")
        
        while True:
            try:
                choice = input("\nChoose wallet type (1-2) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice == '1':
                    return "hd"
                elif choice == '2':
                    return "legacy"
                else:
                    print("❌ Please enter 1, 2, or 'q' to quit.")
            except KeyboardInterrupt:
                return None
    
    def _prompt_security_level(self) -> Optional[int]:
        """Prompt for mnemonic security level"""
        print("\n🛡️  SELECT SECURITY LEVEL:")
        print("1. Standard Security (12 words) 🔒")
        print("   • 128-bit entropy")
        print("   • Good for most users")
        print("   • Easy to write down")
        
        print("\n2. High Security (18 words) 🔒🔒")
        print("   • 192-bit entropy") 
        print("   • Enhanced security")
        print("   • Recommended for large amounts")
        
        print("\n3. Maximum Security (24 words) 🔒🔒🔒")
        print("   • 256-bit entropy")
        print("   • Military-grade security")
        print("   • For maximum protection")
        
        while True:
            try:
                choice = input("\nChoose security level (1-3) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                elif choice == '1':
                    return 12
                elif choice == '2':
                    return 18
                elif choice == '3':
                    return 24
                else:
                    print("❌ Please enter 1, 2, 3, or 'q' to quit.")
            except KeyboardInterrupt:
                return None
    
    def _prompt_creation_password(self) -> Optional[str]:
        """Prompt for wallet encryption password"""
        print("\n🔑 WALLET ENCRYPTION:")
        print("• Password encrypts your wallet file")
        print("• Required for sending transactions") 
        print("• Keep it safe - cannot be recovered!")
        print("• Press Enter to create without password (NOT RECOMMENDED)")
        print("• Type 'cancel' to quit\n")
        
        try:
            while True:
                password = getpass.getpass("Wallet password (recommended): ")
                
                if password.strip().lower() == 'cancel':
                    return None
                
                if not password.strip():  # No password
                    if self._confirm_no_password():
                        return ""
                    else:
                        continue
                
                # Validate password strength
                if not self._validate_password_strength(password):
                    print("❌ Password too weak. Use at least 8 characters with mix of letters, numbers, and symbols.")
                    continue
                
                confirm = getpass.getpass("Confirm password: ")
                if password != confirm:
                    print("❌ Passwords do not match. Please try again.")
                    continue
                
                return password
                
        except KeyboardInterrupt:
            return None
    
    def _confirm_no_password(self) -> bool:
        """Confirm if user really wants no password"""
        print("\n⚠️  SECURITY WARNING:")
        print("• Without password, wallet file is NOT encrypted")
        print("• Anyone with file access can steal your funds")
        print("• You won't be prompted for password when sending")
        
        while True:
            try:
                confirm = input("\nCreate wallet WITHOUT password? (yes/NO): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("⚠️  Proceeding without password encryption...")
                    return True
                elif confirm in ['n', 'no', '']:
                    return False
                else:
                    print("❌ Please enter 'yes' or 'no'")
            except KeyboardInterrupt:
                return False
    
    def _validate_password_strength(self, password: str) -> bool:
        """Validate password meets minimum requirements"""
        if len(password) < 8:
            return False
        
        # Check for character variety
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password) 
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)
        
        # Require at least 3 of 4 character types
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    def _confirm_wallet_creation(self, wallet_type: str, mnemonic_length: int) -> bool:
        """Show final confirmation before creating wallet"""
        print("\n⚠️  FINAL CONFIRMATION:")
        print("────────────────────────────────")
        print(f"Wallet Type:    {wallet_type.upper()}")
        print(f"Security Level: {mnemonic_length} words")
        print("────────────────────────────────")
        print("• You will receive a recovery phrase")
        print("• WRITE IT DOWN and store securely")
        print("• It's the ONLY way to recover funds")
        print("• Never share it with anyone")
        print("────────────────────────────────")
        
        while True:
            try:
                confirm = input("\nCreate new wallet? (yes/NO): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return True
                elif confirm in ['n', 'no', '']:
                    return False
                else:
                    print("❌ Please enter 'yes' or 'no'")
            except KeyboardInterrupt:
                return False
    
    def _execute_wallet_creation(self, wallet_type: str, mnemonic_length: int, password: str) -> str:
        """Execute the actual wallet creation"""
        print("\n🔄 Creating wallet...")
        
        try:
            # Call the RPC method
            result = self.client.create_wallet(wallet_type, password, mnemonic_length)
            
            if not result.get('wallet_id'):
                return "❌ Failed to create wallet. Please try again."
            
            return self._format_success_response(result)
            
        except Exception as e:
            return self.handler._format_rpc_error(e)
    
    def _format_success_response(self, result: Dict[str, Any]) -> str:
        """Format successful wallet creation response"""
        response = "\n✅ WALLET CREATED SUCCESSFULLY"
        response += "\n────────────────────────────────"
        response += f"\nWallet ID:    {result.get('wallet_id', 'Unknown')}"
        response += f"\nWallet Type:  {result.get('wallet_type', 'Unknown').upper()}"
        response += f"\nNetwork:      {result.get('network', 'Unknown')}"
        response += f"\nFirst Address: {result.get('address', 'Unknown')}"
        
        # Show the critical mnemonic information
        if 'mnemonic' in result:
            mnemonic = result['mnemonic']
            response += self._format_mnemonic_display(mnemonic)
        
        response += "\n\n💡 Next steps:"
        response += "\n• Backup your recovery phrase immediately"
        response += "\n• Use 'wallet-info' to see wallet details"
        response += "\n• Use 'balance' to check your balance"
        response += "\n• Use 'address' to generate new addresses"
        
        return response
    
    def _format_mnemonic_display(self, mnemonic: str) -> str:
        """Format mnemonic for secure display"""
        words = mnemonic.split()
        response = "\n\n🔐 CRITICAL SECURITY INFORMATION"
        response += "\n────────────────────────────────"
        response += "\nYOUR RECOVERY PHRASE (WRITE THIS DOWN):"
        response += f"\n┌{'─' * 40}┐"
        
        # Format mnemonic in lines based on word count
        if len(words) == 12:
            lines = [' '.join(words[0:4]), ' '.join(words[4:8]), ' '.join(words[8:12])]
        elif len(words) == 18:
            lines = [' '.join(words[0:6]), ' '.join(words[6:12]), ' '.join(words[12:18])]
        else:  # 24 words
            lines = [' '.join(words[0:6]), ' '.join(words[6:12]), ' '.join(words[12:18]), ' '.join(words[18:24])]
        
        for line in lines:
            response += f"\n│ {line:38} │"
        
        response += f"\n└{'─' * 40}┘"
        
        response += "\n\n⚠️  SECURITY WARNINGS:"
        response += "\n• This is the ONLY way to recover your wallet"
        response += "\n• Store it securely - never digitally"
        response += "\n• Never share with anyone"
        response += "\n• Loss = Permanent loss of all funds"
        
        return response