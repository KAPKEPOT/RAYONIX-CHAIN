# rayonix_node/cli/wallet_commands/load_wallet.py

import getpass
from typing import Optional, Dict, Any
from rayonix_node.cli.command_handler import CommandHandler


class LoadWalletCommand:
    """Interactive wallet loading wizard"""
    
    def __init__(self, command_handler: CommandHandler):
        self.handler = command_handler
        self.client = command_handler.client
    
    def execute(self, args: list) -> str:
        """Execute the wallet loading wizard"""
        try:
            print("\n🔄 WALLET LOADING WIZARD")
            print("────────────────────────────────")
            
            # Step 1: Get wallet type
            wallet_type = self._prompt_wallet_type()
            if not wallet_type:
                return "❌ Wallet loading cancelled."
            
            # Step 2: Get mnemonic phrase
            mnemonic = self._prompt_mnemonic_phrase()
            if not mnemonic:
                return "❌ Wallet loading cancelled."
            
            # Step 3: Get password
            password = self._prompt_wallet_password()
            
            # Step 4: Confirm and load
            if self._confirm_wallet_loading(wallet_type, mnemonic):
                return self._execute_wallet_loading(wallet_type, mnemonic, password)
            else:
                return "❌ Wallet loading cancelled."
                
        except KeyboardInterrupt:
            return "\n❌ Wallet loading cancelled by user."
        except Exception as e:
            return f"❌ Error during wallet loading: {e}"
    
    def _prompt_wallet_type(self) -> Optional[str]:
        """Prompt user for wallet type"""
        print("\n📁 SELECT WALLET TYPE:")
        print("1. HD Wallet (Recommended) - Hierarchical Deterministic")
        print("2. Legacy Wallet - Single key wallet")
        
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
    
    def _prompt_mnemonic_phrase(self) -> Optional[str]:
        """Securely prompt for mnemonic phrase"""
        print("\n🔐 ENTER MNEMONIC PHRASE:")
        print("• Enter your 12, 18, or 24 word recovery phrase")
        print("• Words should be separated by spaces")
        print("• Press Enter twice when finished")
        print("• Type 'cancel' to quit\n")
        
        lines = []
        try:
            while True:
                line = input("Mnemonic phrase: ").strip()
                if line.lower() == 'cancel':
                    return None
                elif line == '':
                    # Empty line means done
                    break
                else:
                    lines.append(line)
            
            mnemonic = ' '.join(lines).strip()
            
            # Validate mnemonic
            if not self._validate_mnemonic_format(mnemonic):
                print("❌ Invalid mnemonic format. Please enter 12, 18, or 24 words.")
                return self._prompt_mnemonic_phrase()  # Retry
            
            return mnemonic
            
        except KeyboardInterrupt:
            return None
    
    def _prompt_wallet_password(self) -> Optional[str]:
        """Prompt for wallet password (optional)"""
        print("\n🔑 WALLET PASSWORD (OPTIONAL):")
        print("• Enter password if your wallet is encrypted")
        print("• Press Enter to skip if no password")
        print("• Type 'cancel' to quit\n")
        
        try:
            password = getpass.getpass("Wallet password (optional): ")
            if password.strip().lower() == 'cancel':
                return None
            
            if password.strip():  # If password provided
                confirm = getpass.getpass("Confirm password: ")
                if password != confirm:
                    print("❌ Passwords do not match. Please try again.")
                    return self._prompt_wallet_password()  # Retry
            
            return password.strip() or None
            
        except KeyboardInterrupt:
            return None
    
    def _validate_mnemonic_format(self, mnemonic: str) -> bool:
        """Basic mnemonic format validation"""
        words = mnemonic.strip().split()
        return len(words) in [12, 18, 24] and all(len(word) >= 3 for word in words)
    
    def _confirm_wallet_loading(self, wallet_type: str, mnemonic: str) -> bool:
        """Show confirmation before loading wallet"""
        words = mnemonic.split()
        masked_mnemonic = f"{words[0]} {words[1]} ... {words[-2]} {words[-1]}"
        
        print("\n⚠️  CONFIRM WALLET LOADING:")
        print("────────────────────────────────")
        print(f"Wallet Type: {wallet_type.upper()}")
        print(f"Mnemonic:    {masked_mnemonic}")
        print(f"Word Count:  {len(words)}")
        print("────────────────────────────────")
        
        while True:
            try:
                confirm = input("\nLoad this wallet? (yes/NO): ").strip().lower()
                if confirm in ['y', 'yes']:
                    return True
                elif confirm in ['n', 'no', '']:
                    return False
                else:
                    print("❌ Please enter 'yes' or 'no'")
            except KeyboardInterrupt:
                return False
    
    def _execute_wallet_loading(self, wallet_type: str, mnemonic: str, password: str) -> str:
        """Execute the actual wallet loading"""
        print("\n🔄 Loading wallet...")
        
        try:
            # Call the RPC method
            result = self.client.load_wallet(mnemonic, password, wallet_type)
            
            if not result.get('loaded', False):
                return "❌ Failed to load wallet. Please check your mnemonic and password."
            
            return self._format_success_response(result)
            
        except Exception as e:
            return self.handler._format_rpc_error(e)
    
    def _format_success_response(self, result: Dict[str, Any]) -> str:
        """Format successful wallet loading response"""
        response = "\n✅ WALLET LOADED SUCCESSFULLY"
        response += "\n────────────────────────────────"
        response += f"\nWallet ID:    {result.get('wallet_id', 'Unknown')}"
        response += f"\nWallet Type:  {result.get('wallet_type', 'Unknown').upper()}"
        response += f"\nNetwork:      {result.get('network', 'Unknown')}"
        response += f"\nAddresses:    {result.get('address_count', 0)}"
        
        # Show first few addresses
        addresses = result.get('addresses', [])
        if addresses:
            response += f"\nFirst Address: {addresses[0]}"
            if len(addresses) > 1:
                response += f"\nSecond Address: {addresses[1]}"
        
        # Show balance if available
        balance = result.get('balance')
        if balance is not None:
            response += f"\nBalance:      {balance:,.6f} RYX"
        
        response += "\n\n💡 Next steps:"
        response += "\n• Use 'wallet-info' to see detailed wallet information"
        response += "\n• Use 'balance' to check your balance"
        response += "\n• Use 'list-addresses' to see all addresses"
        
        return response