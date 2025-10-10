import os
import json
import zipfile
import tempfile
from datetime import datetime
from typing import Optional

from rayonix_wallet.core.exceptions import BackupError
from rayonix_wallet.crypto.encryption import EncryptionManager
from rayonix_wallet.utils.secure import SecureString
from rayonix_wallet.core.types import AddressInfo, Transaction, WalletState

class BackupManager:
    """Backup and restore functionality"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.encryption_manager = EncryptionManager(wallet.config)
    
    def backup(self, backup_path: str, passphrase: Optional[str] = None) -> bool:
        """Create encrypted wallet backup"""
        try:
            if not passphrase and self.wallet.config.passphrase:
                passphrase = self.wallet.config.passphrase
            
            if not passphrase:
                raise BackupError("Passphrase required for backup")
            
            # Create temporary directory for backup files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Export wallet data
                wallet_data = self._export_wallet_data()
                
                # Create metadata
                metadata = {
                    'wallet_id': self.wallet.wallet_id,
                    'backup_date': datetime.utcnow().isoformat(),
                    'version': '1.0',
                    'network': self.wallet.config.network,
                    'wallet_type': self.wallet.config.wallet_type.name,
                    'address_type': self.wallet.config.address_type.name
                }
                
                # Write files to temp directory
                with open(os.path.join(temp_dir, 'wallet_data.json'), 'w') as f:
                    json.dump(wallet_data, f, indent=2)
                
                with open(os.path.join(temp_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create encrypted zip archive
                self._create_encrypted_archive(temp_dir, backup_path, passphrase)
                
                return True
                
        except Exception as e:
            raise BackupError(f"Backup failed: {e}")
    
    def _export_wallet_data(self) -> dict:
        """Export all wallet data for backup"""
        return {
            'addresses': [asdict(addr) for addr in self.wallet.addresses.values()],
            'transactions': [asdict(tx) for tx in self.wallet.transactions.values()],
            'state': asdict(self.wallet.state),
            'config': asdict(self.wallet.config)
        }
    
    def _create_encrypted_archive(self, source_dir: str, output_path: str, passphrase: str):
        """Create encrypted zip archive"""
        salt = os.urandom(16)
        key = self.encryption_manager._derive_key_from_passphrase(passphrase, salt)
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add files to zip
            for filename in os.listdir(source_dir):
                file_path = os.path.join(source_dir, filename)
                if os.path.isfile(file_path):
                    with open(file_path, 'rb') as f:
                        data = f.read()
                    
                    # Encrypt file data
                    encrypted_data = self.encryption_manager.encrypt_data(data, key)
                    zipf.writestr(filename, encrypted_data)
            
            # Add encryption metadata
            metadata = {
                'salt': salt.hex(),
                'algorithm': 'AES-GCM',
                'kdf': 'PBKDF2-HMAC-SHA256',
                'iterations': 100000
            }
            zipf.writestr('encryption.json', json.dumps(metadata))
    
    def restore(self, backup_path: str, passphrase: str) -> bool:
        """Restore wallet from backup"""
        try:
            if not os.path.exists(backup_path):
                raise BackupError("Backup file not found")
            
            # Extract and decrypt backup
            with tempfile.TemporaryDirectory() as temp_dir:
                # Read encryption metadata
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    encryption_info = json.loads(zipf.read('encryption.json').decode())
                    salt = bytes.fromhex(encryption_info['salt'])
                
                # Derive decryption key
                key = self.encryption_manager._derive_key_from_passphrase(
                    passphrase, salt, encryption_info.get('iterations', 100000)
                )
                
                # Extract and decrypt files
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    for filename in zipf.namelist():
                        if filename != 'encryption.json':
                            encrypted_data = zipf.read(filename)
                            decrypted_data = self.encryption_manager.decrypt_data(encrypted_data, key)
                            
                            with open(os.path.join(temp_dir, filename), 'wb') as f:
                                f.write(decrypted_data)
                
                # Import wallet data
                self._import_wallet_data(temp_dir)
                
                return True
                
        except Exception as e:
            raise BackupError(f"Restore failed: {e}")
    
    def _import_wallet_data(self, source_dir: str):
        """Import wallet data from backup"""
        # Read wallet data
        with open(os.path.join(source_dir, 'wallet_data.json'), 'r') as f:
            wallet_data = json.load(f)
        
        # Read metadata
        with open(os.path.join(source_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        # Verify wallet compatibility
        if metadata['wallet_id'] != self.wallet.wallet_id:
            raise BackupError("Backup is for a different wallet")
        
        # Import addresses
        for addr_data in wallet_data['addresses']:
            address_info = AddressInfo(**addr_data)
            self.wallet.db.save_address(address_info)
        
        # Import transactions
        for tx_data in wallet_data['transactions']:
            transaction = Transaction(**tx_data)
            self.wallet.db.save_transaction(transaction)
        
        # Import state
        state = WalletState(**wallet_data['state'])
        self.wallet.db.save_wallet_state(state)
        
        # Update in-memory state
        self.wallet._load_wallet_state()
    
    def export_private_keys(self, passphrase: str) -> dict:
        """Export all private keys (for migration purposes)"""
        if self.wallet.locked:
            raise BackupError("Wallet must be unlocked")
        
        try:
            private_keys = {}
            for address, info in self.wallet.addresses.items():
                if info.derivation_path != 'imported':
                    priv_key = self.wallet.key_manager.export_private_key(address, info.derivation_path)
                    private_keys[address] = priv_key
            
            # Encrypt the private keys
            encrypted_data = self.encryption_manager.encrypt_with_passphrase(
                json.dumps(private_keys).encode(), passphrase
            )
            
            return encrypted_data
            
        except Exception as e:
            raise BackupError(f"Private key export failed: {e}")
    
    def verify_backup(self, backup_path: str, passphrase: str) -> bool:
        """Verify backup integrity"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try to extract and decrypt
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    encryption_info = json.loads(zipf.read('encryption.json').decode())
                    salt = bytes.fromhex(encryption_info['salt'])
                
                key = self.encryption_manager._derive_key_from_passphrase(
                    passphrase, salt, encryption_info.get('iterations', 100000)
                )
                
                with zipfile.ZipFile(backup_path, 'r') as zipf:
                    for filename in zipf.namelist():
                        if filename != 'encryption.json':
                            encrypted_data = zipf.read(filename)
                            self.encryption_manager.decrypt_data(encrypted_data, key)
                
                return True
                
        except Exception:
            return False