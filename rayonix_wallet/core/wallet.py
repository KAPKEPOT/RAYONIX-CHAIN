import threading
import time
import gc
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import asdict

from rayonix_wallet.core.config import WalletConfig
from rayonix_wallet.core.types import WalletType, SecureKeyPair, Transaction, AddressInfo, WalletBalance, WalletState
from rayonix_wallet.core.exceptions import WalletError
from rayonix_wallet.crypto.key_management import KeyManager
from rayonix_wallet.crypto.address import AddressDerivation
from rayonix_wallet.storage.wallet_database import WalletDatabaseAdapter
from database.core.database import AdvancedDatabase
from rayonix_wallet.storage.backup import BackupManager
from rayonix_wallet.services.synchronizer import WalletSynchronizer
from rayonix_wallet.services.transaction import TransactionManager
from rayonix_wallet.services.balance import BalanceCalculator
from rayonix_wallet.services.multisig import MultisigManager
from rayonix_wallet.utils.validation import validate_address_format
from rayonix_wallet.utils.qr_code import generate_qr_code
from rayonix_wallet.interfaces.blockchain import BlockchainInterface
from rayonix_wallet.utils.logging import logger
from rayonix_wallet.core.config import WalletConfig
from rayonix_wallet.utils.secure import SecureString

class RayonixWallet:
    """Advanced cryptographic wallet with enterprise-grade features"""
    
    def __init__(self, config: Optional[WalletConfig] = None, wallet_id: Optional[str] = None):
        self.config = config or WalletConfig()
        self.wallet_id = wallet_id or self._generate_wallet_id()
        
        # Initialize components
        self.db = WalletDatabaseAdapter(AdvancedDatabase(self.config.db_path))
        self.key_manager = KeyManager(self.config)
        self.address_derivation = AddressDerivation(self.config)
        self.transaction_manager = TransactionManager(self)
        self.balance_calculator = BalanceCalculator(self)
        self.multisig_manager = MultisigManager(self)
        self.backup_manager = BackupManager(self)
        self.synchronizer = WalletSynchronizer(self)
        
        # Wallet state
        self.master_key = None
        self._creation_mnemonic = None
        
        # Load state
        self._load_wallet_state()
        
        # Security state
        self.locked = True
        self.failed_attempts = 0
        self.lock_time = None
        
        # Performance
        self.running = False
        self.background_thread = None
        
        logger.info(f"Wallet initialized with ID: {self.wallet_id}")
    
    def is_initialized(self) -> bool:
        """Check if wallet is properly initialized with keys"""
        return self.master_key is not None
    
    def initialize_new_wallet(self) -> str:
        """Initialize a new HD wallet and return the mnemonic"""
        if self.is_initialized():
            raise WalletError("Wallet is already initialized")
        
        mnemonic, wallet_id = self.create_hd_wallet()
        return mnemonic
    
    def _generate_wallet_id(self) -> str:
        """Generate unique wallet ID"""
        import hashlib
        import secrets
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:16]
    
    def _load_wallet_state(self):
        """Load wallet state from database"""
        self.state = self.db.get_wallet_state() or WalletState(
            sync_height=0, last_updated=time.time(), tx_count=0,
            addresses_generated=0, addresses_used=0, total_received=0,
            total_sent=0, security_score=0
        )
        
        self.addresses = {addr.address: addr for addr in self.db.get_all_addresses()}
        self.transactions = {tx.txid: tx for tx in self.db.get_transactions(limit=1000)}
    
    def set_blockchain_reference(self, rayonix_coin_instance: Any) -> bool:
        """Set reference to blockchain instance"""
        try:
            if not hasattr(rayonix_coin_instance, 'utxo_set') or not hasattr(rayonix_coin_instance, 'get_balance'):
                logger.error("Invalid blockchain reference provided")
                return False
            self.rayonix_chain = rayonix_coin_instance
            logger.info("Blockchain reference set successfully")
            
            # Start synchronizer if we have blockchain reference
            self.synchronizer.start()
            return True
        except Exception as e:
            logger.error(f"Failed to set blockchain reference: {e}")
            return False
    
    def create_from_mnemonic(self, mnemonic_phrase: str, passphrase: str = "") -> bool:
        """Create wallet from BIP39 mnemonic phrase"""
        try:
            # Validate mnemonic
            if not self._validate_mnemonic(mnemonic_phrase):
                raise ValueError("Invalid mnemonic phrase")
            
            # Generate seed from mnemonic
            seed = self._mnemonic_to_seed(mnemonic_phrase, passphrase)
            
            # Generate master key from seed using BIP32
            from bip32 import BIP32
            bip32 = BIP32.from_seed(seed)
            
            # Get private key using proper method
            private_key_bytes = self._get_private_key_from_bip32(bip32)
            
            # Create secure key pair
            private_key_secure = SecureString(private_key_bytes)
            
            # Get public key
            public_key_bytes = bip32.get_pubkey_from_path("m")
            
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=public_key_bytes,
                chain_code=bip32.chain_code,
                depth=0,
                index=0,
                parent_fingerprint=b'\x00\x00\x00\x00'
            )
            
            # Store mnemonic securely
            self._creation_mnemonic = SecureString(mnemonic_phrase.encode())
            
            # Generate initial addresses
            self._generate_initial_addresses()
            logger.info("Wallet created from mnemonic successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create wallet from mnemonic: {e}")
            return False

    def _get_private_key_from_bip32(self, bip32_obj) -> bytes:
        """Extract private key from BIP32 object using multiple methods"""
        try:
            # Method 1: Try get_privkey_from_path
            return bip32_obj.get_privkey_from_path("m")
        except Exception:
            try:
                # Method 2: Try get_private_key
                return bip32_obj.get_private_key()
            except Exception:
                try:
                    # Method 3: Try accessing internal attributes
                    if hasattr(bip32_obj, '_privkey'):
                        return bip32_obj._privkey
                    elif hasattr(bip32_obj, 'private_key'):
                        return bip32_obj.private_key
                    else:
                        # Method 4: Derive from extended private key
                        xprv = bip32_obj.get_xpriv()
                        # Extract private key from xprv (remove version bytes and chain code)
                        return bytes.fromhex(xprv[26:90])  # This is implementation-specific
                except Exception as e:
                    raise WalletError(f"Cannot extract private key from BIP32 object: {e}")

    def create_from_private_key(self, private_key: str, wallet_type: WalletType = WalletType.NON_HD) -> bool:
        """Create wallet from private key"""
        try:
            # Decode private key
            priv_key_bytes = self._decode_private_key(private_key)
            
            # Create secure key pair
            private_key_secure = SecureString(priv_key_bytes)
            public_key = self._private_to_public(priv_key_bytes)
            
            key_pair = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=public_key
            )
            
            # Store in memory (for non-HD wallets, we don't use derivation paths)
            self.master_key = key_pair
            
            # Generate address
            address = self._derive_address(public_key, 0, False)
            address_info = AddressInfo(
                address=address,
                index=0,
                derivation_path='m/0/0',
                balance=0,
                received=0,
                sent=0,
                tx_count=0,
                is_used=False,
                is_change=False
            )
            
            # Save to database
            self.db.save_address(address_info)
            self.addresses[address] = address_info
            
            self.config.wallet_type = wallet_type
            logger.info("Wallet created from private key successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create wallet from private key: {e}")
            return False
    
    def create_hd_wallet(self) -> Tuple[str, str]:
        """Create new HD wallet with mnemonic"""
        try:
            # Generate mnemonic
            mnemonic_phrase = self._generate_mnemonic()
            
            # Generate seed
            seed = self._mnemonic_to_seed(mnemonic_phrase, "")
            
            # Generate master key using BIP32 library
            from bip32 import BIP32
            bip32 = BIP32.from_seed(seed)
            
            # Get private key using proper method
            private_key_bytes = self._get_private_key_from_bip32(bip32)
            
            # Create secure key pair
            private_key_secure = SecureString(private_key_bytes)
            
            # Get public key
            public_key_bytes = bip32.get_pubkey_from_path("m")
            
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=public_key_bytes,
                chain_code=bip32.chain_code,
                depth=0,
                index=0,
                parent_fingerprint=b'\x00\x00\x00\x00'
            )
            
            # Store mnemonic securely
            self._creation_mnemonic = SecureString(mnemonic_phrase.encode())
            
            # Generate initial addresses
            self._generate_initial_addresses()
            
            logger.info("HD wallet created successfully")
            return mnemonic_phrase, self.wallet_id
            
        except Exception as e:
            logger.error(f"Failed to create HD wallet: {e}")
            raise
    
    def _generate_mnemonic(self) -> str:
        """Generate BIP39 mnemonic phrase"""
        try:
            import bip39
            return bip39.generate_mnemonic()
        except ImportError:
            # Fallback implementation
            import hashlib
            import secrets
            entropy = secrets.token_bytes(32)
            return hashlib.sha256(entropy).hexdigest()[:32]  # Simplified
    
    def _mnemonic_to_seed(self, mnemonic: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed using BIP39"""
        try:
            import bip39
            return bip39.mnemonic_to_seed(mnemonic, passphrase)
        except ImportError:
            # Fallback implementation
            import hashlib
            return hashlib.pbkdf2_hmac('sha512', mnemonic.encode(), 
                                     f'mnemonic{passphrase}'.encode(), 2048)
    
    def _validate_mnemonic(self, mnemonic: str) -> bool:
        """Validate BIP39 mnemonic"""
        try:
            import bip39
            return bip39.validate_mnemonic(mnemonic)
        except ImportError:
            # Basic validation fallback
            words = mnemonic.split()
            return len(words) in [12, 15, 18, 21, 24]
    
    def _decode_private_key(self, private_key: str) -> bytes:
        """Decode private key from various formats"""
        # Implementation depends on your key format
        try:
            import base58
            return base58.b58decode(private_key)
        except:
            return private_key.encode()
    
    def _private_to_public(self, private_key: bytes) -> bytes:
        """Convert private key to public key"""
        # Implementation depends on your cryptography library
        try:
            from ecdsa import SigningKey, SECP256k1
            sk = SigningKey.from_string(private_key, curve=SECP256k1)
            return sk.verifying_key.to_string()
        except ImportError:
            return private_key  # Simplified fallback
    
    def _derive_address(self, public_key: bytes, index: int, is_change: bool) -> str:
        """Derive address from public key"""
        # Implementation depends on your address format
        try:
            import hashlib
            import base58
            # Simple implementation for demonstration
            hash1 = hashlib.sha256(public_key).digest()
            hash2 = hashlib.new('ripemd160', hash1).digest()
            return base58.b58encode_check(hash2).decode()
        except:
            return f"addr_{index}_{int(is_change)}_{public_key.hex()[:16]}"
    
    def _generate_initial_addresses(self):
        """Generate initial set of addresses"""
        for i in range(self.config.gap_limit):
            self.derive_address(i, False)
            self.derive_address(i, True)
        logger.info(f"Generated {self.config.gap_limit * 2} initial addresses")
    
    def derive_address(self, index: int, is_change: bool = False) -> str:
        """Derive address at specified index"""
        if not self.master_key:
            raise ValueError("Wallet is locked or not initialized")
        
        # Use BIP32 library for proper derivation
        from bip32 import BIP32
        
        # Create BIP32 object from the master private key
        bip32 = BIP32.from_seed(self.master_key.private_key)
        
        # BIP44 derivation path: m/purpose'/coin_type'/account'/change/address_index
        change_index = 1 if is_change else 0
        derivation_path = f"m/44'/0'/{self.config.account_index}'/{change_index}/{index}"
        
        try:
            # Derive child key
            derived_public_key = bip32.get_pubkey_from_path(derivation_path)
            
            # Generate address
            address = self._derive_address(derived_public_key, index, is_change)
            
            # Create address info
            address_info = AddressInfo(
                address=address,
                index=index,
                derivation_path=derivation_path,
                balance=0,
                received=0,
                sent=0,
                tx_count=0,
                is_used=False,
                is_change=is_change
            )
            
            # Save to database
            self.db.save_address(address_info)
            self.addresses[address] = address_info
            
            # Update wallet state
            self.state.addresses_generated += 1
            self.db.save_wallet_state(self.state)
            
            return address
            
        except Exception as e:
            logger.error(f"Failed to derive address: {e}")
            raise
    
    def _save_address_info(self, address: str, index: int, derivation_path: str, is_change: bool):
        """Save address information to database"""
        address_info = AddressInfo(
            address=address,
            index=index,
            derivation_path=derivation_path,
            balance=0,
            received=0,
            sent=0,
            tx_count=0,
            is_used=False,
            is_change=is_change
        )
        
        self.db.save_address(address_info)
        self.addresses[address] = address_info
        self.state.addresses_generated += 1
        self.db.save_wallet_state(self.state)
    
    def unlock(self, passphrase: str, timeout: Optional[int] = None) -> bool:
        """Unlock wallet with passphrase"""
        try:
            if not self.locked:
                return True
            
            if self.failed_attempts >= 5:
                lockout_time = 300
                if self.lock_time and time.time() - self.lock_time < lockout_time:
                    remaining = lockout_time - (time.time() - self.lock_time)
                    logger.warning(f"Wallet locked due to too many failed attempts. Try again in {int(remaining)} seconds")
                    return False
            
            if self.key_manager.verify_passphrase(passphrase):
                self.locked = False
                self.failed_attempts = 0
                self.lock_time = None
                logger.info("Wallet unlocked successfully")
                return True
            else:
                self.failed_attempts += 1
                self.lock_time = time.time()
                logger.warning(f"Failed unlock attempt {self.failed_attempts}")
                return False
        except Exception as e:
            logger.error(f"Unlock failed: {e}")
            self.failed_attempts += 1
            self.lock_time = time.time()
            return False
    
    def lock(self):
        """Lock wallet and clear sensitive data from memory"""
        try:
            self.key_manager.wipe()
            self.locked = True
            gc.collect()
            logger.info("Wallet locked successfully")
        except Exception as e:
            logger.error(f"Lock failed: {e}")
    
    def get_balance(self, address: Optional[str] = None, force_refresh: bool = False) -> WalletBalance:
        """Get wallet balance"""
        return self.balance_calculator.get_balance(address, force_refresh)
    
    def get_transaction_history(self, limit: int = 50, offset: int = 0) -> List[Transaction]:
        """Get transaction history"""
        return self.db.get_transactions(limit, offset)
    
    def send_transaction(self, to_address: str, amount: int, fee_rate: Optional[int] = None, 
                        memo: Optional[str] = None) -> Optional[str]:
        """Send transaction"""
        if self.locked:
            raise WalletError("Wallet is locked")
        
        return self.transaction_manager.send_transaction(to_address, amount, fee_rate, memo)
    
    def backup(self, backup_path: str) -> bool:
        """Backup wallet to encrypted file"""
        if self.locked:
            raise WalletError("Wallet must be unlocked for backup")
        
        return self.backup_manager.backup(backup_path)
    
    def restore(self, backup_path: str, passphrase: str) -> bool:
        """Restore wallet from backup"""
        return self.backup_manager.restore(backup_path, passphrase)
    
    def export_private_key(self, address: str) -> Optional[str]:
        """Export private key for address"""
        if self.locked:
            raise WalletError("Wallet is locked")
        
        if address not in self.addresses:
            raise WalletError("Address not found in wallet")
        
        return self.key_manager.export_private_key(address, self.addresses[address].derivation_path)
    
    def import_address(self, address: str, label: Optional[str] = None) -> bool:
        """Import watch-only address"""
        if address in self.addresses:
            return False
        
        address_info = AddressInfo(
            address=address,
            index=-1,
            derivation_path='imported',
            balance=0,
            received=0,
            sent=0,
            tx_count=0,
            is_used=False,
            is_change=False,
            labels=[label] if label else []
        )
        
        self.db.save_address(address_info)
        self.addresses[address] = address_info
        return True
    
    def get_qr_code(self, address: str, amount: Optional[int] = None, 
                   memo: Optional[str] = None) -> Optional[bytes]:
        """Generate QR code for address or payment request"""
        return generate_qr_code(address, amount, memo, self.config.network)
    
    def validate_address(self, address: str) -> bool:
        """Validate cryptocurrency address"""
        return validate_address_format(address, self.config.address_type, self.config.network)
    
    def set_multisig(self, required: int, public_keys: List[str]) -> bool:
        """Setup multi-signature wallet"""
        return self.multisig_manager.set_multisig(required, public_keys)
    
    def add_cosigner(self, public_key: str) -> bool:
        """Add cosigner to multisig wallet"""
        return self.multisig_manager.add_cosigner(public_key)
    
    def create_multisig_transaction(self, to_address: str, amount: int, 
                                  fee_rate: Optional[int] = None) -> Dict:
        """Create multisig transaction requiring multiple signatures"""
        return self.multisig_manager.create_multisig_transaction(to_address, amount, fee_rate)
    
    def sign_multisig_transaction(self, transaction: Dict) -> Dict:
        """Sign multisig transaction"""
        return self.multisig_manager.sign_multisig_transaction(transaction)
    
    def finalize_multisig_transaction(self, transaction: Dict) -> Optional[str]:
        """Finalize multisig transaction with required signatures"""
        return self.multisig_manager.finalize_multisig_transaction(transaction)
    
    def get_transaction_fee_estimate(self, priority: str = "medium") -> int:
        """Get transaction fee estimate"""
        return self.transaction_manager.get_fee_estimate(priority)
    
    def sweep_private_key(self, private_key: str, to_address: str, 
                         fee_rate: Optional[int] = None) -> Optional[str]:
        """Sweep funds from private key to wallet address"""
        return self.transaction_manager.sweep_private_key(private_key, to_address, fee_rate)
    
    def get_primary_address(self) -> Optional[str]:
        """Get primary address for this wallet"""
        if not self.addresses:
            return None
        return list(self.addresses.keys())[0]
    
    def close(self):
        """Cleanly close wallet"""
        try:
            self.running = False
            self.synchronizer.stop()
            self.lock()
            self.db.close()
            logger.info("Wallet closed successfully")
        except Exception as e:
            logger.error(f"Error closing wallet: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()