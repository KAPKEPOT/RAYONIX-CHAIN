#rayonix_wallet/core/wallet.py
import threading
import time
import gc
import json
import struct
import hashlib
import secrets
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import asdict
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidKey, InvalidTag
from cryptography.hazmat.primitives import serialization
from rayonix_wallet.core.wallet_types import WalletType, SecureKeyPair, Transaction, AddressInfo, WalletBalance, WalletState
from rayonix_wallet.core.exceptions import WalletError, CryptoError, InvalidAddressError
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
from rayonix_wallet.recovery.recovery_manager import RecoveryManager

class ProductionRayonixWallet:
    
    def __init__(self, config: Optional[WalletConfig] = None, wallet_id: Optional[str] = None):
        self.config = config or WalletConfig()
        
        # INITIALIZE CRYPTO BACKEND FIRST
        self._crypto_backend = default_backend()
        self.wallet_id = wallet_id or self._generate_cryptographic_wallet_id()
        
        # Initialize production-grade components
        self.db = WalletDatabaseAdapter(AdvancedDatabase(self.config.db_path))
        self.key_manager = KeyManager(self.config)
        self.address_derivation = AddressDerivation(self.config)
        self.transaction_manager = TransactionManager(self)
        self.balance_calculator = BalanceCalculator(self)
        self.multisig_manager = MultisigManager(self)
        self.backup_manager = BackupManager(self)
        self.synchronizer = WalletSynchronizer(self)
        
        # Cryptographic state management
        self.master_key: Optional[SecureKeyPair] = None
        self._creation_mnemonic: Optional[SecureString] = None
        self._key_derivation_cache: Dict[str, bytes] = {}
        self._address_validation_cache: Dict[str, bool] = {}
        
        # Advanced security state
        self.locked = True
        self.failed_attempts = 0
        self.lock_time: Optional[float] = None
        self.last_successful_auth: Optional[float] = None
        self.auth_attempt_history: List[float] = []
        self.security_events: List[Dict[str, Any]] = []
        
        # Performance and concurrency
        self.running = False
        self.background_thread: Optional[threading.Thread] = None
        self._state_lock = threading.RLock()
        self._aes_gcm_key: Optional[bytes] = None
        
        # Cryptographic context
        self._crypto_context = self._initialize_cryptographic_context()
        
        # Load wallet state with comprehensive validation
        self.recovery_manager = RecoveryManager(self)
        
        try:
        	# Try normal loading first
        	self._load_wallet_state_with_cryptographic_validation()
        except Exception as e:
        	logger.error(f"Wallet initialization failed: {e}")
        	
        	# Automatic recovery without blockchain first
        	recovery_result = self.recovery_manager.auto_recover()
        	if recovery_result['success']:
        		logger.info("Wallet recovered successfully via automatic recovery")
        	else:
        		logger.critical("Automatic recovery failed")
        		# We'll still continue but wallet might need manual recovery
        
        logger.info(f"RAYONIX wallet initialized with ID: {self.wallet_id}")
    
    def _generate_cryptographic_wallet_id(self) -> str:
        """Generate cryptographically secure wallet ID with domain separation"""
        # Generate high-entropy random bytes
        entropy = secrets.token_bytes(64)
        
        # Use HKDF for cryptographic separation
        hkdf = HKDF(
            algorithm=hashes.SHA512(),
            length=32,
            salt=secrets.token_bytes(32),
            #salt=secrets.token_bytes(32),
            info=b'rayonix-wallet-id-generation-v2',
            backend=self._crypto_backend
        )
        
        wallet_id_bytes = hkdf.derive(entropy + struct.pack('>Q', int(time.time())))
        return wallet_id_bytes.hex()
    
    def _initialize_cryptographic_context(self) -> bytes:
        """Initialize cryptographic context for deterministic operations"""
        context_data = (
            self.wallet_id.encode() +
            struct.pack('>I', int(time.time())) +
            secrets.token_bytes(16)
        )
        
        return hashlib.sha3_512(context_data).digest()
    
    def _load_wallet_state_with_cryptographic_validation(self):
        """Load wallet state with comprehensive cryptographic validation"""
        with self._state_lock:
            try:
                # Attempt database repair first
                self._attempt_database_repair()
                
                # Try normal loading
                self.state = self._load_validated_wallet_state()
                self.addresses = self._load_validated_addresses()
                self.transactions = self._load_validated_transactions()
                self._initialize_cryptographic_components()
                
                logger.info("Wallet state loaded successfully")
                
            except Exception as e:
                logger.error(f"Wallet state loading failed: {e}")
                # CRITICAL: Only recreate STATE, not wallet
                logger.warning("Recreating wallet STATE metadata (wallet keys are preserved)")
                self._recreate_wallet_state_metadata()
                
                # Log but don't re-raise - allow system to continue
                self._log_security_event("state_metadata_recovery", f"Recovered state from: {str(e)}")
                
    def _attempt_database_repair(self):
    	"""Attempt to repair database corruption"""
    	try:
    		if hasattr(self.db, 'repair_corrupted_entries'):
    			stats = self.db.repair_corrupted_entries()
    			if stats.get('removed_corrupted', 0) > 0:
    				logger.info(f"Repaired database: removed {stats['removed_corrupted']} corrupted entries")
    	
    	except Exception as e:
    		logger.warning(f"Database repair attempt failed: {e}")
    
    def _recreate_wallet_state_metadata(self):
    	"""Recreate wallet STATE metadata without affecting cryptographic material"""
    	try:
    		# Get existing addresses to preserve them
    		existing_addresses = self._get_existing_addresses_safely()
    		
    		# Create fresh STATE but preserve addresses
    		self.state = WalletState(
    		    sync_height=0,  # Will need to rescan
    		    last_updated=time.time(),
    		    tx_count=len(self.transactions) if hasattr(self, 'transactions') else 0,
    		    addresses_generated=len(existing_addresses),
    		    addresses_used=self._count_used_addresses(existing_addresses),
    		    total_received=0,  # Will be recalculated
    		    total_sent=0,      # Will be recalculated
    		    security_score=self._calculate_current_security_score()
    		)
    		
    		# Save the state metadata
    		self.db.save_wallet_state(self.state)
    		# Preserve existing addresses
    		self.addresses = existing_addresses
    		
    		logger.info("Wallet STATE metadata recreated - cryptographic material preserved")
    	
    	except Exception as e:
    		logger.error(f"Failed to recreate wallet state: {e}")
    		# If this fails, we have bigger problems
    		raise WalletError("Critical wallet state failure")
    
    def _get_existing_addresses_safely(self) -> Dict[str, AddressInfo]:
    	"""Safely get existing addresses without relying on corrupted state"""
    	addresses = {}
    	
    	try:
    		# Try to load from database directly
    		raw_addresses = self.db.get_all_addresses()
    		for addr_data in raw_addresses:
    			try:
    				if isinstance(addr_data, dict):
    					address_info = AddressInfo(**addr_data)
    				else:
    					address_info = addr_data
    				
    				# Basic validation that this is a real address
    				if (hasattr(address_info, 'address') and hasattr(address_info, 'derivation_path')):
    					addresses[address_info.address] = address_info
    			
    			except Exception as e:
    				logger.warning(f"Skipping invalid address entry: {e}")
    				continue
    	except Exception as e:
    		logger.error(f"Failed to load existing addresses: {e}")
    	return addresses
    	
    def _load_validated_wallet_state(self) -> WalletState:
        """Load and cryptographically validate wallet state"""
        state_data = self.db.get_wallet_state()
        
        if not state_data:
            # Initialize new state with cryptographic parameters
            return WalletState(
                sync_height=0,
                last_updated=time.time(),
                tx_count=0,
                addresses_generated=0,
                addresses_used=0,
                total_received=0,
                total_sent=0,
                security_score=self._calculate_initial_security_score()
            )
        
        # Validate state integrity
        if not self._validate_wallet_state_integrity(state_data):
            raise WalletError("Wallet state integrity validation failed")
        
        return state_data
    
    def _validate_wallet_state_integrity(self, state: WalletState) -> bool:
        """Cryptographically validate wallet state integrity"""
        try:
            # Check for reasonable values
            if state.addresses_generated < 0 or state.tx_count < 0:
                return False
            
            if state.total_received < 0 or state.total_sent < 0:
                return False
            
            # Validate security score range
            if not (0 <= state.security_score <= 100):
                return False
            
            # Additional cryptographic checks could be added here
            # such as digital signatures of the state
            
            return True
            
        except Exception:
            return False
    
    def _calculate_initial_security_score(self) -> int:
        """Calculate initial security score based on configuration"""
        score = 0
        
        # Encryption enabled
        if self.config.encryption:
            score += 20
        
        # Strong passphrase required
        if self.config.passphrase and len(self.config.passphrase) >= 12:
            score += 15
        
        # Auto-backup enabled
        if self.config.auto_backup:
            score += 10
        
        # HD wallet
        if self.config.wallet_type == WalletType.HD:
            score += 25
        
        # Additional security features
        score += 30  # Base security
        
        return min(score, 100)
    
    def _load_validated_addresses(self) -> Dict[str, AddressInfo]:
        """Load and cryptographically validate addresses"""
        addresses_data = self.db.get_all_addresses()
        validated_addresses = {}
        
        for addr_data in addresses_data:
            try:
                # Convert to AddressInfo with validation
                if isinstance(addr_data, dict):
                    address_info = self._validate_address_info_cryptographic(addr_data)
                else:
                    address_info = self._validate_address_info_cryptographic(asdict(addr_data))
                
                validated_addresses[address_info.address] = address_info
                
            except Exception as e:
                logger.warning(f"Failed to validate address data: {e}")
                continue
        
        return validated_addresses
    
    def _validate_address_info_cryptographic(self, addr_data: Dict) -> AddressInfo:
        """Cryptographically validate address information"""
        try:
            address_info = AddressInfo(**addr_data)
            
            # Validate address format cryptographically
            if not self._validate_address_cryptographic(address_info.address):
                raise WalletError(f"Cryptographic address validation failed: {address_info.address}")
            
            # Validate derivation path
            if not self._validate_derivation_path_cryptographic(address_info.derivation_path):
                raise WalletError(f"Invalid derivation path: {address_info.derivation_path}")
            
            # Validate numeric ranges
            if address_info.balance < 0 or address_info.received < 0 or address_info.sent < 0:
                raise WalletError("Invalid address amounts")
            
            return address_info
            
        except Exception as e:
            raise WalletError(f"Address info cryptographic validation failed: {e}")
    
    def _validate_address_cryptographic(self, address: str) -> bool:
        """Cryptographic address validation"""
        try:
            return self.address_derivation.validate_address(
                address, self.config.address_type, self.config.network
            )
        except Exception:
            return False
    
    def _validate_derivation_path_cryptographic(self, path: str) -> bool:
        """Cryptographic derivation path validation - FIXED VERSION"""
        if not path.startswith('m/'):
            return False
        
        parts = path.split('/')[1:]
        for part in parts:
            if not self._validate_path_component_cryptographic(part):
                return False
        
        return True
    
    def _validate_path_component_cryptographic(self, component: str) -> bool:
        """Validate derivation path component cryptographically - FIXED"""
        # Handle hardened notation (both ' and h are valid)
        is_hardened = component.endswith("'") or component.endswith("h")
        if is_hardened:
            component = component[:-1]
        
        # Must be numeric
        if not component.isdigit():
            return False
        
        # Check BIP32 range
        try:
            index = int(component)
            if is_hardened:
                return 0 <= index <= 0x7FFFFFFF  # Hardened key range
            else:
                return 0 <= index <= 0xFFFFFFFF  # Normal key range
        except (ValueError, OverflowError):
            return False
    
    def _load_validated_transactions(self) -> Dict[str, Transaction]:
        """Load and cryptographically validate transactions"""
        transactions_data = self.db.get_transactions(limit=1000)
        validated_transactions = {}
        
        for tx_data in transactions_data:
            try:
                if isinstance(tx_data, dict):
                    transaction = self._validate_transaction_cryptographic(tx_data)
                else:
                    transaction = self._validate_transaction_cryptographic(asdict(tx_data))
                
                validated_transactions[transaction.txid] = transaction
                
            except Exception as e:
                logger.warning(f"Failed to validate transaction: {e}")
                continue
        
        return validated_transactions
    
    def _validate_transaction_cryptographic(self, tx_data: Dict) -> Transaction:
        """Cryptographically validate transaction data"""
        try:
            transaction = Transaction(**tx_data)
            
            # Validate transaction ID format
            if not self._validate_transaction_id_cryptographic(transaction.txid):
                raise WalletError(f"Invalid transaction ID: {transaction.txid}")
            
            # Validate addresses
            if not self._validate_address_cryptographic(transaction.from_address):
                raise WalletError(f"Invalid from address: {transaction.from_address}")
            
            if not self._validate_address_cryptographic(transaction.to_address):
                raise WalletError(f"Invalid to address: {transaction.to_address}")
            
            # Validate amounts
            if transaction.amount < 0 or transaction.fee < 0:
                raise WalletError("Invalid transaction amounts")
            
            # Validate status
            if transaction.status not in ['pending', 'confirmed', 'failed']:
                raise WalletError(f"Invalid transaction status: {transaction.status}")
            
            return transaction
            
        except Exception as e:
            raise WalletError(f"Transaction cryptographic validation failed: {e}")
    
    def _validate_transaction_id_cryptographic(self, txid: str) -> bool:
        """Cryptographic transaction ID validation"""
        if not isinstance(txid, str) or len(txid) != 64:
            return False
        
        try:
            # Must be valid hex
            bytes.fromhex(txid)
            
            # Additional cryptographic checks could be added here
            # such as verifying the transaction ID follows expected patterns
            
            return True
            
        except ValueError:
            return False
    
    def _initialize_cryptographic_components(self):
        """Initialize cryptographic components for wallet operations"""
        try:
            # Generate AES-GCM key for sensitive data encryption
            self._aes_gcm_key = self._generate_aes_gcm_key()
            
            # Initialize cryptographic RNG state
            self._initialize_cryptographic_rng()
            
            # Set up cryptographic event logging
            self._setup_cryptographic_logging()
            
        except Exception as e:
            logger.error(f"Cryptographic component initialization failed: {e}")
            raise
    
    def _generate_aes_gcm_key(self) -> bytes:
        """Generate AES-GCM key for sensitive data encryption"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=secrets.token_bytes(16),
            info=b'rayonix-wallet-aes-gcm-key',
            backend=self._crypto_backend
        )
        
        return hkdf.derive(self._crypto_context + secrets.token_bytes(32))
    
    def _initialize_cryptographic_rng(self):
        """Initialize cryptographic random number generator state"""
        # Seed Python's random module with cryptographic entropy
        import random
        random.seed(secrets.token_bytes(32))
        
        # Additional RNG initialization if needed
        pass
    
    def _setup_cryptographic_logging(self):
        """Set up cryptographic event logging"""
        self.security_events = []
        
        # Log wallet initialization event
        self._log_security_event(
            "wallet_initialized",
            f"Wallet {self.wallet_id} initialized with cryptographic components"
        )
    
    def _log_security_event(self, event_type: str, details: str):
        """Log security event with cryptographic timestamp"""
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details,
            'wallet_id': self.wallet_id,
            'session_id': hashlib.sha256(secrets.token_bytes(16)).hexdigest()[:16]
        }
        
        self.security_events.append(event)
        
        # Trim event history
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-500:]
    
    def create_from_mnemonic(self, mnemonic_phrase: str, passphrase: str = "") -> bool:
        """Create wallet from BIP39 mnemonic with complete cryptographic implementation"""
        if self.is_initialized():
            raise WalletError("Wallet is already initialized")
        
        with self._state_lock:
            try:
                # Comprehensive mnemonic cryptographic validation
                if not self._validate_mnemonic_cryptographic(mnemonic_phrase):
                    raise CryptoError("Mnemonic validation failed")
                
                # Use KeyManager for all cryptographic operations
                if not self.key_manager.initialize_from_mnemonic(mnemonic_phrase, passphrase):
                    raise CryptoError("Key manager initialization failed")
                
                # Get master key from KeyManager
                self.master_key = self.key_manager.get_master_key()
                if not self.master_key:
                    raise CryptoError("Failed to get master key from KeyManager")
                
                # Generate initial addresses through proper delegation
                self._generate_addresses_cryptographic()
                
                # Secure mnemonic storage with cryptographic protection
                self._store_mnemonic_cryptographic(mnemonic_phrase, passphrase)
                
                # Update security state
                self._update_security_state_after_creation()
                
                logger.info("Wallet created from mnemonic with complete cryptographic implementation")
                self._log_security_event("wallet_created", "New wallet created from mnemonic")
                
                return True
                
            except Exception as e:
                logger.error(f"Cryptographic wallet creation failed: {e}")
                self._log_security_event("creation_failed", str(e))
                self._secure_cryptographic_cleanup()
                return False
    
    def _validate_mnemonic_cryptographic(self, mnemonic: str) -> bool:
        """Cryptographic mnemonic validation with entropy analysis"""
        try:
            from mnemonic import Mnemonic
            
            mnemo = Mnemonic("english")
            
            # Basic BIP39 validation
            if not mnemo.check(mnemonic):
                return False
            
            # Entropy analysis
            entropy = mnemo.to_entropy(mnemonic)
            if len(entropy) < 16:  # 128-bit minimum
                return False
            
            # Calculate actual entropy bits
            entropy_bits = len(entropy) * 8
            if entropy_bits < 128:
                return False
            
            # Check for common patterns and weak mnemonics
            if self._is_weak_mnemonic(mnemonic):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_weak_mnemonic(self, mnemonic: str) -> bool:
        """Detect weak or predictable mnemonics"""
        words = mnemonic.split()
        
        # Check for repeated words
        if len(set(words)) < len(words) * 0.7:  # 70% unique words minimum
            return True
        
        # Check for sequential patterns
        word_indices = [self._get_word_index(word) for word in words if self._get_word_index(word) is not None]
        if len(word_indices) == len(words):
            # Check for sequential indices
            for i in range(1, len(word_indices)):
                if word_indices[i] == word_indices[i-1] + 1:
                    return True
        
        # Additional weakness checks could be added here
        
        return False
    
    def _get_word_index(self, word: str) -> Optional[int]:
        """Get BIP39 word list index for pattern detection"""
        try:
            from mnemonic import Mnemonic
            mnemo = Mnemonic("english")
            word_list = mnemo.wordlist
            return word_list.index(word) if word in word_list else None
        except Exception:
            return None
    
    def _generate_addresses_cryptographic(self):
        """Generate addresses with cryptographic derivation through proper delegation"""
        with self._state_lock:
            try:
                # Generate receiving addresses
                for i in range(self.config.gap_limit):
                    receiving_address = self.derive_address_cryptographic(i, False)
                    
                    # Generate change addresses
                    change_address = self.derive_address_cryptographic(i, True)
                
                logger.info(f"Generated {self.config.gap_limit * 2} cryptographic addresses")
                
            except Exception as e:
                logger.error(f"Cryptographic address generation failed: {e}")
                raise
    
    def derive_address_cryptographic(self, index: int, is_change: bool = False) -> str:
        """Cryptographic address derivation with proper delegation"""
        if not self.master_key:
            raise WalletError("Wallet is locked or not initialized")
        
        with self._state_lock:
            try:
                #  Use enum value directly - no string conversion             
                cache_key = f"{index}_{is_change}_{self.config.address_type.value}"
                
                if cache_key in self._key_derivation_cache:
                    return self._key_derivation_cache[cache_key]
                
                # DELEGATE to KeyManager for key derivation
                derivation_path = self.key_manager.get_derivation_path(index, is_change)
                derived_public_key = self.key_manager.derive_public_key(derivation_path)
                
                # DELEGATE to AddressDerivation for address generation
                address = self.address_derivation._derive_address(derived_public_key, index, is_change)
                
                # Create comprehensive address info
                address_info = AddressInfo(
                    address=address,
                    index=index,
                    derivation_path=derivation_path,
                    balance=0,
                    received=0,
                    sent=0,
                    tx_count=0,
                    is_used=False,
                    is_change=is_change,
                    labels=[]
                )
                
                # Save to database with cryptographic validation
                if not self.db.save_address(address_info):
                    raise WalletError(f"Failed to save address to database: {address}")
                
                self.addresses[address] = address_info
                
                # Update wallet state cryptographically
                self.state.addresses_generated += 1
                if not self.db.save_wallet_state(self.state):
                    raise WalletError("Failed to save wallet state")
                
                # Cache result with size limits
                if len(self._key_derivation_cache) < 5000:
                    self._key_derivation_cache[cache_key] = address
                
                return address
                
            except Exception as e:
                logger.error(f"Cryptographic address derivation failed for index {index}: {e}")
                self._log_security_event("address_derivation_failed", f"Index {index}: {str(e)}")
                raise WalletError(f"Cryptographic address derivation failed: {e}")
    
    def _store_mnemonic_cryptographic(self, mnemonic: str, passphrase: str):
        """Cryptographic mnemonic storage with encryption"""
        try:
            # Encrypt mnemonic using AES-GCM
            aesgcm = AESGCM(self._aes_gcm_key)
            nonce = secrets.token_bytes(12)  # 96-bit nonce for AES-GCM
            
            # Additional authenticated data for integrity
            aad = self.wallet_id.encode() + struct.pack('>Q', int(time.time()))
            
            # Encrypt mnemonic
            mnemonic_encrypted = aesgcm.encrypt(nonce, mnemonic.encode(), aad)
            
            # Store encrypted mnemonic in secure string
            storage_data = nonce + aad + mnemonic_encrypted
            self._creation_mnemonic = SecureString(storage_data)
            
            # Clear original from memory
            del mnemonic
            del passphrase
            
        except Exception as e:
            raise CryptoError(f"Cryptographic mnemonic storage failed: {e}")
    
    def _update_security_state_after_creation(self):
        """Update security state after wallet creation"""
        self.state.security_score = self._calculate_current_security_score()
        self.db.save_wallet_state(self.state)
        
        self._log_security_event("security_state_updated", "Security state updated after wallet creation")
    
    def _calculate_current_security_score(self) -> int:
        """Calculate current security score based on wallet state"""
        score = self._calculate_initial_security_score()
        
        # Adjust based on actual wallet state
        if len(self.addresses) > 0:
            score += 5
        
        if self.master_key:
            score += 10
        
        # Additional factors...
        
        return min(score, 100)
    
    def _secure_cryptographic_cleanup(self):
        """Cryptographic cleanup of sensitive data"""
        try:
            # Wipe master key
            if self.master_key:
                self.master_key.wipe()
                self.master_key = None
            
            # Wipe mnemonic
            if self._creation_mnemonic:
                self._creation_mnemonic.wipe()
                self._creation_mnemonic = None
            
            # Clear caches
            self._key_derivation_cache.clear()
            self._address_validation_cache.clear()
            
            # Wipe AES key
            if self._aes_gcm_key:
                # Overwrite in memory
                for i in range(len(self._aes_gcm_key)):
                    self._aes_gcm_key = b'\x00' * len(self._aes_gcm_key)
                self._aes_gcm_key = None
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.error(f"Cryptographic cleanup failed: {e}")
    
    def is_initialized(self) -> bool:
        """Check if wallet is cryptographically initialized"""
        return self.master_key is not None and len(self.addresses) > 0
    
    def unlock(self, passphrase: str, timeout: Optional[int] = None) -> bool:
        """Cryptographic wallet unlocking with comprehensive security"""
        with self._state_lock:
            try:
                if not self.locked:
                    return True
                
                # Brute force protection
                if not self._check_brute_force_protection_cryptographic():
                    return False
                
                # Cryptographic passphrase validation
                if not self._validate_passphrase_cryptographic(passphrase):
                    self._record_failed_attempt_cryptographic()
                    return False
                
                # Verify passphrase with key manager
                if not self.key_manager.verify_passphrase(passphrase):
                    self._record_failed_attempt_cryptographic()
                    return False
                
                # Successful authentication
                self._complete_successful_unlock_cryptographic()
                
                logger.info("Wallet unlocked with cryptographic security")
                self._log_security_event("wallet_unlocked", "Successful cryptographic unlock")
                
                return True
                
            except Exception as e:
                logger.error(f"Cryptographic unlock failed: {e}")
                self._record_failed_attempt_cryptographic()
                self._log_security_event("unlock_failed", str(e))
                return False
    
    def _check_brute_force_protection_cryptographic(self) -> bool:
        """Cryptographic brute force protection"""
        current_time = time.time()
        
        # Check temporary lockout
        if self.lock_time and current_time - self.lock_time < 300:  # 5-minute lockout
            remaining = 300 - (current_time - self.lock_time)
            logger.warning(f"Wallet cryptographically locked. Try again in {int(remaining)} seconds")
            return False
        
        # Check attempt history
        recent_attempts = sum(1 for t in self.auth_attempt_history if current_time - t < 3600)  # Last hour
        if recent_attempts >= 10:  # Maximum 10 attempts per hour
            logger.warning("Too many authentication attempts - rate limiting activated")
            return False
        
        return True
    
    def _validate_passphrase_cryptographic(self, passphrase: str) -> bool:
        """Cryptographic passphrase validation"""
        if not isinstance(passphrase, str) or len(passphrase) == 0:
            return False
        
        # Minimum length check
        if len(passphrase) < 8:
            return False
        
        # Additional complexity checks could be added here
        # For now, we rely on the key manager for actual verification
        
        return True
    
    def _record_failed_attempt_cryptographic(self):
        """Record failed authentication attempt cryptographically"""
        current_time = time.time()
        
        self.failed_attempts += 1
        self.auth_attempt_history.append(current_time)
        
        # Trim history
        if len(self.auth_attempt_history) > 100:
            self.auth_attempt_history = self.auth_attempt_history[-50:]
        
        # Activate lockout after 5 failed attempts
        if self.failed_attempts >= 5:
            self.lock_time = current_time
            self._log_security_event("lockout_activated", "5 failed authentication attempts")
    
    def _complete_successful_unlock_cryptographic(self):
        """Complete successful cryptographic unlock"""
        self.locked = False
        self.failed_attempts = 0
        self.lock_time = None
        self.last_successful_auth = time.time()
        self.auth_attempt_history.append(time.time())
        
        # Trim attempt history
        if len(self.auth_attempt_history) > 100:
            self.auth_attempt_history = self.auth_attempt_history[-50:]
    
    def lock(self):
        """Cryptographic wallet locking"""
        with self._state_lock:
            try:
                self.key_manager.wipe()
                self.locked = True
                
                # Additional cryptographic cleanup
                self._secure_cryptographic_cleanup()
                
                # Force garbage collection
                gc.collect()
                
                logger.info("Wallet locked with cryptographic security")
                self._log_security_event("wallet_locked", "Cryptographic lock activated")
                
            except Exception as e:
                logger.error(f"Cryptographic lock failed: {e}")
                self._log_security_event("lock_failed", str(e))

    def get_wallet_type(self) -> WalletType:
    	"""Get the wallet type"""
    	with self._state_lock:
    		if not hasattr(self.config, 'wallet_type') or not self.config.wallet_type:
    			  raise WalletError("Wallet type not configured")
    		return self.config.wallet_type    
    			
    def get_addresses(self) -> Dict[str, AddressInfo]:
    	"""Get all wallet addresses with complete information"""
    	with self._state_lock:
    	      if not self.is_initialized():
    	      	raise WalletError("Wallet is not initialized")
    	      
    	      # Return a defensive copy to prevent external modification
    	      return {addr: AddressInfo(**asdict(info)) for addr, info in self.addresses.items()}
    	      
    def validate_address(self, address: str) -> bool:
        """Cryptographic address validation"""
        if address in self._address_validation_cache:
            return self._address_validation_cache[address]
        
        try:
            # Use production validation
            is_valid = self.address_derivation.validate_address(
                address, self.config.address_type, self.config.network
            )
            
            # Cache result
            self._address_validation_cache[address] = is_valid
            
            return is_valid
            
        except Exception:
            return False
           
    def get_balance(self, address: Optional[str] = None, force_refresh: bool = False):
        """ Get balance for wallet or specific address"""
        try:
        	if not hasattr(self, 'balance_calculator') or not self.balance_calculator:
        		return self._create_error_balance("Balance calculator not available")
        	
        	# Check if we have blockchain connection
        	if hasattr(self.balance_calculator, 'rayonix_chain') and self.balance_calculator.rayonix_chain:
        		return self.balance_calculator.get_balance(force_refresh=force_refresh)
        	
        	else:
        		# Fallback to offline balance
        		logger.warning("Using offline balance calculation - no blockchain connection")
        		return self.balance_calculator._get_offline_balance()
        
        except Exception as e:
        	logger.error(f"Error getting balance: {e}")
        	return self._create_error_balance(str(e))
        
    def get_address_balance(self, address: str) -> int:
    	"""Get balance for specific address"""
    	try:
    		# First try through balance calculator if available
    		if (hasattr(self, 'balance_calculator') and hasattr(self.balance_calculator, 'rayonix_chain') and self.balance_calculator.rayonix_chain):
    			return self.balance_calculator.rayonix_chain.get_address_balance(address)
    		
    		#else:
    			# Fallback: check local wallet state
    			#if address in self.addresses:
    				#return self.addresses[address].balance
    			#return 0
    	except Exception as e:
    		logger.error(f"Error getting address balance for {address}: {e}")
    		return 0
    	
    def _create_error_balance(self, error_message: str):
    	"""Create an error balance response"""
    	from rayonix_wallet.core.wallet_types import WalletBalance
    	return WalletBalance(
    	    total=-1,
    	    confirmed=-1,
    	    unconfirmed=-1,
    	    locked=-1,
    	    available=-1,
    	    by_address={},
    	    tokens={},
    	    error=error_message,
    	    error_type="balance_calculation_error"
    	)
    
    def connect_to_blockchain(self, rayonix_chain):
    	"""Connect wallet to blockchain for advanced recovery"""
    	self.rayonix_chain = rayonix_chain
    	self.recovery_manager = RecoveryManager(self, rayonix_chain)
    
    def is_connected_to_blockchain(self) -> bool:
    	"""Check if wallet is properly connected to a blockchain"""
    	return (hasattr(self, 'balance_calculator') and 
            hasattr(self.balance_calculator, 'rayonix_chain') and 
            self.balance_calculator.rayonix_chain is not None)
    	
    def get_security_report(self) -> Dict[str, Any]:
        """Get comprehensive cryptographic security report"""
        return {
            'wallet_id': self.wallet_id,
            'security_score': self.state.security_score,
            'authentication_attempts': len(self.auth_attempt_history),
            'failed_attempts': self.failed_attempts,
            'last_successful_auth': self.last_successful_auth,
            'is_locked': self.locked,
            'address_count': len(self.addresses),
            'transaction_count': len(self.transactions),
            'recent_security_events': self.security_events[-10:] if self.security_events else [],
            'cryptographic_status': 'active' if self._aes_gcm_key else 'inactive',
            'key_derivation_cache_size': len(self._key_derivation_cache),
            'validation_cache_size': len(self._address_validation_cache)
        }
    def _count_used_addresses(self, addresses: Dict[str, Any]) -> int:
    	try:
    		count = 0
    		for address_info in addresses.values():
    			# Check if address is used based on various criteria
    			if hasattr(address_info, 'tx_count') and address_info.tx_count > 0:
    				count += 1
    			elif hasattr(address_info, 'balance') and address_info.balance > 0:
    				count += 1
    			elif hasattr(address_info, 'received') and address_info.received > 0:
    				count += 1
    			elif hasattr(address_info, 'sent') and address_info.sent > 0:
    				count += 1
    			elif hasattr(address_info, 'is_used') and address_info.is_used:
    				count += 1
    		return count
    	except Exception as e:
    		logger.error(f"Error counting used addresses: {e}")
    		return 0
    
    def get_wallet_info(self) -> Dict[str, Any]:
    	"""Get comprehensive wallet information for API responses"""
    	with self._state_lock:
    		info = {
    		    'wallet_id': self.wallet_id,
    		    'wallet_type': self.config.wallet_type.name if self.config.wallet_type else 'UNKNOWN',
    		    'wallet_type_code': self.config.wallet_type.value if self.config.wallet_type else -1,
    		    'address_type': self.config.address_type.name if self.config.address_type else 'UNKNOWN',
    		    'network': getattr(self.config, 'network', 'Unknown'),
    		    'is_initialized': self.is_initialized(),
    		    'is_locked': self.locked,
    		    'address_count': len(self.addresses),
    		    'transaction_count': len(self.transactions),
    		}
    		
    		# Add security information if available
    		if hasattr(self, 'state'):
    			info.update({
    			    'security_score': self.state.security_score,
    			    'sync_height': self.state.sync_height,
    			    'last_updated': self.state.last_updated
    			})
    		return info
    
    def close(self):
        """Cryptographic wallet shutdown"""
        try:
            self.running = False
            self.synchronizer.stop()
            self.lock()
            self.db.close()
            
            # Final cryptographic cleanup
            self._secure_cryptographic_cleanup()
            
            logger.info("Wallet closed with cryptographic security")
            self._log_security_event("wallet_closed", "Cryptographic shutdown completed")
            
        except Exception as e:
            logger.error(f"Cryptographic wallet close failed: {e}")
            self._log_security_event("close_failed", str(e))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class RayonixWallet(ProductionRayonixWallet):
    def __init__(self, config: Optional[WalletConfig] = None, wallet_id: Optional[str] = None):
        super().__init__(config, wallet_id)
    
    def initialize_new_wallet(self) -> str:
        """Backward compatibility method"""
        if self.is_initialized():
            raise WalletError("Wallet is already initialized")
        
        # Generate new mnemonic
        mnemonic = self._generate_new_mnemonic_cryptographic()
        # Store the mnemonic BEFORE calling create_from_mnemonic
        temp_mnemonic = mnemonic  # Keep reference
        
        # Create wallet from mnemonic
        if not self.create_from_mnemonic(mnemonic, ""):
            raise WalletError("Failed to initialize new wallet")
        
        # Return the mnemonic BEFORE it gets wiped
        
        return temp_mnemonic
    
    def _generate_new_mnemonic_cryptographic(self) -> str:
        """Generate new BIP39 mnemonic cryptographically"""
        try:
            from mnemonic import Mnemonic
            
            mnemo = Mnemonic("english")
            
            # Generate 256 bits of entropy for maximum security
            entropy = secrets.token_bytes(32)
            mnemonic = mnemo.to_mnemonic(entropy)
            
            # Validate the generated mnemonic
            if not self._validate_mnemonic_cryptographic(mnemonic):
                raise CryptoError("Generated mnemonic failed cryptographic validation")
            
            return mnemonic
            
        except Exception as e:
            raise CryptoError(f"Cryptographic mnemonic generation failed: {e}")
    
    def create_hd_wallet(self) -> Tuple[str, str]:
        """Backward compatibility method"""
        mnemonic = self.initialize_new_wallet()
        return mnemonic, self.wallet_id
    
    def create_new_wallet(self) -> str:
        """Backward compatibility method"""
        return self.initialize_new_wallet()

    def get_addresses(self) -> Dict[str, AddressInfo]:
    	"""Backward compatibility method"""
    	return super().get_addresses()
    	
    def get_wallet_type(self) -> WalletType:
    	"""Get wallet type - backward compatibility method"""
    	return super().get_wallet_type()
    
    def get_wallet_info(self) -> Dict[str, Any]:
    	"""Get wallet information - backward compatibility method"""
    	return super().get_wallet_info()