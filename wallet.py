# wallet.py
import json
import os
import hashlib
import base58
import binascii
import secrets
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature, InvalidKey
from utxo import UTXO, Transaction as UTXOTransaction, UTXOSet
import hmac
import bech32
import ecdsa
from ecdsa import SECP256k1, SigningKey, VerifyingKey
from ecdsa.curves import Curve
from ecdsa.util import randrange_from_seed__trytryagain
from bip32 import BIP32, HARDENED_INDEX
from mnemonic import Mnemonic
from base64 import b64encode, b64decode
import qrcode
import io
import requests
from datetime import datetime, timedelta
import logging
import sqlite3
import contextlib
from pathlib import Path
import secure
from secure import SecureString, secure_dump, secure_load
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RayonixWallet")

class WalletType(Enum):
    """Types of wallets"""
    HD = auto()          # Hierarchical Deterministic (BIP32/44)
    NON_HD = auto()      # Single key pair
    MULTISIG = auto()    # Multi-signature
    WATCH_ONLY = auto()  # Watch-only (public keys only)
    HARDWARE = auto()    # Hardware wallet integration
    SMART_CONTRACT = auto()  # Smart contract wallet

class KeyDerivation(Enum):
    """Key derivation standards"""
    BIP32 = auto()       # Hierarchical Deterministic Wallets
    BIP39 = auto()       # Mnemonic code for generating deterministic keys
    BIP44 = auto()       # Multi-Account Hierarchy for Deterministic Wallets
    BIP49 = auto()       # Derivation scheme for P2WPKH-nested-in-P2SH
    BIP84 = auto()       # Derivation scheme for P2WPKH
    ELECTRUM = auto()    # Electrum-style derivation

class AddressType(Enum):
    """Cryptocurrency address types"""
    P2PKH = auto()       # Pay to Public Key Hash (legacy)
    P2SH = auto()        # Pay to Script Hash
    P2WPKH = auto()     # Pay to Witness Public Key Hash (native SegWit)
    P2WSH = auto()       # Pay to Witness Script Hash
    P2TR = auto()        # Pay to Taproot (Taproot)
    BECH32 = auto()      # Bech32 addresses
    ETHEREUM = auto()    # Ethereum-style addresses
    RAYONIX = auto()     # Rayonix-style addresses
    CONTRACT = auto()    # Smart contract addresses

@dataclass
class WalletConfig:
    """Wallet configuration"""
    wallet_type: WalletType = WalletType.HD
    key_derivation: KeyDerivation = KeyDerivation.BIP44
    address_type: AddressType = AddressType.RAYONIX
    encryption: bool = True
    compression: bool = True
    passphrase: Optional[str] = None
    network: str = "mainnet"
    account_index: int = 0
    change_index: int = 0
    gap_limit: int = 20
    auto_backup: bool = True
    backup_interval: int = 86400  # 24 hours
    price_alerts: bool = False
    transaction_fees: Dict[str, int] = field(default_factory=lambda: {
        "low": 1, "medium": 2, "high": 5
    })
    db_path: str = "wallet.db"
    sync_interval: int = 300  # 5 minutes

@dataclass
class SecureKeyPair:
    """Cryptographic key pair with secure memory management"""
    _private_key: SecureString
    public_key: bytes
    chain_code: Optional[bytes] = None
    depth: int = 0
    index: int = 0
    parent_fingerprint: bytes = b'\x00\x00\x00\x00'
    curve: Curve = field(default_factory=lambda: SECP256k1)
    
    @property
    def private_key(self) -> bytes:
        return self._private_key.get_value()
    
    def wipe(self):
        """Securely wipe private key from memory"""
        self._private_key.wipe()
        
    def __del__(self):
        self.wipe()

@dataclass
class Transaction:
    """Wallet transaction"""
    txid: str
    amount: int
    fee: int
    confirmations: int
    timestamp: int
    block_height: Optional[int]
    from_address: str
    to_address: str
    status: str  # pending, confirmed, failed
    direction: str  # sent, received
    memo: Optional[str] = None
    exchange_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AddressInfo:
    """Address information"""
    address: str
    index: int
    derivation_path: str
    balance: int
    received: int
    sent: int
    tx_count: int
    is_used: bool
    is_change: bool
    labels: List[str] = field(default_factory=list)

@dataclass
class WalletBalance:
    """Wallet balance information"""
    total: int
    confirmed: int
    unconfirmed: int
    locked: int
    available: int
    by_address: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    offline_mode: bool = False
    last_online_update: Optional[float] = None
    data_freshness: Optional[str] = None
    confidence_level: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    offline_estimated: bool = False
    estimation_confidence: Optional[str] = None
    reconstruction_confidence: Optional[str] = None

@dataclass
class WalletState:
    """Wallet state and statistics"""
    sync_height: int
    last_updated: float
    tx_count: int
    addresses_generated: int
    addresses_used: int
    total_received: int
    total_sent: int
    security_score: int

class WalletDatabase:
    """Database layer for wallet persistence"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            # Addresses table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS addresses (
                    address TEXT PRIMARY KEY,
                    index INTEGER NOT NULL,
                    derivation_path TEXT NOT NULL,
                    balance INTEGER DEFAULT 0,
                    received INTEGER DEFAULT 0,
                    sent INTEGER DEFAULT 0,
                    tx_count INTEGER DEFAULT 0,
                    is_used BOOLEAN DEFAULT FALSE,
                    is_change BOOLEAN DEFAULT FALSE,
                    labels TEXT DEFAULT '[]',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Transactions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    amount INTEGER NOT NULL,
                    fee INTEGER NOT NULL,
                    confirmations INTEGER DEFAULT 0,
                    timestamp INTEGER NOT NULL,
                    block_height INTEGER,
                    from_address TEXT NOT NULL,
                    to_address TEXT NOT NULL,
                    status TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    memo TEXT,
                    exchange_rate REAL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Wallet state table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS wallet_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    sync_height INTEGER DEFAULT 0,
                    last_updated REAL NOT NULL,
                    tx_count INTEGER DEFAULT 0,
                    addresses_generated INTEGER DEFAULT 0,
                    addresses_used INTEGER DEFAULT 0,
                    total_received INTEGER DEFAULT 0,
                    total_sent INTEGER DEFAULT 0,
                    security_score INTEGER DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # UTXO index table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS wallet_utxos (
                    txid TEXT NOT NULL,
                    vout INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    script_pubkey TEXT NOT NULL,
                    confirmations INTEGER DEFAULT 0,
                    spent BOOLEAN DEFAULT FALSE,
                    spent_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (txid, vout)
                )
            ''')
            
            conn.commit()
    
    @contextlib.contextmanager
    def _get_connection(self):
        """Get database connection with context management"""
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
        finally:
            conn.close()
    
    def save_address(self, address_info: AddressInfo):
        """Save address to database"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO addresses 
                (address, index, derivation_path, balance, received, sent, tx_count, is_used, is_change, labels)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                address_info.address,
                address_info.index,
                address_info.derivation_path,
                address_info.balance,
                address_info.received,
                address_info.sent,
                address_info.tx_count,
                address_info.is_used,
                address_info.is_change,
                json.dumps(address_info.labels)
            ))
            conn.commit()
    
    def get_address(self, address: str) -> Optional[AddressInfo]:
        """Get address from database"""
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT * FROM addresses WHERE address = ?', (address,))
            row = cursor.fetchone()
            if row:
                return AddressInfo(
                    address=row[0],
                    index=row[1],
                    derivation_path=row[2],
                    balance=row[3],
                    received=row[4],
                    sent=row[5],
                    tx_count=row[6],
                    is_used=bool(row[7]),
                    is_change=bool(row[8]),
                    labels=json.loads(row[9])
                )
        return None
    
    def get_all_addresses(self) -> List[AddressInfo]:
        """Get all addresses from database"""
        addresses = []
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT * FROM addresses ORDER BY index')
            for row in cursor.fetchall():
                addresses.append(AddressInfo(
                    address=row[0],
                    index=row[1],
                    derivation_path=row[2],
                    balance=row[3],
                    received=row[4],
                    sent=row[5],
                    tx_count=row[6],
                    is_used=bool(row[7]),
                    is_change=bool(row[8]),
                    labels=json.loads(row[9])
                ))
        return addresses
    
    def save_transaction(self, transaction: Transaction):
        """Save transaction to database"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO transactions 
                (txid, amount, fee, confirmations, timestamp, block_height, from_address, to_address, status, direction, memo, exchange_rate, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.txid,
                transaction.amount,
                transaction.fee,
                transaction.confirmations,
                transaction.timestamp,
                transaction.block_height,
                transaction.from_address,
                transaction.to_address,
                transaction.status,
                transaction.direction,
                transaction.memo,
                transaction.exchange_rate,
                json.dumps(transaction.metadata)
            ))
            conn.commit()
    
    def get_transaction(self, txid: str) -> Optional[Transaction]:
        """Get transaction from database"""
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT * FROM transactions WHERE txid = ?', (txid,))
            row = cursor.fetchone()
            if row:
                return Transaction(
                    txid=row[0],
                    amount=row[1],
                    fee=row[2],
                    confirmations=row[3],
                    timestamp=row[4],
                    block_height=row[5],
                    from_address=row[6],
                    to_address=row[7],
                    status=row[8],
                    direction=row[9],
                    memo=row[10],
                    exchange_rate=row[11],
                    metadata=json.loads(row[12])
                )
        return None
    
    def get_transactions(self, limit: int = 50, offset: int = 0) -> List[Transaction]:
        """Get transactions from database"""
        transactions = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                'SELECT * FROM transactions ORDER BY timestamp DESC LIMIT ? OFFSET ?',
                (limit, offset)
            )
            for row in cursor.fetchall():
                transactions.append(Transaction(
                    txid=row[0],
                    amount=row[1],
                    fee=row[2],
                    confirmations=row[3],
                    timestamp=row[4],
                    block_height=row[5],
                    from_address=row[6],
                    to_address=row[7],
                    status=row[8],
                    direction=row[9],
                    memo=row[10],
                    exchange_rate=row[11],
                    metadata=json.loads(row[12])
                ))
        return transactions
    
    def save_wallet_state(self, state: WalletState):
        """Save wallet state to database"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO wallet_state 
                (id, sync_height, last_updated, tx_count, addresses_generated, addresses_used, total_received, total_sent, security_score)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                state.sync_height,
                state.last_updated,
                state.tx_count,
                state.addresses_generated,
                state.addresses_used,
                state.total_received,
                state.total_sent,
                state.security_score
            ))
            conn.commit()
    
    def get_wallet_state(self) -> Optional[WalletState]:
        """Get wallet state from database"""
        with self._get_connection() as conn:
            cursor = conn.execute('SELECT * FROM wallet_state WHERE id = 1')
            row = cursor.fetchone()
            if row:
                return WalletState(
                    sync_height=row[1],
                    last_updated=row[2],
                    tx_count=row[3],
                    addresses_generated=row[4],
                    addresses_used=row[5],
                    total_received=row[6],
                    total_sent=row[7],
                    security_score=row[8]
                )
        return None
    
    def save_utxo(self, utxo: UTXO, address: str):
        """Save UTXO to wallet index"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO wallet_utxos 
                (txid, vout, address, amount, script_pubkey, confirmations, spent, spent_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                utxo.tx_hash,
                utxo.vout,
                address,
                utxo.amount,
                utxo.script_pubkey.hex() if hasattr(utxo.script_pubkey, 'hex') else utxo.script_pubkey,
                utxo.confirmations,
                utxo.spent,
                utxo.spent_by
            ))
            conn.commit()
    
    def get_utxos(self, address: Optional[str] = None) -> List[Dict]:
        """Get UTXOs from wallet index"""
        utxos = []
        query = 'SELECT * FROM wallet_utxos WHERE spent = FALSE'
        params = ()
        
        if address:
            query += ' AND address = ?'
            params = (address,)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                utxos.append({
                    'txid': row[0],
                    'vout': row[1],
                    'address': row[2],
                    'amount': row[3],
                    'script_pubkey': row[4],
                    'confirmations': row[5],
                    'spent': bool(row[6]),
                    'spent_by': row[7]
                })
        return utxos
    
    def mark_utxo_spent(self, txid: str, vout: int, spent_by: str):
        """Mark UTXO as spent"""
        with self._get_connection() as conn:
            conn.execute(
                'UPDATE wallet_utxos SET spent = TRUE, spent_by = ? WHERE txid = ? AND vout = ?',
                (spent_by, txid, vout)
            )
            conn.commit()

class WalletSynchronizer:
    """Service for synchronizing wallet state with blockchain"""
    
    def __init__(self, wallet: 'RayonixWallet'):
        self.wallet = wallet
        self.running = False
        self.thread = None
    
    def start(self):
        """Start synchronization service"""
        self.running = True
        self.thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.thread.start()
        logger.info("Wallet synchronizer started")
    
    def stop(self):
        """Stop synchronization service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info("Wallet synchronizer stopped")
    
    def _sync_worker(self):
        """Background synchronization worker"""
        while self.running:
            try:
                self.sync_with_blockchain()
                time.sleep(self.wallet.config.sync_interval)
            except Exception as e:
                logger.error(f"Synchronization error: {e}")
                time.sleep(60)  # Wait before retrying after error
    
    def sync_with_blockchain(self, start_height: Optional[int] = None):
        """Synchronize wallet with blockchain"""
        if not self.wallet.rayonix_coin:
            logger.warning("Cannot sync: No blockchain reference")
            return
        
        # Get current wallet state
        current_state = self.wallet.db.get_wallet_state() or WalletState(
            sync_height=0, last_updated=time.time(), tx_count=0,
            addresses_generated=0, addresses_used=0, total_received=0,
            total_sent=0, security_score=0
        )
        
        # Determine start height for sync
        start_block = start_height or current_state.sync_height
        blockchain = self.wallet.rayonix_coin.blockchain
        current_height = len(blockchain)
        
        if start_block >= current_height:
            logger.debug("Wallet already synchronized with latest block")
            return
        
        logger.info(f"Synchronizing wallet from block {start_block} to {current_height}")
        
        # Process blocks
        for block_height in range(start_block, current_height):
            if not self.running:
                break
                
            block = blockchain[block_height]
            self._process_block(block, block_height)
        
        # Update wallet state
        new_state = WalletState(
            sync_height=current_height,
            last_updated=time.time(),
            tx_count=current_state.tx_count,  # Will be updated by _process_block
            addresses_generated=current_state.addresses_generated,
            addresses_used=current_state.addresses_used,
            total_received=current_state.total_received,
            total_sent=current_state.total_sent,
            security_score=current_state.security_score
        )
        self.wallet.db.save_wallet_state(new_state)
        self.wallet.state = new_state
        
        logger.info(f"Wallet synchronization completed up to block {current_height}")
    
    def _process_block(self, block: Dict, block_height: int):
        """Process a block and update wallet state"""
        for tx_data in block.get('transactions', []):
            self._process_transaction(tx_data, block_height)
    
    def _process_transaction(self, tx_data: Dict, block_height: int):
        """Process a transaction and update wallet state"""
        txid = tx_data.get('txid')
        if not txid:
            return
        
        # Check if transaction involves any wallet addresses
        wallet_addresses = {addr: info for addr, info in self.wallet.addresses.items()}
        relevant = False
        
        # Check inputs (if we're spending)
        for vin in tx_data.get('vin', []):
            if 'address' in vin and vin['address'] in wallet_addresses:
                relevant = True
                break
        
        # Check outputs (if we're receiving)
        for vout in tx_data.get('vout', []):
            if 'address' in vout and vout['address'] in wallet_addresses:
                relevant = True
                break
        
        if not relevant:
            return
        
        # Create or update transaction record
        transaction = self.wallet.db.get_transaction(txid) or Transaction(
            txid=txid,
            amount=0,
            fee=tx_data.get('fee', 0),
            confirmations=len(self.wallet.rayonix_coin.blockchain) - block_height,
            timestamp=tx_data.get('timestamp', int(time.time())),
            block_height=block_height,
            from_address="",  # Will be populated below
            to_address="",    # Will be populated below
            status="confirmed",
            direction="unknown"
        )
        
        # Update transaction details
        total_received = 0
        total_sent = 0
        from_addresses = set()
        to_addresses = set()
        
        # Process inputs (funds being spent)
        for vin in tx_data.get('vin', []):
            if 'address' in vin and vin['address'] in wallet_addresses:
                amount = vin.get('value', 0)
                total_sent += amount
                from_addresses.add(vin['address'])
                
                # Mark UTXOs as spent
                if 'txid' in vin and 'vout' in vin:
                    self.wallet.db.mark_utxo_spent(vin['txid'], vin['vout'], txid)
        
        # Process outputs (funds being received)
        for vout in tx_data.get('vout', []):
            if 'address' in vout and vout['address'] in wallet_addresses:
                amount = vout.get('value', 0)
                total_received += amount
                to_addresses.add(vout['address'])
                
                # Add new UTXO to wallet index
                utxo = UTXO(
                    tx_hash=txid,
                    vout=vout['n'],
                    amount=amount,
                    script_pubkey=vout.get('scriptPubKey', ''),
                    address=vout['address'],
                    confirmations=transaction.confirmations,
                    spent=False
                )
                self.wallet.db.save_utxo(utxo, vout['address'])
        
        # Determine transaction direction and amount
        if total_received > 0 and total_sent == 0:
            transaction.direction = "received"
            transaction.amount = total_received
            transaction.from_address = ",".join(from_addresses) if from_addresses else "external"
            transaction.to_address = ",".join(to_addresses)
        elif total_sent > 0 and total_received == 0:
            transaction.direction = "sent"
            transaction.amount = total_sent
            transaction.from_address = ",".join(from_addresses)
            transaction.to_address = ",".join(to_addresses) if to_addresses else "external"
        else:
            # Self-transfer or mixed transaction
            transaction.direction = "transfer"
            transaction.amount = max(total_received, total_sent)
            transaction.from_address = ",".join(from_addresses)
            transaction.to_address = ",".join(to_addresses)
        
        # Save transaction
        self.wallet.db.save_transaction(transaction)
        
        # Update address balances and stats
        for address in from_addresses.union(to_addresses):
            if address in wallet_addresses:
                addr_info = wallet_addresses[address]
                # Update address stats based on transaction
                if address in from_addresses:
                    addr_info.sent += total_sent
                    addr_info.balance -= total_sent
                if address in to_addresses:
                    addr_info.received += total_received
                    addr_info.balance += total_received
                
                addr_info.tx_count += 1
                addr_info.is_used = True
                
                # Save updated address info
                self.wallet.db.save_address(addr_info)
                self.wallet.addresses[address] = addr_info
        
        # Update wallet state
        state = self.wallet.state
        state.tx_count += 1
        if transaction.direction == "received":
            state.total_received += transaction.amount
        elif transaction.direction == "sent":
            state.total_sent += transaction.amount
        
        state.addresses_used = len([a for a in self.wallet.addresses.values() if a.is_used])
        state.last_updated = time.time()
        
        self.wallet.db.save_wallet_state(state)

class RayonixWallet:
    """Advanced cryptographic wallet with enterprise-grade features"""
    
    def __init__(self, config: Optional[WalletConfig] = None, wallet_id: Optional[str] = None):
        self.config = config or WalletConfig()
        self.wallet_id = wallet_id or self._generate_wallet_id()
        
        # Initialize database
        self.db = WalletDatabase(self.config.db_path)
        
        # Load state from database
        self.state = self.db.get_wallet_state() or WalletState(
            sync_height=0, last_updated=time.time(), tx_count=0,
            addresses_generated=0, addresses_used=0, total_received=0,
            total_sent=0, security_score=0
        )
        
        # Load addresses from database
        self.addresses = {addr.address: addr for addr in self.db.get_all_addresses()}
        
        # Load transactions from database (limited to recent ones)
        self.transactions = {tx.txid: tx for tx in self.db.get_transactions(limit=1000)}
        
        # Security
        self.master_key: Optional[SecureKeyPair] = None
        self.encryption_key: Optional[SecureString] = None
        self.lock_time: Optional[float] = None
        self.failed_attempts = 0
        self.locked = True  # Start locked by default
        
        # Cache and performance
        self.address_cache: Dict[str, str] = {}
        self.transaction_cache: Dict[str, List[Transaction]] = {}
        self.balance_cache: Optional[WalletBalance] = None
        
        # Multi-signature
        self.multisig_config: Optional[Dict] = None
        self.cosigners: List[str] = []
        
        # Hardware wallet integration
        self.hardware_wallet: Optional[Any] = None
        
        # Blockchain integration
        self.rayonix_coin: Optional[Any] = None
        
        # Synchronizer
        self.synchronizer = WalletSynchronizer(self)
        
        # Background tasks
        self.background_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Balance-specific initialization
        self._balance_lock = threading.RLock()
        self._last_balance_update: float = 0
        self._balance_calculation_in_progress: bool = False
        self._last_balance_error: Optional[Dict] = None
        self._balance_update_attempts: int = 0
        
        logger.info(f"Wallet initialized with ID: {self.wallet_id}")
    
    def set_blockchain_reference(self, rayonix_coin_instance: Any) -> bool:
        """Set reference to blockchain instance"""
        try:
            if not hasattr(rayonix_coin_instance, 'utxo_set') or not hasattr(rayonix_coin_instance, 'get_balance'):
                logger.error("Invalid blockchain reference provided")
                return False
            self.rayonix_coin = rayonix_coin_instance
            logger.info("Blockchain reference set successfully")
            
            # Start synchronizer if we have blockchain reference
            self.synchronizer.start()
            return True
        except Exception as e:
            logger.error(f"Failed to set blockchain reference: {e}")
            return False
    
    def _generate_wallet_id(self) -> str:
        """Generate unique wallet ID"""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()[:16]
    
    def _initialize_wallet(self):
        """Initialize wallet components"""
        # Setup encryption if enabled
        if self.config.encryption and self.config.passphrase:
            self.encryption_key = SecureString(self._derive_encryption_key(self.config.passphrase))
        
        # Initialize hardware wallet if configured
        if self.config.wallet_type == WalletType.HARDWARE:
            self._initialize_hardware_wallet()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _derive_encryption_key(self, passphrase: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from passphrase"""
        salt = salt or os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(passphrase.encode())
    
    def _initialize_hardware_wallet(self):
        """Initialize hardware wallet integration"""
        try:
            # This would integrate with actual hardware wallets like Ledger, Trezor
            # For production, use proper hardware wallet libraries
            self.hardware_wallet = {
                'connected': False,
                'model': 'Simulated',
                'version': '1.0.0'
            }
            logger.info("Hardware wallet simulation initialized")
        except Exception as e:
            logger.error(f"Hardware wallet initialization failed: {e}")
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        self.running = True
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
    
    def _background_worker(self):
        """Background worker for wallet maintenance"""
        while self.running:
            try:
                # Auto-backup if enabled
                if self.config.auto_backup:
                    self._auto_backup()
                
                # Price alerts if enabled
                if self.config.price_alerts:
                    self._check_price_alerts()
                
                # Cleanup old cache entries
                self._cleanup_cache()
                
                # Check for wallet lock timeout
                self._check_lock_timeout()
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(60)
    
    def _auto_backup(self):
        """Automatically backup wallet"""
        backup_dir = os.path.join(os.path.expanduser("~"), ".rayonix", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_file = os.path.join(backup_dir, f"wallet_{self.wallet_id}_{int(time.time())}.backup")
        self.backup(backup_file)
        logger.info(f"Auto-backup created: {backup_file}")
    
    def _check_price_alerts(self):
        """Check cryptocurrency price alerts"""
        # This would integrate with price APIs
        pass
    
    def _cleanup_cache(self):
        """Cleanup old cache entries"""
        current_time = time.time()
        # Remove entries older than 1 hour
        self.transaction_cache = {
            k: v for k, v in self.transaction_cache.items()
            if current_time - self._get_cache_timestamp(k) < 3600
        }
    
    def _check_lock_timeout(self):
        """Check and reset wallet lock timeout"""
        if self.locked and self.lock_time and time.time() - self.lock_time > 3600:  # 1 hour lock
            self.unlock(self.config.passphrase or "")
    
    def create_from_mnemonic(self, mnemonic_phrase: str, passphrase: str = "") -> bool:
        """Create wallet from BIP39 mnemonic phrase"""
        try:
            # Validate mnemonic
            if not self._validate_mnemonic(mnemonic_phrase):
                raise ValueError("Invalid mnemonic phrase")
            
            # Generate seed from mnemonic
            seed = self._mnemonic_to_seed(mnemonic_phrase, passphrase)
            
            # Generate master key from seed using BIP32 library
            bip32 = BIP32.from_seed(seed)
            
            # Create secure key pair
            private_key_secure = SecureString(bip32.private_key)
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=bip32.public_key,
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
            bip32 = BIP32.from_seed(seed)
            
            # Create secure key pair
            private_key_secure = SecureString(bip32.private_key)
            self.master_key = SecureKeyPair(
                _private_key=private_key_secure,
                public_key=bip32.public_key,
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
    
    def _generate_mnemonic(self, strength: int = 256) -> str:
        """Generate BIP39 mnemonic phrase"""
        mnemo = Mnemonic("english")
        return mnemo.generate(strength=strength)
    
    def _validate_mnemonic(self, mnemonic_phrase: str) -> bool:
        """Validate BIP39 mnemonic phrase"""
        mnemo = Mnemonic("english")
        return mnemo.check(mnemonic_phrase)
    
    def _mnemonic_to_seed(self, mnemonic_phrase: str, passphrase: str = "") -> bytes:
        """Convert mnemonic to seed using BIP39"""
        mnemo = Mnemonic("english")
        return mnemo.to_seed(mnemonic_phrase, passphrase)
    
    def _generate_initial_addresses(self):
        """Generate initial set of addresses"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Generate first 20 addresses for receiving and change
        for i in range(self.config.gap_limit):
            # Receiving addresses
            receiving_address = self.derive_address(i, False)
            
            # Change addresses
            change_address = self.derive_address(i, True)
        
        logger.info(f"Generated {self.config.gap_limit * 2} initial addresses")
    
    def derive_address(self, index: int, is_change: bool = False) -> str:
        """Derive address at specified index"""
        if not self.master_key:
            raise ValueError("Wallet is locked or not initialized")
        
        # Use BIP32 library for proper derivation
        bip32 = BIP32.from_seed(self.master_key.private_key)
        
        # BIP44 derivation path: m/purpose'/coin_type'/account'/change/address_index
        change_index = 1 if is_change else 0
        derivation_path = f"m/44'/0'/{self.config.account_index}'/{change_index}/{index}"
        
        try:
            # Derive child key
            derived_private_key = bip32.get_privkey_from_path(derivation_path)
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
    
    def _derive_address(self, public_key: bytes, index: int, is_change: bool) -> str:
        """Derive address from public key based on address type"""
        if self.config.address_type == AddressType.RAYONIX:
            # Rayonix-specific address derivation
            return self._derive_rayonix_address(public_key)
        elif self.config.address_type == AddressType.P2PKH:
            return self._derive_p2pkh_address(public_key)
        elif self.config.address_type == AddressType.P2WPKH:
            return self._derive_p2wpkh_address(public_key)
        elif self.config.address_type == AddressType.BECH32:
            return self._derive_bech32_address(public_key)
        else:
            # Default to Rayonix address
            return self._derive_rayonix_address(public_key)
    
    def _derive_rayonix_address(self, public_key: bytes) -> str:
        """Derive Rayonix-specific address"""
        # Hash public key with SHA256
        sha_hash = hashlib.sha256(public_key).digest()
        
        # Hash again with RIPEMD160
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        # Add network prefix (0x3C for mainnet, 0x6F for testnet)
        network_byte = b'\x3C' if self.config.network == "mainnet" else b'\x6F'
        payload = network_byte + ripemd_hash
        
        # Double SHA256 for checksum
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        # Base58 encode
        address_bytes = payload + checksum
        return base58.b58encode(address_bytes).decode('ascii')
    
    def _derive_p2pkh_address(self, public_key: bytes) -> str:
        """Derive P2PKH address (legacy Bitcoin-style)"""
        # Hash public key with SHA256
        sha_hash = hashlib.sha256(public_key).digest()
        
        # Hash again with RIPEMD160
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        # Add version byte (0x00 for mainnet, 0x6F for testnet)
        version_byte = b'\x00' if self.config.network == "mainnet" else b'\x6F'
        payload = version_byte + ripemd_hash
        
        # Double SHA256 for checksum
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        # Base58 encode
        address_bytes = payload + checksum
        return base58.b58encode(address_bytes).decode('ascii')
    
    def _derive_p2wpkh_address(self, public_key: bytes) -> str:
        """Derive P2WPKH address (native SegWit)"""
        # Hash public key with SHA256
        sha_hash = hashlib.sha256(public_key).digest()
        
        # Hash again with RIPEMD160
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        # Bech32 encoding
        hrp = "bc" if self.config.network == "mainnet" else "tb"
        return bech32.encode(hrp, 0, ripemd_hash)
    
    def _derive_bech32_address(self, public_key: bytes) -> str:
        """Derive Bech32 address"""
        # Hash public key with SHA256
        sha_hash = hashlib.sha256(public_key).digest()
        
        # Hash again with RIPEMD160
        ripemd_hash = hashlib.new('ripemd160', sha_hash).digest()
        
        # Bech32 encoding
        hrp = "ray" if self.config.network == "mainnet" else "tray"
        return bech32.encode(hrp, 0, ripemd_hash)
    
    def unlock(self, passphrase: str, timeout: Optional[int] = None) -> bool:
        """Unlock wallet with passphrase"""
        try:
            # Check if already unlocked
            if not self.locked:
                return True
            
            # Check for too many failed attempts
            if self.failed_attempts >= 5:
                lockout_time = 300  # 5 minutes
                if self.lock_time and time.time() - self.lock_time < lockout_time:
                    remaining = lockout_time - (time.time() - self.lock_time)
                    logger.warning(f"Wallet locked due to too many failed attempts. Try again in {int(remaining)} seconds")
                    return False
            
            # Verify passphrase (this would use proper key derivation in production)
            if self._verify_passphrase(passphrase):
                self.locked = False
                self.failed_attempts = 0
                self.lock_time = None
                
                # Derive encryption key if needed
                if self.config.encryption:
                    self.encryption_key = SecureString(self._derive_encryption_key(passphrase))
                
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
            # Wipe master key from memory
            if self.master_key:
                self.master_key.wipe()
                self.master_key = None
            
            # Wipe encryption key
            if self.encryption_key:
                self.encryption_key.wipe()
                self.encryption_key = None
            
            # Clear sensitive caches
            self.address_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            self.locked = True
            logger.info("Wallet locked successfully")
            
        except Exception as e:
            logger.error(f"Lock failed: {e}")
    
    def _verify_passphrase(self, passphrase: str) -> bool:
        """Verify wallet passphrase"""
        # In a real implementation, this would verify against a stored hash
        # For demonstration, we'll use a simple check
        expected_hash = hashlib.sha256(passphrase.encode()).hexdigest()
        
        # This would be stored securely during wallet creation
        stored_hash = hashlib.sha256((self.config.passphrase or "").encode()).hexdigest()
        
        return expected_hash == stored_hash
    
    def get_balance(self, address: Optional[str] = None, force_refresh: bool = False) -> WalletBalance:
        """Get wallet balance with advanced error handling and offline capabilities"""
        with self._balance_lock:
            current_time = time.time()
            
            # Check if we should use cached balance
            if (not force_refresh and self.balance_cache and 
                current_time - self._last_balance_update < 300):  # 5 minute cache
                return self._enhance_cached_balance(self.balance_cache)
            
            # Prevent concurrent balance calculations
            if self._balance_calculation_in_progress:
                if self.balance_cache:
                    return self._enhance_cached_balance(self.balance_cache, is_stale=True)
                return self._create_offline_balance()
            
            self._balance_calculation_in_progress = True
            
            try:
                balance = self._calculate_balance(address)
                self.balance_cache = balance
                self._last_balance_update = current_time
                self._balance_update_attempts = 0
                self._last_balance_error = None
                return balance
                
            except Exception as e:
                self._handle_balance_error(e)
                if self.balance_cache:
                    return self._enhance_cached_balance(self.balance_cache, is_stale=True)
                return self._create_offline_balance()
                
            finally:
                self._balance_calculation_in_progress = False
    
    def _calculate_balance(self, address: Optional[str] = None) -> WalletBalance:
        """Calculate wallet balance using UTXO index"""
        try:
            # Use our synchronized UTXO index for fast balance calculation
            utxos = self.db.get_utxos(address)
            
            total = 0
            confirmed = 0
            unconfirmed = 0
            by_address: Dict[str, int] = {}
            
            for utxo in utxos:
                amount = utxo['amount']
                total += amount
                
                if utxo['confirmations'] >= 6:  # Consider confirmed after 6 blocks
                    confirmed += amount
                else:
                    unconfirmed += amount
                
                # Track balance by address
                addr = utxo['address']
                by_address[addr] = by_address.get(addr, 0) + amount
            
            # Get locked balance (if any)
            locked = self._get_locked_balance()
            
            return WalletBalance(
                total=total,
                confirmed=confirmed,
                unconfirmed=unconfirmed,
                locked=locked,
                available=total - locked,
                by_address=by_address,
                offline_mode=False,
                last_online_update=time.time(),
                data_freshness="live"
            )
            
        except Exception as e:
            logger.error(f"Balance calculation failed: {e}")
            raise
    
    def _get_locked_balance(self) -> int:
        """Get locked/unspendable balance"""
        # This would check for locked UTXOs, time-locked transactions, etc.
        return 0
    
    def _handle_balance_error(self, error: Exception):
        """Handle balance calculation errors"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        self._last_balance_error = {
            'type': error_type,
            'message': error_msg,
            'timestamp': time.time(),
            'attempt': self._balance_update_attempts + 1
        }
        
        self._balance_update_attempts += 1
        
        logger.error(f"Balance error ({self._balance_update_attempts}): {error_type}: {error_msg}")
    
    def _enhance_cached_balance(self, balance: WalletBalance, is_stale: bool = False) -> WalletBalance:
        """Enhance cached balance with freshness information"""
        enhanced = WalletBalance(**balance.__dict__)
        
        if is_stale:
            enhanced.offline_mode = True
            enhanced.data_freshness = "stale"
            enhanced.warning = "Using cached data - unable to connect to network"
            enhanced.estimation_confidence = "medium"
        else:
            enhanced.data_freshness = "cached"
            enhanced.estimation_confidence = "high"
        
        return enhanced
    
    def _create_offline_balance(self) -> WalletBalance:
        """Create offline balance estimate"""
        # Try to reconstruct from last known state
        last_state = self.db.get_wallet_state()
        
        if last_state:
            return WalletBalance(
                total=last_state.total_received - last_state.total_sent,
                confirmed=last_state.total_received - last_state.total_sent,
                unconfirmed=0,
                locked=0,
                available=last_state.total_received - last_state.total_sent,
                offline_mode=True,
                last_online_update=self._last_balance_update,
                data_freshness="reconstructed",
                warning="Operating in offline mode - balances may be inaccurate",
                error=self._last_balance_error['message'] if self._last_balance_error else None,
                error_type=self._last_balance_error['type'] if self._last_balance_error else None,
                offline_estimated=True,
                estimation_confidence="low",
                reconstruction_confidence="medium"
            )
        
        # Fallback to zero balance
        return WalletBalance(
            total=0,
            confirmed=0,
            unconfirmed=0,
            locked=0,
            available=0,
            offline_mode=True,
            last_online_update=None,
            data_freshness="unknown",
            warning="No balance data available - completely offline",
            error=self._last_balance_error['message'] if self._last_balance_error else "No connection",
            error_type=self._last_balance_error['type'] if self._last_balance_error else "ConnectionError",
            offline_estimated=False,
            estimation_confidence="none"
        )
    
    def get_transaction_history(self, limit: int = 50, offset: int = 0) -> List[Transaction]:
        """Get transaction history"""
        return self.db.get_transactions(limit, offset)
    
    def send_transaction(self, to_address: str, amount: int, fee_rate: Optional[int] = None, 
                        memo: Optional[str] = None) -> Optional[str]:
        """Send transaction with proper signing"""
        if self.locked:
            raise ValueError("Wallet is locked")
        
        if not self.master_key:
            raise ValueError("Master key not available")
        
        # Validate amount
        if amount <= 0:
            raise ValueError("Amount must be positive")
        
        # Get available UTXOs
        utxos = self._select_utxos(amount, fee_rate or self.config.transaction_fees["medium"])
        
        if not utxos:
            raise ValueError("Insufficient funds")
        
        # Create transaction
        transaction = self._create_transaction(to_address, amount, utxos, fee_rate, memo)
        
        # Sign transaction
        signed_tx = self._sign_transaction(transaction)
        
        # Broadcast transaction
        txid = self._broadcast_transaction(signed_tx)
        
        if txid:
            # Update local state
            self._update_after_transaction(signed_tx, txid)
            
            # Start immediate synchronization
            threading.Thread(target=self.synchronizer.sync_with_blockchain, daemon=True).start()
        
        return txid
    
    def _select_utxos(self, amount: int, fee_rate: int) -> List[Dict]:
        """Select UTXOs for transaction using coin selection algorithm"""
        available_utxos = self.db.get_utxos()
        
        # Sort UTXOs by confirmations (prefer more confirmed)
        sorted_utxos = sorted(available_utxos, key=lambda x: x['confirmations'], reverse=True)
        
        selected_utxos = []
        total_selected = 0
        
        # Simple largest-first algorithm (could be improved with more sophisticated algorithms)
        for utxo in sorted_utxos:
            if total_selected >= amount + self._estimate_fee(len(selected_utxos) + 1, 2, fee_rate):
                break
                
            selected_utxos.append(utxo)
            total_selected += utxo['amount']
        
        if total_selected < amount + self._estimate_fee(len(selected_utxos), 2, fee_rate):
            return []  # Insufficient funds
        
        return selected_utxos
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: int) -> int:
        """Estimate transaction fee"""
        # Base transaction size + inputs + outputs
        base_size = 10  # bytes
        input_size = 180  # bytes per input (approx)
        output_size = 34   # bytes per output (approx)
        
        total_size = base_size + (input_count * input_size) + (output_count * output_size)
        return total_size * fee_rate
    
    def _create_transaction(self, to_address: str, amount: int, utxos: List[Dict], 
                          fee_rate: int, memo: Optional[str]) -> Dict:
        """Create transaction structure"""
        total_input = sum(utxo['amount'] for utxo in utxos)
        fee = self._estimate_fee(len(utxos), 2, fee_rate)  # 2 outputs: recipient + change
        change_amount = total_input - amount - fee
        
        if change_amount < 0:
            raise ValueError("Insufficient funds after fee calculation")
        
        # Generate change address
        change_address = self.derive_address(self.config.change_index, True)
        self.config.change_index += 1
        
        transaction = {
            'version': 1,
            'locktime': 0,
            'vin': [],
            'vout': [
                {
                    'value': amount,
                    'address': to_address,
                    'script_pubkey': self._address_to_script(to_address)
                }
            ],
            'metadata': {
                'memo': memo,
                'fee_rate': fee_rate,
                'created': int(time.time())
            }
        }
        
        # Add change output if needed
        if change_amount > 0:
            transaction['vout'].append({
                'value': change_amount,
                'address': change_address,
                'script_pubkey': self._address_to_script(change_address)
            })
        
        # Add inputs
        for utxo in utxos:
            transaction['vin'].append({
                'txid': utxo['txid'],
                'vout': utxo['vout'],
                'script_sig': '',
                'sequence': 0xffffffff
            })
        
        return transaction
    
    def _address_to_script(self, address: str) -> str:
        """Convert address to scriptPubKey"""
        # This would implement proper script generation based on address type
        if address.startswith('1'):
            return f"76a914{hashlib.new('ripemd160', hashlib.sha256(address.encode()).digest()).hexdigest()}88ac"
        elif address.startswith('3'):
            return f"a914{hashlib.new('ripemd160', hashlib.sha256(address.encode()).digest()).hexdigest()}87"
        else:
            # Default to P2PKH
            return f"76a914{hashlib.new('ripemd160', hashlib.sha256(address.encode()).digest()).hexdigest()}88ac"
    
    def _sign_transaction(self, transaction: Dict) -> Dict:
        """Sign transaction with proper canonical signing"""
        if not self.master_key:
            raise ValueError("Master key not available")
        
        signed_tx = transaction.copy()
        signed_tx['vin'] = signed_tx['vin'].copy()
        
        # Sign each input
        for i, tx_input in enumerate(signed_tx['vin']):
            # Get the UTXO being spent
            utxo = self._get_utxo(tx_input['txid'], tx_input['vout'])
            if not utxo:
                raise ValueError(f"UTXO not found: {tx_input['txid']}:{tx_input['vout']}")
            
            # Create signature hash
            sighash = self._create_signature_hash(transaction, i, utxo['script_pubkey'])
            
            # Sign the hash
            signature = self._sign_data(sighash, self.master_key.private_key)
            
            # Create scriptSig
            public_key = self.master_key.public_key.hex()
            script_sig = self._create_script_sig(signature, public_key)
            
            tx_input['script_sig'] = script_sig
            tx_input['witness'] = []  # For SegWit
            
        return signed_tx
    
    def _get_utxo(self, txid: str, vout: int) -> Optional[Dict]:
        """Get UTXO from wallet index"""
        utxos = self.db.get_utxos()
        for utxo in utxos:
            if utxo['txid'] == txid and utxo['vout'] == vout:
                return utxo
        return None
    
    def _create_signature_hash(self, transaction: Dict, input_index: int, script_pubkey: str) -> bytes:
        """Create signature hash following Rayonix protocol"""
        # This implements the canonical transaction signing process
        # following Rayonix protocol specifications
        
        # Serialize transaction for signing
        serialized = self._serialize_for_signing(transaction, input_index, script_pubkey)
        
        # Double SHA256 hash
        return hashlib.sha256(hashlib.sha256(serialized).digest()).digest()
    
    def _serialize_for_signing(self, transaction: Dict, input_index: int, script_pubkey: str) -> bytes:
        """Serialize transaction for signing following Rayonix protocol"""
        # Implementation of canonical transaction serialization
        # This would follow Rayonix's specific serialization rules
        
        version = transaction['version'].to_bytes(4, 'little')
        input_count = len(transaction['vin']).to_bytes(1, 'little')
        
        # Serialize inputs
        inputs_data = b''
        for i, tx_in in enumerate(transaction['vin']):
            txid = bytes.fromhex(tx_in['txid'])[::-1]  # Reverse for little-endian
            vout = tx_in['vout'].to_bytes(4, 'little')
            
            if i == input_index:
                script = bytes.fromhex(script_pubkey)
                script_len = len(script).to_bytes(1, 'little')
            else:
                script = b''
                script_len = b'\x00'
            
            sequence = tx_in.get('sequence', 0xffffffff).to_bytes(4, 'little')
            
            inputs_data += txid + vout + script_len + script + sequence
        
        # Serialize outputs
        output_count = len(transaction['vout']).to_bytes(1, 'little')
        outputs_data = b''
        for tx_out in transaction['vout']:
            value = tx_out['value'].to_bytes(8, 'little')
            script_pubkey = bytes.fromhex(tx_out.get('script_pubkey', ''))
            script_len = len(script_pubkey).to_bytes(1, 'little')
            outputs_data += value + script_len + script_pubkey
        
        locktime = transaction.get('locktime', 0).to_bytes(4, 'little')
        sighash_type = b'\x01\x00\x00\x80'  # SIGHASH_ALL
        
        return version + input_count + inputs_data + output_count + outputs_data + locktime + sighash_type
    
    def _sign_data(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using ECDSA with proper canonical signatures"""
        # Use cryptography library for proper signing
        private_key_obj = ec.derive_private_key(
            int.from_bytes(private_key, 'big'),
            ec.SECP256K1(),
            default_backend()
        )
        
        signature = private_key_obj.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        
        # Ensure canonical signature
        return self._ensure_canonical_signature(signature)
    
    def _ensure_canonical_signature(self, signature: bytes) -> bytes:
        """Ensure signature is canonical (low S value)"""
        # Parse DER signature
        r_len = signature[3]
        r = signature[4:4+r_len]
        s_start = 4 + r_len + 2
        s_len = signature[s_start-1]
        s = signature[s_start:s_start+s_len]
        
        # Convert S to integer
        s_int = int.from_bytes(s, 'big')
        
        # SECP256k1 curve order
        curve_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        
        # Ensure S is low (canonical)
        if s_int > curve_order // 2:
            s_int = curve_order - s_int
            s = s_int.to_bytes((s_int.bit_length() + 7) // 8, 'big')
            
            # Reconstruct DER signature
            der_sig = b'\x30' + len(signature).to_bytes(1, 'big')
            der_sig += b'\x02' + r_len.to_bytes(1, 'big') + r
            der_sig += b'\x02' + s_len.to_bytes(1, 'big') + s
            
            return der_sig
        
        return signature
    
    def _create_script_sig(self, signature: bytes, public_key: str) -> str:
        """Create scriptSig for transaction"""
        # Push signature and public key
        sig_push = f"{len(signature):02x}{signature.hex()}"
        pubkey_push = f"{len(public_key)//2:02x}{public_key}"
        
        return f"{sig_push}{pubkey_push}"
    
    def _broadcast_transaction(self, transaction: Dict) -> Optional[str]:
        """Broadcast transaction to network"""
        if not self.rayonix_coin:
            raise ValueError("No blockchain reference available")
        
        try:
            # Convert to Rayonix transaction format
            rayonix_tx = self._convert_to_rayonix_transaction(transaction)
            
            # Add to mempool
            txid = self.rayonix_coin.add_transaction_to_mempool(rayonix_tx)
            
            logger.info(f"Transaction broadcasted: {txid}")
            return txid
            
        except Exception as e:
            logger.error(f"Transaction broadcast failed: {e}")
            return None
    
    def _convert_to_rayonix_transaction(self, transaction: Dict) -> Any:
        """Convert internal transaction format to Rayonix transaction"""
        # This would convert to the specific transaction format expected by Rayonix
        # For now, we'll assume it's compatible
        
        # Create UTXOTransaction object
        tx = UTXOTransaction()
        tx.tx_inputs = []
        tx.tx_outputs = []
        
        # Add inputs
        for vin in transaction['vin']:
            tx_input = {
                'prev_tx_hash': vin['txid'],
                'prev_tx_out_index': vin['vout'],
                'script_sig': vin['script_sig'],
                'sequence': vin.get('sequence', 0xffffffff)
            }
            tx.tx_inputs.append(tx_input)
        
        # Add outputs
        for vout in transaction['vout']:
            tx_output = {
                'value': vout['value'],
                'script_pubkey': vout['script_pubkey']
            }
            tx.tx_outputs.append(tx_output)
        
        tx.version = transaction['version']
        tx.lock_time = transaction.get('locktime', 0)
        
        return tx
    
    def _update_after_transaction(self, transaction: Dict, txid: str):
        """Update wallet state after transaction"""
        # Create transaction record
        tx_record = Transaction(
            txid=txid,
            amount=sum(vout['value'] for vout in transaction['vout']),
            fee=self._calculate_transaction_fee(transaction),
            confirmations=0,
            timestamp=int(time.time()),
            block_height=None,
            from_address="self",  # Would be actual from addresses
            to_address=transaction['vout'][0]['address'],
            status="pending",
            direction="sent",
            memo=transaction['metadata'].get('memo')
        )
        
        # Save transaction
        self.db.save_transaction(tx_record)
        self.transactions[txid] = tx_record
        
        # Mark UTXOs as spent
        for vin in transaction['vin']:
            self.db.mark_utxo_spent(vin['txid'], vin['vout'], txid)
        
        # Update wallet state
        self.state.tx_count += 1
        self.state.total_sent += tx_record.amount
        self.db.save_wallet_state(self.state)
    
    def _calculate_transaction_fee(self, transaction: Dict) -> int:
        """Calculate transaction fee"""
        input_total = sum(self._get_utxo_value(vin['txid'], vin['vout']) for vin in transaction['vin'])
        output_total = sum(vout['value'] for vout in transaction['vout'])
        return input_total - output_total
    
    def _get_utxo_value(self, txid: str, vout: int) -> int:
        """Get UTXO value"""
        utxo = self._get_utxo(txid, vout)
        return utxo['amount'] if utxo else 0
    
    def backup(self, backup_path: str) -> bool:
        """Backup wallet to encrypted file"""
        try:
            if self.locked:
                raise ValueError("Wallet must be unlocked for backup")
            
            # Prepare backup data
            backup_data = {
                'wallet_id': self.wallet_id,
                'config': asdict(self.config),
                'addresses': [asdict(addr) for addr in self.addresses.values()],
                'transactions': [asdict(tx) for tx in self.transactions.values()],
                'state': asdict(self.state),
                'backup_timestamp': int(time.time())
            }
            
            # Encrypt backup
            encrypted_data = self._encrypt_backup(backup_data)
            
            # Write to file
            with open(backup_path, 'wb') as f:
                f.write(encrypted_data)
            
            logger.info(f"Wallet backed up to: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return False
    
    def _encrypt_backup(self, data: Dict) -> bytes:
        """Encrypt wallet backup"""
        if not self.encryption_key:
            raise ValueError("Encryption key not available")
        
        # Serialize data
        json_data = json.dumps(data).encode()
        
        # Generate IV
        iv = os.urandom(16)
        
        # Encrypt
        cipher = Cipher(
            algorithms.AES(self.encryption_key.get_value()),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad data
        pad_length = 16 - (len(json_data) % 16)
        padded_data = json_data + bytes([pad_length] * pad_length)
        
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        
        return iv + encrypted
    
    def restore(self, backup_path: str, passphrase: str) -> bool:
        """Restore wallet from backup"""
        try:
            # Read backup file
            with open(backup_path, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt backup
            backup_data = self._decrypt_backup(encrypted_data, passphrase)
            
            # Restore wallet state
            self.wallet_id = backup_data['wallet_id']
            self.config = WalletConfig(**backup_data['config'])
            
            # Restore addresses
            self.addresses = {}
            for addr_data in backup_data['addresses']:
                addr_info = AddressInfo(**addr_data)
                self.addresses[addr_info.address] = addr_info
                self.db.save_address(addr_info)
            
            # Restore transactions
            self.transactions = {}
            for tx_data in backup_data['transactions']:
                transaction = Transaction(**tx_data)
                self.transactions[transaction.txid] = transaction
                self.db.save_transaction(transaction)
            
            # Restore state
            self.state = WalletState(**backup_data['state'])
            self.db.save_wallet_state(self.state)
            
            logger.info("Wallet restored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def _decrypt_backup(self, encrypted_data: bytes, passphrase: str) -> Dict:
        """Decrypt wallet backup"""
        # Extract IV
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        
        # Derive key from passphrase
        key = self._derive_encryption_key(passphrase, iv)
        
        # Decrypt
        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        
        # Remove padding
        pad_length = padded_data[-1]
        if pad_length < 1 or pad_length > 16:
            raise ValueError("Invalid padding")
        
        json_data = padded_data[:-pad_length]
        
        return json.loads(json_data.decode())
    
    def export_private_key(self, address: str) -> Optional[str]:
        """Export private key for address"""
        if self.locked:
            raise ValueError("Wallet is locked")
        
        if address not in self.addresses:
            raise ValueError("Address not found in wallet")
        
        addr_info = self.addresses[address]
        
        # For HD wallets, derive the private key from the derivation path
        if self.config.wallet_type == WalletType.HD and self.master_key:
            bip32 = BIP32.from_seed(self.master_key.private_key)
            private_key = bip32.get_privkey_from_path(addr_info.derivation_path)
            return private_key.hex()
        
        # For non-HD wallets, return the master key
        elif self.config.wallet_type == WalletType.NON_HD and self.master_key:
            return self.master_key.private_key.hex()
        
        return None
    
    def import_address(self, address: str, label: Optional[str] = None) -> bool:
        """Import watch-only address"""
        if address in self.addresses:
            return False
        
        address_info = AddressInfo(
            address=address,
            index=-1,  # No derivation index for imported addresses
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
        
        logger.info(f"Imported watch-only address: {address}")
        return True
    
    def get_qr_code(self, address: str, amount: Optional[int] = None, 
                   memo: Optional[str] = None) -> Optional[bytes]:
        """Generate QR code for address or payment request"""
        try:
            # Create payment URI
            uri = f"rayonix:{address}"
            if amount:
                uri += f"?amount={amount / 100000000:.8f}"  # Convert to base units
            if memo:
                uri += f"&memo={memo}"
            
            # Generate QR code
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(uri)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            
            return img_bytes.getvalue()
            
        except Exception as e:
            logger.error(f"QR code generation failed: {e}")
            return None
    
    def validate_address(self, address: str) -> bool:
        """Validate cryptocurrency address"""
        try:
            # Check address format based on configured address type
            if self.config.address_type == AddressType.RAYONIX:
                return self._validate_rayonix_address(address)
            elif self.config.address_type == AddressType.P2PKH:
                return self._validate_p2pkh_address(address)
            elif self.config.address_type == AddressType.P2WPKH:
                return self._validate_bech32_address(address)
            else:
                return self._validate_rayonix_address(address)
                
        except Exception:
            return False
    
    def _validate_rayonix_address(self, address: str) -> bool:
        """Validate Rayonix address"""
        try:
            # Decode base58
            decoded = base58.b58decode(address)
            
            # Check length
            if len(decoded) != 25:
                return False
            
            # Extract payload and checksum
            payload = decoded[:-4]
            checksum = decoded[-4:]
            
            # Verify checksum
            calculated_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            return checksum == calculated_checksum
            
        except Exception:
            return False
    
    def _validate_p2pkh_address(self, address: str) -> bool:
        """Validate P2PKH address"""
        try:
            decoded = base58.b58decode(address)
            return len(decoded) == 25
        except Exception:
            return False
    
    def _validate_bech32_address(self, address: str) -> bool:
        """Validate Bech32 address"""
        try:
            hrp, data, spec = bech32.decode(address)
            return hrp in ['bc', 'tb', 'ray', 'tray'] and data is not None
        except Exception:
            return False
    
    def set_multisig(self, required: int, public_keys: List[str]) -> bool:
        """Setup multi-signature wallet"""
        try:
            if len(public_keys) < required:
                raise ValueError("Not enough public keys for required signatures")
            
            self.multisig_config = {
                'required': required,
                'public_keys': public_keys,
                'total': len(public_keys)
            }
            
            # Generate multisig address
            redeem_script = self._create_multisig_redeem_script(required, public_keys)
            multisig_address = self._derive_multisig_address(redeem_script)
            
            # Save as watch-only address
            self.import_address(multisig_address, "Multisig")
            
            logger.info(f"Multisig wallet configured: {required} of {len(public_keys)}")
            return True
            
        except Exception as e:
            logger.error(f"Multisig setup failed: {e}")
            return False
    
    def _create_multisig_redeem_script(self, required: int, public_keys: List[str]) -> str:
        """Create multisig redeem script"""
        # OP_M required OP_N [pubkeys] OP_N OP_CHECKMULTISIG
        op_m = bytes([80 + required])  # OP_1 = 0x51, OP_2 = 0x52, etc.
        op_n = bytes([80 + len(public_keys)])
        op_checkmultisig = b'\xae'
        
        script = op_m
        for pubkey in public_keys:
            pubkey_bytes = bytes.fromhex(pubkey)
            script += bytes([len(pubkey_bytes)]) + pubkey_bytes
        script += op_n + op_checkmultisig
        
        return script.hex()
    
    def _derive_multisig_address(self, redeem_script: str) -> str:
        """Derive multisig address from redeem script"""
        script_hash = hashlib.sha256(bytes.fromhex(redeem_script)).digest()
        ripemd_hash = hashlib.new('ripemd160', script_hash).digest()
        
        # P2SH address format (version byte 0x05 for mainnet)
        version_byte = b'\x05' if self.config.network == "mainnet" else b'\xc4'
        payload = version_byte + ripemd_hash
        
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        address_bytes = payload + checksum
        
        return base58.b58encode(address_bytes).decode('ascii')
    
    def add_cosigner(self, public_key: str) -> bool:
        """Add cosigner to multisig wallet"""
        if not self.multisig_config:
            raise ValueError("Multisig not configured")
        
        if public_key in self.multisig_config['public_keys']:
            return False
        
        self.multisig_config['public_keys'].append(public_key)
        self.multisig_config['total'] += 1
        
        # Regenerate multisig address
        redeem_script = self._create_multisig_redeem_script(
            self.multisig_config['required'],
            self.multisig_config['public_keys']
        )
        multisig_address = self._derive_multisig_address(redeem_script)
        
        # Update address
        if multisig_address in self.addresses:
            del self.addresses[multisig_address]
        self.import_address(multisig_address, "Multisig")
        
        logger.info(f"Cosigner added: {public_key}")
        return True
    
    def create_multisig_transaction(self, to_address: str, amount: int, 
                                  fee_rate: Optional[int] = None) -> Dict:
        """Create multisig transaction requiring multiple signatures"""
        if not self.multisig_config:
            raise ValueError("Multisig not configured")
        
        # Create unsigned transaction
        transaction = self._create_transaction(to_address, amount, [], fee_rate or self.config.transaction_fees["medium"], None)
        
        # Add multisig metadata
        transaction['multisig'] = {
            'required': self.multisig_config['required'],
            'public_keys': self.multisig_config['public_keys'],
            'signatures': [],
            'redeem_script': self._create_multisig_redeem_script(
                self.multisig_config['required'],
                self.multisig_config['public_keys']
            )
        }
        
        return transaction
    
    def sign_multisig_transaction(self, transaction: Dict) -> Dict:
        """Sign multisig transaction"""
        if self.locked or not self.master_key:
            raise ValueError("Wallet is locked")
        
        if 'multisig' not in transaction:
            raise ValueError("Not a multisig transaction")
        
        # Create signature hash
        sighash = self._create_signature_hash(transaction, 0, transaction['multisig']['redeem_script'])
        
        # Sign with our private key
        signature = self._sign_data(sighash, self.master_key.private_key)
        
        # Add signature to transaction
        transaction['multisig']['signatures'].append({
            'public_key': self.master_key.public_key.hex(),
            'signature': signature.hex()
        })
        
        return transaction
    
    def finalize_multisig_transaction(self, transaction: Dict) -> Optional[str]:
        """Finalize multisig transaction with required signatures"""
        if 'multisig' not in transaction:
            raise ValueError("Not a multisig transaction")
        
        multisig_info = transaction['multisig']
        
        # Check if we have enough signatures
        if len(multisig_info['signatures']) < multisig_info['required']:
            raise ValueError(f"Need {multisig_info['required']} signatures, got {len(multisig_info['signatures'])}")
        
        # Create final scriptSig
        script_sig = self._create_multisig_script_sig(multisig_info['signatures'], multisig_info['redeem_script'])
        
        # Add scriptSig to transaction
        for vin in transaction['vin']:
            vin['script_sig'] = script_sig
        
        # Remove multisig metadata
        del transaction['multisig']
        
        # Broadcast transaction
        return self._broadcast_transaction(transaction)
    
    def _create_multisig_script_sig(self, signatures: List[Dict], redeem_script: str) -> str:
        """Create scriptSig for multisig transaction"""
        # OP_0 [signatures] [redeem_script]
        script_sig = '00'  # OP_0
        
        for sig in signatures:
            signature_hex = sig['signature']
            script_sig += f"{len(signature_hex)//2:02x}{signature_hex}"
        
        redeem_script_bytes = bytes.fromhex(redeem_script)
        script_sig += f"{len(redeem_script_bytes):02x}{redeem_script}"
        
        return script_sig
    
    def get_transaction_fee_estimate(self, priority: str = "medium") -> int:
        """Get transaction fee estimate"""
        fee_rate = self.config.transaction_fees.get(priority, self.config.transaction_fees["medium"])
        
        # Estimate typical transaction size
        typical_size = 250  # bytes for typical transaction
        
        return fee_rate * typical_size
    
    def sweep_private_key(self, private_key: str, to_address: str, 
                         fee_rate: Optional[int] = None) -> Optional[str]:
        """Sweep funds from private key to wallet address"""
        try:
            # Create temporary wallet from private key
            temp_wallet = RayonixWallet(WalletConfig(wallet_type=WalletType.NON_HD))
            if not temp_wallet.create_from_private_key(private_key):
                raise ValueError("Invalid private key")
            
            # Get balance
            balance = temp_wallet.get_balance()
            if balance.total <= 0:
                raise ValueError("No funds to sweep")
            
            # Create transaction spending all funds
            transaction = temp_wallet._create_sweep_transaction(to_address, fee_rate or self.config.transaction_fees["high"])
            
            # Sign transaction
            signed_tx = temp_wallet._sign_transaction(transaction)
            
            # Broadcast transaction
            txid = temp_wallet._broadcast_transaction(signed_tx)
            
            # Cleanup
            temp_wallet.lock()
            
            logger.info(f"Funds swept: {txid}")
            return txid
            
        except Exception as e:
            logger.error(f"Sweep failed: {e}")
            return None
    
    def _create_sweep_transaction(self, to_address: str, fee_rate: int) -> Dict:
        """Create transaction spending all available funds"""
        utxos = self.db.get_utxos()
        total_input = sum(utxo['amount'] for utxo in utxos)
        
        if total_input <= 0:
            raise ValueError("No funds available")
        
        # Estimate fee
        fee = self._estimate_fee(len(utxos), 1, fee_rate)  # 1 output
        
        if total_input <= fee:
            raise ValueError("Insufficient funds to pay fee")
        
        amount = total_input - fee
        
        transaction = {
            'version': 1,
            'locktime': 0,
            'vin': [],
            'vout': [
                {
                    'value': amount,
                    'address': to_address,
                    'script_pubkey': self._address_to_script(to_address)
                }
            ]
        }
        
        # Add inputs
        for utxo in utxos:
            transaction['vin'].append({
                'txid': utxo['txid'],
                'vout': utxo['vout'],
                'script_sig': '',
                'sequence': 0xffffffff
            })
        
        return transaction
    
    def close(self):
        """Cleanly close wallet"""
        try:
            # Stop background tasks
            self.running = False
            self.synchronizer.stop()
            
            # Lock wallet
            self.lock()
            
            # Close database
            if hasattr(self.db, 'close'):
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

# Utility functions for secure operations
def generate_mnemonic(strength: int = 256) -> str:
    """Generate BIP39 mnemonic phrase"""
    mnemo = Mnemonic("english")
    return mnemo.generate(strength=strength)

def validate_mnemonic(mnemonic_phrase: str) -> bool:
    """Validate BIP39 mnemonic phrase"""
    mnemo = Mnemonic("english")
    return mnemo.check(mnemonic_phrase)

def mnemonic_to_seed(mnemonic_phrase: str, passphrase: str = "") -> bytes:
    """Convert mnemonic to seed using BIP39"""
    mnemo = Mnemonic("english")
    return mnemo.to_seed(mnemonic_phrase, passphrase)

def create_hd_wallet(mnemonic_phrase: Optional[str] = None, passphrase: str = "", 
                    network: str = "mainnet") -> RayonixWallet:
    """Convenience function to create HD wallet"""
    wallet = RayonixWallet(WalletConfig(
        wallet_type=WalletType.HD,
        key_derivation=KeyDerivation.BIP44,
        network=network
    ))
    
    if mnemonic_phrase:
        if not wallet.create_from_mnemonic(mnemonic_phrase, passphrase):
            raise ValueError("Failed to create wallet from mnemonic")
    else:
        mnemonic, _ = wallet.create_hd_wallet()
        logger.info(f"New wallet created with mnemonic: {mnemonic}")
    
    return wallet

def create_wallet_from_private_key(private_key: str, network: str = "mainnet") -> RayonixWallet:
    """Convenience function to create wallet from private key"""
    wallet = RayonixWallet(WalletConfig(
        wallet_type=WalletType.NON_HD,
        network=network
    ))
    
    if not wallet.create_from_private_key(private_key):
        raise ValueError("Failed to create wallet from private key")
    
    return wallet

# Example usage and testing
if __name__ == "__main__":
    # Create a new HD wallet
    wallet = create_hd_wallet()
    
    # Unlock wallet (in real usage, this would use a proper passphrase)
    wallet.unlock("")
    
    # Get wallet balance
    balance = wallet.get_balance()
    print(f"Wallet balance: {balance.total} satoshis")
    
    # Generate a new address
    address = wallet.derive_address(0)
    print(f"New address: {address}")
    
    # Lock wallet when done
    wallet.lock()
    
    # Close wallet
    wallet.close()
                
        