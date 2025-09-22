import sqlite3
import json
import threading
from typing import List, Dict, Optional, Any
from contextlib import contextmanager

from rayonix_wallet.core.types import AddressInfo, Transaction, WalletState
from rayonix_wallet.core.exceptions import DatabaseError

class WalletDatabase:
    """SQLite database layer for wallet storage"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        self._connection_pool = {}
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        thread_id = threading.get_ident()
        if thread_id not in self._connection_pool:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connection_pool[thread_id] = conn
        return self._connection_pool[thread_id]
    
    @contextmanager
    def _transaction(self):
        """Context manager for database transactions"""
        conn = self._get_connection()
        with self._lock:
            try:
                yield conn
                conn.commit()
            except Exception as e:
                conn.rollback()
                raise DatabaseError(f"Transaction failed: {e}")
    
    def _init_database(self):
        """Initialize database schema"""
        with self._transaction() as conn:
            # Wallet metadata table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS wallet_metadata (
                    id INTEGER PRIMARY KEY,
                    wallet_id TEXT UNIQUE NOT NULL,
                    wallet_type TEXT NOT NULL,
                    network TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_modified INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    config TEXT NOT NULL
                )
            ''')
            
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
                    is_used INTEGER DEFAULT 0,
                    is_change INTEGER DEFAULT 0,
                    labels TEXT DEFAULT '[]',
                    created_at INTEGER NOT NULL,
                    last_used INTEGER
                )
            ''')
            
            # Transactions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    txid TEXT PRIMARY KEY,
                    amount INTEGER NOT NULL,
                    fee INTEGER DEFAULT 0,
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
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
            ''')
            
            # Wallet state table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS wallet_state (
                    id INTEGER PRIMARY KEY,
                    sync_height INTEGER DEFAULT 0,
                    last_updated REAL NOT NULL,
                    tx_count INTEGER DEFAULT 0,
                    addresses_generated INTEGER DEFAULT 0,
                    addresses_used INTEGER DEFAULT 0,
                    total_received INTEGER DEFAULT 0,
                    total_sent INTEGER DEFAULT 0,
                    security_score INTEGER DEFAULT 0
                )
            ''')
            
            # UTXO table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS utxos (
                    txid TEXT NOT NULL,
                    vout INTEGER NOT NULL,
                    address TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    script_pubkey TEXT NOT NULL,
                    confirmations INTEGER DEFAULT 0,
                    is_spent INTEGER DEFAULT 0,
                    spent_txid TEXT,
                    created_at INTEGER NOT NULL,
                    PRIMARY KEY (txid, vout)
                )
            ''')
            
            # Create indexes
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addresses_derivation ON addresses(derivation_path)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_addresses_used ON addresses(is_used)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_tx_address ON transactions(from_address, to_address)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_utxo_address ON utxos(address)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_utxo_spent ON utxos(is_spent)')
    
    def save_address(self, address_info: AddressInfo):
        """Save address information"""
        with self._transaction() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO addresses 
                (address, index, derivation_path, balance, received, sent, tx_count, 
                 is_used, is_change, labels, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                address_info.address,
                address_info.index,
                address_info.derivation_path,
                address_info.balance,
                address_info.received,
                address_info.sent,
                address_info.tx_count,
                int(address_info.is_used),
                int(address_info.is_change),
                json.dumps(address_info.labels),
                int(time.time()),
                int(time.time()) if address_info.is_used else None
            ))
    
    def get_address(self, address: str) -> Optional[AddressInfo]:
        """Get address information"""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute('SELECT * FROM addresses WHERE address = ?', (address,)).fetchone()
            
            if row:
                return AddressInfo(
                    address=row['address'],
                    index=row['index'],
                    derivation_path=row['derivation_path'],
                    balance=row['balance'],
                    received=row['received'],
                    sent=row['sent'],
                    tx_count=row['tx_count'],
                    is_used=bool(row['is_used']),
                    is_change=bool(row['is_change']),
                    labels=json.loads(row['labels'])
                )
            return None
    
    def get_all_addresses(self) -> List[AddressInfo]:
        """Get all addresses"""
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute('SELECT * FROM addresses ORDER BY index, is_change').fetchall()
            
            addresses = []
            for row in rows:
                addresses.append(AddressInfo(
                    address=row['address'],
                    index=row['index'],
                    derivation_path=row['derivation_path'],
                    balance=row['balance'],
                    received=row['received'],
                    sent=row['sent'],
                    tx_count=row['tx_count'],
                    is_used=bool(row['is_used']),
                    is_change=bool(row['is_change']),
                    labels=json.loads(row['labels'])
                ))
            return addresses
    
    def save_transaction(self, transaction: Transaction):
        """Save transaction"""
        with self._transaction() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO transactions 
                (txid, amount, fee, confirmations, timestamp, block_height, from_address, 
                 to_address, status, direction, memo, exchange_rate, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(transaction.metadata),
                int(time.time()),
                int(time.time())
            ))
    
    def get_transactions(self, limit: int = 50, offset: int = 0) -> List[Transaction]:
        """Get transactions with pagination"""
        with self._lock:
            conn = self._get_connection()
            rows = conn.execute('''
                SELECT * FROM transactions 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset)).fetchall()
            
            transactions = []
            for row in rows:
                transactions.append(Transaction(
                    txid=row['txid'],
                    amount=row['amount'],
                    fee=row['fee'],
                    confirmations=row['confirmations'],
                    timestamp=row['timestamp'],
                    block_height=row['block_height'],
                    from_address=row['from_address'],
                    to_address=row['to_address'],
                    status=row['status'],
                    direction=row['direction'],
                    memo=row['memo'],
                    exchange_rate=row['exchange_rate'],
                    metadata=json.loads(row['metadata'])
                ))
            return transactions
    
    def save_wallet_state(self, state: WalletState):
        """Save wallet state"""
        with self._transaction() as conn:
            conn.execute('DELETE FROM wallet_state')
            conn.execute('''
                INSERT INTO wallet_state 
                (sync_height, last_updated, tx_count, addresses_generated, 
                 addresses_used, total_received, total_sent, security_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
    
    def get_wallet_state(self) -> Optional[WalletState]:
        """Get wallet state"""
        with self._lock:
            conn = self._get_connection()
            row = conn.execute('SELECT * FROM wallet_state LIMIT 1').fetchone()
            
            if row:
                return WalletState(
                    sync_height=row['sync_height'],
                    last_updated=row['last_updated'],
                    tx_count=row['tx_count'],
                    addresses_generated=row['addresses_generated'],
                    addresses_used=row['addresses_used'],
                    total_received=row['total_received'],
                    total_sent=row['total_sent'],
                    security_score=row['security_score']
                )
            return None
    
    def save_utxo(self, txid: str, vout: int, address: str, amount: int, 
                 script_pubkey: str, confirmations: int = 0):
        """Save UTXO"""
        with self._transaction() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO utxos 
                (txid, vout, address, amount, script_pubkey, confirmations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (txid, vout, address, amount, script_pubkey, confirmations, int(time.time())))
    
    def mark_utxo_spent(self, txid: str, vout: int, spent_txid: str):
        """Mark UTXO as spent"""
        with self._transaction() as conn:
            conn.execute('''
                UPDATE utxos 
                SET is_spent = 1, spent_txid = ? 
                WHERE txid = ? AND vout = ?
            ''', (spent_txid, txid, vout))
    
    def get_utxos(self, address: Optional[str] = None, unspent_only: bool = True) -> List[Dict]:
        """Get UTXOs"""
        with self._lock:
            conn = self._get_connection()
            
            query = 'SELECT * FROM utxos'
            params = []
            
            if address:
                query += ' WHERE address = ?'
                params.append(address)
                if unspent_only:
                    query += ' AND is_spent = 0'
            elif unspent_only:
                query += ' WHERE is_spent = 0'
            
            rows = conn.execute(query, params).fetchall()
            
            utxos = []
            for row in rows:
                utxos.append({
                    'txid': row['txid'],
                    'vout': row['vout'],
                    'address': row['address'],
                    'amount': row['amount'],
                    'script_pubkey': row['script_pubkey'],
                    'confirmations': row['confirmations'],
                    'is_spent': bool(row['is_spent']),
                    'spent_txid': row['spent_txid']
                })
            return utxos
    
    def close(self):
        """Close database connections"""
        with self._lock:
            for conn in self._connection_pool.values():
                conn.close()
            self._connection_pool.clear()
    
    def vacuum(self):
        """Optimize database"""
        with self._transaction() as conn:
            conn.execute('VACUUM')
    
    def backup(self, backup_path: str):
        """Create database backup"""
        import shutil
        with self._lock:
            self._get_connection().execute('BEGIN IMMEDIATE')
            try:
                shutil.copy2(self.db_path, backup_path)
                self._get_connection().execute('COMMIT')
            except Exception as e:
                self._get_connection().execute('ROLLBACK')
                raise DatabaseError(f"Backup failed: {e}")