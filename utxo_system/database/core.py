# utxo_system/database/core.py
import plyvel
import json
import struct
import threading
from typing import List, Dict, Set, Tuple, Optional, Iterator, Any
from contextlib import contextmanager
from utxo_system.models.utxo import UTXO
from utxo_system.models.transaction import Transaction
from utxo_system.utils.logging_config import logger
from utxo_system.database.indexing import AddressIndexer
from utxo_system.database.serialization import serialize_utxo, deserialize_utxo

# Database key prefixes
UTXO_PREFIX = b'u:'
ADDRESS_INDEX_PREFIX = b'a:'
SPENT_UTXO_PREFIX = b's:'
METADATA_PREFIX = b'm:'
LAST_BLOCK_HEIGHT_KEY = b'm:last_block_height'

class UTXOSet:
    def __init__(self, db_path: str = './utxo_db'):
        self.db_path = db_path
        self.db = plyvel.DB(db_path, create_if_missing=True)
        self.lock = threading.RLock()
        self.indexer = AddressIndexer(self.db)
        
    def close(self):
        """Close the database connection"""
        self.db.close()
    
    def _get_utxo_key(self, utxo_id: str) -> bytes:
        return UTXO_PREFIX + utxo_id.encode('utf-8')
    
    def _get_spent_utxo_key(self, utxo_id: str) -> bytes:
        return SPENT_UTXO_PREFIX + utxo_id.encode('utf-8')
    
    @contextmanager
    def atomic_write(self):
        """Context manager for atomic database operations"""
        batch = self.db.write_batch()
        try:
            yield batch
            batch.write()
        except Exception:
            batch.clear()
            raise
    
    def add_utxo(self, utxo: UTXO, batch=None):
        """Add a UTXO to the database"""
        utxo_id = utxo.id
        utxo_key = self._get_utxo_key(utxo_id)
        
        write_batch = batch if batch else self.db
        write_batch.put(utxo_key, serialize_utxo(utxo))
        self.indexer.add_utxo_to_address(utxo.address, utxo_id, write_batch)
    
    def spend_utxo(self, utxo_id: str, batch=None):
        """Mark a UTXO as spent"""
        with self.lock:
            utxo_key = self._get_utxo_key(utxo_id)
            utxo_data = self.db.get(utxo_key)
            
            if not utxo_data:
                return False
            
            utxo = deserialize_utxo(utxo_data)
            utxo.spent = True
            
            write_batch = batch if batch else self.db
            
            # Move to spent UTXOs
            spent_key = self._get_spent_utxo_key(utxo_id)
            write_batch.put(spent_key, serialize_utxo(utxo))
            write_batch.delete(utxo_key)
            
            # Update address index
            self.indexer.remove_utxo_from_address(utxo.address, utxo_id, write_batch)
            
            return True
    
    def get_utxos_for_address(self, address: str, current_block_height: int = 0, 
                             current_time: int = 0) -> List[UTXO]:
        """Get all spendable UTXOs for an address"""
        with self.lock:
            utxo_ids = self.indexer.get_utxos_for_address(address)
            utxos = []
            
            for utxo_id in utxo_ids:
                utxo = self.get_utxo(utxo_id)
                if utxo and utxo.is_spendable(current_block_height, current_time):
                    utxos.append(utxo)
            
            return utxos
    
    def get_balance(self, address: str, current_block_height: int = 0, 
                   current_time: int = 0) -> int:
        """Get the balance for an address"""
        utxos = self.get_utxos_for_address(address, current_block_height, current_time)
        return sum(utxo.amount for utxo in utxos)
    
    def find_spendable_utxos(self, address: str, amount: int, 
                            current_block_height: int = 0, current_time: int = 0) -> Tuple[List[UTXO], int]:
        """Find UTXOs to spend for a given amount"""
        utxos = self.get_utxos_for_address(address, current_block_height, current_time)
        utxos.sort(key=lambda x: x.amount, reverse=True)
        
        total = 0
        selected = []
        
        for utxo in utxos:
            if total >= amount:
                break
            selected.append(utxo)
            total += utxo.amount
        
        return selected, total
    
    def get_utxo(self, utxo_id: str) -> Optional[UTXO]:
        """Get a UTXO by ID"""
        with self.lock:
            # First check unspent UTXOs
            utxo_key = self._get_utxo_key(utxo_id)
            utxo_data = self.db.get(utxo_key)
            
            if utxo_data:
                return deserialize_utxo(utxo_data)
            
            # Then check spent UTXOs
            spent_key = self._get_spent_utxo_key(utxo_id)
            spent_data = self.db.get(spent_key)
            
            if spent_data:
                return deserialize_utxo(spent_data)
            
            return None
    
    def process_transaction(self, transaction: Transaction, block_height: int = 0):
        """
        Process a transaction by spending inputs and creating new UTXOs from outputs.
        """
        with self.atomic_write() as batch:
            # Spend the inputs
            for tx_input in transaction.inputs:
                utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
                self.spend_utxo(utxo_id, batch)
            
            # Create new UTXOs from outputs
            for i, output in enumerate(transaction.outputs):
                utxo = UTXO(
                    tx_hash=transaction.hash,
                    output_index=i,
                    address=output.address,
                    amount=output.amount
                )
                
                # Set locktime if specified in output
                utxo.locktime = output.locktime
                
                # Record the block height when this UTXO was created
                utxo.created_at_block = block_height
                
                self.add_utxo(utxo, batch)
    
    def process_block_transactions(self, transactions: List[Transaction], block_height: int) -> bool:
        """
        Process all transactions in a block atomically.
        """
        with self.atomic_write() as batch:
            # Process all transactions
            for tx in transactions:
                # Spend the inputs
                for tx_input in tx.inputs:
                    utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
                    self.spend_utxo(utxo_id, batch)
                
                # Create new UTXOs from outputs
                for i, output in enumerate(tx.outputs):
                    utxo = UTXO(
                        tx_hash=tx.hash,
                        output_index=i,
                        address=output.address,
                        amount=output.amount
                    )
                    
                    # Set locktime if specified in output
                    utxo.locktime = output.locktime
                    
                    # Record the block height when this UTXO was created
                    utxo.created_at_block = block_height
                    
                    self.add_utxo(utxo, batch)
            
            # Update last processed block height
            self.db.put(LAST_BLOCK_HEIGHT_KEY, struct.pack('>I', block_height))
            
            return True
    
    def get_last_processed_block_height(self) -> int:
        """Get the last block height that was processed"""
        data = self.db.get(LAST_BLOCK_HEIGHT_KEY)
        if data:
            return struct.unpack('>I', data)[0]
        return 0
    
    def rebuild_indexes(self):
        """Rebuild all indexes from UTXO data"""
        with self.lock:
            with self.atomic_write() as batch:
                self.indexer.rebuild_indexes(batch)
    
    def compact_database(self):
        """Compact the database to reclaim space"""
        self.db.compact_range()
    
    def get_stats(self) -> Dict:
        """Get statistics about the UTXO set"""
        with self.lock:
            stats = {
                'total_utxos': 0,
                'total_spent_utxos': 0,
                'total_addresses': self.indexer.get_address_count(),
                'total_size_mb': 0
            }
            
            # Count UTXOs
            for _, _ in self.db.iterator(prefix=UTXO_PREFIX):
                stats['total_utxos'] += 1
            
            # Count spent UTXOs
            for _, _ in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                stats['total_spent_utxos'] += 1
            
            # Estimate database size
            stats['total_size_mb'] = sum(1 for _ in self.db.iterator()) * 0.0001
            
            return stats