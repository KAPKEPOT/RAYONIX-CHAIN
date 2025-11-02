# utxo_system/core/utxoset.py
import plyvel
import json
import struct
import threading
import pickle
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

    # === State Management Methods Required by StateManager ===
    
    def create_snapshot(self) -> Any:
        """Create a snapshot of the current UTXO set state"""
        with self.lock:
            # Create a comprehensive snapshot of all UTXO data
            snapshot = {
                'last_block_height': self.get_last_processed_block_height(),
                'utxos': {},
                'spent_utxos': {},
                'address_index': {}
            }
            
            # Capture all unspent UTXOs
            for key, value in self.db.iterator(prefix=UTXO_PREFIX):
                utxo_id = key.decode('utf-8').split(':', 1)[1]
                snapshot['utxos'][utxo_id] = value
            
            # Capture all spent UTXOs
            for key, value in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                utxo_id = key.decode('utf-8').split(':', 1)[1]
                snapshot['spent_utxos'][utxo_id] = value
            
            # Capture address index
            for key, value in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                address = key.decode('utf-8').split(':', 1)[1]
                snapshot['address_index'][address] = value
            
            return snapshot
    
    def restore_snapshot(self, snapshot: Any):
        """Restore the UTXO set from a snapshot"""
        with self.lock:
            with self.atomic_write() as batch:
                # Clear existing data
                for key, _ in self.db.iterator(prefix=UTXO_PREFIX):
                    batch.delete(key)
                for key, _ in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                    batch.delete(key)
                for key, _ in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                    batch.delete(key)
                
                # Restore UTXOs
                for utxo_id, utxo_data in snapshot['utxos'].items():
                    key = self._get_utxo_key(utxo_id)
                    batch.put(key, utxo_data)
                
                # Restore spent UTXOs
                for utxo_id, utxo_data in snapshot['spent_utxos'].items():
                    key = self._get_spent_utxo_key(utxo_id)
                    batch.put(key, utxo_data)
                
                # Restore address index
                for address, index_data in snapshot['address_index'].items():
                    key = ADDRESS_INDEX_PREFIX + address.encode('utf-8')
                    batch.put(key, index_data)
                
                # Restore last block height
                if 'last_block_height' in snapshot:
                    batch.put(LAST_BLOCK_HEIGHT_KEY, 
                             struct.pack('>I', snapshot['last_block_height']))
    
    def to_bytes(self) -> bytes:
        """Serialize the entire UTXO set to bytes for persistence"""
        snapshot = self.create_snapshot()
        return pickle.dumps(snapshot)
    
    def from_bytes(self, data: bytes):
        """Deserialize the UTXO set from bytes"""
        snapshot = pickle.loads(data)
        self.restore_snapshot(snapshot)
    
    def calculate_hash(self) -> str:
        """Calculate a hash representing the current state of the UTXO set"""
        import hashlib
        
        with self.lock:
            # Create a hash of all UTXO data for integrity checking
            hasher = hashlib.sha256()
            
            # Hash all unspent UTXOs
            for key, value in sorted(self.db.iterator(prefix=UTXO_PREFIX)):
                hasher.update(key)
                hasher.update(value)
            
            # Hash all spent UTXOs
            for key, value in sorted(self.db.iterator(prefix=SPENT_UTXO_PREFIX)):
                hasher.update(key)
                hasher.update(value)
            
            # Hash address index
            for key, value in sorted(self.db.iterator(prefix=ADDRESS_INDEX_PREFIX)):
                hasher.update(key)
                hasher.update(value)
            
            # Hash last block height
            last_height = self.get_last_processed_block_height()
            hasher.update(struct.pack('>I', last_height))
            
            return hasher.hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the UTXO set"""
        try:
            with self.lock:
                # Verify that all indexed UTXOs exist
                for key, value in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                    address = key.decode('utf-8').split(':', 1)[1]
                    utxo_ids = json.loads(value.decode('utf-8'))
                    
                    for utxo_id in utxo_ids:
                        utxo = self.get_utxo(utxo_id)
                        if not utxo:
                            logger.error(f"UTXO {utxo_id} indexed for address {address} but not found")
                            return False
                        if utxo.address != address:
                            logger.error(f"UTXO {utxo_id} address mismatch: expected {address}, got {utxo.address}")
                            return False
                
                # Verify that no UTXO is both spent and unspent
                utxo_ids = set()
                for key, _ in self.db.iterator(prefix=UTXO_PREFIX):
                    utxo_id = key.decode('utf-8').split(':', 1)[1]
                    if utxo_id in utxo_ids:
                        logger.error(f"Duplicate UTXO found: {utxo_id}")
                        return False
                    utxo_ids.add(utxo_id)
                
                for key, _ in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                    utxo_id = key.decode('utf-8').split(':', 1)[1]
                    if utxo_id in utxo_ids:
                        logger.error(f"UTXO {utxo_id} exists in both spent and unspent sets")
                        return False
                
                return True
                
        except Exception as e:
            logger.error(f"UTXO set integrity verification failed: {e}")
            return False
    
    def get_utxo_count(self) -> int:
        """Get the total number of unspent UTXOs"""
        count = 0
        for _, _ in self.db.iterator(prefix=UTXO_PREFIX):
            count += 1
        return count
    
    def revert_transaction(self, transaction: Transaction) -> bool:
        """Revert a transaction by restoring inputs and removing outputs"""
        with self.atomic_write() as batch:
            try:
                # Remove outputs (new UTXOs created by this transaction)
                for i in range(len(transaction.outputs)):
                    utxo_id = f"{transaction.hash}:{i}"
                    
                    # Delete from UTXO set
                    utxo_key = self._get_utxo_key(utxo_id)
                    utxo_data = self.db.get(utxo_key)
                    
                    if utxo_data:
                        utxo = deserialize_utxo(utxo_data)
                        batch.delete(utxo_key)
                        self.indexer.remove_utxo_from_address(utxo.address, utxo_id, batch)
                
                # Restore inputs (UTXOs spent by this transaction)
                for tx_input in transaction.inputs:
                    utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
                    spent_key = self._get_spent_utxo_key(utxo_id)
                    spent_data = self.db.get(spent_key)
                    
                    if spent_data:
                        utxo = deserialize_utxo(spent_data)
                        utxo.spent = False
                        
                        # Move back to unspent UTXOs
                        utxo_key = self._get_utxo_key(utxo_id)
                        batch.put(utxo_key, serialize_utxo(utxo))
                        batch.delete(spent_key)
                        
                        # Restore to address index
                        self.indexer.add_utxo_to_address(utxo.address, utxo_id, batch)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to revert transaction {transaction.hash}: {e}")
                return False
    
    def revert_transaction_by_hash(self, tx_hash: str) -> bool:
        """Revert a transaction by its hash"""
        # This would need to lookup the transaction first, then call revert_transaction
        # Implementation depends on transaction storage
        logger.warning(f"revert_transaction_by_hash not fully implemented for tx: {tx_hash}")
        return False
       
    def get_total_supply(self) -> int:
    	"""Calculate total coin supply by summing all unspent UTXOs"""
    	with self.lock:
    		total = 0
    		
    		# Sum all unspent UTXOs
    		for key, value in self.db.iterator(prefix=UTXO_PREFIX):
    			try:
    				utxo = deserialize_utxo(value)
    				total += utxo.amount
    			except Exception as e:
    				logger.error(f"Error deserializing UTXO {key}: {e}")
    				raise RuntimeError(f"Failed to calculate total supply: UTXO deserialization error at {key}") from e
    		
    		logger.info(f"Total supply calculated from UTXO set: {total}")
    		return total
    	
    def get_circulating_supply(self) -> int:
    	"""Get circulating supply (same as total supply for UTXO-based system)"""
    	return self.get_total_supply()
    