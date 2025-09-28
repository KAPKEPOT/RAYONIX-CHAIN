# utxo_system/database/indexing.py
import json
from typing import List, Set, Optional, Any
import plyvel

ADDRESS_INDEX_PREFIX = b'a:'

class AddressIndexer:
    def __init__(self, db: plyvel.DB):
        self.db = db
    
    def _get_address_key(self, address: str) -> bytes:
        return ADDRESS_INDEX_PREFIX + address.encode('utf-8')
    
    def add_utxo_to_address(self, address: str, utxo_id: str, batch=None) -> None:
        address_key = self._get_address_key(address)
        write_batch = batch if batch else self.db
        
        # FIX: Use the original database for read operations, batch only for writes
        if batch:
            # When using batch, read from the main database
            existing_utxos = self._get_utxos_for_address_raw(address_key, self.db)
        else:
            # When not using batch, read from the provided db handle
            existing_utxos = self._get_utxos_for_address_raw(address_key, write_batch)
        
        existing_utxos.add(utxo_id)
        write_batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
    
    def remove_utxo_from_address(self, address: str, utxo_id: str, batch=None) -> None:
        address_key = self._get_address_key(address)
        write_batch = batch if batch else self.db
        
        # FIX: Use the original database for read operations, batch only for writes
        if batch:
            existing_utxos = self._get_utxos_for_address_raw(address_key, self.db)
        else:
            existing_utxos = self._get_utxos_for_address_raw(address_key, write_batch)
        
        existing_utxos.discard(utxo_id)
        
        if existing_utxos:
            write_batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
        else:
            write_batch.delete(address_key)
    
    def _get_utxos_for_address_raw(self, address_key: bytes, db_handle) -> Set[str]:
        existing_data = db_handle.get(address_key)
        if existing_data:
            return set(json.loads(existing_data.decode('utf-8')))
        return set()
    
    def get_utxos_for_address(self, address: str) -> List[str]:
        address_key = self._get_address_key(address)
        existing_data = self.db.get(address_key)
        
        if not existing_data:
            return []
        
        return json.loads(existing_data.decode('utf-8'))
    
    def get_address_count(self) -> int:
        count = 0
        for _, _ in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
            count += 1
        return count
    
    def rebuild_indexes(self, batch) -> None:
        """Rebuild address indexes from UTXO data"""
        # Clear existing address indexes
        for key, _ in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
            batch.delete(key)
        
        # Rebuild from UTXOs
        for key, value in self.db.iterator(prefix=b'u:'):
            from ..models.utxo import UTXO
            utxo = UTXO.deserialize(value)
            utxo_id = key[len(b'u:'):].decode('utf-8')
            self.add_utxo_to_address(utxo.address, utxo_id, batch)