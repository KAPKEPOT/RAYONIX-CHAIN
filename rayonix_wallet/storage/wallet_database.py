# rayonix_wallet/storage/wallet_database.py
import logging
from typing import List, Optional, Iterator, Tuple, Any, Dict
from database.core.database import AdvancedDatabase

logger = logging.getLogger(__name__)

class WalletDatabaseAdapter:
    """Adapter layer for wallet-specific database operations"""
    
    def __init__(self, database: AdvancedDatabase):
        self.db = database
        
    def _make_serializable(self, obj: Any) -> Any:
        """Convert dataclass objects to serializable formats"""
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        else:
            return obj
            
    def get_wallet_state(self):
        """Get wallet state from database"""
        try:
            return self.db.get(b'wallet_state')
        except Exception as e:
            logger.warning(f"Failed to get wallet state: {e}")
            return None
    
    def save_wallet_state(self, state) -> bool:
        """Save wallet state to database"""
        try:
            serializable_state = self._make_serializable(state)
            return self.db.put(b'wallet_state', serializable_state)
        except Exception as e:
            logger.error(f"Failed to save wallet state: {e}")
            return False
    
    def get_all_addresses(self) -> List:
        """Get all addresses from database"""
        addresses = []
        try:
            for key, value in self.iterate(prefix=b'address_'):
                addresses.append(value)
            return addresses
        except Exception as e:
            logger.error(f"Failed to get all addresses: {e}")
            return []
    
    def save_address(self, address_info) -> bool:
        """Save address information to database - FIXED"""
        try:
            # Convert dataclass to serializable dict
            serializable_data = self._make_serializable(address_info)
            key = f"address_{address_info.address}".encode()
            return self.db.put(key, serializable_data)
        except Exception as e:
            logger.error(f"Failed to save address: {e}")
            return False
    
    def get_transactions(self, limit: int = 1000, offset: int = 0) -> List:
        """Get transactions from database"""
        transactions = []
        count = 0
        try:
            for key, value in self.iterate(prefix=b'tx_'):
                if count >= offset:
                    transactions.append(value)
                count += 1
                if len(transactions) >= limit:
                    break
            return transactions
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")
            return []
    
    def save_transaction(self, transaction) -> bool:
        """Save transaction to database - FIXED"""
        try:
            serializable_data = self._make_serializable(transaction)
            key = f"tx_{transaction.txid}".encode()
            return self.db.put(key, serializable_data)
        except Exception as e:
            logger.error(f"Failed to save transaction: {e}")
            return False
    
    def iterate(self, prefix: bytes = b'') -> Iterator[Tuple[bytes, Any]]:
        """Iterate through database entries with prefix"""
        try:
            return self.db.iterate(prefix=prefix)
        except Exception as e:
            logger.error(f"Database iteration failed: {e}")
            # Return empty iterator instead of raising
            return iter([])
    
    def repair_corrupted_entries(self) -> Dict[str, int]:
        """Attempt to repair corrupted database entries"""
        try:
            if hasattr(self.db, 'repair_corrupted_entries'):
                return self.db.repair_corrupted_entries()
            else:
                logger.warning("Database repair not supported")
                return {'repaired': 0, 'removed_corrupted': 0, 'total_scanned': 0}
        except Exception as e:
            logger.error(f"Database repair failed: {e}")
            return {'repaired': 0, 'removed_corrupted': 0, 'total_scanned': 0}
    
    def close(self):
        """Close database connection"""
        try:
            self.db.close()
        except Exception as e:
            logger.error(f"Failed to close database: {e}")