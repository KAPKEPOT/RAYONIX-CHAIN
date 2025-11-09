#rayonix_wallet/storage/wallet_database.py
from typing import List, Optional
from database.core.database import AdvancedDatabase

class WalletDatabaseAdapter:
    """Adapter layer for wallet-specific database operations"""
    
    def __init__(self, database: AdvancedDatabase):
        self.db = database
    
    def get_wallet_state(self):
        """Get wallet state from database"""
        try:
            return self.db.get(b'wallet_state')
        except Exception:
            return None
    
    def save_wallet_state(self, state) -> bool:
        """Save wallet state to database"""
        return self.db.put(b'wallet_state', state)
    
    def get_all_addresses(self) -> List:
        """Get all addresses from database"""
        addresses = []
        try:
            for key, value in self.db.iterate(prefix=b'address_'):
                addresses.append(value)
            return addresses
        except Exception:
            return []
    
    def save_address(self, address_info) -> bool:
        """Save address information to database"""
        key = f"address_{address_info.address}".encode()
        return self.db.put(key, address_info)
    
    def get_transactions(self, limit: int = 1000, offset: int = 0) -> List:
        """Get transactions from database"""
        transactions = []
        count = 0
        try:
            for key, value in self.db.iterate(prefix=b'tx_'):
                if count >= offset:
                    transactions.append(value)
                count += 1
                if len(transactions) >= limit:
                    break
            return transactions
        except Exception:
            return []
    
    def save_transaction(self, transaction) -> bool:
        """Save transaction to database"""
        key = f"tx_{transaction.txid}".encode()
        return self.db.put(key, transaction)
    
    def close(self):
        """Close database connection"""
        self.db.close()