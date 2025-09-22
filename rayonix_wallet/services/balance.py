from typing import Dict, Optional
from rayonix_wallet.core.types import WalletBalance
from rayonix_wallet.core.exceptions import WalletError

class BalanceCalculator:
    """Balance calculation and management"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
    
    def get_balance(self, address: Optional[str] = None, force_refresh: bool = False) -> WalletBalance:
        """Calculate wallet balance"""
        try:
            cache_key = f"balance_{address or 'total'}"
            
            # Check cache
            if not force_refresh and cache_key in self._cache:
                cached_balance, timestamp = self._cache[cache_key]
                if time.time() - timestamp < self._cache_timeout:
                    return cached_balance
            
            if address:
                balance = self._get_address_balance(address)
            else:
                balance = self._get_total_balance()
            
            # Update cache
            self._cache[cache_key] = (balance, time.time())
            
            return balance
            
        except Exception as e:
            logger.error(f"Balance calculation failed: {e}")
            return WalletBalance(
                total=0,
                confirmed=0,
                unconfirmed=0,
                locked=0,
                available=0,
                error=str(e),
                error_type="calculation_error"
            )
    
    def _get_total_balance(self) -> WalletBalance:
        """Calculate total wallet balance"""
        total = 0
        confirmed = 0
        unconfirmed = 0
        locked = 0
        by_address = {}
        
        for address, info in self.wallet.addresses.items():
            address_balance = self._get_address_balance_from_info(info)
            total += address_balance
            by_address[address] = address_balance
            
            # For simplicity, assume all balances are confirmed
            # Real implementation would separate confirmed/unconfirmed
            confirmed += address_balance
        
        # Calculate available balance (excluding locked amounts)
        available = total - locked
        
        return WalletBalance(
            total=total,
            confirmed=confirmed,
            unconfirmed=unconfirmed,
            locked=locked,
            available=available,
            by_address=by_address
        )
    
    def _get_address_balance(self, address: str) -> WalletBalance:
        """Calculate balance for specific address"""
        if address not in self.wallet.addresses:
            raise WalletError("Address not found in wallet")
        
        address_info = self.wallet.addresses[address]
        balance = self._get_address_balance_from_info(address_info)
        
        return WalletBalance(
            total=balance,
            confirmed=balance,
            unconfirmed=0,
            locked=0,
            available=balance,
            by_address={address: balance}
        )
    
    def _get_address_balance_from_info(self, address_info: Dict) -> int:
        """Get balance from address information"""
        # Use the balance stored in address info
        # This gets updated during synchronization
        return address_info.balance
    
    def refresh_balances(self) -> bool:
        """Refresh all balances from blockchain"""
        try:
            if hasattr(self.wallet, 'synchronizer'):
                return self.wallet.synchronizer.synchronize()
            return False
        except Exception as e:
            logger.error(f"Balance refresh failed: {e}")
            return False
    
    def get_balance_history(self, timeframe: str = "30d") -> Dict:
        """Get balance history over time"""
        # This would typically query historical data from the database
        # or from blockchain analysis
        
        # Placeholder implementation
        return {
            'timestamps': [],
            'balances': [],
            'timeframe': timeframe
        }
    
    def estimate_balance(self, include_pending: bool = True) -> int:
        """Estimate balance including pending transactions"""
        total_balance = self.get_balance().total
        
        if include_pending:
            # Add pending incoming transactions
            for tx in self.wallet.transactions.values():
                if tx.status == 'pending' and tx.direction == 'in':
                    total_balance += tx.amount
            
            # Subtract pending outgoing transactions
            for tx in self.wallet.transactions.values():
                if tx.status == 'pending' and tx.direction == 'out':
                    total_balance -= tx.amount
        
        return total_balance
    
    def clear_cache(self):
        """Clear balance cache"""
        self._cache.clear()