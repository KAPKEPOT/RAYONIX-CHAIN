# blockchain/transactions/mempool.py
import time
import heapq
from typing import Dict, List, Any, Optional, Tuple
from sortedcontainers import SortedDict

from blockchain.models.transaction import Transaction

class Mempool:
    """Advanced mempool management with priority scheduling"""
    
    def __init__(self, max_size: int = 10000, expiry_time: int = 3600):
        self.max_size = max_size
        self.expiry_time = expiry_time
        self.transactions: SortedDict = SortedDict()  # tx_hash -> (transaction, timestamp, fee_rate)
        self.fee_rate_index = []  # Min-heap for fee rate tracking
        self.size_bytes = 0
        self.lock = threading.RLock()
    
    def add_transaction(self, transaction: Transaction, fee_rate: float) -> bool:
        """Add transaction to mempool"""
        with self.lock:
            if transaction.hash in self.transactions:
                return False  # Already exists
            
            if len(self.transactions) >= self.max_size:
                if not self._make_space():
                    return False  # Couldn't make space
            
            timestamp = time.time()
            self.transactions[transaction.hash] = (transaction, timestamp, fee_rate)
            heapq.heappush(self.fee_rate_index, (fee_rate, timestamp, transaction.hash))
            
            # Update size tracking
            tx_size = len(transaction.to_bytes())
            self.size_bytes += tx_size
            
            return True
    
    def _make_space(self) -> bool:
        """Make space in mempool by removing low fee rate transactions"""
        with self.lock:
            if not self.fee_rate_index:
                return False
            
            # Remove lowest fee rate transaction
            while self.fee_rate_index and len(self.transactions) >= self.max_size:
                fee_rate, timestamp, tx_hash = heapq.heappop(self.fee_rate_index)
                if tx_hash in self.transactions:
                    self._remove_transaction(tx_hash)
            
            return len(self.transactions) < self.max_size
    
    def _remove_transaction(self, tx_hash: str):
        """Remove transaction from mempool"""
        with self.lock:
            if tx_hash in self.transactions:
                transaction, timestamp, fee_rate = self.transactions[tx_hash]
                del self.transactions[tx_hash]
                
                # Update size tracking
                tx_size = len(transaction.to_bytes())
                self.size_bytes -= tx_size
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction by hash"""
        with self.lock:
            if tx_hash in self.transactions:
                return self.transactions[tx_hash][0]
            return None
    
    def has_transaction(self, tx_hash: str) -> bool:
        """Check if transaction exists in mempool"""
        with self.lock:
            return tx_hash in self.transactions
    
    def remove_transaction(self, tx_hash: str) -> bool:
        """Remove transaction from mempool"""
        with self.lock:
            if tx_hash in self.transactions:
                self._remove_transaction(tx_hash)
                return True
            return False
    
    def remove_transactions(self, tx_hashes: List[str]) -> int:
        """Remove multiple transactions from mempool"""
        count = 0
        with self.lock:
            for tx_hash in tx_hashes:
                if self.remove_transaction(tx_hash):
                    count += 1
        return count
    
    def get_priority_transactions(self, limit: int = 100) -> List[Transaction]:
        """Get transactions with highest fee rates"""
        with self.lock:
            # Get top fee rate transactions
            top_transactions = []
            fee_heap = self.fee_rate_index.copy()
            
            while fee_heap and len(top_transactions) < limit:
                fee_rate, timestamp, tx_hash = heapq.heappop(fee_heap)
                if tx_hash in self.transactions:
                    transaction = self.transactions[tx_hash][0]
                    top_transactions.append(transaction)
            
            return top_transactions
    
    def clean_expired_transactions(self) -> int:
        """Remove expired transactions from mempool"""
        current_time = time.time()
        expired_hashes = []
        
        with self.lock:
            for tx_hash, (transaction, timestamp, fee_rate) in self.transactions.items():
                if current_time - timestamp > self.expiry_time:
                    expired_hashes.append(tx_hash)
            
            return self.remove_transactions(expired_hashes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get mempool statistics"""
        with self.lock:
            fee_rates = [fee_rate for _, _, fee_rate in self.transactions.values()]
            timestamps = [timestamp for _, timestamp, _ in self.transactions.values()]
            
            return {
                'transaction_count': len(self.transactions),
                'total_size_bytes': self.size_bytes,
                'average_fee_rate': sum(fee_rates) / len(fee_rates) if fee_rates else 0,
                'min_fee_rate': min(fee_rates) if fee_rates else 0,
                'max_fee_rate': max(fee_rates) if fee_rates else 0,
                'oldest_timestamp': min(timestamps) if timestamps else 0,
                'newest_timestamp': max(timestamps) if timestamps else 0
            }
    
    def get_transactions_for_block(self, max_size: int, min_fee_rate: float = 0) -> List[Transaction]:
        """Get transactions suitable for block inclusion"""
        transactions = []
        current_size = 0
        
        with self.lock:
            # Process transactions in fee rate order (highest first)
            fee_heap = sorted(self.fee_rate_index, reverse=True)
            
            for fee_rate, timestamp, tx_hash in fee_heap:
                if current_size >= max_size:
                    break
                
                if tx_hash in self.transactions:
                    transaction, _, _ = self.transactions[tx_hash]
                    tx_size = len(transaction.to_bytes())
                    
                    if fee_rate >= min_fee_rate and current_size + tx_size <= max_size:
                        transactions.append(transaction)
                        current_size += tx_size
        
        return transactions
    
    def estimate_fee_rate(self, confirmation_target: int) -> float:
        """Estimate fee rate for confirmation target"""
        with self.lock:
            if not self.transactions:
                return 0.0
            
            # Sort transactions by fee rate
            sorted_txs = sorted(self.transactions.items(), key=lambda x: x[1][2], reverse=True)
            
            # Calculate cumulative size
            cumulative_size = 0
            target_size = confirmation_target * 1000000  # Assume 1MB blocks
            
            for tx_hash, (transaction, timestamp, fee_rate) in sorted_txs:
                tx_size = len(transaction.to_bytes())
                cumulative_size += tx_size
                
                if cumulative_size >= target_size:
                    return fee_rate
            
            # If we didn't reach target size, return lowest fee rate
            return sorted_txs[-1][1][2] if sorted_txs else 0.0