# blockchain/transactions/transaction_manager.py
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from sortedcontainers import SortedDict

from utxo_system.models.transaction import Transaction
from utxo_system.models.utxo import UTXO
from blockchain.models.transaction_results import TransactionCreationResult
from blockchain.validation.validation_manager import ValidationManager
from blockchain.fees.fee_estimator import FeeEstimator

logger = logging.getLogger(__name__)

class TransactionManager:
    """Advanced transaction creation and fee estimation"""
    
    def __init__(self, state_manager: Any, wallet: Any, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.wallet = wallet
        self.config = config
        # Convert dataclass to dict if needed
        if hasattr(config, '__dataclass_fields__'):
        	from dataclasses import asdict
        	config_dict = asdict(config)
        else:
        	config_dict = config
  
        
        self.fee_estimator = FeeEstimator(state_manager, config)
        self.coin_selection_strategies = self._initialize_coin_selection_strategies()
        self.mempool: SortedDict = SortedDict()  # tx_hash -> (transaction, timestamp, fee_rate)
        self.validation_manager = ValidationManager(state_manager, config)
        # Use attribute access or dict get with fallback
        if hasattr(config, 'max_mempool_size'):
        	self.mempool_size_limit = config.max_mempool_size
        else:
        	self.mempool_size_limit = config_dict.get('max_mempool_size', 10000)
        	
        if hasattr(config, 'mempool_expiry_time'):
        	self.mempool_expiry_time = config.mempool_expiry_time
        else:
        	self.mempool_expiry_time = config_dict.get('mempool_expiry_time', 3600) 
    
    def _initialize_coin_selection_strategies(self) -> Dict[str, callable]:
        """Initialize coin selection strategies"""
        return {
            'default': self._default_coin_selection,
            'privacy': self._privacy_coin_selection,
            'efficiency': self._efficiency_coin_selection,
            'consolidation': self._consolidation_coin_selection,
            'random': self._random_coin_selection
        }
    
    def create_transaction(self, from_address: str, to_address: str, amount: int,
                         fee_strategy: str = 'default', coin_selection: str = 'default',
                         memo: Optional[str] = None, locktime: int = 0, 
                         **kwargs) -> TransactionCreationResult:
        """Create a transaction with advanced options"""
        try:
            # Get UTXOs for sender
            utxos = self.state_manager.utxo_set.get_utxos_for_address(from_address)
            if not utxos:
                return TransactionCreationResult(
                    success=False,
                    transaction=None,
                    fee_estimate=0,
                    selected_utxos=[],
                    change_amount=0,
                    error_message="No spendable funds"
                )
            
            # Estimate fee
            fee_estimate = self.fee_estimator.estimate_fee(fee_strategy)
            
            # Select coins based on strategy
            coin_selector = self.coin_selection_strategies.get(coin_selection, self._default_coin_selection)
            selected_utxos, total_input, change_amount = coin_selector(utxos, amount, fee_estimate)
            
            if total_input < amount + fee_estimate:
                return TransactionCreationResult(
                    success=False,
                    transaction=None,
                    fee_estimate=fee_estimate,
                    selected_utxos=selected_utxos,
                    change_amount=change_amount,
                    error_message="Insufficient funds"
                )
            
            # Create transaction
            transaction = self._build_transaction(
                from_address, to_address, amount, selected_utxos, 
                change_amount, fee_estimate, memo, locktime, **kwargs
            )
            
            # Sign transaction
            signed_transaction = self.wallet.sign_transaction(transaction)
            
            # Calculate transaction metrics
            tx_size = len(signed_transaction.to_bytes())
            network_fee = total_input - amount - change_amount
            
            return TransactionCreationResult(
                success=True,
                transaction=signed_transaction,
                fee_estimate=fee_estimate,
                selected_utxos=selected_utxos,
                change_amount=change_amount,
                total_input=total_input,
                total_output=amount + change_amount,
                network_fee=network_fee,
                size_bytes=tx_size
            )
            
        except Exception as e:
            logger.error(f"Transaction creation failed: {e}", exc_info=True)
            return TransactionCreationResult(
                success=False,
                transaction=None,
                fee_estimate=0,
                selected_utxos=[],
                change_amount=0,
                error_message=str(e)
            )
    
    def _build_transaction(self, from_address: str, to_address: str, amount: int,
                          selected_utxos: List[UTXO], change_amount: int, 
                          fee_estimate: int, memo: Optional[str], locktime: int,
                          **kwargs) -> Transaction:
        """Build transaction from components"""
        from blockchain.models.transaction import Transaction, TransactionInput, TransactionOutput
        
        # Create transaction inputs
        inputs = []
        for utxo in selected_utxos:
            inputs.append(TransactionInput(
                tx_hash=utxo.tx_hash,
                output_index=utxo.output_index,
                signature=None,  # Will be signed later
                public_key=None  # Will be set during signing
            ))
        
        # Create transaction outputs
        outputs = [
            TransactionOutput(
                address=to_address,
                amount=amount,
                locktime=locktime
            )
        ]
        
        # Add change output if needed
        if change_amount > 0:
            change_address = self.wallet.get_change_address(from_address)
            outputs.append(TransactionOutput(
                address=change_address,
                amount=change_amount,
                locktime=0
            ))
        
        # Create transaction
        return Transaction(
            inputs=inputs,
            outputs=outputs,
            locktime=locktime,
            version=2,
            memo=memo,
            **kwargs
        )
    
    def _default_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Default coin selection strategy (largest first)"""
        sorted_utxos = sorted(utxos, key=lambda x: x.amount, reverse=True)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            if total >= amount + fee:
                break
            selected.append(utxo)
            total += utxo.amount
        
        change = total - amount - fee
        return selected, total, change
    
    def _privacy_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Privacy-focused coin selection (minimize address reuse)"""
        # Prefer UTXOs that haven't been spent from recently
        sorted_utxos = sorted(utxos, key=lambda x: x.age, reverse=True)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            if total >= amount + fee:
                break
            selected.append(utxo)
            total += utxo.amount
        
        change = total - amount - fee
        return selected, total, change
    
    def _efficiency_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Efficiency-focused coin selection (minimize UTXO count)"""
        # Try to find a single UTXO that covers the amount
        for utxo in sorted(utxos, key=lambda x: x.amount, reverse=True):
            if utxo.amount >= amount + fee:
                change = utxo.amount - amount - fee
                return [utxo], utxo.amount, change
        
        # Fall back to default strategy
        return self._default_coin_selection(utxos, amount, fee)
    
    def _consolidation_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Consolidation strategy (spend many small UTXOs)"""
        sorted_utxos = sorted(utxos, key=lambda x: x.amount)
        selected = []
        total = 0
        
        for utxo in sorted_utxos:
            selected.append(utxo)
            total += utxo.amount
            if total >= amount + fee:
                break
        
        change = total - amount - fee
        return selected, total, change
    
    def _random_coin_selection(self, utxos: List[UTXO], amount: int, fee: int) -> Tuple[List[UTXO], int, int]:
        """Random coin selection for privacy"""
        import random
        
        random.shuffle(utxos)
        selected = []
        total = 0
        
        for utxo in utxos:
            if total >= amount + fee:
                break
            selected.append(utxo)
            total += utxo.amount
        
        change = total - amount - fee
        return selected, total, change
    
    def add_to_mempool(self, transaction: Transaction) -> bool:
        """Add transaction to mempool"""
        try:
            # Validate transaction
            validation_result = self.validation_manager.validate_transaction(transaction)
            if not validation_result.is_valid:
                logger.warning(f"Transaction validation failed: {validation_result.errors}")
                return False
            
            # Calculate fee rate
            input_sum = sum(inp.amount for inp in transaction.inputs)
            output_sum = sum(out.amount for out in transaction.outputs)
            fee = input_sum - output_sum
            size = len(transaction.to_bytes())
            fee_rate = fee / size if size > 0 else 0
            
            # Check mempool size limit
            if len(self.mempool) >= self.mempool_size_limit:
                # Remove lowest fee rate transactions
                self._evict_low_fee_transactions()
            
            # Add to mempool sorted by fee rate
            self.mempool[transaction.hash] = (transaction, time.time(), fee_rate)
            
            logger.info(f"Transaction added to mempool: {transaction.hash}, fee rate: {fee_rate:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add transaction to mempool: {e}")
            return False
    
    def _evict_low_fee_transactions(self, target_size: Optional[int] = None):
        """Evict low fee rate transactions from mempool"""
        if target_size is None:
            target_size = self.mempool_size_limit * 0.9  # Keep 90% of capacity
        
        if len(self.mempool) <= target_size:
            return
        
        # Sort transactions by fee rate (lowest first)
        sorted_txs = sorted(self.mempool.items(), key=lambda x: x[1][2])
        
        # Remove lowest fee rate transactions
        for tx_hash, _ in sorted_txs[:len(self.mempool) - int(target_size)]:
            del self.mempool[tx_hash]
            logger.debug(f"Evicted transaction from mempool: {tx_hash}")
    
    def get_mempool_transactions(self, limit: int = 1000) -> List[Transaction]:
        """Get transactions from mempool sorted by fee rate"""
        # Sort by fee rate (highest first)
        sorted_txs = sorted(self.mempool.items(), key=lambda x: x[1][2], reverse=True)
        
        transactions = []
        for tx_hash, (tx, timestamp, fee_rate) in sorted_txs:
            transactions.append(tx)
            if len(transactions) >= limit:
                break
        
        return transactions
    
    def get_mempool_stats(self) -> Dict[str, Any]:
        """Get mempool statistics"""
        total_size = sum(len(tx.to_bytes()) for tx, _, _ in self.mempool.values())
        fee_rates = [fee_rate for _, _, fee_rate in self.mempool.values()]
        
        return {
            'transaction_count': len(self.mempool),
            'total_size_bytes': total_size,
            'average_fee_rate': sum(fee_rates) / len(fee_rates) if fee_rates else 0,
            'min_fee_rate': min(fee_rates) if fee_rates else 0,
            'max_fee_rate': max(fee_rates) if fee_rates else 0,
            'oldest_transaction': min(timestamp for _, timestamp, _ in self.mempool.values()) if self.mempool else 0
        }
    
    def remove_from_mempool(self, transaction_hashes: List[str]):
        """Remove transactions from mempool"""
        for tx_hash in transaction_hashes:
            if tx_hash in self.mempool:
                del self.mempool[tx_hash]
                logger.debug(f"Removed transaction from mempool: {tx_hash}")
    
    def clean_mempool(self):
        """Clean expired transactions from mempool"""
        current_time = time.time()
        expired_hashes = []
        
        for tx_hash, (tx, timestamp, fee_rate) in self.mempool.items():
            if current_time - timestamp > self.mempool_expiry_time:
                expired_hashes.append(tx_hash)
        
        self.remove_from_mempool(expired_hashes)
        logger.info(f"Cleaned {len(expired_hashes)} expired transactions from mempool")
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction from mempool by hash"""
        if tx_hash in self.mempool:
            return self.mempool[tx_hash][0]
        return None
    
    def has_transaction(self, tx_hash: str) -> bool:
        """Check if transaction is in mempool"""
        return tx_hash in self.mempool