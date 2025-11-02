# utxo_system/validation/transaction_validator.py
from typing import Tuple
from utxo_system.models.transaction import Transaction
from utxo_system.core.utxoset import UTXOSet
from utxo_system.exceptions import ValidationError
from utxo_system.utils.logging_config import logger

class TransactionValidator:
    """Advanced transaction validation with comprehensive checks"""
    
    @staticmethod
    def validate_transaction(transaction: Transaction, utxo_set: Any, 
                           current_height: int = 0) -> Tuple[bool, List[str]]:
        """Comprehensive transaction validation"""
        errors = []
        
        # Basic structure validation
        if not transaction.validate_structure():
            errors.append("Invalid transaction structure")
            return False, errors
        
        # Check if transaction is final
        if not transaction.is_final(current_height):
            errors.append("Transaction is not final")
        
        # Verify signatures (skip for coinbase)
        if not transaction.is_coinbase():
            if not transaction.verify_all_signatures(utxo_set):
                errors.append("Invalid signatures")
        
        # Check for double spends
        double_spend_error = TransactionValidator._check_double_spends(transaction, utxo_set)
        if double_spend_error:
            errors.append(double_spend_error)
        
        # Validate fee
        fee = transaction.calculate_fee(utxo_set)
        if fee < 0:
            errors.append("Negative transaction fee")
        
        # Check output amounts
        for i, output in enumerate(transaction.outputs):
            if output.amount <= 0:
                errors.append(f"Output {i} has invalid amount: {output.amount}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _check_double_spends(transaction: Transaction, utxo_set: Any) -> Optional[str]:
        """Check for double spending attempts"""
        for inp in transaction.inputs:
            if inp.is_coinbase():
                continue
                
            utxo_id = inp.get_outpoint()
            try:
                utxo = utxo_set.get_utxo(utxo_id)
                if utxo and getattr(utxo, 'is_spent', False):
                    return f"Double spend detected: {utxo_id}"
            except Exception as e:
                logger.warning(f"Error checking double spend for {utxo_id}: {e}")
        
        return None


# Utility functions for transaction handling
def create_transaction_hash(data: bytes) -> str:
    """Create double SHA256 hash of transaction data"""
    return hashlib.sha256(hashlib.sha256(data).digest()).hexdigest()

def validate_transaction_format(transaction_data: Dict) -> bool:
    """Validate transaction data format"""
    required_fields = {'version', 'inputs', 'outputs', 'locktime'}
    return all(field in transaction_data for field in required_fields)

def calculate_transaction_weight(transaction: Transaction) -> int:
    """Calculate transaction weight for fee calculation"""
    base_size = transaction.calculate_size()
    total_size = base_size
    
    # Add witness size if present
    if transaction.witness_data:
        witness_size = sum(len(str(w)) for w_list in transaction.witness_data.values() 
                          for w in w_list) if isinstance(transaction.witness_data, dict) else 0
        total_size += witness_size
    
    return (base_size * 3 + total_size) // 4
