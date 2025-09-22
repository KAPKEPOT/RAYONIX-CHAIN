# utxo_system/validation/transaction_validator.py
from typing import Tuple
from utxo_system.models.transaction import Transaction
from utxo_system.database.core import UTXOSet
from utxo_system.exceptions import ValidationError
from utxo_system.utils.logging_config import logger

def validate_transaction(transaction: Transaction, utxo_set: UTXOSet, 
                        current_block_height: int = 0) -> Tuple[bool, str]:
    """
    Validate a transaction before processing it.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check all inputs exist and are spendable
    for i, tx_input in enumerate(transaction.inputs):
        utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
        utxo = utxo_set.get_utxo(utxo_id)
        
        if not utxo:
            return False, f"Input UTXO {utxo_id} does not exist"
        
        if utxo.spent:
            return False, f"Input UTXO {utxo_id} is already spent"
        
        if not utxo.is_spendable(current_block_height, 0):
            return False, f"Input UTXO {utxo_id} is not spendable yet"
        
        # Verify signature
        if not transaction.verify_input_signature(i, utxo_set):
            return False, f"Invalid signature for input {i}"
    
    # Check output amounts are positive
    for output in transaction.outputs:
        if output.amount <= 0:
            return False, "Output amount must be positive"
    
    # Check fee is reasonable (at least non-negative)
    fee = transaction.calculate_fee(utxo_set)
    if fee < 0:
        return False, "Transaction has negative fee"
    
    # Additional validation: check for dust outputs
    for output in transaction.outputs:
        if output.amount < 546:  # Minimum dust amount (similar to Bitcoin)
            logger.warning(f"Dust output detected: {output.amount}")
    
    return True, ""