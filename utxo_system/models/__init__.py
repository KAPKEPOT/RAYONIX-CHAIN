# utxo_system/models/__init__.py
from utxo_system.models.utxo import UTXO
from utxo_system.models.transaction import Transaction, TransactionInput, TransactionOutput

__all__ = ['UTXO', 'Transaction', 'TransactionInput', 'TransactionOutput']