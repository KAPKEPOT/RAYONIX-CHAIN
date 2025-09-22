# utxo_system/exceptions/custom_errors.py
class SerializationError(Exception):
    """Raised when serialization fails"""
    pass

class DeserializationError(Exception):
    """Raised when deserialization fails"""
    pass

class ValidationError(Exception):
    """Raised when validation fails"""
    pass

class DatabaseError(Exception):
    """Raised for database-related errors"""
    pass

class InsufficientFundsError(Exception):
    """Raised when there are insufficient funds for a transaction"""
    pass