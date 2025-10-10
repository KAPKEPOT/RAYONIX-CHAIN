# exceptions.py

class DatabaseError(Exception):
    """Base database exception"""
    pass

class KeyNotFoundError(DatabaseError):
    """Raised when a key is not found"""
    pass

class SerializationError(DatabaseError):
    """Raised when serialization/deserialization fails"""
    pass

class IndexError(DatabaseError):
    """Base index error"""
    pass

class IndexCorruptionError(IndexError):
    """Raised when index corruption is detected"""
    pass

class TransactionError(DatabaseError):
    """Raised when transaction operations fail"""
    pass

class CompressionError(DatabaseError):
    """Raised when compression/decompression fails"""
    pass

class EncryptionError(DatabaseError):
    """Raised when encryption/decryption fails"""
    pass

class ConcurrencyError(DatabaseError):
    """Raised when concurrency issues occur"""
    pass

class IntegrityError(DatabaseError):
    """Raised when data integrity is compromised"""
    pass
    
# Add this to your existing database/utils/exceptions.py file

class DuplicateKeyError(DatabaseError):
    """Raised when a duplicate key violation occurs"""
    pass    