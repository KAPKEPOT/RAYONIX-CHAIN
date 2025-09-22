class WalletError(Exception):
    """Base exception for wallet errors"""
    pass

class DatabaseError(WalletError):
    """Database-related errors"""
    pass

class CryptoError(WalletError):
    """Cryptography-related errors"""
    pass

class SyncError(WalletError):
    """Synchronization errors"""
    pass

class InsufficientFundsError(WalletError):
    """Insufficient funds error"""
    pass

class InvalidAddressError(WalletError):
    """Invalid address error"""
    pass

class TransactionError(WalletError):
    """Transaction-related errors"""
    pass

class BackupError(WalletError):
    """Backup/restore errors"""
    pass

class MultisigError(WalletError):
    """Multi-signature errors"""
    pass