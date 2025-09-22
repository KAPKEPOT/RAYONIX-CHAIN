from contextlib import contextmanager
from typing import Any, Generator
import threading
import time
import logging

from database.utils.exceptions import TransactionError

logger = logging.getLogger(__name__)

@contextmanager
def transaction(db, timeout: int = 30) -> Generator[Any, None, None]:
    """
    Transaction context manager with timeout and rollback support
    """
    start_time = time.time()
    transaction_id = f"tx_{int(time.time() * 1000)}_{threading.get_ident()}"
    
    try:
        logger.info(f"Starting transaction {transaction_id}")
        
        # Start transaction (implementation depends on database type)
        if hasattr(db, 'begin_transaction'):
            tx = db.begin_transaction()
        else:
            # For databases without native transactions, use write batch
            tx = db.db.write_batch() if hasattr(db.db, 'write_batch') else None
        
        yield tx
        
        # Commit transaction
        if tx and hasattr(tx, 'write'):
            tx.write()
            logger.info(f"Committed transaction {transaction_id}")
        elif tx and hasattr(tx, 'commit'):
            tx.commit()
            logger.info(f"Committed transaction {transaction_id}")
        
    except Exception as e:
        # Rollback transaction
        if tx and hasattr(tx, 'clear'):
            tx.clear()
            logger.info(f"Rolled back transaction {transaction_id}")
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TransactionError(f"Transaction timeout after {elapsed:.2f}s: {e}")
        else:
            raise TransactionError(f"Transaction failed: {e}")
    
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Transaction {transaction_id} completed in {elapsed:.3f}s")