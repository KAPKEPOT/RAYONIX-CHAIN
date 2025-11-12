# fixed_transaction.py
from contextlib import contextmanager
import time
import threading
import logging
from database.utils.exceptions import TransactionError

logger = logging.getLogger(__name__)

@contextmanager
def transaction(db, timeout: int = 30):
    """
    Fixed transaction context manager with proper rollback support
    """
    start_time = time.time()
    transaction_id = f"tx_{int(time.time() * 1000)}_{threading.get_ident()}"
    
    # Store original state for rollback
    original_values = {}
    modified_keys = set()
    
    try:
        logger.info(f"Starting transaction {transaction_id}")
        
        def transactional_put(key, value, **kwargs):
            """Transactional put that tracks changes for rollback"""
            # Store original value for rollback
            if key not in original_values:
                try:
                    original_values[key] = db.get(key, use_cache=False)
                except:
                    original_values[key] = None  # Key didn't exist
            
            modified_keys.add(key)
            # Actually perform the operation
            return db.db.put(key, db._prepare_value_for_storage(db._serialize_value(value), kwargs.get('ttl')))
        
        def transactional_delete(key, **kwargs):
            """Transactional delete that tracks changes for rollback"""
            if key not in original_values:
                try:
                    original_values[key] = db.get(key, use_cache=False)
                except:
                    original_values[key] = None
            
            modified_keys.add(key)
            return db.db.delete(key)
        
        # Replace database methods with transactional versions
        original_put = db.put
        original_delete = db.delete
        original_get = db.get
        
        db.put = transactional_put
        db.delete = transactional_delete
        
        # Use a more defensive get during transactions
        def transactional_get(key, **kwargs):
            return original_get(key, **kwargs)
        
        db.get = transactional_get
        
        yield db  # Yield the modified db instance
        
        # Commit - just clear the transactional wrappers
        logger.info(f"Committed transaction {transaction_id}")
        
    except Exception as e:
        # Rollback - restore original values
        logger.info(f"Rolling back transaction {transaction_id}")
        
        with db.db.write_batch() as batch:
            for key in modified_keys:
                original_value = original_values.get(key)
                if original_value is None:
                    # Key was created in transaction, delete it
                    batch.delete(key)
                else:
                    # Restore original value
                    batch.put(key, db._prepare_value_for_storage(db._serialize_value(original_value), None))
        
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TransactionError(f"Transaction timeout after {elapsed:.2f}s: {e}")
        else:
            raise TransactionError(f"Transaction failed: {e}")
    
    finally:
        # Restore original methods
        if hasattr(db, 'put'):
            db.put = original_put
        if hasattr(db, 'delete'):  
            db.delete = original_delete
        if hasattr(db, 'get'):
            db.get = original_get
        
        elapsed = time.time() - start_time
        logger.info(f"Transaction {transaction_id} completed in {elapsed:.3f}s")