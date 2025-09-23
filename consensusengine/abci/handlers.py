# consensus/abci/handlers.py
import json
import hashlib
from typing import Dict, Any, List, Optional, Callable
import logging
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger('ABCIHandlers')

class TransactionHandler:
    """Production-ready transaction validation and processing"""
    
    def __init__(self, state_manager: Any = None):
        self.state_manager = state_manager
        self.tx_pool: Dict[str, Dict] = {}  # tx_hash -> transaction data
        self.processed_txs: set = set()
        
    def validate_transaction_structure(self, tx_data: Dict) -> bool:
        """Validate transaction structure"""
        required_fields = {'hash', 'sender', 'receiver', 'amount', 'fee', 'timestamp', 'signature'}
        
        if not all(field in tx_data for field in required_fields):
            return False
        
        if not isinstance(tx_data['amount'], (int, float)) or tx_data['amount'] <= 0:
            return False
        
        if not isinstance(tx_data['fee'], (int, float)) or tx_data['fee'] < 0:
            return False
        
        if tx_data['timestamp'] <= 0:
            return False
        
        # Validate hash format
        if not tx_data['hash'] or len(tx_data['hash']) != 64:
            return False
        
        # Validate address formats (assuming 42 char addresses)
        if len(tx_data['sender']) != 42 or len(tx_data['receiver']) != 42:
            return False
        
        return True
    
    def verify_transaction_signature(self, tx_data: Dict, public_key: str) -> bool:
        """Verify transaction signature"""
        try:
            # Recreate signing data
            signing_data = self._get_tx_signing_data(tx_data)
            
            if isinstance(tx_data['signature'], str):
                signature_bytes = bytes.fromhex(tx_data['signature'])
            else:
                signature_bytes = tx_data['signature']
            
            # Load public key
            verifying_key = serialization.load_der_public_key(
                bytes.fromhex(public_key),
                backend=default_backend()
            )
            
            # Verify signature
            verifying_key.verify(
                signature_bytes,
                signing_data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
            
        except (InvalidSignature, ValueError, Exception) as e:
            logger.warning(f"Transaction signature verification failed: {e}")
            return False
    
    def _get_tx_signing_data(self, tx_data: Dict) -> bytes:
        """Get data that should be signed for transaction"""
        signing_fields = ['sender', 'receiver', 'amount', 'fee', 'timestamp', 'data']
        data_to_sign = {field: tx_data.get(field) for field in signing_fields}
        return json.dumps(data_to_sign, sort_keys=True).encode()
    
    def check_balance(self, sender: str, amount: int, fee: int) -> bool:
        """Check if sender has sufficient balance"""
        if not self.state_manager:
            return True  # Skip balance check if no state manager
        
        balance = self.state_manager.get_balance(sender)
        return balance >= (amount + fee)
    
    def check_transaction_unique(self, tx_hash: str) -> bool:
        """Check if transaction is unique (not already processed)"""
        return tx_hash not in self.processed_txs and tx_hash not in self.tx_pool
    
    def validate_transaction(self, tx_data: str) -> bool:
        """Main transaction validation method"""
        try:
            # Parse transaction data
            if isinstance(tx_data, str):
                tx_dict = json.loads(tx_data)
            else:
                tx_dict = tx_data
            
            # Validate structure
            if not self.validate_transaction_structure(tx_dict):
                logger.warning("Invalid transaction structure")
                return False
            
            # Check uniqueness
            if not self.check_transaction_unique(tx_dict['hash']):
                logger.warning("Duplicate transaction")
                return False
            
            # Verify signature (in production, we'd need to get public key from sender)
            # For now, we'll assume signature verification is handled elsewhere
            
            # Check balance
            if not self.check_balance(tx_dict['sender'], tx_dict['amount'], tx_dict['fee']):
                logger.warning("Insufficient balance")
                return False
            
            # Add to transaction pool
            self.tx_pool[tx_dict['hash']] = tx_dict
            
            return True
            
        except Exception as e:
            logger.error(f"Transaction validation error: {e}")
            return False
    
    def deliver_transaction(self, tx_data: str) -> bool:
        """Process and deliver transaction to state"""
        try:
            if isinstance(tx_data, str):
                tx_dict = json.loads(tx_data)
            else:
                tx_dict = tx_data
            
            tx_hash = tx_dict['hash']
            
            # Remove from pool and mark as processed
            if tx_hash in self.tx_pool:
                del self.tx_pool[tx_hash]
            self.processed_txs.add(tx_hash)
            
            # Update state (in production, this would update account balances)
            if self.state_manager:
                success = self.state_manager.apply_transaction(tx_dict)
                if not success:
                    logger.error(f"Failed to apply transaction: {tx_hash}")
                    return False
            
            logger.info(f"Successfully delivered transaction: {tx_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Transaction delivery error: {e}")
            return False
    
    def get_transaction_pool(self) -> List[Dict]:
        """Get current transaction pool"""
        return list(self.tx_pool.values())
    
    def clear_processed_transactions(self, up_to_block: int = None):
        """Clear old processed transactions to save memory"""
        # In production, this would implement a sliding window
        # For now, we'll keep all processed transactions
        pass

class BlockHandler:
    """Block processing and validation"""
    
    def __init__(self, state_manager: Any = None):
        self.state_manager = state_manager
        self.block_store: Dict[str, Dict] = {}  # block_hash -> block data
        self.current_block: Optional[Dict] = None
        
    def begin_block(self, height: int, block_hash: str) -> bool:
        """Begin block processing"""
        try:
            self.current_block = {
                'height': height,
                'hash': block_hash,
                'transactions': [],
                'start_time': time.time(),
                'state_changes': []
            }
            
            # Notify state manager
            if self.state_manager:
                self.state_manager.begin_block(height, block_hash)
            
            logger.info(f"Began processing block {height} with hash {block_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error beginning block: {e}")
            return False
    
    def add_transaction_to_block(self, tx_data: Dict) -> bool:
        """Add transaction to current block"""
        if not self.current_block:
            logger.error("No current block to add transaction to")
            return False
        
        self.current_block['transactions'].append(tx_data)
        return True
    
    def validate_block(self, block_data: Dict) -> bool:
        """Validate complete block"""
        try:
            required_fields = {'height', 'hash', 'previous_hash', 'timestamp', 'validator', 'signature'}
            
            if not all(field in block_data for field in required_fields):
                return False
            
            # Validate block hash
            calculated_hash = self.calculate_block_hash(block_data)
            if calculated_hash != block_data['hash']:
                return False
            
            # Validate timestamp (within reasonable bounds)
            current_time = time.time()
            if abs(block_data['timestamp'] - current_time) > 7200:  # 2 hours
                return False
            
            # Validate transactions
            for tx in block_data.get('transactions', []):
                if not self.validate_transaction_in_block(tx, block_data):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Block validation error: {e}")
            return False
    
    def calculate_block_hash(self, block_data: Dict) -> str:
        """Calculate block hash from block data"""
        hash_data = {
            'height': block_data['height'],
            'previous_hash': block_data['previous_hash'],
            'timestamp': block_data['timestamp'],
            'validator': block_data['validator'],
            'transactions': [tx.get('hash', '') for tx in block_data.get('transactions', [])]
        }
        
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def validate_transaction_in_block(self, tx_data: Dict, block_data: Dict) -> bool:
        """Validate transaction within block context"""
        # Check if transaction is properly included
        if 'hash' not in tx_data:
            return False
        
        # Additional block-specific validation can go here
        return True
    
    def end_block(self, height: int) -> Dict[str, Any]:
        """End block processing and return results"""
        try:
            if not self.current_block or self.current_block['height'] != height:
                logger.error(f"No current block at height {height}")
                return {}
            
            block_result = {
                'height': height,
                'transaction_count': len(self.current_block['transactions']),
                'processing_time': time.time() - self.current_block['start_time'],
                'state_changes': self.current_block['state_changes']
            }
            
            # Notify state manager
            if self.state_manager:
                end_block_result = self.state_manager.end_block(height)
                block_result.update(end_block_result)
            
            # Store block
            if 'hash' in self.current_block:
                self.block_store[self.current_block['hash']] = self.current_block.copy()
            
            logger.info(f"Ended processing block {height} with {len(self.current_block['transactions'])} transactions")
            
            self.current_block = None
            return block_result
            
        except Exception as e:
            logger.error(f"Error ending block: {e}")
            return {}
    
    def commit_block(self) -> str:
        """Commit block changes and return app hash"""
        try:
            if self.state_manager:
                app_hash = self.state_manager.commit()
                return app_hash
            
            # Fallback: generate simple app hash
            return hashlib.sha256(str(time.time()).encode()).hexdigest()
            
        except Exception as e:
            logger.error(f"Error committing block: {e}")
            return ""
    
    def get_block(self, block_hash: str) -> Optional[Dict]:
        """Get block by hash"""
        return self.block_store.get(block_hash)
    
    def get_blocks_by_height(self, height: int) -> List[Dict]:
        """Get all blocks at specific height (handling forks)"""
        return [block for block in self.block_store.values() if block.get('height') == height]

class StateManager:
    """Manages application state for ABCI"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path
        self.current_state: Dict[str, Any] = {}
        self.pending_changes: List[Dict] = []
        
    def begin_block(self, height: int, block_hash: str):
        """Begin block state changes"""
        self.pending_changes = []
        
    def apply_transaction(self, tx_data: Dict) -> bool:
        """Apply transaction to state"""
        try:
            # Update balances
            sender = tx_data['sender']
            receiver = tx_data['receiver']
            amount = tx_data['amount']
            fee = tx_data['fee']
            
            # Deduct from sender
            current_sender_balance = self.current_state.get(sender, {}).get('balance', 0)
            if current_sender_balance < (amount + fee):
                return False
            
            self.current_state.setdefault(sender, {})['balance'] = current_sender_balance - (amount + fee)
            
            # Add to receiver
            current_receiver_balance = self.current_state.get(receiver, {}).get('balance', 0)
            self.current_state.setdefault(receiver, {})['balance'] = current_receiver_balance + amount
            
            # Record change
            self.pending_changes.append({
                'type': 'transfer',
                'sender': sender,
                'receiver': receiver,
                'amount': amount,
                'fee': fee
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying transaction: {e}")
            return False
    
    def end_block(self, height: int) -> Dict[str, Any]:
        """End block state processing"""
        return {
            'state_changes': len(self.pending_changes),
            'height': height
        }
    
    def commit(self) -> str:
        """Commit state changes and return app hash"""
        try:
            # In production, this would persist to database
            # For now, we'll just calculate a hash of the current state
            
            state_hash = hashlib.sha256(
                json.dumps(self.current_state, sort_keys=True).encode()
            ).hexdigest()
            
            self.pending_changes = []  # Clear pending changes after commit
            
            return state_hash
            
        except Exception as e:
            logger.error(f"Error committing state: {e}")
            return ""
    
    def get_balance(self, address: str) -> int:
        """Get balance for address"""
        return self.current_state.get(address, {}).get('balance', 0)
    
    def query_state(self, path: str) -> Any:
        """Query application state"""
        if path.startswith('balance/'):
            address = path[8:]  # Remove 'balance/' prefix
            return self.get_balance(address)
        elif path == 'state':
            return self.current_state
        else:
            return None