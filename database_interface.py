# database_interface.py
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from .utxo import Transaction, Block

class BlockchainDatabase(ABC):
    """Production-ready blockchain database interface"""
    
    @abstractmethod
    def get_block(self, block_hash: str) -> Optional[Block]:
        """Get block by hash with caching"""
        pass
    
    @abstractmethod
    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get block by height"""
        pass
    
    @abstractmethod
    def put_block(self, block: Block, batch=None):
        """Atomically store block"""
        pass
    
    @abstractmethod
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction by hash"""
        pass
    
    @abstractmethod
    def get_blocks_range(self, start_height: int, end_height: int) -> List[Block]:
        """Get range of blocks efficiently"""
        pass
    
    @abstractmethod
    def get_chain_head(self) -> Optional[Block]:
        """Get current chain head"""
        pass
    
    @abstractmethod
    def put_chain_head(self, block_hash: str):
        """Update chain head"""
        pass

class BlockchainDatabaseImpl(BlockchainDatabase):
    """Production-ready blockchain database implementation using AdvancedDatabase"""
    
    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        """
        Initialize blockchain database
        
        Args:
            db_path: Path to database storage
            config: Database configuration (optional)
        """
        self.db = AdvancedDatabase(db_path, config)
        self._setup_indexes()
    
    def _setup_indexes(self):
        """Setup blockchain-specific indexes"""
        # Block hash index
        self.db.create_index("block_hash", IndexConfig(
            index_type=IndexType.BTREE,
            unique=True,
            fields=["hash"]
        ))
        
        # Block height index
        self.db.create_index("block_height", IndexConfig(
            index_type=IndexType.BTREE,
            unique=True,
            fields=["height"]
        ))
        
        # Transaction hash index
        self.db.create_index("transaction_hash", IndexConfig(
            index_type=IndexType.BTREE,
            unique=True,
            fields=["tx_hash"]
        ))
        
        # Block parent hash index (for chain traversal)
        self.db.create_index("parent_hash", IndexConfig(
            index_type=IndexType.BTREE,
            unique=False,
            fields=["parent_hash"]
        ))
        
        # Address index (for UTXO/account queries)
        self.db.create_index("address", IndexConfig(
            index_type=IndexType.BTREE,
            unique=False,
            fields=["addresses"]
        ))
        
        # Timestamp index
        self.db.create_index("timestamp", IndexConfig(
            index_type=IndexType.BTREE,
            unique=False,
            fields=["timestamp"]
        ))
    
    def get_block(self, block_hash: str) -> Optional[Block]:
        """Get block by hash with caching"""
        try:
            # Use the block_hash index for efficient lookup
            results = self.db.query("block_hash", block_hash, limit=1)
            if results:
                return self._deserialize_block(results[0])
            
            # Fallback to direct key lookup
            block_data = self.db.get(f"block:{block_hash}".encode())
            if block_data:
                return self._deserialize_block(block_data)
            
            return None
            
        except KeyNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting block {block_hash}: {e}")
            return None
    
    def get_block_by_height(self, height: int) -> Optional[Block]:
        """Get block by height"""
        try:
            # Use the block_height index
            results = self.db.query("block_height", height, limit=1)
            if results:
                return self._deserialize_block(results[0])
            
            # Fallback to scanning (should be rare if index works properly)
            for key, value in self.db.iterate(prefix=b"block:"):
                block = self._deserialize_block(value)
                if block and block.height == height:
                    return block
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting block at height {height}: {e}")
            return None
    
    def put_block(self, block: Block, batch=None):
        """Atomically store block"""
        try:
            block_data = self._serialize_block(block)
            block_key = f"block:{block.hash}".encode()
            
            # Prepare index updates
            index_updates = {
                "block_hash": {"new_values": [block.hash]},
                "block_height": {"new_values": [block.height]},
                "parent_hash": {"new_values": [block.parent_hash]},
                "timestamp": {"new_values": [block.timestamp]}
            }
            
            # Extract addresses from transactions for indexing
            addresses = self._extract_addresses_from_block(block)
            if addresses:
                index_updates["address"] = {"new_values": addresses}
            
            if batch is not None:
                # Add to existing batch
                batch.append(BatchOperation(
                    op_type='put',
                    key=block_key,
                    value=block_data,
                    index_updates=index_updates
                ))
            else:
                # Single operation
                self.db.put(block_key, block_data, update_indexes=True)
                
                # Also store transactions
                self._store_transactions(block)
                
        except Exception as e:
            logger.error(f"Error storing block {block.hash}: {e}")
            raise DatabaseError(f"Failed to store block: {e}")
    
    def get_transaction(self, tx_hash: str) -> Optional[Transaction]:
        """Get transaction by hash"""
        try:
            # Use transaction hash index
            results = self.db.query("transaction_hash", tx_hash, limit=1)
            if results:
                return self._deserialize_transaction(results[0])
            
            # Direct lookup
            tx_data = self.db.get(f"tx:{tx_hash}".encode())
            if tx_data:
                return self._deserialize_transaction(tx_data)
            
            return None
            
        except KeyNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error getting transaction {tx_hash}: {e}")
            return None
    
    def get_blocks_range(self, start_height: int, end_height: int) -> List[Block]:
        """Get range of blocks efficiently"""
        blocks = []
        
        try:
            # Use block_height index for range query
            # This is a simplified approach - in production you'd want a more efficient range query
            for height in range(start_height, end_height + 1):
                block = self.get_block_by_height(height)
                if block:
                    blocks.append(block)
                else:
                    # Gap in blockchain, break early
                    break
                    
        except Exception as e:
            logger.error(f"Error getting blocks range {start_height}-{end_height}: {e}")
        
        return blocks
    
    def get_chain_head(self) -> Optional[Block]:
        """Get current chain head"""
        try:
            head_hash = self.db.get(b"chain:head")
            if head_hash:
                return self.get_block(head_hash.decode())
            return None
        except Exception as e:
            logger.error(f"Error getting chain head: {e}")
            return None
    
    def put_chain_head(self, block_hash: str):
        """Update chain head"""
        try:
            self.db.put(b"chain:head", block_hash.encode())
        except Exception as e:
            logger.error(f"Error updating chain head to {block_hash}: {e}")
            raise DatabaseError(f"Failed to update chain head: {e}")
    
    def _serialize_block(self, block: Block) -> bytes:
        """Serialize block for storage"""
        # Convert block to dict for serialization
        block_dict = {
            'hash': block.hash,
            'height': block.height,
            'parent_hash': block.parent_hash,
            'timestamp': block.timestamp,
            'transactions': [self._serialize_transaction(tx) for tx in block.transactions],
            'nonce': block.nonce,
            'difficulty': block.difficulty,
            'merkle_root': block.merkle_root
        }
        return block_dict
    
    def _deserialize_block(self, block_data: Any) -> Block:
        """Deserialize block from storage"""
        if isinstance(block_data, Block):
            return block_data
        
        # Convert dict back to Block object
        block = Block(
            hash=block_data['hash'],
            height=block_data['height'],
            parent_hash=block_data['parent_hash'],
            timestamp=block_data['timestamp'],
            transactions=[self._deserialize_transaction(tx) for tx in block_data['transactions']],
            nonce=block_data['nonce'],
            difficulty=block_data['difficulty'],
            merkle_root=block_data['merkle_root']
        )
        return block
    
    def _serialize_transaction(self, tx: Transaction) -> dict:
        """Serialize transaction for storage"""
        return {
            'hash': tx.hash,
            'inputs': [{'address': inp.address, 'amount': inp.amount} for inp in tx.inputs],
            'outputs': [{'address': out.address, 'amount': out.amount} for out in tx.outputs],
            'timestamp': tx.timestamp,
            'fee': tx.fee
        }
    
    def _deserialize_transaction(self, tx_data: Any) -> Transaction:
        """Deserialize transaction from storage"""
        if isinstance(tx_data, Transaction):
            return tx_data
        
        tx = Transaction(
            hash=tx_data['hash'],
            inputs=[Input(address=inp['address'], amount=inp['amount']) for inp in tx_data['inputs']],
            outputs=[Output(address=out['address'], amount=out['amount']) for out in tx_data['outputs']],
            timestamp=tx_data['timestamp'],
            fee=tx_data['fee']
        )
        return tx
    
    def _store_transactions(self, block: Block):
        """Store all transactions from a block"""
        batch_ops = []
        
        for tx in block.transactions:
            tx_data = self._serialize_transaction(tx)
            tx_key = f"tx:{tx.hash}".encode()
            
            # Extract addresses for indexing
            addresses = set()
            for inp in tx.inputs:
                addresses.add(inp.address)
            for out in tx.outputs:
                addresses.add(out.address)
            
            index_updates = {
                "transaction_hash": {"new_values": [tx.hash]},
                "address": {"new_values": list(addresses)}
            }
            
            batch_ops.append(BatchOperation(
                op_type='put',
                key=tx_key,
                value=tx_data,
                index_updates=index_updates
            ))
        
        # Store all transactions in a batch
        if batch_ops:
            self.db.batch_write(batch_ops)
    
    def _extract_addresses_from_block(self, block: Block) -> List[str]:
        """Extract all unique addresses from a block's transactions"""
        addresses = set()
        for tx in block.transactions:
            for inp in tx.inputs:
                addresses.add(inp.address)
            for out in tx.outputs:
                addresses.add(out.address)
        return list(addresses)
    
    def close(self):
        """Close the database"""
        self.db.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Example usage:
if __name__ == "__main__":
    # Initialize blockchain database
    blockchain_db = BlockchainDatabaseImpl("./blockchain_data")
    
    try:
        # Store a block
        block = Block(
            hash="abc123",
            height=1,
            parent_hash="000000",
            timestamp=time.time(),
            transactions=[Transaction(...)],
            nonce=12345,
            difficulty=1000,
            merkle_root="merkle123"
        )
        
        blockchain_db.put_block(block)
        blockchain_db.put_chain_head(block.hash)
        
        # Retrieve block
        retrieved_block = blockchain_db.get_block("abc123")
        print(f"Block height: {retrieved_block.height}")
        
        # Get chain head
        head = blockchain_db.get_chain_head()
        print(f"Chain head: {head.hash if head else 'None'}")
        
    finally:
        blockchain_db.close()