# blockchain/production/block_producer.py
import time
import asyncio
import logging
from typing import Dict, List, Any, Optional

from blockchain.models.block import Block, BlockHeader
from utxo_system.models.transaction import Transaction
from blockchain.validation.validation_manager import ValidationManager

logger = logging.getLogger(__name__)

class BlockProducer:
    """Handles block production for Proof-of-Stake consensus"""
    
    def __init__(self, state_manager: Any, validation_manager: ValidationManager, 
                 config: Dict[str, Any], wallet: Any):
        self.state_manager = state_manager
        self.validation_manager = validation_manager
        self.config = config
        self.wallet = wallet
        self.last_block_time = 0
        self.block_production_count = 0
        self.production_stats = {
            'successful_blocks': 0,
            'failed_blocks': 0,
            'average_production_time': 0,
            'total_fees_earned': 0
        }
    
    async def create_new_block(self) -> Optional[Block]:
        """Create a new block if eligible"""
        try:
            # Check if we should produce a block
            if not self._should_produce_block():
                return None
            
            # Get transactions from mempool
            transactions = self._select_transactions_for_block()
            if not transactions and not self._should_create_empty_block():
                return None
            
            # Create block header
            header = self._create_block_header(transactions)
            
            # Create block
            block = Block(
                header=header,
                transactions=transactions,
                hash=header.calculate_hash(),
                chainwork=self._calculate_block_work(header),
                size=self._calculate_block_size(header, transactions)
            )
            
            # Sign block
            signed_block = await self._sign_block(block)
            
            # Validate block before broadcasting
            validation_result = self.validation_manager.validate_block(signed_block, ValidationLevel.CONSENSUS)
            if not validation_result.is_valid:
                logger.error(f"Self-created block validation failed: {validation_result.errors}")
                self.production_stats['failed_blocks'] += 1
                return None
            
            self.production_stats['successful_blocks'] += 1
            self.block_production_count += 1
            self.last_block_time = time.time()
            
            # Update production statistics
            fees_earned = sum(tx.fee for tx in transactions if hasattr(tx, 'fee'))
            self.production_stats['total_fees_earned'] += fees_earned
            
            logger.info(f"Created new block: #{header.height} with {len(transactions)} transactions, fees: {fees_earned}")
            return signed_block
            
        except Exception as e:
            logger.error(f"Block creation failed: {e}", exc_info=True)
            self.production_stats['failed_blocks'] += 1
            return None
    
    def _should_produce_block(self) -> bool:
        """Check if we should produce a block"""
        current_time = time.time()
        
        # Check block time interval
        if current_time - self.last_block_time < self.config['block_time_target']:
            return False
        
        # Check if wallet has addresses
        if not self.wallet or not self.wallet.addresses:
            return False
        
        # Check if we're eligible to produce block (validator status)
        validator_address = list(self.wallet.addresses.keys())[0]
        if not self.state_manager.consensus.is_validator_eligible(validator_address, current_time):
            return False
        
        return True
    
    def _select_transactions_for_block(self) -> List[Transaction]:
        """Select transactions for inclusion in block"""
        try:
            mempool = self.state_manager.transaction_manager.mempool
            max_block_size = self.config['max_block_size']
            max_transactions = self.config.get('max_block_transactions', 1000)
            
            # Get transactions sorted by fee rate (highest first)
            sorted_txs = sorted(
                [(tx, fee_rate) for tx, _, fee_rate in mempool.values()],
                key=lambda x: x[1],
                reverse=True
            )
            
            selected_transactions = []
            current_size = 0
            
            for transaction, fee_rate in sorted_txs:
                tx_size = len(transaction.to_bytes())
                
                # Check if transaction fits in block
                if current_size + tx_size > max_block_size:
                    continue
                
                # Check transaction count limit
                if len(selected_transactions) >= max_transactions:
                    break
                
                # Validate transaction
                validation_result = self.validation_manager.validate_transaction(transaction, ValidationLevel.FULL)
                if not validation_result.is_valid:
                    continue
                
                selected_transactions.append(transaction)
                current_size += tx_size
            
            return selected_transactions
            
        except Exception as e:
            logger.error(f"Transaction selection failed: {e}")
            return []
    
    def _should_create_empty_block(self) -> bool:
        """Check if we should create an empty block"""
        # Don't create too many empty blocks consecutively
        if self.block_production_count > 0 and self.production_stats['successful_blocks'] == 0:
            return False
        
        # Create empty block if it's been too long since last block
        time_since_last_block = time.time() - self.last_block_time
        if time_since_last_block > self.config['block_time_target'] * 2:
            return True
        
        return False
    
    def _create_block_header(self, transactions: List[Transaction]) -> BlockHeader:
        """Create block header"""
        current_head = self.state_manager.database.get('chain_head')
        if not current_head:
            raise ValueError("No current chain head found")
        
        validator_address = list(self.wallet.addresses.keys())[0]
        
        return BlockHeader(
            version=2,
            height=current_head.header.height + 1,
            previous_hash=current_head.hash,
            merkle_root=self._calculate_merkle_root(transactions),
            timestamp=int(time.time()),
            difficulty=self.state_manager.consensus.calculate_difficulty(current_head.header.height + 1),
            nonce=0,
            validator=validator_address
        )
    
    def _calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """Calculate merkle root for transactions"""
        from blockchain.utils.merkle import MerkleTree
        
        if not transactions:
            return '0' * 64  # Empty merkle root
        
        tx_hashes = [tx.hash for tx in transactions]
        return MerkleTree(tx_hashes).get_root_hash()
    
    def _calculate_block_work(self, header: BlockHeader) -> int:
        """Calculate block work value"""
        # This would be based on your specific consensus algorithm
        # For PoS, this might be based on stake amount and other factors
        return 2 ** 256 // (header.difficulty + 1)
    
    def _calculate_block_size(self, header: BlockHeader, transactions: List[Transaction]) -> int:
        """Calculate block size in bytes"""
        header_size = len(header.to_bytes())
        transactions_size = sum(len(tx.to_bytes()) for tx in transactions)
        return header_size + transactions_size
    
    async def _sign_block(self, block: Block) -> Block:
        """Sign block with validator key"""
        try:
            signature = self.wallet.sign_data(block.hash.encode())
            block.header.signature = signature
            return block
        except Exception as e:
            logger.error(f"Block signing failed: {e}")
            raise
    
    def get_production_stats(self) -> Dict[str, Any]:
        """Get block production statistics"""
        success_rate = (
            self.production_stats['successful_blocks'] / 
            (self.production_stats['successful_blocks'] + self.production_stats['failed_blocks'])
            if (self.production_stats['successful_blocks'] + self.production_stats['failed_blocks']) > 0 
            else 0
        )
        
        return {
            **self.production_stats,
            'success_rate': success_rate,
            'last_block_time': self.last_block_time,
            'total_blocks_produced': self.block_production_count
        }
    
    def reset_stats(self):
        """Reset production statistics"""
        self.production_stats = {
            'successful_blocks': 0,
            'failed_blocks': 0,
            'average_production_time': 0,
            'total_fees_earned': 0
        }
        self.block_production_count = 0