# tasks/mempool_task.py - Mempool processing

import asyncio
import time
import logging
from typing import Dict, List

logger = logging.getLogger("rayonix_node.mempool")

class MempoolTask:
    """Handles mempool management and transaction processing"""
    
    def __init__(self, node: 'RayonixNode'):
        self.node = node
        self.last_cleanup = time.time()
        self.transactions_processed = 0
    
    async def process_mempool(self):
        """Main mempool processing loop"""
        while self.node.running:
            try:
                # Process mempool transactions
                await self._process_transactions()
                
                # Clean up old transactions periodically
                if time.time() - self.last_cleanup > 300:  # Every 5 minutes
                    await self._cleanup_mempool()
                    self.last_cleanup = time.time()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in mempool processing: {e}")
                await asyncio.sleep(5)
    
    async def _process_transactions(self):
        """Process transactions in the mempool"""
        mempool = self.node.rayonix_chain.mempool
        if not mempool:
            return
        
        # Process transactions in order of fee (highest first)
        sorted_txs = sorted(
            mempool.values(), 
            key=lambda tx: self._calculate_transaction_priority(tx), 
            reverse=True
        )
        
        processed_count = 0
        for transaction in sorted_txs:
            if not self.node.running:
                break
            
            try:
                # Check if transaction is still valid
                if not self.node.rayonix_chain._validate_transaction(transaction):
                    logger.debug(f"Removing invalid transaction: {transaction.get('hash')}")
                    self.node.rayonix_chain._remove_from_mempool(transaction.get('hash'))
                    continue
                
                # Attempt to include in next block if mining/staking
                if self._should_include_in_block():
                    # This would typically be handled by the block creation process
                    pass
                
                processed_count += 1
                self.transactions_processed += 1
                
                # Limit processing per iteration to avoid blocking
                if processed_count >= 100:
                    break
                    
            except Exception as e:
                logger.error(f"Error processing transaction {transaction.get('hash')}: {e}")
                continue
    
    async def _cleanup_mempool(self):
        """Clean up old or invalid transactions from mempool"""
        mempool = self.node.rayonix_chain.mempool
        if not mempool:
            return
        
        current_time = time.time()
        removed_count = 0
        
        for tx_hash, transaction in list(mempool.items()):
            if not self.node.running:
                break
            
            try:
                # Remove transactions older than 24 hours
                tx_time = transaction.get('timestamp', 0)
                if current_time - tx_time > 86400:  # 24 hours
                    self.node.rayonix_chain._remove_from_mempool(tx_hash)
                    removed_count += 1
                    continue
                
                # Remove invalid transactions
                if not self.node.rayonix_chain._validate_transaction(transaction):
                    self.node.rayonix_chain._remove_from_mempool(tx_hash)
                    removed_count += 1
                    continue
                    
            except Exception as e:
                logger.error(f"Error cleaning up transaction {tx_hash}: {e}")
                continue
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} transactions from mempool")
    
    def _calculate_transaction_priority(self, transaction: Dict) -> float:
        """Calculate transaction priority for processing"""
        # Base priority on fee per byte
        fee = self._calculate_transaction_fee(transaction)
        size = len(str(transaction).encode('utf-8'))
        
        if size > 0:
            return fee / size
        else:
            return 0
    
    def _calculate_transaction_fee(self, transaction: Dict) -> float:
        """Calculate transaction fee"""
        # This is a simplified implementation
        # In a real implementation, this would calculate actual fee based on inputs/outputs
        return transaction.get('fee', 0)
    
    def _should_include_in_block(self) -> bool:
        """Check if we should include transactions in a new block"""
        # For miners/stakers
        return self.node.get_config_value('consensus.consensus_type') in ['pos', 'pow']
    
    def get_mempool_stats(self) -> Dict:
        """Get mempool statistics"""
        mempool = self.node.rayonix_chain.mempool
        return {
            'transaction_count': len(mempool) if mempool else 0,
            'total_size': sum(len(str(tx).encode('utf-8')) for tx in mempool.values()) if mempool else 0,
            'transactions_processed': self.transactions_processed,
            'last_cleanup': self.last_cleanup
        }