#rayonix_wallet/recovery/blockchain_rescanner.py
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RescanProgress:
    current_height: int = 0
    target_height: int = 0
    percentage: float = 0.0
    processed_blocks: int = 0
    found_transactions: int = 0
    estimated_time_remaining: str = "unknown"

class BlockchainRescanner:
    """RAYONIX blockchain rescanning integrated with actual UTXO system"""
    
    def __init__(self, wallet, rayonix_chain=None):
        self.wallet = wallet
        self.rayonix_chain = rayonix_chain
        self.is_rescanning = False
        self.progress = RescanProgress()
        self._rescan_start_time = None
    
    def full_rescan(self, start_height: int = 0) -> Dict[str, Any]:
        """
        Full blockchain rescan using actual RAYONIX chain
        """
        if not self.rayonix_chain:
            return {
                'success': False, 
                'error': 'RAYONIX chain not available for rescan'
            }
        
        logger.warning(f"Starting FULL blockchain rescan from height {start_height}")
        
        self.is_rescanning = True
        self._rescan_start_time = time.time()
        
        try:
            # Get current blockchain height from actual chain
            chain_height = self.rayonix_chain.get_block_count()
            if chain_height <= 0:
                return {
                    'success': False, 
                    'error': 'Cannot get blockchain height'
                }
            
            self.progress.target_height = chain_height
            self.progress.current_height = start_height
            
            # Reset wallet state for rescan
            self._prepare_wallet_for_rescan()
            
            # Perform actual rescan
            results = self._perform_blockchain_rescan(start_height, chain_height)
            
            # Finalize wallet state
            self._finalize_rescan(results)
            
            logger.info(f"Full rescan completed: {results}")
            return {
                'success': True,
                'strategy': 'full_rescan',
                'results': results,
                'final_height': chain_height,
                'duration_seconds': time.time() - self._rescan_start_time
            }
            
        except Exception as e:
            logger.error(f"Full rescan failed: {e}")
            return {'success': False, 'error': str(e)}
        finally:
            self.is_rescanning = False
    
    def _prepare_wallet_for_rescan(self):
        """Prepare wallet for rescan - reset state but keep addresses"""
        logger.info("Preparing wallet for blockchain rescan...")
        
        # Reset wallet state
        self.wallet.state.sync_height = 0
        self.wallet.state.tx_count = 0
        self.wallet.state.total_received = 0
        self.wallet.state.total_sent = 0
        
        # Clear transaction cache (will be rebuilt from blockchain)
        self.wallet.transactions = {}
        
        # Reset address balances (will be recalculated)
        for address_info in self.wallet.addresses.values():
            address_info.balance = 0
            address_info.received = 0
            address_info.sent = 0
            address_info.tx_count = 0
            address_info.is_used = False
        
        logger.info("Wallet prepared for rescan")
    
    def _perform_blockchain_rescan(self, start_height: int, end_height: int) -> Dict[str, Any]:
        """Perform actual blockchain rescan using RAYONIX chain"""
        results = {
            'blocks_processed': 0,
            'transactions_found': 0,
            'addresses_found': set(),
            'total_received': 0,
            'start_height': start_height,
            'end_height': end_height
        }
        
        logger.info(f"Rescanning blocks {start_height} to {end_height}")
        
        for height in range(start_height, end_height + 1):
            if not self.is_rescanning:
                logger.warning("Rescan interrupted by user")
                break
            
            # Update progress
            self._update_progress(height, end_height)
            
            try:
                # Get actual block from RAYONIX chain
                block = self.rayonix_chain.get_block_by_height(height)
                if not block:
                    logger.warning(f"Block {height} not found in blockchain")
                    continue
                
                # Process block transactions
                block_result = self._process_block_transactions(block, height)
                if block_result:
                    results['blocks_processed'] += 1
                    results['transactions_found'] += block_result['transactions_found']
                    results['total_received'] += block_result['total_received']
                    results['addresses_found'].update(block_result['addresses_found'])
                
                # Update wallet sync height
                self.wallet.state.sync_height = height
                
                # Log progress every 100 blocks
                if height % 100 == 0:
                    logger.info(f"Rescan progress: {height}/{end_height} "
                              f"({height/end_height*100:.1f}%) - "
                              f"Found {results['transactions_found']} transactions")
                    
            except Exception as e:
                logger.error(f"Error processing block {height}: {e}")
                # Continue with next block
                continue
        
        # Convert set to list for JSON serialization
        results['addresses_found'] = list(results['addresses_found'])
        
        return results
    
    def _process_block_transactions(self, block, block_height: int) -> Optional[Dict[str, Any]]:
        """Process transactions in a block and find wallet-related ones"""
        try:
            block_result = {
                'transactions_found': 0,
                'total_received': 0,
                'addresses_found': set(),
                'block_height': block_height
            }
            
            # Get transactions from block
            transactions = getattr(block, 'transactions', [])
            if not transactions:
                return block_result
            
            for tx in transactions:
                tx_result = self._process_transaction(tx, block_height)
                if tx_result and tx_result['is_ours']:
                    block_result['transactions_found'] += 1
                    block_result['total_received'] += tx_result.get('received', 0)
                    block_result['addresses_found'].update(tx_result.get('our_addresses', []))
            
            return block_result
            
        except Exception as e:
            logger.error(f"Failed to process block {block_height}: {e}")
            return None
    
    def _process_transaction(self, tx, block_height: int) -> Optional[Dict[str, Any]]:
        """Process a transaction and check if it involves our wallet addresses"""
        try:
            tx_result = {
                'is_ours': False,
                'received': 0,
                'our_addresses': [],
                'txid': getattr(tx, 'hash', 'unknown')
            }
            
            # Get our wallet addresses
            our_addresses = set(self.wallet.addresses.keys())
            if not our_addresses:
                return tx_result
            
            # Check transaction outputs (receiving)
            outputs = getattr(tx, 'outputs', [])
            for output in outputs:
                output_address = getattr(output, 'address', None)
                if output_address and output_address in our_addresses:
                    tx_result['is_ours'] = True
                    amount = getattr(output, 'amount', 0)
                    tx_result['received'] += amount
                    tx_result['our_addresses'].append(output_address)
                    
                    # Update address state
                    self._update_address_for_receiving(output_address, amount, tx, block_height)
            
            # Note: Input processing would require UTXO lookup to see if we're spending our own outputs
            # This is more complex and requires full UTXO set access
            
            if tx_result['is_ours']:
                # Add to wallet transactions
                self._add_transaction_to_wallet(tx, block_height, tx_result['received'])
            
            return tx_result
            
        except Exception as e:
            logger.error(f"Failed to process transaction: {e}")
            return None
    
    def _update_address_for_receiving(self, address: str, amount: int, tx, block_height: int):
        """Update address state when receiving funds"""
        if address in self.wallet.addresses:
            addr_info = self.wallet.addresses[address]
            addr_info.balance += amount
            addr_info.received += amount
            addr_info.tx_count += 1
            addr_info.is_used = True
            
            # Update wallet totals
            self.wallet.state.total_received += amount
    
    def _add_transaction_to_wallet(self, tx, block_height: int, received_amount: int):
        """Add transaction to wallet's transaction list"""
        try:
            from rayonix_wallet.core.wallet_types import Transaction
            
            # Create wallet transaction record
            wallet_tx = Transaction(
                txid=getattr(tx, 'hash', ''),
                from_address=getattr(tx, 'from_address', ''),
                to_address=', '.join(self._get_output_addresses(tx)),
                amount=received_amount,
                fee=getattr(tx, 'fee', 0),
                timestamp=getattr(tx, 'timestamp', time.time()),
                block_height=block_height,
                status='confirmed'
            )
            
            # Add to wallet
            self.wallet.transactions[wallet_tx.txid] = wallet_tx
            self.wallet.state.tx_count = len(self.wallet.transactions)
            
        except Exception as e:
            logger.error(f"Failed to add transaction to wallet: {e}")
    
    def _get_output_addresses(self, tx) -> List[str]:
        """Extract output addresses from transaction"""
        addresses = []
        for output in getattr(tx, 'outputs', []):
            addr = getattr(output, 'address', None)
            if addr:
                addresses.append(addr)
        return addresses
    
    def _update_progress(self, current_height: int, target_height: int):
        """Update rescan progress information"""
        self.progress.current_height = current_height
        self.progress.target_height = target_height
        
        if target_height > 0:
            self.progress.percentage = (current_height / target_height) * 100
            
            # Calculate ETA
            if self._rescan_start_time and current_height > 0:
                elapsed = time.time() - self._rescan_start_time
                blocks_per_second = current_height / elapsed
                remaining_blocks = target_height - current_height
                
                if blocks_per_second > 0:
                    remaining_seconds = remaining_blocks / blocks_per_second
                    if remaining_seconds < 60:
                        self.progress.estimated_time_remaining = f"{remaining_seconds:.0f}s"
                    elif remaining_seconds < 3600:
                        self.progress.estimated_time_remaining = f"{remaining_seconds/60:.1f}m"
                    else:
                        self.progress.estimated_time_remaining = f"{remaining_seconds/3600:.1f}h"
    
    def get_rescan_progress(self) -> RescanProgress:
        """Get current rescan progress"""
        return self.progress
    
    def stop_rescan(self):
        """Stop ongoing rescan"""
        logger.info("Stopping blockchain rescan...")
        self.is_rescanning = False