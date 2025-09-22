import threading
import time
from typing import List, Dict, Optional
from rayonix_wallet.core.types import Transaction, AddressInfo
from rayonix_wallet.core.exceptions import SyncError
from rayonix_wallet.storage.database import WalletDatabase

class WalletSynchronizer:
    """Blockchain synchronization service"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.running = False
        self.sync_thread = None
        self.last_sync_time = 0
        self.sync_interval = wallet.config.sync_interval
        self.current_block_height = 0
    
    def start(self):
        """Start synchronization service"""
        if self.running:
            return
        
        self.running = True
        self.sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self.sync_thread.start()
    
    def stop(self):
        """Stop synchronization service"""
        self.running = False
        if self.sync_thread:
            self.sync_thread.join(timeout=5.0)
    
    def _sync_loop(self):
        """Main synchronization loop"""
        while self.running:
            try:
                if time.time() - self.last_sync_time >= self.sync_interval:
                    self.synchronize()
                    self.last_sync_time = time.time()
                
                time.sleep(1)
            except Exception as e:
                logger.error(f"Synchronization error: {e}")
                time.sleep(30)
    
    def synchronize(self, force_full_sync: bool = False) -> bool:
        """Synchronize wallet with blockchain"""
        try:
            if not hasattr(self.wallet, 'blockchain_interface'):
                raise SyncError("Blockchain interface not available")
            
            # Get current blockchain height
            current_height = self.wallet.blockchain_interface.get_block_height()
            self.current_block_height = current_height
            
            # Update addresses and transactions
            self._update_addresses()
            self._update_transactions()
            
            # Update wallet state
            self._update_wallet_state()
            
            logger.info(f"Synchronization completed at block {current_height}")
            return True
            
        except Exception as e:
            logger.error(f"Synchronization failed: {e}")
            return False
    
    def _update_addresses(self):
        """Update address information from blockchain"""
        for address_info in self.wallet.addresses.values():
            try:
                # Get address balance and transaction count
                balance_info = self.wallet.blockchain_interface.get_address_balance(address_info.address)
                transactions = self.wallet.blockchain_interface.get_address_transactions(address_info.address)
                
                # Update address info
                address_info.balance = balance_info['balance']
                address_info.received = balance_info['received']
                address_info.sent = balance_info['sent']
                address_info.tx_count = len(transactions)
                address_info.is_used = address_info.tx_count > 0
                
                # Save to database
                self.wallet.db.save_address(address_info)
                
            except Exception as e:
                logger.warning(f"Failed to update address {address_info.address}: {e}")
    
    def _update_transactions(self):
        """Update transaction information from blockchain"""
        # Get all transactions for wallet addresses
        all_transactions = {}
        
        for address in self.wallet.addresses.keys():
            try:
                transactions = self.wallet.blockchain_interface.get_address_transactions(address)
                for tx in transactions:
                    if tx['txid'] not in all_transactions:
                        all_transactions[tx['txid']] = tx
            except Exception as e:
                logger.warning(f"Failed to get transactions for {address}: {e}")
        
        # Process transactions
        for tx_data in all_transactions.values():
            try:
                transaction = self._create_transaction_from_data(tx_data)
                self.wallet.db.save_transaction(transaction)
                self.wallet.transactions[transaction.txid] = transaction
            except Exception as e:
                logger.warning(f"Failed to process transaction {tx_data['txid']}: {e}")
    
    def _create_transaction_from_data(self, tx_data: Dict) -> Transaction:
        """Create Transaction object from blockchain data"""
        # Determine transaction direction
        is_incoming = any(
            output.get('address') in self.wallet.addresses 
            for output in tx_data.get('vout', [])
        )
        
        is_outgoing = any(
            input.get('address') in self.wallet.addresses 
            for input in tx_data.get('vin', [])
        )
        
        direction = 'in' if is_incoming and not is_outgoing else 'out'
        
        # Calculate amount
        amount = 0
        if direction == 'in':
            for output in tx_data.get('vout', []):
                if output.get('address') in self.wallet.addresses:
                    amount += output.get('value', 0)
        else:
            for input in tx_data.get('vin', []):
                if input.get('address') in self.wallet.addresses:
                    amount += input.get('value', 0)
        
        return Transaction(
            txid=tx_data['txid'],
            amount=amount,
            fee=tx_data.get('fee', 0),
            confirmations=tx_data.get('confirmations', 0),
            timestamp=tx_data.get('time', int(time.time())),
            block_height=tx_data.get('blockheight'),
            from_address=self._get_from_address(tx_data),
            to_address=self._get_to_address(tx_data),
            status='confirmed' if tx_data.get('confirmations', 0) > 0 else 'pending',
            direction=direction,
            memo=None,
            exchange_rate=None
        )
    
    def _get_from_address(self, tx_data: Dict) -> str:
        """Extract from address from transaction data"""
        for input in tx_data.get('vin', []):
            if 'address' in input:
                return input['address']
        return 'unknown'
    
    def _get_to_address(self, tx_data: Dict) -> str:
        """Extract to address from transaction data"""
        for output in tx_data.get('vout', []):
            if 'address' in output:
                return output['address']
        return 'unknown'
    
    def _update_wallet_state(self):
        """Update wallet state after synchronization"""
        state = self.wallet.state
        state.sync_height = self.current_block_height
        state.last_updated = time.time()
        state.tx_count = len(self.wallet.transactions)
        
        # Count used addresses
        used_addresses = sum(1 for addr in self.wallet.addresses.values() if addr.is_used)
        state.addresses_used = used_addresses
        
        # Calculate totals
        total_received = 0
        total_sent = 0
        for tx in self.wallet.transactions.values():
            if tx.direction == 'in':
                total_received += tx.amount
            else:
                total_sent += tx.amount
        
        state.total_received = total_received
        state.total_sent = total_sent
        
        self.wallet.db.save_wallet_state(state)
    
    def rescan_blockchain(self, from_height: int = 0) -> bool:
        """Rescan blockchain from specific height"""
        try:
            logger.info(f"Starting blockchain rescan from height {from_height}")
            
            # Clear transaction cache
            self.wallet.transactions.clear()
            
            # Reset sync height
            self.wallet.state.sync_height = from_height
            self.wallet.db.save_wallet_state(self.wallet.state)
            
            # Force full sync
            return self.synchronize(force_full_sync=True)
            
        except Exception as e:
            logger.error(f"Blockchain rescan failed: {e}")
            return False
    
    def get_sync_status(self) -> Dict:
        """Get synchronization status"""
        return {
            'running': self.running,
            'current_height': self.current_block_height,
            'wallet_height': self.wallet.state.sync_height,
            'last_sync': self.last_sync_time,
            'addresses_count': len(self.wallet.addresses),
            'transactions_count': len(self.wallet.transactions),
            'sync_progress': self._calculate_sync_progress()
        }
    
    def _calculate_sync_progress(self) -> float:
        """Calculate synchronization progress percentage"""
        if self.current_block_height == 0:
            return 0.0
        
        progress = (self.wallet.state.sync_height / self.current_block_height) * 100
        return min(progress, 100.0)