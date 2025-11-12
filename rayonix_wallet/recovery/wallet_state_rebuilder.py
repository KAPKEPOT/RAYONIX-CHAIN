#rayonix_wallet/recovery/wallet_state_rebuilder.py
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RebuildStats:
    addresses_recovered: int = 0
    transactions_recovered: int = 0
    wallet_state_rebuilt: bool = False
    errors_encountered: int = 0

class WalletStateRebuilder:
    """Rebuild wallet state from available data with real RAYONIX integration"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.stats = RebuildStats()
    
    def rebuild_wallet_state(self) -> Dict[str, Any]:
        """
        Rebuild wallet state by scanning database and integrating with blockchain
        """
        logger.warning("Rebuilding wallet state from available data...")
        
        try:
            # Step 1: Recover addresses from database
            recovered_addresses = self._recover_addresses_from_database()
            self.stats.addresses_recovered = len(recovered_addresses)
            
            # Step 2: Recover transactions from database
            recovered_transactions = self._recover_transactions_from_database()
            self.stats.transactions_recovered = len(recovered_transactions)
            
            # Step 3: Create new wallet state
            new_state = self._create_new_wallet_state(recovered_addresses, recovered_transactions)
            self.stats.wallet_state_rebuilt = True
            
            # Step 4: Apply recovered data to wallet
            self._apply_recovered_data(new_state, recovered_addresses, recovered_transactions)
            
            logger.info(f"Wallet state rebuilt: {self.stats}")
            return {
                'success': True,
                'operation': 'rebuild_wallet_state',
                'stats': self.stats.__dict__,
                'addresses_recovered': self.stats.addresses_recovered,
                'transactions_recovered': self.stats.transactions_recovered
            }
            
        except Exception as e:
            logger.error(f"Wallet state rebuild failed: {e}")
            self.stats.errors_encountered += 1
            return {
                'success': False,
                'operation': 'rebuild_wallet_state',
                'error': str(e),
                'stats': self.stats.__dict__
            }
    
    def rebuild_from_blockchain(self, rayonix_chain) -> Dict[str, Any]:
        """
        Rebuild wallet state entirely from blockchain data
        This is the most reliable but slowest method
        """
        logger.warning("Rebuilding wallet state from blockchain data...")
        
        try:
            if not rayonix_chain:
                return {'success': False, 'error': 'Blockchain not available'}
            
            # Get all our addresses
            our_addresses = list(self.wallet.addresses.keys())
            if not our_addresses:
                return {'success': False, 'error': 'No addresses available to scan'}
            
            # Reset wallet state
            self._reset_wallet_state()
            
            # Scan blockchain for our addresses
            scan_results = self._scan_blockchain_for_addresses(rayonix_chain, our_addresses)
            
            # Update wallet with found data
            self._update_wallet_from_scan(scan_results)
            
            logger.info(f"Blockchain rebuild completed: {scan_results}")
            return {
                'success': True,
                'operation': 'rebuild_from_blockchain',
                'scan_results': scan_results,
                'addresses_scanned': len(our_addresses)
            }
            
        except Exception as e:
            logger.error(f"Blockchain rebuild failed: {e}")
            return {
                'success': False,
                'operation': 'rebuild_from_blockchain',
                'error': str(e)
            }
    
    def _recover_addresses_from_database(self) -> Dict[str, Any]:
        """Recover addresses from database with validation"""
        addresses = {}
        
        try:
            raw_addresses = self.wallet.db.get_all_addresses()
            if not raw_addresses:
                logger.warning("No addresses found in database")
                return addresses
            
            for addr_data in raw_addresses:
                try:
                    # Validate and convert address data
                    address_info = self._validate_and_convert_address(addr_data)
                    if address_info:
                        addresses[address_info.address] = address_info
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid address data: {e}")
                    continue
                    
            logger.info(f"Recovered {len(addresses)} addresses from database")
            return addresses
            
        except Exception as e:
            logger.error(f"Failed to recover addresses: {e}")
            return {}
    
    def _recover_transactions_from_database(self) -> Dict[str, Any]:
        """Recover transactions from database with validation"""
        transactions = {}
        
        try:
            raw_transactions = self.wallet.db.get_transactions(limit=10000)
            if not raw_transactions:
                logger.warning("No transactions found in database")
                return transactions
            
            for tx_data in raw_transactions:
                try:
                    # Validate and convert transaction data
                    transaction = self._validate_and_convert_transaction(tx_data)
                    if transaction:
                        transactions[transaction.txid] = transaction
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid transaction data: {e}")
                    continue
                    
            logger.info(f"Recovered {len(transactions)} transactions from database")
            return transactions
            
        except Exception as e:
            logger.error(f"Failed to recover transactions: {e}")
            return {}
    
    def _validate_and_convert_address(self, addr_data) -> Optional[Any]:
        """Validate address data and convert to proper object"""
        try:
            if isinstance(addr_data, dict):
                from rayonix_wallet.core.wallet_types import AddressInfo
                
                # Validate required fields
                required_fields = ['address', 'index', 'derivation_path']
                if not all(field in addr_data for field in required_fields):
                    return None
                
                return AddressInfo(**addr_data)
                
            elif hasattr(addr_data, 'address') and hasattr(addr_data, 'index'):
                # Already a valid AddressInfo object
                return addr_data
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Address validation failed: {e}")
            return None
    
    def _validate_and_convert_transaction(self, tx_data) -> Optional[Any]:
        """Validate transaction data and convert to proper object"""
        try:
            if isinstance(tx_data, dict):
                from rayonix_wallet.core.wallet_types import Transaction
                
                # Validate required fields
                required_fields = ['txid', 'amount']
                if not all(field in tx_data for field in required_fields):
                    return None
                
                return Transaction(**tx_data)
                
            elif hasattr(tx_data, 'txid') and hasattr(tx_data, 'amount'):
                # Already a valid Transaction object
                return tx_data
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Transaction validation failed: {e}")
            return None
    
    def _create_new_wallet_state(self, addresses: Dict[str, Any], transactions: Dict[str, Any]):
        """Create new wallet state from recovered data"""
        from rayonix_wallet.core.wallet_types import WalletState
        
        # Calculate statistics
        total_received = sum(tx.amount for tx in transactions.values() if hasattr(tx, 'amount'))
        used_addresses = sum(1 for addr in addresses.values() if getattr(addr, 'is_used', False))
        
        return WalletState(
            sync_height=0,  # Will need rescan
            last_updated=time.time(),
            tx_count=len(transactions),
            addresses_generated=len(addresses),
            addresses_used=used_addresses,
            total_received=total_received,
            total_sent=0,  # Hard to determine without full UTXO tracking
            security_score=getattr(self.wallet, '_calculate_initial_security_score', lambda: 50)()
        )
    
    def _apply_recovered_data(self, new_state, addresses: Dict[str, Any], transactions: Dict[str, Any]):
        """Apply recovered data to wallet"""
        # Update wallet state
        self.wallet.state = new_state
        self.wallet.db.save_wallet_state(new_state)
        
        # Update addresses
        self.wallet.addresses = addresses
        for address_info in addresses.values():
            try:
                self.wallet.db.save_address(address_info)
            except Exception as e:
                logger.warning(f"Failed to save address {address_info.address}: {e}")
        
        # Update transactions
        self.wallet.transactions = transactions
        for transaction in transactions.values():
            try:
                self.wallet.db.save_transaction(transaction)
            except Exception as e:
                logger.warning(f"Failed to save transaction {transaction.txid}: {e}")
    
    def _reset_wallet_state(self):
        """Reset wallet state to clean slate"""
        from rayonix_wallet.core.wallet_types import WalletState
        
        self.wallet.state = WalletState(
            sync_height=0,
            last_updated=time.time(),
            tx_count=0,
            addresses_generated=0,
            addresses_used=0,
            total_received=0,
            total_sent=0,
            security_score=50
        )
        
        self.wallet.transactions = {}
    
    def _scan_blockchain_for_addresses(self, rayonix_chain, addresses: List[str]) -> Dict[str, Any]:
        """Scan blockchain for transactions involving our addresses"""
        results = {
            'addresses_found': set(),
            'transactions_found': 0,
            'total_received': 0,
            'blocks_scanned': 0,
            'scan_range': 'full'
        }
        
        try:
            current_height = rayonix_chain.get_block_count()
            if current_height <= 0:
                return results
            
            # Convert to set for faster lookups
            address_set = set(addresses)
            
            # Scan recent blocks first (most likely to have our transactions)
            start_height = max(0, current_height - 10000)  # Last 10,000 blocks
            logger.info(f"Scanning blocks {start_height} to {current_height}")
            
            for height in range(start_height, current_height + 1):
                block = rayonix_chain.get_block_by_height(height)
                if not block:
                    continue
                
                block_result = self._scan_block_for_addresses(block, address_set, height)
                if block_result['transactions_found'] > 0:
                    results['transactions_found'] += block_result['transactions_found']
                    results['total_received'] += block_result['total_received']
                    results['addresses_found'].update(block_result['addresses_found'])
                
                results['blocks_scanned'] += 1
                
                # Progress logging
                if height % 1000 == 0:
                    logger.info(f"Blockchain scan: {height}/{current_height} - "
                              f"Found {results['transactions_found']} transactions")
            
            results['addresses_found'] = list(results['addresses_found'])
            return results
            
        except Exception as e:
            logger.error(f"Blockchain scan failed: {e}")
            return results
    
    def _scan_block_for_addresses(self, block, address_set: set, block_height: int) -> Dict[str, Any]:
        """Scan a single block for our addresses"""
        result = {
            'transactions_found': 0,
            'total_received': 0,
            'addresses_found': set()
        }
        
        try:
            transactions = getattr(block, 'transactions', [])
            for tx in transactions:
                # Check outputs for our addresses
                outputs = getattr(tx, 'outputs', [])
                for output in outputs:
                    output_address = getattr(output, 'address', '')
                    if output_address in address_set:
                        result['transactions_found'] += 1
                        result['total_received'] += getattr(output, 'amount', 0)
                        result['addresses_found'].add(output_address)
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to scan block {block_height}: {e}")
            return result
    
    def _update_wallet_from_scan(self, scan_results: Dict[str, Any]):
        """Update wallet with scan results"""
        # Update addresses that were found
        for address in scan_results['addresses_found']:
            if address in self.wallet.addresses:
                self.wallet.addresses[address].is_used = True
        
        # Update wallet state
        self.wallet.state.total_received = scan_results['total_received']
        self.wallet.state.addresses_used = len(scan_results['addresses_found'])
        self.wallet.state.tx_count = scan_results['transactions_found']
        
        # Save updated state
        self.wallet.db.save_wallet_state(self.wallet.state)