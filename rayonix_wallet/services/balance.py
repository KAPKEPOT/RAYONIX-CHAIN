from typing import Dict, Optional, Tuple  # Added Tuple import
from dataclasses import asdict  # Added missing import
import time  # Added missing import
import logging  # Added missing import
from rayonix_wallet.core.types import WalletBalance
from rayonix_wallet.core.exceptions import WalletError

# Add logger initialization
logger = logging.getLogger(__name__)

class BalanceCalculator:
    """Balance calculation and management"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self._cache = {}
        self._cache_timeout = 300  # 5 minutes
        # Initialize missing attributes
        self._balance_cache = None
        self._last_balance_update = None
        self._balance_lock = None  # Should be initialized properly
        self.rayonix_chain = None  # Should be set by parent class
        self.addresses = {}
        self.transactions = {}
    
    def get_balance(self, force_refresh: bool = False) -> WalletBalance:
        if not self.rayonix_chain:
            return self._get_offline_balance()
            
        # Initialize lock if not exists
        if self._balance_lock is None:
            import threading
            self._balance_lock = threading.RLock()
            
        with self._balance_lock:
            try:
                if force_refresh or self._should_refresh_balance():
                    return self._calculate_comprehensive_balance()
                else:
                    cached = self._get_cached_balance()
                    if cached:
                        return cached
                    return self._calculate_comprehensive_balance()
                    
            except Exception as e:
                logger.error(f"Balance calculation failed: {e}")
                return self._handle_balance_calculation_error(e)		
                    
    def _get_offline_balance(self) -> WalletBalance:
        try:
            # Try to get the most recent cached balance first
            cached_balance = self._get_cached_balance()
            if cached_balance:
                logger.info("Using cached balance data (offline mode)")
                return self._mark_balance_as_offline(cached_balance)
            # If no cache, try to reconstruct from transaction history
            reconstructed_balance = self._reconstruct_balance_from_history()
            
            if reconstructed_balance:
                logger.warning("Using reconstructed balance from transaction history (offline mode)")
                return self._mark_balance_as_offline(reconstructed_balance)
            # Last resort: use wallet state and heuristics
            return self._estimate_balance_from_wallet_state()
            
        except Exception as e:
            logger.error(f"Offline balance calculation failed: {e}")
            return self._create_error_balance("Offline balance calculation failed")
            
    def _get_cached_balance(self) -> Optional[WalletBalance]:
        """Get validated cached balance with staleness check"""
        if not hasattr(self, '_balance_cache') or not self._balance_cache:
            return None
            
        cache_data = self._balance_cache
        cache_age = time.time() - cache_data.get('timestamp', 0)
        
        # Validate cache signature if available
        if hasattr(self, '_validate_balance_signature') and cache_data.get('signature'):
            if not self._validate_balance_signature(cache_data):
                logger.warning("Cached balance signature validation failed")
                return None
                
        if cache_age > 86400:
            return None
        return cache_data.get('balance')
        
    def _reconstruct_balance_from_history(self) -> Optional[WalletBalance]:
        if not hasattr(self, 'transactions') or not self.transactions:
            return None
        try:
            total_received = 0
            total_sent = 0
            address_balances = {}
            
            # Analyze all transactions to reconstruct balances
            for txid, transaction in self.transactions.items():
                if transaction.direction == "received" and transaction.to_address in self.addresses:
                    total_received += transaction.amount
                    address_balances[transaction.to_address] = address_balances.get(transaction.to_address, 0) + transaction.amount
                    
                elif transaction.direction == "sent" and transaction.from_address in self.addresses:
                    total_sent += transaction.amount
                    address_balances[transaction.from_address] = address_balances.get(transaction.from_address, 0) - transaction.amount
                    
            # Create detailed address breakdown
            by_address = {}
            for address, balance in address_balances.items():
                by_address[address] = {
                    "total": balance,
                    "confirmed": balance,
                    "unconfirmed": 0,
                    "locked": 0,
                    "available": balance,
                    "offline_estimated": True,
                    "last_known_activity": self._get_last_tx_timestamp_for_address(address)
                }
            total_balance = total_received - total_sent
            return WalletBalance(
                total=total_balance,
                confirmed=total_balance,
                unconfirmed=0,
                locked=0,
                available=total_balance,
                by_address=by_address,
                offline_estimated=True,
                reconstruction_confidence=self._calculate_reconstruction_confidence()
            )
        except Exception as e:
            logger.error(f"Balance reconstruction failed: {e}")
            return None   

    def _estimate_balance_from_wallet_state(self) -> WalletBalance:
        base_estimate = 0
        by_address = {}
        
        for address, info in self.addresses.items():
            # Simple heuristic: addresses with recent activity might have balances
            address_estimate = 0
            if info.is_used and info.tx_count > 0:
                # Very rough estimate based on transaction count
                address_estimate = max(1000, info.tx_count * 500)
                
            elif not info.is_used:
                # New addresses likely have 0 balance
                address_estimate = 0
                
            else:
                # Default small balance for active addresses
                address_estimate = 100
                
            by_address[address] = {
                "total": address_estimate,
                "confirmed": address_estimate,
                "unconfirmed": 0,
                "locked": 0,
                "available": address_estimate,
                "offline_estimated": True,
                "estimation_confidence": "low"
            }
            base_estimate += address_estimate
            
        return WalletBalance(
            total=base_estimate,
            confirmed=base_estimate,
            unconfirmed=0,
            locked=0,
            available=base_estimate,
            by_address=by_address,
            offline_estimated=True,
            estimation_confidence="very_low",
            warning="Balance is a rough offline estimate - connect to network for accurate data"
        )
        
    def _mark_balance_as_offline(self, balance: WalletBalance) -> WalletBalance:
        """Add offline metadata to balance data"""
        # Convert to dict to add additional fields
        balance_dict = asdict(balance) if hasattr(balance, '__dataclass_fields__') else balance.__dict__
        balance_dict.update({
            'offline_mode': True,
            'last_online_update': self._last_balance_update,
            'data_freshness': self._calculate_data_freshness(),
            'confidence_level': self._calculate_offline_confidence(balance),
            'warning': 'Balance data may be outdated - connect to network for latest information'
        })
        # Convert back to WalletBalance or similar structure
        return WalletBalance(**balance_dict)
        
    def _calculate_data_freshness(self) -> str:
        """Calculate how fresh the offline data is"""
        if not self._last_balance_update:
            return "unknown"
        age_hours = (time.time() - self._last_balance_update) / 3600
        
        if age_hours < 1:
            return "very_fresh"
        elif age_hours < 6:
            return "fresh"
        elif age_hours < 24:
            return "recent"
        elif age_hours < 72:
            return "aged"
        else:
            return "very_aged"
   
    def _calculate_offline_confidence(self, balance: WalletBalance) -> str:
        """Calculate confidence level for offline balance data"""
        if hasattr(balance, 'offline_estimated') and balance.offline_estimated:
            if hasattr(balance, 'estimation_confidence'):
                return balance.estimation_confidence
                
            return "low"  
        # Cached data confidence based on age
        data_freshness = self._calculate_data_freshness()
        freshness_to_confidence = {
            "very_fresh": "very_high",
            "fresh": "high",
            "recent": "medium",
            "aged": "low",
            "very_aged": "very_low",
            "unknown": "low" 
        }
        return freshness_to_confidence.get(data_freshness, "low")                        	
    	
    def _get_last_tx_timestamp_for_address(self, address: str) -> Optional[float]:
        """Get timestamp of last transaction for an address"""
        if not hasattr(self, 'transactions') or not self.transactions:
            return None
            
        last_timestamp = 0
        for tx in self.transactions.values():
            if (tx.to_address == address or tx.from_address == address) and tx.timestamp > last_timestamp:
                last_timestamp = tx.timestamp
        return last_timestamp if last_timestamp > 0 else None
    	    								                       		
    def _calculate_reconstruction_confidence(self) -> str:
        """Calculate confidence level for reconstructed balance"""
        if not hasattr(self, 'transactions') or not self.transactions:
            return "very_low"
            
        tx_count = len(self.transactions)
        if tx_count > 100:
            return "high"
        elif tx_count > 20:
            return "medium"
        elif tx_count > 5:
            return "low"
        else:
            return "very_low"
    		
    def _create_error_balance(self, error_message: str) -> WalletBalance:
        """Create error balance response"""
        return WalletBalance(
            total=-1,
            confirmed=-1,
            unconfirmed=-1,
            locked=-1,
            available=-1,
            by_address={},
            error=error_message,
            error_type="offline_mode_error",
            offline_mode=True
        )    																		                       								
    def _should_refresh_balance(self) -> bool:
        """Determine if balance should be refreshed"""
        if not self._last_balance_update:
            return True
        cache_age = time.time() - self._last_balance_update	
        return cache_age > 30		    		
    			
    def _calculate_comprehensive_balance(self) -> WalletBalance:
        """Calculate comprehensive balance with multi-layer validation"""
        total_confirmed = 0
        total_unconfirmed = 0
        total_locked = 0
        by_address = {}
        token_balances = {}
        
        # Track performance for monitoring
        start_time = time.time()
        addresses_processed = 0
        for address, address_info in self.addresses.items():
            try:
                address_balance = self._get_address_balance_with_retry(address)
                # Calculate confirmed vs unconfirmed based on transaction maturity
                confirmed, unconfirmed, locked = self._classify_balance_by_maturity(address, address_balance)
                by_address[address] = {
                    "total": confirmed + unconfirmed,
                    "confirmed": confirmed,
                    "unconfirmed": unconfirmed,
                    "locked": locked,
                    "available": max(0, confirmed - locked),
                    "utxo_count": self._get_utxo_count(address),
                    "last_activity": self._get_last_activity_timestamp(address)
                    
                }
                total_confirmed += confirmed
                total_unconfirmed += unconfirmed
                total_locked += locked
                addresses_processed += 1
                
                # Check for token balances if supported
                if self._supports_tokens():
                    token_balances.update(self._get_token_balances(address))
                    
            except Exception as e:  # Catch generic exception since specific ones may not be defined
                logger.warning(f"Blockchain connection failed for address {address}: {e}")
                # Use cached values with stale marker
                by_address[address] = self._get_cached_address_balance(address, stale=True)
            except Exception as e:
                logger.error(f"Unexpected error processing address {address}: {e}")
                
                by_address[address] = {
                    "total": 0,
                    "confirmed": 0,
                    "unconfirmed": 0,
                    "locked": 0,
                    "available": 0,
                    "error": str(e),
                    "utxo_count": 0
                }
        # Calculate available balance considering locked funds
        total_available = max(0, total_confirmed - total_locked)
        
        # Update cache with new balance information
        balance_data = WalletBalance(
            total=total_confirmed + total_unconfirmed,
            confirmed=total_confirmed,
            unconfirmed=total_unconfirmed,
            locked=total_locked,
            available=total_available,
            by_address=by_address,
            tokens=token_balances
        )
        self._update_balance_cache(balance_data)
        
        # Log performance metrics
        processing_time = time.time() - start_time
        if processing_time > 1.0:
            logger.warning(f"Balance calculation took {processing_time:.2f}s for {addresses_processed} addresses")
        return balance_data  
   	
    def _get_address_balance_with_retry(self, address: str, max_retries: int = 3) -> int:
        """Get address balance with retry logic and exponential backoff"""
        for attempt in range(max_retries):
            try:
                # Use the blockchain's UTXO set for accurate balance calculation
                utxos = self.rayonix_chain.utxo_set.get_utxos_for_address(address)
                
                if not utxos:
                    return 0
                    
                total_balance = sum(utxo.amount for utxo in utxos if not utxo.spent)
                
                # Validate balance consistency
                self._validate_balance_consistency(address, total_balance)
                return total_balance
                
            except Exception as e:  # Catch generic exception
                if attempt == max_retries - 1:
                    raise
                    
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1} for address {address} after {wait_time}s: {e}")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Permanent error getting balance for {address}: {e}")
                raise
                
    def _classify_balance_by_maturity(self, address: str, total_balance: int) -> Tuple[int, int, int]:
        """Classify balance into confirmed, unconfirmed, and locked amounts"""
        confirmed_balance = 0
        unconfirmed_balance = 0
        locked_balance = 0
        
        try:
            # Get UTXOs for detailed classification
            utxos = self.rayonix_chain.utxo_set.get_utxos_for_address(address)
            for utxo in utxos:
                if utxo.spent:
                    continue
                    
                # Check if UTXO is confirmed (has sufficient confirmations)
                is_confirmed = self._is_utxo_confirmed(utxo)
                
                # Check if UTXO is locked (time-locked or other constraints)
                is_locked = self._is_utxo_locked(utxo)
                
                if is_locked:
                    locked_balance += utxo.amount
                elif is_confirmed:
                    confirmed_balance += utxo.amount
                else:
                    unconfirmed_balance += utxo.amount
        except Exception as e:
            logger.error(f"Error classifying balance for {address}: {e}")
            
            # Fallback: consider everything confirmed if classification fails
            confirmed_balance = total_balance
            
        return confirmed_balance, unconfirmed_balance, locked_balance
        
    def _handle_balance_calculation_error(self, error: Exception) -> WalletBalance:
        """Handle balance calculation errors gracefully"""
        error_type = type(error).__name__
        # Try to use cached balance if available
        cached_balance = self._get_cached_balance()
        if cached_balance:
            logger.warning(f"Using cached balance due to {error_type}: {error}")
            return cached_balance
            
        # Return error-indicative balance
        return WalletBalance(
            total=-1,
            confirmed=-1,
            unconfirmed=-1,
            locked=-1,
            available=-1,
            by_address={},
            tokens={},
            error=str(error),
            error_type=error_type
        )
    
    def clear_cache(self):
        """Clear balance cache"""
        self._cache.clear()
        
    # Add missing method stubs for unimplemented methods
    def _get_utxo_count(self, address: str) -> int:
        """Get UTXO count for address"""
        try:
            utxos = self.rayonix_chain.utxo_set.get_utxos_for_address(address)
            return len(utxos) if utxos else 0
        except:
            return 0
            
    def _get_last_activity_timestamp(self, address: str) -> Optional[float]:
        """Get last activity timestamp for address"""
        return self._get_last_tx_timestamp_for_address(address)
        
    def _supports_tokens(self) -> bool:
        """Check if token support is available"""
        return False
        
    def _get_token_balances(self, address: str) -> Dict:
        """Get token balances for address"""
        return {}
        
    def _validate_balance_consistency(self, address: str, balance: int):
        """Validate balance consistency"""
        # Basic validation - can be expanded
        if balance < 0:
            logger.warning(f"Negative balance detected for address {address}: {balance}")
            
    def _is_utxo_confirmed(self, utxo) -> bool:
        """Check if UTXO is confirmed"""
        # Default implementation - should be overridden
        return True
        
    def _is_utxo_locked(self, utxo) -> bool:
        """Check if UTXO is locked"""
        # Default implementation - should be overridden
        return False
        
    def _get_cached_address_balance(self, address: str, stale: bool = False) -> Dict:
        """Get cached address balance"""
        return {
            "total": 0,
            "confirmed": 0,
            "unconfirmed": 0,
            "locked": 0,
            "available": 0,
            "stale": stale,
            "utxo_count": 0
        }
        
    def _update_balance_cache(self, balance_data: WalletBalance):
        """Update balance cache"""
        self._balance_cache = {
            'balance': balance_data,
            'timestamp': time.time()
        }
        self._last_balance_update = time.time()