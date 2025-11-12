#rayonix_wallet/recovery/corruption_detector.py

import logging
from typing import Dict, List, Set, Any, Optional
from enum import Enum, auto
import time

logger = logging.getLogger(__name__)

class CorruptionType(Enum):
    WALLET_STATE_CORRUPTED = auto()
    ADDRESS_INDEX_CORRUPTED = auto() 
    TRANSACTION_DATA_CORRUPTED = auto()
    DATABASE_STRUCTURE_CORRUPTED = auto()
    SEVERE_CORRUPTION = auto()

class CorruptionDetector:
    """RAYONIX corruption detection and diagnosis"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.corruption_flags: Set[CorruptionType] = set()
    
    def diagnose_corruption(self) -> Dict[str, Any]:
        """Comprehensive corruption diagnosis"""
        logger.info("Starting RAYONIX corruption diagnosis...")
        
        diagnosis = {
            'corrupted_components': [],
            'recovery_strategy': None,
            'severity': 'none',
            'corrupted_entries_count': 0,
            'database_health': 'unknown'
        }
        
        # Test database connectivity first
        db_health = self._test_database_connectivity()
        if not db_health['healthy']:
            diagnosis['database_health'] = 'unhealthy'
            diagnosis['corrupted_components'].append('database_connection')
            self.corruption_flags.add(CorruptionType.DATABASE_STRUCTURE_CORRUPTED)
        
        # Test wallet state
        wallet_state_ok = self._test_wallet_state()
        if not wallet_state_ok:
            self.corruption_flags.add(CorruptionType.WALLET_STATE_CORRUPTED)
            diagnosis['corrupted_components'].append('wallet_state')
        
        # Test address data
        address_data_ok = self._test_address_data()
        if not address_data_ok:
            self.corruption_flags.add(CorruptionType.ADDRESS_INDEX_CORRUPTED)
            diagnosis['corrupted_components'].append('address_data')
        
        # Count actual corrupted entries
        corrupted_count = self._scan_for_corrupted_entries()
        diagnosis['corrupted_entries_count'] = corrupted_count
        
        # Determine recovery strategy
        diagnosis.update(self._determine_recovery_strategy())
        
        logger.info(f"Corruption diagnosis complete: {diagnosis}")
        return diagnosis
    
    def _test_database_connectivity(self) -> Dict[str, Any]:
        """Test basic database operations"""
        try:
            # Test basic read/write
            test_key = b'_health_check_'
            test_value = b'ok'
            
            # Write test
            write_ok = self.wallet.db.put(test_key, test_value)
            if not write_ok:
                return {'healthy': False, 'error': 'write_failed'}
            
            # Read test
            read_value = self.wallet.db.get(test_key)
            if read_value != test_value:
                return {'healthy': False, 'error': 'read_verify_failed'}
            
            # Cleanup
            self.wallet.db.delete(test_key)
            
            return {'healthy': True}
            
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    def _test_wallet_state(self) -> bool:
        """Test wallet state integrity"""
        try:
            state = self.wallet.db.get_wallet_state()
            if not state:
                logger.warning("No wallet state found")
                return False
            
            # Validate state structure
            required_fields = ['sync_height', 'last_updated', 'security_score']
            if not all(hasattr(state, field) for field in required_fields):
                logger.warning("Wallet state missing required fields")
                return False
            
            # Validate data types
            if not isinstance(state.sync_height, int) or state.sync_height < 0:
                logger.warning("Invalid sync height in wallet state")
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Wallet state test failed: {e}")
            return False
    
    def _test_address_data(self) -> bool:
        """Test address data integrity"""
        try:
            addresses = self.wallet.db.get_all_addresses()
            if addresses is None:
                logger.warning("No addresses found")
                return False
            
            # Test a sample of addresses
            test_count = min(5, len(addresses))
            for i in range(test_count):
                addr = addresses[i]
                if not hasattr(addr, 'address') or not addr.address:
                    logger.warning(f"Invalid address at index {i}")
                    return False
                if not hasattr(addr, 'index') or not isinstance(addr.index, int):
                    logger.warning(f"Invalid address index at index {i}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Address data test failed: {e}")
            return False
    
    def _scan_for_corrupted_entries(self) -> int:
        """Scan for obviously corrupted database entries"""
        corrupted_count = 0
        max_scan = 1000  # Don't scan entire DB
        
        try:
            scan_count = 0
            for key, value in self.wallet.db.iterate():
                if scan_count >= max_scan:
                    break
                    
                try:
                    # Try to process the value
                    if hasattr(self.wallet.db, '_extract_value_from_storage'):
                        extracted = self.wallet.db._extract_value_from_storage(value)
                        # If we get here, the entry is probably OK
                except Exception as e:
                    corrupted_count += 1
                    logger.debug(f"Corrupted entry: {key} - {e}")
                
                scan_count += 1
                
        except Exception as e:
            logger.warning(f"Error during corruption scan: {e}")
        
        return corrupted_count
    
    def _determine_recovery_strategy(self) -> Dict[str, Any]:
        """Determine recovery strategy based on corruption severity"""
        if not self.corruption_flags:
            return {
                'severity': 'none', 
                'recovery_strategy': 'continue_normal',
                'action_required': False
            }
        
        # Only wallet state corrupted
        if (CorruptionType.WALLET_STATE_CORRUPTED in self.corruption_flags and 
            len(self.corruption_flags) == 1):
            return {
                'severity': 'low', 
                'recovery_strategy': 'rebuild_wallet_state',
                'action_required': True
            }
        
        # Address or transaction data corrupted
        if (CorruptionType.ADDRESS_INDEX_CORRUPTED in self.corruption_flags or
            CorruptionType.TRANSACTION_DATA_CORRUPTED in self.corruption_flags):
            return {
                'severity': 'medium', 
                'recovery_strategy': 'rebuild_data_indexes',
                'action_required': True
            }
        
        # Database structure corrupted
        if CorruptionType.DATABASE_STRUCTURE_CORRUPTED in self.corruption_flags:
            return {
                'severity': 'high', 
                'recovery_strategy': 'rebuild_database',
                'action_required': True
            }
        
        # Multiple corruption types
        if len(self.corruption_flags) >= 2:
            return {
                'severity': 'high', 
                'recovery_strategy': 'full_recovery',
                'action_required': True
            }
        
        return {
            'severity': 'unknown', 
            'recovery_strategy': 'full_recovery',
            'action_required': True
        }