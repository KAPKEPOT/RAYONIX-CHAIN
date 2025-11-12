#rayonix_wallet/recovery/recovery_manager.py
import logging
from typing import Dict, Any, Optional
from rayonix_wallet.recovery.corruption_detector import CorruptionDetector, CorruptionType
from rayonix_wallet.recovery.blockchain_rescanner import BlockchainRescanner
from rayonix_wallet.recovery.wallet_state_rebuilder import WalletStateRebuilder

logger = logging.getLogger(__name__)

class RecoveryManager:
    """RAYONIX wallet recovery management system"""
    
    def __init__(self, wallet, rayonix_chain=None):
        self.wallet = wallet
        self.rayonix_chain = rayonix_chain
        self.detector = CorruptionDetector(wallet)
        self.rescanner = BlockchainRescanner(wallet, rayonix_chain) if rayonix_chain else None
        self.rebuilder = WalletStateRebuilder(wallet)
    
    def auto_recover(self) -> Dict[str, Any]:
        """Automatic recovery based on corruption diagnosis"""
        logger.warning("Initiating RAYONIX automatic recovery...")
        
        try:
            # Step 1: Diagnose corruption
            diagnosis = self.detector.diagnose_corruption()
            
            # Step 2: Execute recovery strategy
            recovery_result = self._execute_recovery_strategy(diagnosis)
            
            # Step 3: Verify recovery
            verification = self._verify_recovery()
            
            return {
                'success': verification['overall_health'],
                'diagnosis': diagnosis,
                'recovery_result': recovery_result,
                'verification': verification,
                'recommendation': self._get_recommendation(verification)
            }
            
        except Exception as e:
            logger.error(f"Automatic recovery failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'recommendation': 'manual_recovery_required'
            }
    
    def _execute_recovery_strategy(self, diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate recovery strategy"""
        strategy = diagnosis.get('recovery_strategy', 'full_recovery')
        severity = diagnosis.get('severity', 'unknown')
        
        logger.info(f"Executing recovery: {strategy} (severity: {severity})")
        
        if strategy == 'continue_normal':
            return {'action': 'none', 'reason': 'no_corruption_detected'}
        
        elif strategy == 'rebuild_wallet_state':
            return self.rebuilder.rebuild_wallet_state()
        
        elif strategy == 'rebuild_data_indexes':
            return self._rebuild_data_indexes()
        
        elif strategy == 'rebuild_database':
            return self._rebuild_database_structure()
        
        elif strategy == 'full_recovery':
            return self._perform_full_recovery()
        
        else:
            logger.warning(f"Unknown recovery strategy: {strategy}")
            return self._perform_full_recovery()  # Default to full recovery
    
    def _rebuild_data_indexes(self) -> Dict[str, Any]:
        """Rebuild wallet data indexes"""
        logger.info("Rebuilding wallet data indexes...")
        
        try:
            # Rebuild address index
            address_result = self.rebuilder._recover_addresses_from_database()
            
            # Rebuild transaction index  
            transaction_result = self.rebuilder._recover_transactions_from_database()
            
            # Create new state
            new_state = self.rebuilder._create_new_wallet_state(
                address_result, transaction_result
            )
            
            # Apply to wallet
            self.rebuilder._apply_recovered_data(new_state, address_result, transaction_result)
            
            return {
                'success': True,
                'operation': 'rebuild_data_indexes',
                'addresses_recovered': len(address_result),
                'transactions_recovered': len(transaction_result)
            }
            
        except Exception as e:
            logger.error(f"Data index rebuild failed: {e}")
            return {
                'success': False,
                'operation': 'rebuild_data_indexes',
                'error': str(e)
            }
    
    def _rebuild_database_structure(self) -> Dict[str, Any]:
        """Rebuild database structure"""
        logger.warning("Rebuilding database structure...")
        
        try:
            # This is more aggressive - we recreate the entire wallet state
            result = self.rebuilder.rebuild_wallet_state()
            
            # If we have blockchain access, try to enrich with real data
            if result['success'] and self.rayonix_chain:
                blockchain_result = self.rebuilder.rebuild_from_blockchain(self.rayonix_chain)
                result['blockchain_enrichment'] = blockchain_result
            
            return result
            
        except Exception as e:
            logger.error(f"Database structure rebuild failed: {e}")
            return {
                'success': False,
                'operation': 'rebuild_database_structure',
                'error': str(e)
            }
    
    def _perform_full_recovery(self) -> Dict[str, Any]:
        """Perform full wallet recovery"""
        logger.warning("Performing FULL wallet recovery...")
        
        try:
            # Step 1: Rebuild from database
            db_result = self.rebuilder.rebuild_wallet_state()
            
            # Step 2: If blockchain available, rescan
            blockchain_result = None
            if self.rescanner and db_result['success']:
                blockchain_result = self.rescanner.full_rescan()
            
            return {
                'success': db_result['success'] and (blockchain_result is None or blockchain_result['success']),
                'database_recovery': db_result,
                'blockchain_rescan': blockchain_result,
                'operation': 'full_recovery'
            }
            
        except Exception as e:
            logger.error(f"Full recovery failed: {e}")
            return {
                'success': False,
                'operation': 'full_recovery',
                'error': str(e)
            }
    
    def _verify_recovery(self) -> Dict[str, Any]:
        """Verify recovery was successful"""
        logger.info("Verifying recovery...")
        
        # Re-run diagnosis
        post_recovery_diagnosis = self.detector.diagnose_corruption()
        
        # Check if corruption is resolved
        corruption_resolved = (
            len(post_recovery_diagnosis['corrupted_components']) == 0 and
            post_recovery_diagnosis['corrupted_entries_count'] == 0
        )
        
        # Check wallet functionality
        wallet_functional = self._test_wallet_functionality()
        
        return {
            'overall_health': corruption_resolved and wallet_functional,
            'corruption_resolved': corruption_resolved,
            'wallet_functional': wallet_functional,
            'remaining_issues': post_recovery_diagnosis['corrupted_components'],
            'recommended_action': 'rescan' if not corruption_resolved else 'normal_operation'
        }
    
    def _test_wallet_functionality(self) -> bool:
        """Test basic wallet functionality"""
        try:
            # Test address access
            addresses = self.wallet.get_addresses()
            if addresses is None:
                return False
            
            # Test state access
            state = self.wallet.state
            if not state:
                return False
            
            # Test basic operations
            can_get_balance = hasattr(self.wallet, 'get_balance')
            can_validate_address = hasattr(self.wallet, 'validate_address')
            
            return can_get_balance and can_validate_address
            
        except Exception as e:
            logger.warning(f"Wallet functionality test failed: {e}")
            return False
    
    def _get_recommendation(self, verification: Dict[str, Any]) -> str:
        """Get recommendation based on recovery results"""
        if verification['overall_health']:
            return 'recovery_successful'
        elif verification['corruption_resolved'] and not verification['wallet_functional']:
            return 'wallet_needs_rescan'
        elif not verification['corruption_resolved']:
            return 'manual_recovery_required'
        else:
            return 'expert_assistance_recommended'
    
    def get_recovery_options(self) -> Dict[str, Any]:
        """Get available recovery options"""
        diagnosis = self.detector.diagnose_corruption()
        
        options = {
            'quick_recovery': {
                'description': 'Rebuild wallet state from database',
                'estimated_time': 'seconds',
                'risk': 'low',
                'data_preservation': 'high'
            },
            'blockchain_rescan': {
                'description': 'Rescan blockchain to rebuild transaction history',
                'estimated_time': 'minutes to hours',
                'risk': 'low', 
                'data_preservation': 'medium',
                'requirements': 'blockchain_connection'
            },
            'full_recovery': {
                'description': 'Complete wallet rebuild from blockchain',
                'estimated_time': 'hours',
                'risk': 'medium',
                'data_preservation': 'low',
                'requirements': 'blockchain_connection'
            }
        }
        
        return {
            'current_state': diagnosis,
            'available_options': options,
            'recommended_option': diagnosis.get('recovery_strategy', 'full_recovery')
        }
    
    def execute_custom_recovery(self, option: str, **kwargs) -> Dict[str, Any]:
        """Execute custom recovery option"""
        logger.info(f"Executing custom recovery: {option}")
        
        if option == 'quick_recovery':
            return self.rebuilder.rebuild_wallet_state()
        
        elif option == 'blockchain_rescan' and self.rescanner:
            start_height = kwargs.get('start_height', 0)
            return self.rescanner.full_rescan(start_height)
        
        elif option == 'full_recovery':
            return self._perform_full_recovery()
        
        elif option == 'nuke_and_rescan' and self.rescanner:
            # Most aggressive option
            nuke_result = self.rebuilder._reset_wallet_state()
            rescan_result = self.rescanner.full_rescan()
            return {
                'success': rescan_result['success'],
                'nuke_operation': nuke_result,
                'rescan_operation': rescan_result
            }
        
        else:
            return {
                'success': False,
                'error': f'Unsupported recovery option: {option}'
            }