# smart_contract/core/contract_manager.py
import os
import time
import threading
import logging
import secrets
import pickle
import asyncio
import statistics
from typing import Dict, Any, Optional, List, Set
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import plyvel
import aiohttp

from smart_contract.core.contract import SmartContract
from ..core.execution_result import ExecutionResult
from smart_contract.core.gas_system.gas_meter import GasMeter
from smart_contract.security.contract_security import ContractSecurity
from smart_contract.database.leveldb_manager import LevelDBManager
from smart_contract.wasm.bytecode_validator import WASMBytecodeValidator
from smart_contract.utils.validation_utils import validate_contract_id, validate_address
from smart_contract.exceptions.contract_errors import (
    ContractDeploymentError, ContractExecutionError, ContractNotFoundError
)
from smart_contract.exceptions.security_errors import SecurityViolationError
from smart_contract.types.enums import ContractType, ContractState, ContractSecurityLevel
from smart_contract.core.storage.contract_storage import ContractStorage  # Missing import

logger = logging.getLogger("SmartContract.ContractManager")

class ContractManager:
    """Advanced contract manager with atomic state transitions and inter-contract calls"""
    
    def __init__(self, db_path: str = "contracts_db", config: Optional[Dict] = None, max_workers: int = 50):
        # Ensure db_path is a string
        if not isinstance(db_path, str):
            logger.warning(f"db_path is not a string, converting: {type(db_path)} -> str")
            db_path = str(db_path)
        
        # Ensure path exists
        os.makedirs(db_path, exist_ok=True)
        
        self.contracts: Dict[str, SmartContract] = {}
        self.db = LevelDBManager(db_path)
        self.security = ContractSecurity()
        self.call_stack: List[Dict] = []
        self.execution_cache: Dict[str, Any] = {}
        self.state_journal: Dict[str, Dict] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.RLock()
        self.gas_price = 5  # Base gas price in RXY
        self.update_errors = 0
        self.gas_price_history = deque(maxlen=1000)
        self.running = True
        
        # Gas price configuration
        self.GAS_PRICE_CONFIG = {
            'min_gas_price': 1,
            'max_gas_price': 100,
            'base_gas_price': 5,
            'adjustment_sensitivity': 0.2,
            'update_interval': 30,
            'emergency_update_interval': 5,
            'max_mempool_size': 10000,
            'target_block_utilization': 0.7,
        }
        
        # Load contracts from database
        self._load_contracts_from_db()
        
        # Start background tasks
        self._start_background_tasks()
       
    def set_blockchain_reference(self, blockchain):
    	"""Set reference to the blockchain for contract integration"""
    	self.blockchain = blockchain
    	logger.info("Blockchain reference set for contract manager")
    	
    def set_consensus_engine(self, consensus):
    	"""Set reference to consensus engine"""
    	self.consensus = consensus
    	logger.info("Consensus engine reference set for contract manager")
    	
    def set_state_manager(self, state_manager):
    	"""Set reference to state manager"""
    	self.state_manager = state_manager
    	logger.info("State manager reference set for contract manager")
    	
    def initialize(self):
    	"""Initialize contract manager after all references are set"""
    	logger.info("Contract manager initialization completed")
    	if hasattr(self, 'blockchain'):
    		logger.info("Contract manager fully integrated with blockchain")
    		
    def create_snapshot(self) -> Dict[str, Any]:
    	try:
    		with self.lock:
    			snapshot = {
    			    'contracts': {},
    			    'timestamp': time.time(),
    			    'snapshot_id': f"snapshot_{time.time()}_{secrets.token_hex(8)}"
    			}
    			# Create deep copies of contract states
    			for contract_id, contract in self.contracts.items():
    				snapshot['contracts'][contract_id] = {
    				    'storage': pickle.dumps(contract.storage),
    				    'balance': contract.balance,
    				    'state': contract.state.value if hasattr(contract.state, 'value') else contract.state,
    				    'version': contract.version
    				}
    			logger.debug(f"Created contract snapshot with {len(snapshot['contracts'])} contracts")
    			return snapshot
    	except Exception as e:
    		logger.error(f"Failed to create contract snapshot: {e}")
    		return {
    		    'contracts': {},
    		    'timestamp': time.time(),
    		    'snapshot_id': f"error_snapshot_{time.time()}"
    		}
    		
    def restore_snapshot(self, snapshot: Dict[str, Any]) -> bool:
    	try:
    		with self.lock:
    			if not snapshot or 'contracts' not in snapshot:
    				logger.error("Invalid snapshot provided for restoration")
    				return False
    			restored_count = 0
    			for contract_id, contract_data in snapshot['contracts'].items():
    				try:
    					if contract_id in self.contracts:
    						contract = self.contracts[contract_id]
    						# Restore contract state
    						if 'storage' in contract_data:
    							contract.storage = pickle.loads(contract_data['storage'])
    						if 'balance' in contract_data:
    							contract.balance = contract_data['balance']
    						if 'state' in contract_data:
    							# Handle both enum and string states
    							state_value = contract_data['state']
    							if isinstance(state_value, str):
    								contract.state = ContractState[state_value]
    							else:
    								contract.state = ContractState(state_value)
    						if 'version' in contract_data:
    							contract.version = contract_data['version']
    						restored_count += 1
    						# Persist the restored state to database
    						self.db.save_contract(contract)
    				except Exception as e:
    					logger.error(f"Failed to restore contract {contract_id}: {e}")
    					continue
    			logger.info(f"Restored {restored_count} contracts from snapshot")
    			return restored_count > 0
    	except Exception as e:
    		logger.error(f"Failed to restore contract snapshot: {e}")
    		return False    		
    
    def _load_contracts_from_db(self) -> None:
        """Load contracts from database"""
        try:
            contracts_data = self.db.load_all_contracts()
            for contract_id, contract_data in contracts_data.items():
                try:
                    from smart_contract.core.contract_storage import ContractStorage
                    contract = SmartContract(
                        contract_id=contract_data['contract_id'],
                        owner=contract_data['owner'],
                        contract_type=ContractType[contract_data['contract_type']],
                        wasm_bytecode=contract_data['wasm_bytecode'],
                        initial_balance=contract_data['balance']
                    )
                    contract.storage = contract_data.get('storage', ContractStorage())
                    contract.state = ContractState[contract_data.get('state', 'ACTIVE')]
                    contract.version = contract_data.get('version', '1.0.0')
                    self.contracts[contract_id] = contract
                except Exception as e:
                    logger.error(f"Failed to load contract {contract_id}: {e}")
        except Exception as e:
            logger.error(f"Error loading contracts from database: {e}")
            
    
  
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Start gas price updater
        self.gas_price_task = asyncio.create_task(self.update_gas_price_async())
        
        # Start journal cleanup task
        self.cleanup_task = asyncio.create_task(self.periodic_cleanup())
        
        logger.info("Background tasks started")
    
    async def update_gas_price_async(self):
        """Async gas price update routine"""
        while self.running:
            try:
                # Fetch network conditions
                network_stats = await self.fetch_network_conditions()
                
                # Get local statistics
                local_stats = self.get_local_stats()
                
                # Calculate new gas price
                new_price = self.calculate_dynamic_gas_price(network_stats, local_stats)
                
                # Update gas price atomically
                with self.lock:
                    old_price = self.gas_price
                    self.gas_price = new_price
                    self._last_gas_price_update = time.time()
                
                # Log price change
                if old_price != new_price:
                    logger.info(
                        f"Gas price updated: {old_price} → {new_price} RXY | "
                        f"Mempool: {network_stats['mempool_size']} | "
                        f"Utilization: {network_stats['block_utilization']:.1%} | "
                        f"Validators: {network_stats['validator_count']}"
                    )
                
                # Update metrics
                self._update_gas_metrics(new_price, network_stats, local_stats)
                
                # Determine next update interval
                update_interval = self.determine_update_interval(network_stats)
                
                # Reset error counter
                self.update_errors = 0
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.update_errors += 1
                logger.error(f"Gas price update failed (attempt {self.update_errors}): {e}")
                
                # Exponential backoff
                backoff_time = min(300, 2 ** min(self.update_errors, 8))
                await asyncio.sleep(backoff_time)
                
                # Reset to base price after many errors
                if self.update_errors > 10:
                    with self.lock:
                        self.gas_price = self.GAS_PRICE_CONFIG['base_gas_price']
                    logger.warning("Reset gas price to base due to persistent errors")
    
    async def fetch_network_conditions(self) -> Dict[str, Any]:
        """Fetch current network conditions from peers"""
        # Implementation would query network peers
        return {
            'mempool_size': 0,
            'pending_transactions': 0,
            'average_fee_rate': self.GAS_PRICE_CONFIG['base_gas_price'],
            'block_utilization': 0.5,
            'network_latency': 100,
            'validator_count': 10
        }
    
    def get_local_stats(self) -> Dict[str, Any]:
        """Get local node statistics"""
        return {
            'mempool_size': 0,
            'pending_transactions': 0,
            'local_fee_estimate': self.GAS_PRICE_CONFIG['base_gas_price'],
            'block_production_rate': 1.0,
            'node_connectivity': 10
        }
    
    def calculate_dynamic_gas_price(self, network_stats: Dict, local_stats: Dict) -> int:
        """Calculate dynamic gas price based on network conditions"""
        # Simplified implementation - would use real network data
        base_price = self.GAS_PRICE_CONFIG['base_gas_price']
        congestion = min(1.0, network_stats['mempool_size'] / self.GAS_PRICE_CONFIG['max_mempool_size'])
        congestion_factor = 1.0 + (congestion * 2.0)
        
        new_price = int(base_price * congestion_factor)
        return max(self.GAS_PRICE_CONFIG['min_gas_price'], 
                  min(self.GAS_PRICE_CONFIG['max_gas_price'], new_price))
    
    def determine_update_interval(self, network_stats: Dict) -> int:
        """Determine update interval based on network conditions"""
        congestion = network_stats['mempool_size'] / self.GAS_PRICE_CONFIG['max_mempool_size']
        if congestion > 0.8:
            return self.GAS_PRICE_CONFIG['emergency_update_interval']
        return self.GAS_PRICE_CONFIG['update_interval']
    
    def _update_gas_metrics(self, new_price: int, network_stats: Dict, local_stats: Dict) -> None:
        """Update gas price metrics"""
        metrics = {
            'current_gas_price': new_price,
            'timestamp': time.time(),
            'network_stats': network_stats,
            'local_stats': local_stats,
            'update_errors': self.update_errors
        }
        self.gas_price_history.append(metrics)
    
    async def periodic_cleanup(self):
        """Periodic cleanup of old journals and cache"""
        while self.running:
            try:
                self.cleanup_old_journals()
                self.cleanup_execution_cache()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Periodic cleanup failed: {e}")
                await asyncio.sleep(60)
    
    def deploy_contract(self, contract_id: str, owner: str, contract_type: ContractType,
                       wasm_bytecode: bytes, initial_balance: int = 0) -> bool:
        """Deploy a new contract with WASM bytecode"""
        with self.lock:
            validate_contract_id(contract_id)
            validate_address(owner)
            
            if contract_id in self.contracts:
                logger.error(f"Contract {contract_id} already exists")
                return False
            
            # Validate WASM bytecode
            if not WASMBytecodeValidator.validate(wasm_bytecode):
                logger.error(f"Invalid WASM bytecode for contract {contract_id}")
                return False
            
            # Security check
            if self.security.is_blacklisted(owner):
                logger.error(f"Owner {owner} is blacklisted")
                return False
            
            try:
                # Create new contract
                contract = SmartContract(
                    contract_id, owner, contract_type, wasm_bytecode, initial_balance
                )
                
                # Store in memory and database
                self.contracts[contract_id] = contract
                self.db.save_contract(contract)
                
                logger.info(f"Contract {contract_id} deployed by {owner}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to deploy contract {contract_id}: {e}")
                raise ContractDeploymentError(f"Failed to deploy contract: {e}")
    
    def execute_function(self, contract_id: str, function_name: str, caller: str,
                       args: Dict[str, Any], gas_limit: int = 1000000) -> ExecutionResult:
        """Execute a contract function with atomic state transitions"""
        journal_id = self._create_state_journal(contract_id)
        
        try:
            with self.lock:
                # Security checks
                if not self._pre_execution_checks(contract_id, caller, function_name, args):
                    return ExecutionResult(
                        success=False,
                        error="Security check failed",
                        gas_used=0
                    )
                
                # Get contract
                contract = self.contracts.get(contract_id)
                if not contract:
                    return ExecutionResult(
                        success=False,
                        error=f"Contract {contract_id} not found",
                        gas_used=0
                    )
                
                # Initialize gas meter
                gas_meter = GasMeter(gas_limit, self.gas_price)
                
                # Execute the function
                result = contract.execute_function(
                    function_name, caller, args, gas_meter, self
                )
                
                # Post-execution checks
                if result.success:
                    security_ok, error_msg = self.security.check_resource_limits(
                        result.execution_time,
                        result.memory_used,
                        result.gas_used,
                        contract.storage.get_total_size()
                    )
                    
                    if not security_ok:
                        result.success = False
                        result.error = f"Resource limit exceeded: {error_msg}"
                
                # Commit or revert state changes
                if result.success:
                    self._commit_state_journal(journal_id)
                    self.db.save_contract(contract)  # Persist changes
                else:
                    self._revert_state_journal(journal_id)
                
                return result
                
        except Exception as e:
            self._revert_state_journal(journal_id)
            logger.error(f"Unexpected error executing {function_name} on {contract_id}: {e}")
            raise ContractExecutionError(f"Execution failed: {e}")
    
    def _pre_execution_checks(self, contract_id: str, caller: str, 
                            function_name: str, args: Dict[str, Any]) -> bool:
        """Perform comprehensive pre-execution security checks"""
        try:
            # Blacklist check
            if self.security.is_blacklisted(caller):
                logger.warning(f"Blacklisted caller {caller} attempted to execute {function_name}")
                return False
            
            # Input validation
            for arg_name, arg_value in args.items():
                is_valid, error = self.security.validate_input(
                    arg_value, 
                    context={'caller': caller, 'operation': function_name, 'contract': contract_id}
                )
                if not is_valid:
                    logger.warning(f"Invalid input {arg_name} from {caller}: {error}")
                    return False
            
            # Rate limiting
            if not self.security.check_rate_limit(caller, f"execute_{function_name}", 1):
                logger.warning(f"Rate limit exceeded for {caller} executing {function_name}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-execution checks failed: {e}")
            return False
    
    def _create_state_journal(self, contract_id: str) -> str:
        """Create a journal for atomic state transitions"""
        journal_id = f"journal_{contract_id}_{time.time()}_{secrets.token_hex(4)}"
        
        with self.lock:
            contract = self.contracts.get(contract_id)
            if contract:
                journal_state = {
                    'storage': pickle.dumps(contract.storage),
                    'balance': contract.balance,
                    'state': contract.state,
                    'version': contract.version
                }
                self.state_journal[journal_id] = journal_state
        
        return journal_id
    
    def _commit_state_journal(self, journal_id: str) -> None:
        """Commit state changes from journal"""
        with self.lock:
            if journal_id in self.state_journal:
                del self.state_journal[journal_id]
    
    def _revert_state_journal(self, journal_id: str) -> None:
        """Revert state changes using journal"""
        with self.lock:
            if journal_id in self.state_journal:
                journal = self.state_journal[journal_id]
                contract_id = journal_id.split('_')[1]
                
                if contract_id in self.contracts:
                    contract = self.contracts[contract_id]
                    contract.storage = pickle.loads(journal['storage'])
                    contract.balance = journal['balance']
                    contract.state = journal['state']
                    contract.version = journal['version']
                
                del self.state_journal[journal_id]
    
    def call_contract(self, from_contract: str, to_contract: str, function_name: str,
                     args: Dict[str, Any], gas_limit: int) -> ExecutionResult:
        """Secure inter-contract call implementation"""
        call_id = self._push_call_stack(from_contract, to_contract, function_name, gas_limit)
        
        try:
            result = self.execute_function(to_contract, function_name, from_contract, args, gas_limit)
            self._update_call_stack(call_id, result)
            return result
            
        except Exception as e:
            self._pop_call_stack(call_id)
            logger.error(f"Inter-contract call failed: {e}")
            raise ContractExecutionError(f"Inter-contract call failed: {e}")
    
    def _push_call_stack(self, from_contract: str, to_contract: str, 
                        function_name: str, gas_limit: int) -> str:
        """Push a call to the call stack"""
        call_id = f"call_{from_contract}_{to_contract}_{time.time()}_{secrets.token_hex(4)}"
        
        call_info = {
            'id': call_id,
            'from': from_contract,
            'to': to_contract,
            'function': function_name,
            'gas_limit': gas_limit,
            'start_time': time.time(),
            'status': 'executing'
        }
        
        with self.lock:
            self.call_stack.append(call_info)
        
        return call_id
    
    def _update_call_stack(self, call_id: str, result: ExecutionResult) -> None:
        """Update call stack with execution result"""
        with self.lock:
            for call in self.call_stack:
                if call['id'] == call_id:
                    call['end_time'] = time.time()
                    call['status'] = 'completed' if result.success else 'failed'
                    call['gas_used'] = result.gas_used
                    call['error'] = result.error
                    break
    
    def _pop_call_stack(self, call_id: str) -> None:
        """Remove a call from the call stack"""
        with self.lock:
            self.call_stack = [call for call in self.call_stack if call['id'] != call_id]
    
    def get_contract(self, contract_id: str) -> Optional[SmartContract]:
        """Get contract by ID"""
        return self.contracts.get(contract_id)
        
             
    
    def get_contract_stats(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a contract"""
        contract = self.contracts.get(contract_id)
        if not contract:
            return None
        return contract.get_stats()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        with self.lock:
            total_contracts = len(self.contracts)
            total_balance = sum(contract.balance for contract in self.contracts.values())
            total_storage = sum(contract.storage.get_total_size() for contract in self.contracts.values())
            
            return {
                'total_contracts': total_contracts,
                'total_balance': total_balance,
                'total_storage_bytes': total_storage,
                'active_calls': len(self.call_stack),
                'pending_journals': len(self.state_journal),
                'threat_level': self.security.threat_intelligence.get_current_threat_level(),
                'gas_price': self.gas_price,
                'blacklisted_addresses': len(self.security.blacklisted_addresses),
                'update_errors': self.update_errors
            }
    
    def cleanup_old_journals(self) -> None:
        """Clean up old state journals"""
        with self.lock:
            current_time = time.time()
            journals_to_remove = []
            
            for journal_id in self.state_journal:
                journal_time = float(journal_id.split('_')[3])
                if current_time - journal_time > 3600:  # 1 hour
                    journals_to_remove.append(journal_id)
            
            for journal_id in journals_to_remove:
                del self.state_journal[journal_id]
    
    def cleanup_execution_cache(self) -> None:
        """Clean up old execution cache entries"""
        with self.lock:
            current_time = time.time()
            cache_keys_to_remove = []
            
            for key, (timestamp, _) in self.execution_cache.items():
                if current_time - timestamp > 300:  # 5 minutes
                    cache_keys_to_remove.append(key)
            
            for key in cache_keys_to_remove:
                del self.execution_cache[key]
    
    def stop(self):
        """Stop the contract manager and cleanup resources"""
        self.running = False
        if hasattr(self, 'gas_price_task'):
            self.gas_price_task.cancel()
        if hasattr(self, 'cleanup_task'):
            self.cleanup_task.cancel()
        
        self.thread_pool.shutdown()
        self.db.close()
        
        logger.info("Contract manager stopped")
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.stop()
        except:
            pass