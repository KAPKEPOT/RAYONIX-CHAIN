# smart_contract/core/contract.py
import time
import logging
import psutil
import wasmtime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ..types.enums import ContractType, ContractState, ContractSecurityLevel
from ..core.storage.contract_storage import ContractStorage
from ..core.execution_result import ExecutionResult
from ..core.gas_system.gas_meter import GasMeter
from ..wasm.wasm_executor import WASMExecutor
from ..utils.validation_utils import validate_contract_id, validate_address
from ..exceptions.contract_errors import ContractUpgradeError

logger = logging.getLogger("SmartContract.Core")

@dataclass
class ContractConfig:
    """Configuration for smart contract deployment and execution"""
    max_memory_usage: int = 100 * 1024 * 1024  # 100MB
    max_execution_time: int = 30  # seconds
    wasm_engine_config: Dict[str, Any] = field(default_factory=lambda: {
        'memory_pages': 65536,
        'max_instances': 1000,
        'cache_size': 100 * 1024 * 1024  # 100MB
    })
    storage_config: Dict[str, Any] = field(default_factory=lambda: {
        'compression_enabled': True,
        'encryption_enabled': True,
        'max_storage_size': 10 * 1024 * 1024 * 1024  # 10GB
    })

class SmartContract:
    """Production-ready smart contract implementation with WebAssembly execution"""
    
    def __init__(self, contract_id: str, owner: str, contract_type: ContractType, 
                 wasm_bytecode: bytes, initial_balance: int = 0, config: Optional[ContractConfig] = None):
        validate_contract_id(contract_id)
        validate_address(owner)
        
        self.contract_id = contract_id
        self.owner = owner
        self.contract_type = contract_type
        self.wasm_bytecode = wasm_bytecode
        self.balance = initial_balance
        self.config = config or ContractConfig()
        
        self.storage = ContractStorage(
            compression_enabled=self.config.storage_config['compression_enabled'],
            encryption_enabled=self.config.storage_config['encryption_enabled'],
            max_size=self.config.storage_config['max_storage_size']
        )
        self.storage.allowed_writers.add(owner)
        
        self.state = ContractState.ACTIVE
        self.created_at = time.time()
        self.last_modified = time.time()
        self.version = "1.0.0"
        self.security_level = ContractSecurityLevel.MEDIUM
        self.gas_optimizer = None  # Will be initialized when needed
        
        # Initialize WASM executor
        self.wasm_executor = WASMExecutor(
            wasm_bytecode, 
            engine_config=self.config.wasm_engine_config
        )
        
        # Performance metrics
        self.execution_count = 0
        self.total_gas_used = 0
        self.total_execution_time = 0
        self.average_gas_per_call = 0
        
        logger.info(f"Contract {contract_id} initialized with type {contract_type.name}")

    def execute_function(self, function_name: str, caller: str, args: Dict[str, Any], 
                        gas_meter: GasMeter, contract_manager: Any) -> ExecutionResult:
        """
        Execute a contract function using the WebAssembly virtual machine
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Check contract state
            if self.state != ContractState.ACTIVE:
                return ExecutionResult(
                    success=False,
                    error=f"Contract is not active. Current state: {self.state.name}",
                    gas_used=gas_meter.gas_used
                )
            
            # Execute via WASM executor
            result = self.wasm_executor.execute_function(
                function_name=function_name,
                caller=caller,
                args=args,
                gas_meter=gas_meter,
                contract_manager=contract_manager,
                contract_id=self.contract_id,
                storage=self.storage
            )
            
            # Update performance metrics
            self._update_metrics(result, start_time, start_memory)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing function {function_name}: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                gas_used=gas_meter.gas_used,
                execution_time=execution_time,
                memory_used=self._get_memory_usage() - start_memory
            )
    
    def _update_metrics(self, result: ExecutionResult, start_time: float, start_memory: int) -> None:
        """Update contract performance metrics"""
        self.execution_count += 1
        self.total_gas_used += result.gas_used
        self.total_execution_time += result.execution_time
        
        if self.execution_count > 0:
            self.average_gas_per_call = self.total_gas_used / self.execution_count
        
        # Update last modified timestamp
        self.last_modified = time.time()
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage of the process"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def upgrade_contract(self, new_wasm_bytecode: bytes, upgrade_reason: str, 
                        caller: str) -> bool:
        """Upgrade contract with new WASM bytecode"""
        if caller != self.owner:
            return False
        
        if self.state != ContractState.ACTIVE:
            return False
        
        try:
            # Create backup
            backup_bytecode = self.wasm_bytecode
            backup_executor = self.wasm_executor
            
            # Update to new bytecode
            self.wasm_bytecode = new_wasm_bytecode
            self.wasm_executor = WASMExecutor(
                new_wasm_bytecode,
                engine_config=self.config.wasm_engine_config
            )
            
            # Update version and timestamp
            self.version = self._increment_version(self.version)
            self.last_modified = time.time()
            
            logger.info(f"Contract {self.contract_id} upgraded by {caller}. Reason: {upgrade_reason}")
            return True
            
        except Exception as e:
            # Restore backup on failure
            self.wasm_bytecode = backup_bytecode
            self.wasm_executor = backup_executor
            logger.error(f"Contract upgrade failed: {e}")
            raise ContractUpgradeError(f"Contract upgrade failed: {e}")
    
    def _increment_version(self, current_version: str) -> str:
        """Increment semantic version"""
        parts = current_version.split('.')
        if len(parts) == 3:
            try:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
            except ValueError:
                pass
        return current_version
    
    def transfer_ownership(self, new_owner: str, caller: str) -> bool:
        """Transfer contract ownership"""
        if caller != self.owner:
            return False
        
        validate_address(new_owner)
        
        old_owner = self.owner
        self.owner = new_owner
        self.storage.allowed_writers.discard(old_owner)
        self.storage.allowed_writers.add(new_owner)
        self.last_modified = time.time()
        
        logger.info(f"Contract {self.contract_id} ownership transferred from {old_owner} to {new_owner}")
        return True
    
    def destroy_contract(self, caller: str, reason: str) -> bool:
        """Destroy the contract and clean up resources"""
        if caller != self.owner:
            return False
        
        self.state = ContractState.DESTROYED
        self.storage.clear_storage()
        self.wasm_executor.cleanup()
        self.wasm_bytecode = b''
        self.last_modified = time.time()
        
        logger.info(f"Contract {self.contract_id} destroyed by {caller}. Reason: {reason}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for serialization"""
        return {
            'contract_id': self.contract_id,
            'owner': self.owner,
            'contract_type': self.contract_type.name,
            'balance': self.balance,
            'state': self.state.name,
            'created_at': self.created_at,
            'last_modified': self.last_modified,
            'version': self.version,
            'security_level': self.security_level.name,
            'storage': self.storage.to_dict(),
            'wasm_bytecode_size': len(self.wasm_bytecode),
            'execution_count': self.execution_count,
            'total_gas_used': self.total_gas_used,
            'average_gas_per_call': self.average_gas_per_call
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get contract statistics"""
        return {
            'contract_id': self.contract_id,
            'owner': self.owner,
            'balance': self.balance,
            'state': self.state.name,
            'storage_size': self.storage.get_total_size(),
            'storage_entries': self.storage.get_entry_count(),
            'audit_log_entries': self.storage.get_audit_log_size(),
            'version': self.version,
            'age_days': (time.time() - self.created_at) / 86400,
            'last_modified_days': (time.time() - self.last_modified) / 86400,
            'execution_count': self.execution_count,
            'total_gas_used': self.total_gas_used,
            'average_gas_per_call': self.average_gas_per_call
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'wasm_executor'):
                self.wasm_executor.cleanup()
        except Exception as e:
            logger.warning(f"Error during contract cleanup: {e}")