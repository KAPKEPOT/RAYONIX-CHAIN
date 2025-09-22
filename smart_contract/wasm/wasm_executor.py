# smart_contract/wasm/wasm_executor.py
import time
import logging
import wasmtime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from ..core.gas_system.gas_meter import GasMeter
from ..core.storage.contract_storage import ContractStorage
from ..core.execution_result import ExecutionResult
from .wasm_host_functions import WASMHostFunctions
from .bytecode_validator import WASMBytecodeValidator

logger = logging.getLogger("SmartContract.WASMExecutor")

@dataclass
class WASMConfig:
    """Configuration for WASM execution"""
    engine_config: Dict[str, Any] = None
    memory_pages: int = 65536  # 4GB maximum
    max_instances: int = 1000
    cache_size: int = 100 * 1024 * 1024  # 100MB
    timeout: int = 30  # seconds

class WASMExecutor:
    """Advanced WebAssembly executor for smart contract execution"""
    
    def __init__(self, wasm_bytecode: bytes, config: Optional[WASMConfig] = None):
        self.wasm_bytecode = wasm_bytecode
        self.config = config or WASMConfig()
        
        # Initialize WASM engine
        self.engine = wasmtime.Engine()
        
        # Initialize store
        self.store = wasmtime.Store(self.engine)
        
        # Initialize module
        self.module = wasmtime.Module(self.engine, wasm_bytecode)
        
        # Initialize linker
        self.linker = wasmtime.Linker(self.engine)
        
        # Instance cache
        self.instance_cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Performance metrics
        self.execution_count = 0
        self.total_execution_time = 0
        self.average_execution_time = 0
        
        logger.info("WASMExecutor initialized")
    
    def execute_function(self, function_name: str, caller: str, args: Dict[str, Any],
                        gas_meter: GasMeter, contract_manager: Any, 
                        contract_id: str, storage: ContractStorage) -> ExecutionResult:
        """Execute a WASM function with comprehensive error handling"""
        start_time = time.time()
        result = ExecutionResult(success=False)
        
        try:
            # Validate function name
            if not self._validate_function_name(function_name):
                result.error = f"Invalid function name: {function_name}"
                return result
            
            # Create host functions
            host_functions = WASMHostFunctions(gas_meter, storage, contract_manager)
            
            # Setup WASM instance
            instance = self._create_instance(host_functions)
            if instance is None:
                result.error = "Failed to create WASM instance"
                return result
            
            # Prepare arguments
            wasm_args = self._prepare_arguments(args)
            
            # Execute the function
            wasm_result = self._execute_wasm_function(instance, function_name, wasm_args, gas_meter)
            
            # Process result
            if wasm_result is not None:
                result.success = True
                result.return_value = wasm_result
            else:
                result.error = "WASM execution returned null"
            
        except Exception as e:
            result.error = f"WASM execution failed: {e}"
            logger.error(f"WASM execution error: {e}")
        
        finally:
            # Update metrics
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            result.gas_used = gas_meter.gas_used
            
            self._update_metrics(execution_time)
            
            # Cleanup
            if 'host_functions' in locals():
                host_functions.cleanup()
        
        return result
    
    def _create_instance(self, host_functions: WASMHostFunctions) -> Optional[Any]:
        """Create a WASM instance with host functions"""
        try:
            # Add host functions to linker
            for name, func in host_functions.functions.items():
                self.linker.define_func("env", name, func)
            
            # Instantiate module
            instance = self.linker.instantiate(self.store, self.module)
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create WASM instance: {e}")
            return None
    
    def _execute_wasm_function(self, instance: Any, function_name: str, 
                             args: List[Any], gas_meter: GasMeter) -> Any:
        """Execute a WASM function with gas metering"""
        try:
            # Get the function
            func = instance.get_func(self.store, function_name)
            if func is None:
                raise ValueError(f"Function {function_name} not found")
            
            # Execute with timeout
            start_time = time.time()
            result = func(self.store, *args)
            
            # Check gas during execution
            self._check_gas_during_execution(gas_meter, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"WASM function execution failed: {e}")
            raise
    
    def _prepare_arguments(self, args: Dict[str, Any]) -> List[Any]:
        """Prepare arguments for WASM function call"""
        wasm_args = []
        
        for key, value in args.items():
            if isinstance(value, int):
                wasm_args.append(wasmtime.Val.i32(value))
            elif isinstance(value, float):
                wasm_args.append(wasmtime.Val.f64(value))
            elif isinstance(value, str):
                # Convert string to bytes and then to WASM memory reference
                wasm_args.append(self._string_to_wasm(value))
            elif isinstance(value, bytes):
                wasm_args.append(self._bytes_to_wasm(value))
            else:
                # Convert other types to string
                wasm_args.append(self._string_to_wasm(str(value)))
        
        return wasm_args
    
    def _string_to_wasm(self, text: str) -> Any:
        """Convert string to WASM memory format"""
        # This would allocate memory and copy the string
        # For now, return placeholder
        return wasmtime.Val.i32(len(text))  # Placeholder
    
    def _bytes_to_wasm(self, data: bytes) -> Any:
        """Convert bytes to WASM memory format"""
        # This would allocate memory and copy the bytes
        # For now, return placeholder
        return wasmtime.Val.i32(len(data))  # Placeholder
    
    def _check_gas_during_execution(self, gas_meter: GasMeter, start_time: float) -> None:
        """Check gas usage during execution and enforce limits"""
        current_time = time.time()
        execution_time = current_time - start_time
        
        # Check execution time limit
        if execution_time > self.config.timeout:
            raise TimeoutError(f"Execution timeout after {execution_time}s")
        
        # Check gas limit
        if gas_meter.get_remaining_gas() <= 0:
            from ..core.gas_system.out_of_gas_error import OutOfGasError
            raise OutOfGasError("Out of gas during execution")
    
    def _validate_function_name(self, function_name: str) -> bool:
        """Validate WASM function name"""
        if not function_name or len(function_name) > 256:
            return False
        
        # Basic validation - should be alphanumeric with underscores
        import re
        return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name))
    
    def _update_metrics(self, execution_time: float) -> None:
        """Update performance metrics"""
        self.execution_count += 1
        self.total_execution_time += execution_time
        self.average_execution_time = self.total_execution_time / self.execution_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get executor metrics"""
        return {
            'execution_count': self.execution_count,
            'total_execution_time': self.total_execution_time,
            'average_execution_time': self.average_execution_time,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) 
                            if (self.cache_hits + self.cache_misses) > 0 else 0,
            'module_size': len(self.wasm_bytecode),
            'instance_cache_size': len(self.instance_cache)
        }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.instance_cache.clear()
        self.linker = None
        self.module = None
        self.store = None
        self.engine = None
        logger.info("WASMExecutor cleanup completed")
    
    def __del__(self):
        """Cleanup resources on destruction"""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during WASMExecutor cleanup: {e}")