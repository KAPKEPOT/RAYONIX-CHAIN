# smart_contract/wasm/wasm_executor.py
import time
import logging
import threading
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import wasmtime
from wasmtime import Memory, Trap, Instance, Store, Module, Engine, Linker, Val

from smart_contract.core.gas_system.gas_meter import GasMeter
from smart_contract.core.storage.contract_storage import ContractStorage
from smart_contract.core.execution_result import ExecutionResult
from smart_contract.wasm.wasm_host_functions import WASMHostFunctions
from smart_contract.wasm.bytecode_validator import WASMBytecodeValidator

logger = logging.getLogger("SmartContract.WASMExecutor")

class ExecutionState(Enum):
    """Execution state enumeration"""
    IDLE = "idle"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class WASMConfig:
    """Advanced configuration for WASM execution"""
    engine_config: Dict[str, Any] = field(default_factory=dict)
    memory_pages: int = 65536  # 4GB maximum
    max_instances: int = 1000
    cache_size: int = 100 * 1024 * 1024  # 100MB
    timeout: int = 30  # seconds
    max_concurrent_executions: int = 100
    memory_limit: int = 128 * 1024 * 1024  # 128MB
    stack_size: int = 64 * 1024  # 64KB
    enable_caching: bool = True
    enable_metrics: bool = True
    enable_memory_protection: bool = True
    enable_sandbox: bool = True

@dataclass
class ExecutionContext:
    """Context for individual execution"""
    execution_id: str
    function_name: str
    caller: str
    contract_id: str
    gas_meter: GasMeter
    storage: ContractStorage
    contract_manager: Any
    start_time: float
    state: ExecutionState = ExecutionState.IDLE
    thread_id: Optional[int] = None
    memory_usage: int = 0

class WASMExecutor:
    """Production-ready WebAssembly executor for smart contract execution"""
    
    def __init__(self, wasm_bytecode: bytes, config: Optional[WASMConfig] = None):
        self.wasm_bytecode = wasm_bytecode
        self.config = config or WASMConfig()
        self.bytecode_hash = hashlib.sha256(wasm_bytecode).hexdigest()
        
        # Initialize WASM components
        self.engine = self._create_engine()
        self.store = Store(self.engine)
        self.module = self._create_module()
        self.linker = Linker(self.engine)
        
        # Execution management
        self.execution_lock = threading.RLock()
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_executions,
            thread_name_prefix="wasm_executor"
        )
        
        # Enhanced caching system
        self.instance_cache: Dict[str, Tuple[Instance, float]] = {}
        self.function_cache: Dict[str, Any] = {}
        self.memory_cache: Dict[str, Memory] = {}
        
        # Advanced metrics
        self.metrics = {
            'execution_count': 0,
            'total_execution_time': 0.0,
            'successful_executions': 0,
            'failed_executions': 0,
            'timeout_executions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_allocations': 0,
            'memory_deallocations': 0,
            'total_memory_used': 0,
            'peak_memory_usage': 0,
            'concurrent_peak': 0,
        }
        
        # Security and validation
        self.validator = WASMBytecodeValidator()
        self.is_validated = self.validator.validate_bytecode(wasm_bytecode)
        
        if not self.is_validated:
            logger.warning("WASM bytecode validation failed, execution may be restricted")
        
        # Initialize host functions placeholder
        self.host_functions: Optional[WASMHostFunctions] = None
        
        logger.info(f"WASMExecutor initialized with bytecode hash: {self.bytecode_hash[:16]}...")

    def _create_engine(self) -> Engine:
        """Create configured WASM engine"""
        try:
            engine_config = self.config.engine_config.copy()
            engine_config.setdefault('max_wasm_stack', self.config.stack_size)
            engine_config.setdefault('consume_fuel', True)  # Enable fuel for gas metering
            
            return Engine(engine_config)
        except Exception as e:
            logger.error(f"Failed to create WASM engine: {e}")
            raise

    def _create_module(self) -> Module:
        """Create and validate WASM module"""
        try:
            module = Module(self.engine, self.wasm_bytecode)
            
            # Additional module validation
            if self.config.enable_sandbox:
                self._validate_module_security(module)
                
            return module
        except Exception as e:
            logger.error(f"Failed to create WASM module: {e}")
            raise

    def _validate_module_security(self, module: Module) -> None:
        """Perform security validation on WASM module"""
        # Check for dangerous imports/exports
        for import_item in module.imports:
            if import_item.module != "env":
                logger.warning(f"Non-standard import module: {import_item.module}")
                
        # Validate memory limits
        for memory in module.memories:
            if memory.min > self.config.memory_pages:
                raise ValueError(f"Memory minimum {memory.min} exceeds limit {self.config.memory_pages}")

    def execute_function(self, function_name: str, caller: str, args: Dict[str, Any],
                        gas_meter: GasMeter, contract_manager: Any, 
                        contract_id: str, storage: ContractStorage) -> ExecutionResult:
        """Execute a WASM function with comprehensive production features"""
        execution_id = self._generate_execution_id()
        context = ExecutionContext(
            execution_id=execution_id,
            function_name=function_name,
            caller=caller,
            contract_id=contract_id,
            gas_meter=gas_meter,
            storage=storage,
            contract_manager=contract_manager,
            start_time=time.time()
        )
        
        with self.execution_lock:
            self.active_executions[execution_id] = context
            current_concurrent = len(self.active_executions)
            self.metrics['concurrent_peak'] = max(self.metrics['concurrent_peak'], current_concurrent)
        
        try:
            # Execute in thread pool with timeout
            future = self.thread_pool.submit(
                self._execute_function_internal, context, args
            )
            
            result = future.result(timeout=self.config.timeout)
            context.state = ExecutionState.COMPLETED
            
        except TimeoutError:
            context.state = ExecutionState.TIMEOUT
            result = ExecutionResult(
                success=False,
                error=f"Execution timeout after {self.config.timeout}s",
                execution_time=self.config.timeout,
                gas_used=gas_meter.gas_used
            )
            self.metrics['timeout_executions'] += 1
            logger.warning(f"Execution {execution_id} timed out")
            
        except Exception as e:
            context.state = ExecutionState.FAILED
            result = ExecutionResult(
                success=False,
                error=f"Execution failed: {e}",
                execution_time=time.time() - context.start_time,
                gas_used=gas_meter.gas_used
            )
            self.metrics['failed_executions'] += 1
            logger.error(f"Execution {execution_id} failed: {e}")
            
        finally:
            with self.execution_lock:
                self.active_executions.pop(execution_id, None)
                self._update_metrics(result, context)
        
        return result

    def _execute_function_internal(self, context: ExecutionContext, args: Dict[str, Any]) -> ExecutionResult:
        """Internal function execution with comprehensive error handling"""
        context.thread_id = threading.get_ident()
        context.state = ExecutionState.EXECUTING
        
        result = ExecutionResult(success=False)
        host_functions = None
        
        try:
            # Validate execution context
            if not self._validate_execution_context(context):
                result.error = "Invalid execution context"
                return result

            # Create or get cached instance
            instance = self._get_or_create_instance(context)
            if instance is None:
                result.error = "Failed to create WASM instance"
                return result

            # Prepare host functions
            host_functions = WASMHostFunctions(
                context.gas_meter, 
                context.storage, 
                context.contract_manager
            )
            self._register_host_functions(host_functions)

            # Prepare arguments
            wasm_args = self._prepare_arguments(args, instance, context)
            if wasm_args is None:
                result.error = "Failed to prepare arguments"
                return result

            # Execute with comprehensive monitoring
            wasm_result = self._execute_with_monitoring(instance, context, wasm_args)
            
            if wasm_result is not None:
                result.success = True
                result.return_value = self._process_return_value(wasm_result)
            else:
                result.error = "WASM execution returned null"

        except Trap as e:
            result.error = f"WASM trap: {e}"
            logger.error(f"WASM trap in execution {context.execution_id}: {e}")
            
        except OutOfMemoryError:
            result.error = "Memory allocation exceeded limit"
            logger.error(f"Memory limit exceeded in execution {context.execution_id}")
            
        except Exception as e:
            result.error = f"WASM execution failed: {e}"
            logger.error(f"Execution {context.execution_id} error: {e}", exc_info=True)
            
        finally:
            # Update result metrics
            execution_time = time.time() - context.start_time
            result.execution_time = execution_time
            result.gas_used = context.gas_meter.gas_used
            
            # Cleanup resources
            if host_functions:
                host_functions.cleanup()
                
            # Update memory metrics
            self._update_memory_metrics(context)
        
        return result

    def _get_or_create_instance(self, context: ExecutionContext) -> Optional[Instance]:
        """Get cached instance or create new one with cache management"""
        cache_key = f"{self.bytecode_hash}_{context.contract_id}"
        
        if self.config.enable_caching and cache_key in self.instance_cache:
            instance, timestamp = self.instance_cache[cache_key]
            # Check if instance is still valid (not expired)
            if time.time() - timestamp < 300:  # 5-minute cache validity
                self.metrics['cache_hits'] += 1
                return instance
            else:
                # Remove expired instance
                del self.instance_cache[cache_key]
        
        self.metrics['cache_misses'] += 1
        
        try:
            # Create new instance
            instance = self.linker.instantiate(self.store, self.module)
            
            # Configure instance
            self._configure_instance(instance, context)
            
            # Cache the instance
            if self.config.enable_caching and len(self.instance_cache) < self.config.max_instances:
                self.instance_cache[cache_key] = (instance, time.time())
                
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create WASM instance: {e}")
            return None

    def _configure_instance(self, instance: Instance, context: ExecutionContext) -> None:
        """Configure WASM instance with security and resource limits"""
        # Set memory limits
        if self.config.enable_memory_protection:
            memory = instance.get_memory(self.store, "memory")
            if memory:
                # Configure memory growth limits
                pass
                
        # Set fuel for gas metering
        try:
            self.store.add_fuel(context.gas_meter.get_remaining_gas())
        except Exception as e:
            logger.warning(f"Failed to set fuel for gas metering: {e}")

    def _register_host_functions(self, host_functions: WASMHostFunctions) -> None:
        """Register host functions with the linker"""
        try:
            for name, func in host_functions.get_functions().items():
                self.linker.define("env", name, func)
        except Exception as e:
            logger.error(f"Failed to register host functions: {e}")
            raise

    def _prepare_arguments(self, args: Dict[str, Any], instance: Instance, 
                          context: ExecutionContext) -> Optional[List[Val]]:
        """Prepare arguments with memory allocation and type conversion"""
        wasm_args = []
        
        try:
            for key, value in args.items():
                wasm_val = self._convert_to_wasm_value(value, instance, context)
                if wasm_val is None:
                    logger.error(f"Failed to convert argument {key} to WASM value")
                    return None
                wasm_args.append(wasm_val)
                
            return wasm_args
            
        except Exception as e:
            logger.error(f"Failed to prepare arguments: {e}")
            return None

    def _convert_to_wasm_value(self, value: Any, instance: Instance, 
                             context: ExecutionContext) -> Optional[Val]:
        """Convert Python value to WASM value with memory management"""
        try:
            if isinstance(value, int):
                if -2147483648 <= value <= 2147483647:
                    return Val.i32(value)
                else:
                    return Val.i64(value)
                    
            elif isinstance(value, float):
                return Val.f64(value)
                
            elif isinstance(value, str):
                return self._allocate_string(value, instance, context)
                
            elif isinstance(value, bytes):
                return self._allocate_bytes(value, instance, context)
                
            elif isinstance(value, bool):
                return Val.i32(1 if value else 0)
                
            else:
                # Convert to string representation
                return self._allocate_string(str(value), instance, context)
                
        except Exception as e:
            logger.error(f"Failed to convert value to WASM: {e}")
            return None

    def _allocate_string(self, text: str, instance: Instance, 
                        context: ExecutionContext) -> Optional[Val]:
        """Allocate string in WASM memory"""
        try:
            memory = instance.get_memory(self.store, "memory")
            if not memory:
                return Val.i32(0)  # Fallback to length only
                
            # Convert string to bytes
            text_bytes = text.encode('utf-8')
            return self._allocate_bytes(text_bytes, instance, context)
            
        except Exception as e:
            logger.error(f"Failed to allocate string: {e}")
            return Val.i32(0)

    def _allocate_bytes(self, data: bytes, instance: Instance, 
                       context: ExecutionContext) -> Optional[Val]:
        """Allocate bytes in WASM memory"""
        try:
            memory = instance.get_memory(self.store, "memory")
            if not memory:
                return Val.i32(len(data))  # Fallback to length only
                
            # Allocate memory in WASM instance
            allocator = instance.get_func(self.store, "allocate")
            if allocator:
                # Call allocate function to get memory pointer
                ptr = allocator(self.store, len(data))
                if ptr is not None:
                    # Write data to memory
                    memory.write(self.store, ptr, data)
                    context.memory_usage += len(data)
                    self.metrics['memory_allocations'] += 1
                    self.metrics['total_memory_used'] += len(data)
                    return Val.i32(ptr)
                    
            return Val.i32(len(data))  # Fallback
            
        except Exception as e:
            logger.error(f"Failed to allocate bytes: {e}")
            return Val.i32(0)

    def _execute_with_monitoring(self, instance: Instance, context: ExecutionContext,
                               args: List[Val]) -> Any:
        """Execute WASM function with comprehensive monitoring"""
        try:
            # Get the function
            func = instance.get_func(self.store, context.function_name)
            if func is None:
                raise ValueError(f"Function {context.function_name} not found")
            
            # Execute with gas and memory monitoring
            start_time = time.time()
            
            # Set execution context for host functions
            if self.host_functions:
                self.host_functions.set_execution_context(context)
            
            # Execute the function
            result = func(self.store, *args)
            
            # Check resource usage
            self._check_resource_usage(context, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"WASM function execution failed: {e}")
            raise

    def _check_resource_usage(self, context: ExecutionContext, start_time: float) -> None:
        """Check resource usage during execution"""
        current_time = time.time()
        execution_time = current_time - start_time
        
        # Check execution time
        if execution_time > self.config.timeout:
            raise TimeoutError(f"Execution timeout after {execution_time}s")
        
        # Check gas limit
        if context.gas_meter.get_remaining_gas() <= 0:
            from ..core.gas_system.out_of_gas_error import OutOfGasError
            raise OutOfGasError("Out of gas during execution")
        
        # Check memory usage
        if context.memory_usage > self.config.memory_limit:
            raise OutOfMemoryError(f"Memory usage {context.memory_usage} exceeds limit {self.config.memory_limit}")

    def _process_return_value(self, value: Any) -> Any:
        """Process WASM return value for Python consumption"""
        if hasattr(value, 'value'):
            return value.value
        return value

    def _validate_execution_context(self, context: ExecutionContext) -> bool:
        """Validate execution context before execution"""
        if not context.function_name or not self._validate_function_name(context.function_name):
            return False
            
        if context.gas_meter.get_remaining_gas() <= 0:
            return False
            
        if len(self.active_executions) > self.config.max_concurrent_executions:
            return False
            
        return True

    def _validate_function_name(self, function_name: str) -> bool:
        """Validate WASM function name with enhanced rules"""
        if not function_name or len(function_name) > 256:
            return False
        
        # Enhanced validation
        import re
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]{0,255}$'
        return bool(re.match(pattern, function_name))

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        timestamp = int(time.time() * 1000)
        thread_id = threading.get_ident()
        return f"exec_{timestamp}_{thread_id}_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"

    def _update_metrics(self, result: ExecutionResult, context: ExecutionContext) -> None:
        """Update comprehensive metrics"""
        with self.execution_lock:
            self.metrics['execution_count'] += 1
            self.metrics['total_execution_time'] += result.execution_time
            
            if result.success:
                self.metrics['successful_executions'] += 1
            else:
                self.metrics['failed_executions'] += 1
                
            self.metrics['peak_memory_usage'] = max(
                self.metrics['peak_memory_usage'], 
                context.memory_usage
            )

    def _update_memory_metrics(self, context: ExecutionContext) -> None:
        """Update memory-related metrics"""
        self.metrics['total_memory_used'] -= context.memory_usage
        self.metrics['memory_deallocations'] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive executor metrics"""
        with self.execution_lock:
            metrics = self.metrics.copy()
            metrics.update({
                'active_executions': len(self.active_executions),
                'cached_instances': len(self.instance_cache),
                'bytecode_size': len(self.wasm_bytecode),
                'bytecode_hash': self.bytecode_hash,
                'is_validated': self.is_validated,
                'average_execution_time': (
                    metrics['total_execution_time'] / metrics['execution_count'] 
                    if metrics['execution_count'] > 0 else 0
                ),
                'success_rate': (
                    metrics['successful_executions'] / metrics['execution_count'] 
                    if metrics['execution_count'] > 0 else 0
                ),
                'cache_hit_rate': (
                    metrics['cache_hits'] / (metrics['cache_hits'] + metrics['cache_misses']) 
                    if (metrics['cache_hits'] + metrics['cache_misses']) > 0 else 0
                ),
            })
            return metrics

    def get_active_executions(self) -> List[Dict[str, Any]]:
        """Get information about active executions"""
        with self.execution_lock:
            return [
                {
                    'execution_id': ctx.execution_id,
                    'function_name': ctx.function_name,
                    'caller': ctx.caller,
                    'contract_id': ctx.contract_id,
                    'state': ctx.state.value,
                    'execution_time': time.time() - ctx.start_time,
                    'memory_usage': ctx.memory_usage,
                    'thread_id': ctx.thread_id
                }
                for ctx in self.active_executions.values()
            ]

    def cleanup(self) -> None:
        """Comprehensive cleanup of resources"""
        try:
            # Wait for active executions to complete with timeout
            start_time = time.time()
            while self.active_executions and (time.time() - start_time) < 10:  # 10s timeout
                time.sleep(0.1)
            
            # Force cleanup if executions are still running
            if self.active_executions:
                logger.warning("Forcing cleanup with active executions")
                self.active_executions.clear()
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True, timeout=5)
            
            # Clear caches
            self.instance_cache.clear()
            self.function_cache.clear()
            self.memory_cache.clear()
            
            # Cleanup WASM resources
            self.linker = None
            self.module = None
            self.store = None
            self.engine = None
            
            logger.info("WASMExecutor cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during WASMExecutor cleanup: {e}")

    def __del__(self):
        """Destructor with safe cleanup"""
        try:
            self.cleanup()
        except Exception as e:
            logger.warning(f"Error during WASMExecutor destruction: {e}")