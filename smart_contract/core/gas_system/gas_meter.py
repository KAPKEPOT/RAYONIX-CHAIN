# smart_contract/core/gas_system/gas_meter.py
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .out_of_gas_error import OutOfGasError

logger = logging.getLogger("SmartContract.GasMeter")

@dataclass
class GasCostConfig:
    """Configuration for gas costs of various operations"""
    # Base operation costs
    BASE_CALL_COST: int = 21000
    BASE_STORAGE_COST: int = 20000
    BASE_COMPUTATION_COST: int = 100
    
    # Memory operations
    MEMORY_LOAD_COST: int = 3
    MEMORY_STORE_COST: int = 5
    MEMORY_ALLOC_COST: int = 2
    
    # Computation operations
    ADD_COST: int = 3
    MUL_COST: int = 5
    DIV_COST: int = 8
    MOD_COST: int = 8
    EXP_COST: int = 10
    
    # Storage operations
    SLOAD_COST: int = 200
    SSTORE_COST: int = 5000
    SLOAD_REFUND: int = 150
    SSTORE_REFUND: int = 15000
    
    # Cryptographic operations
    SHA3_COST: int = 30
    SHA3_WORD_COST: int = 6
    KECCAK256_COST: int = 30
    KECCAK256_WORD_COST: int = 6
    EC_RECOVER_COST: int = 3000
    EC_ADD_COST: int = 500
    EC_MUL_COST: int = 40000
    EC_PAIRING_COST: int = 80000
    EC_PAIRING_POINT_COST: int = 100000
    
    # Log operations
    LOG_BASE_COST: int = 375
    LOG_TOPIC_COST: int = 375
    LOG_DATA_COST: int = 8
    
    # Contract operations
    CREATE_COST: int = 32000
    CALL_COST: int = 700
    CALL_VALUE_COST: int = 9000
    CALL_STIPEND: int = 2300
    SELFDESTRUCT_COST: int = 5000
    SELFDESTRUCT_REFUND: int = 24000
    
    # WASM-specific costs
    WASM_INSTANTIATE_COST: int = 10000
    WASM_FUNCTION_CALL_COST: int = 200
    WASM_MEMORY_ACCESS_COST: int = 3
    WASM_TABLE_ACCESS_COST: int = 3
    WASM_GLOBAL_ACCESS_COST: int = 3
    
    # Dynamic pricing factors
    MEMORY_EXPANSION_FACTOR: float = 0.1
    COMPUTATION_INTENSITY_FACTOR: float = 1.5
    STORAGE_COMPLEXITY_FACTOR: float = 2.0

class GasMeter:
    """Advanced gas metering system with dynamic pricing and optimization"""
    
    def __init__(self, gas_limit: int, gas_price: int = 1, config: Optional[GasCostConfig] = None):
        self.gas_limit = gas_limit
        self.gas_price = gas_price
        self.config = config or GasCostConfig()
        
        self.gas_used = 0
        self.gas_refunded = 0
        self.breakdown: Dict[str, int] = {}
        self.operation_stack: list = []
        self.start_time = time.time()
        self.memory_usage = 0
        self.storage_operations = 0
        self.computation_intensity = 0
        
        # Performance tracking
        self.operation_count = 0
        self.average_operation_cost = 0
        self.peak_memory_usage = 0
        
        logger.debug(f"GasMeter initialized with limit: {gas_limit}, price: {gas_price}")
    
    def consume_gas(self, operation: str, cost: int, details: Optional[Dict] = None) -> None:
        """Consume gas for an operation with detailed tracking"""
        if cost < 0:
            self.gas_refunded += abs(cost)
            return
        
        # Apply dynamic pricing
        adjusted_cost = self._apply_dynamic_pricing(operation, cost, details)
        
        # Check gas limit
        if self.gas_used + adjusted_cost > self.gas_limit:
            raise OutOfGasError(
                f"Out of gas: {self.gas_used + adjusted_cost} > {self.gas_limit} "
                f"while performing {operation}"
            )
        
        # Update gas usage
        self.gas_used += adjusted_cost
        self.breakdown[operation] = self.breakdown.get(operation, 0) + adjusted_cost
        
        # Update performance metrics
        self.operation_count += 1
        self.average_operation_cost = (
            (self.average_operation_cost * (self.operation_count - 1) + adjusted_cost) 
            / self.operation_count
        )
        
        # Track operation
        self.operation_stack.append({
            'operation': operation,
            'cost': adjusted_cost,
            'timestamp': time.time(),
            'details': details or {}
        })
        
        logger.debug(f"Gas consumed: {operation} - {adjusted_cost} gas (total: {self.gas_used})")
    
    def _apply_dynamic_pricing(self, operation: str, base_cost: int, details: Optional[Dict]) -> int:
        """Apply dynamic pricing based on current execution context"""
        adjusted_cost = base_cost
        
        # Memory-intensive operations
        if operation in ['memory_load', 'memory_store', 'memory_alloc']:
            memory_factor = 1.0 + (self.memory_usage / (1024 * 1024)) * self.config.MEMORY_EXPANSION_FACTOR
            adjusted_cost = int(base_cost * memory_factor)
        
        # Computation-intensive operations
        elif operation in ['computation', 'math_operation']:
            intensity_factor = 1.0 + (self.computation_intensity * self.config.COMPUTATION_INTENSITY_FACTOR)
            adjusted_cost = int(base_cost * intensity_factor)
        
        # Storage operations
        elif operation in ['storage_read', 'storage_write']:
            complexity_factor = 1.0 + (self.storage_operations * self.config.STORAGE_COMPLEXITY_FACTOR)
            adjusted_cost = int(base_cost * complexity_factor)
        
        return max(1, adjusted_cost)
    
    def refund_gas(self, amount: int, reason: str) -> None:
        """Refund gas with specified reason"""
        if amount > 0:
            self.gas_refunded += amount
            logger.debug(f"Gas refunded: {reason} - {amount} gas")
    
    def get_remaining_gas(self) -> int:
        """Get remaining gas"""
        return max(0, self.gas_limit - self.gas_used)
    
    def get_effective_gas_used(self) -> int:
        """Get effective gas used after refunds"""
        return max(0, self.gas_used - self.gas_refunded)
    
    def get_total_cost(self) -> int:
        """Get total cost in base currency"""
        return self.get_effective_gas_used() * self.gas_price
    
    def update_memory_usage(self, bytes_used: int) -> None:
        """Update memory usage tracking"""
        self.memory_usage = bytes_used
        self.peak_memory_usage = max(self.peak_memory_usage, bytes_used)
    
    def update_computation_intensity(self, intensity: float) -> None:
        """Update computation intensity factor"""
        self.computation_intensity = max(0.0, min(1.0, intensity))
    
    def increment_storage_operations(self) -> None:
        """Increment storage operation count"""
        self.storage_operations += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive gas metrics"""
        return {
            'gas_used': self.gas_used,
            'gas_refunded': self.gas_refunded,
            'effective_gas_used': self.get_effective_gas_used(),
            'remaining_gas': self.get_remaining_gas(),
            'total_cost': self.get_total_cost(),
            'operation_count': self.operation_count,
            'average_operation_cost': self.average_operation_cost,
            'peak_memory_usage': self.peak_memory_usage,
            'storage_operations': self.storage_operations,
            'computation_intensity': self.computation_intensity,
            'execution_time': time.time() - self.start_time,
            'gas_breakdown': self.breakdown.copy(),
            'gas_price': self.gas_price
        }
    
    def create_checkpoint(self) -> Dict:
        """Create a checkpoint for potential rollback"""
        return {
            'gas_used': self.gas_used,
            'gas_refunded': self.gas_refunded,
            'operation_count': self.operation_count,
            'average_operation_cost': self.average_operation_cost,
            'memory_usage': self.memory_usage,
            'storage_operations': self.storage_operations,
            'computation_intensity': self.computation_intensity
        }
    
    def restore_checkpoint(self, checkpoint: Dict) -> None:
        """Restore state from checkpoint"""
        self.gas_used = checkpoint['gas_used']
        self.gas_refunded = checkpoint['gas_refunded']
        self.operation_count = checkpoint['operation_count']
        self.average_operation_cost = checkpoint['average_operation_cost']
        self.memory_usage = checkpoint['memory_usage']
        self.storage_operations = checkpoint['storage_operations']
        self.computation_intensity = checkpoint['computation_intensity']
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - log final metrics"""
        if exc_type is None:
            logger.info(f"Gas usage completed: {self.get_effective_gas_used()} gas used, "
                       f"{self.gas_refunded} gas refunded")
        return False