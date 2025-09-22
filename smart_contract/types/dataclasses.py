# smart_contract/types/dataclasses.py
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime

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
    gas_config: Dict[str, Any] = field(default_factory=lambda: {
        'base_gas_price': 5,
        'max_gas_per_call': 10 * 1000 * 1000,
        'gas_optimization_enabled': True
    })
    security_config: Dict[str, Any] = field(default_factory=lambda: {
        'threat_detection_enabled': True,
        'rate_limiting_enabled': True,
        'input_validation_enabled': True
    })

@dataclass
class SecurityPolicy:
    """Security policy for contract execution"""
    allowed_operations: List[str] = field(default_factory=lambda: ['*'])
    denied_operations: List[str] = field(default_factory=list)
    max_gas_per_operation: Dict[str, int] = field(default_factory=dict)
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'calls_per_minute': 1000,
        'storage_ops_per_minute': 500,
        'max_concurrent_calls': 100
    })
    resource_limits: Dict[str, int] = field(default_factory=lambda: {
        'max_memory_mb': 100,
        'max_storage_mb': 10240,
        'max_execution_time_seconds': 30
    })
    allowed_callees: List[str] = field(default_factory=lambda: ['*'])
    denied_callees: List[str] = field(default_factory=list)
    require_authentication: bool = True
    audit_logging: bool = True

@dataclass
class ExecutionContext:
    """Context for contract execution"""
    caller: str
    contract_id: str
    function_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    gas_limit: int = 1000000
    gas_price: int = 5
    call_depth: int = 0
    parent_call: Optional['ExecutionContext'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContractMetadata:
    """Metadata for smart contracts"""
    name: str
    version: str
    author: str
    description: str = ""
    license: str = "MIT"
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    dependencies: Dict[str, str] = field(default_factory=dict)
    audit_reports: List[str] = field(default_factory=list)
    security_score: float = 0.0
    gas_efficiency: float = 0.0

@dataclass
class GasMetrics:
    """Gas usage metrics"""
    total_gas_used: int = 0
    average_gas_per_call: float = 0.0
    max_gas_per_call: int = 0
    min_gas_per_call: int = 0
    gas_by_operation: Dict[str, int] = field(default_factory=dict)
    optimization_savings: int = 0
    refunds_issued: int = 0

@dataclass
class PerformanceMetrics:
    """Performance metrics for contracts"""
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = 0.0
    memory_usage_stats: Dict[str, float] = field(default_factory=dict)
    storage_usage_stats: Dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    error_breakdown: Dict[str, int] = field(default_factory=dict)