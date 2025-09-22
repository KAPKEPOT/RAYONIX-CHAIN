# smart_contract/__init__.py
from .core.contract import SmartContract
from .core.contract_manager import ContractManager
from .core.execution_result import ExecutionResult
from .core.gas_system.gas_meter import GasMeter
from .core.gas_system.gas_optimizer import GasOptimizer
from .core.gas_system.out_of_gas_error import OutOfGasError
from .core.storage.contract_storage import ContractStorage
from .security.contract_security import ContractSecurity
from .security.behavioral_analyzer import BehavioralAnalyzer
from .security.threat_intelligence import ThreatIntelligenceFeed
from .security.validators.input_validator import InputValidator
from .security.validators.domain_validator import DomainValidator
from .security.validators.ip_validator import IPValidator
from .wasm.wasm_host_functions import WASMHostFunctions
from .wasm.wasm_executor import WASMExecutor
from .wasm.bytecode_validator import WASMBytecodeValidator
from .types.enums import ContractType, ContractState, ContractSecurityLevel
from .types.dataclasses import ContractConfig, SecurityPolicy
from .utils.cryptography_utils import (
    encrypt_data, decrypt_data, derive_key, generate_encryption_key,
    validate_signature, generate_key_pair
)
from .utils.serialization_utils import (
    serialize_contract, deserialize_contract, compress_data, decompress_data
)
from .utils.validation_utils import (
    validate_address, validate_wasm_bytecode, validate_contract_id,
    validate_function_name, validate_gas_limit
)
from .utils.network_utils import (
    fetch_network_conditions, resolve_domain, validate_ip_address
)
from .database.leveldb_manager import LevelDBManager
from .exceptions.contract_errors import (
    ContractDeploymentError, ContractExecutionError,
    ContractUpgradeError, ContractNotFoundError
)
from .exceptions.security_errors import (
    SecurityViolationError, RateLimitExceededError,
    BlacklistedAddressError, InputValidationError
)

__version__ = "1.0.0"
__all__ = [
    'SmartContract', 'ContractManager', 'ExecutionResult', 'GasMeter',
    'GasOptimizer', 'OutOfGasError', 'ContractStorage', 'ContractSecurity',
    'BehavioralAnalyzer', 'ThreatIntelligenceFeed', 'InputValidator',
    'DomainValidator', 'IPValidator', 'WASMHostFunctions', 'WASMExecutor',
    'WASMBytecodeValidator', 'ContractType', 'ContractState',
    'ContractSecurityLevel', 'ContractConfig', 'SecurityPolicy',
    'encrypt_data', 'decrypt_data', 'derive_key', 'generate_encryption_key',
    'validate_signature', 'generate_key_pair', 'serialize_contract',
    'deserialize_contract', 'compress_data', 'decompress_data',
    'validate_address', 'validate_wasm_bytecode', 'validate_contract_id',
    'validate_function_name', 'validate_gas_limit', 'fetch_network_conditions',
    'resolve_domain', 'validate_ip_address', 'LevelDBManager',
    'ContractDeploymentError', 'ContractExecutionError', 'ContractUpgradeError',
    'ContractNotFoundError', 'SecurityViolationError', 'RateLimitExceededError',
    'BlacklistedAddressError', 'InputValidationError'
]