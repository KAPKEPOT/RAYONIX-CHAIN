# smart_contract/exceptions/contract_errors.py
class ContractError(Exception):
    """Base class for all contract-related exceptions"""
    pass

class ContractDeploymentError(ContractError):
    """Error during contract deployment"""
    
    def __init__(self, message: str, bytecode_hash: str = None):
        self.bytecode_hash = bytecode_hash
        super().__init__(f"DeploymentError: {message}")

class ContractExecutionError(ContractError):
    """Error during contract execution"""
    
    def __init__(self, message: str, contract_address: str = None, method: str = None):
        self.contract_address = contract_address
        self.method = method
        super().__init__(f"ExecutionError {contract_address or 'Unknown'}.{method or 'Unknown'}: {message}")

class ContractValidationError(ContractError):
    """Contract validation failed"""
    
    def __init__(self, message: str, validation_type: str = "general"):
        self.validation_type = validation_type
        super().__init__(f"ValidationError[{validation_type}]: {message}")

class ContractUpgradeError(ContractError):
    """Contract upgrade failed"""
    
    def __init__(self, message: str, contract_address: str, new_version: str):
        self.contract_address = contract_address
        self.new_version = new_version
        super().__init__(f"UpgradeError {contract_address} to {new_version}: {message}")

class ContractNotFoundError(ContractError):
    """Contract not found"""
    
    def __init__(self, contract_address: str):
        self.contract_address = contract_address
        super().__init__(f"Contract not found: {contract_address}")

class ContractStateError(ContractError):
    """Invalid contract state"""
    
    def __init__(self, message: str, contract_address: str, expected_state: str = None, actual_state: str = None):
        self.contract_address = contract_address
        self.expected_state = expected_state
        self.actual_state = actual_state
        
        state_info = ""
        if expected_state and actual_state:
            state_info = f" (expected: {expected_state}, actual: {actual_state})"
        
        super().__init__(f"StateError {contract_address}: {message}{state_info}")

class ContractCompilationError(ContractError):
    """Contract compilation failed"""
    
    def __init__(self, message: str, source_hash: str = None, compiler: str = None):
        self.source_hash = source_hash
        self.compiler = compiler
        super().__init__(f"CompilationError: {message}")