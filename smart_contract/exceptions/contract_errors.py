# smart_contract/exceptions/contract_errors.py
class ContractDeploymentError(Exception):
    """Exception raised when contract deployment fails"""
    def __init__(self, message: str, contract_id: str = None):
        super().__init__(message)
        self.contract_id = contract_id
        self.message = message

class ContractExecutionError(Exception):
    """Exception raised when contract execution fails"""
    def __init__(self, message: str, contract_id: str = None, function_name: str = None):
        super().__init__(message)
        self.contract_id = contract_id
        self.function_name = function_name
        self.message = message

class ContractUpgradeError(Exception):
    """Exception raised when contract upgrade fails"""
    def __init__(self, message: str, contract_id: str = None, version: str = None):
        super().__init__(message)
        self.contract_id = contract_id
        self.version = version
        self.message = message

class ContractNotFoundError(Exception):
    """Exception raised when contract is not found"""
    def __init__(self, message: str, contract_id: str = None):
        super().__init__(message)
        self.contract_id = contract_id
        self.message = message