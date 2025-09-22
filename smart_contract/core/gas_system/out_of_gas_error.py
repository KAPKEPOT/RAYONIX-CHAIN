# smart_contract/core/gas_system/out_of_gas_error.py
class OutOfGasError(Exception):
    """Exception raised when a contract execution runs out of gas"""
    
    def __init__(self, message: str, gas_used: int = 0, gas_limit: int = 0):
        super().__init__(message)
        self.gas_used = gas_used
        self.gas_limit = gas_limit
        self.message = message
    
    def __str__(self) -> str:
        return f"{self.message} (Used: {self.gas_used}, Limit: {self.gas_limit})"
    
    def to_dict(self) -> dict:
        return {
            'error': 'OutOfGasError',
            'message': self.message,
            'gas_used': self.gas_used,
            'gas_limit': self.gas_limit
        }