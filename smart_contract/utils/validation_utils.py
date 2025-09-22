# smart_contract/utils/validation_utils.py
import re
import logging
from typing import Optional

logger = logging.getLogger("SmartContract.Validation")

def validate_address(address: str) -> bool:
    """Validate blockchain address format"""
    if not address or not isinstance(address, str):
        return False
    
    # Ethereum address pattern
    eth_pattern = r'^0x[a-fA-F0-9]{40}$'
    
    # Other blockchain patterns could be added here
    
    if re.match(eth_pattern, address):
        return True
    
    # Add validation for other address types as needed
    return False

def validate_wasm_bytecode(bytecode: bytes) -> bool:
    """Validate WASM bytecode basic structure"""
    if not bytecode or len(bytecode) < 8:
        return False
    
    # Check WASM magic number
    if bytecode[:4] != b'\x00asm':
        return False
    
    # Check WASM version
    if bytecode[4:8] != b'\x01\x00\x00\x00':
        return False
    
    return True

def validate_contract_id(contract_id: str) -> bool:
    """Validate contract ID format"""
    if not contract_id or not isinstance(contract_id, str):
        return False
    
    # Alphanumeric with underscores and hyphens
    pattern = r'^[a-zA-Z0-9_-]{1,256}$'
    return bool(re.match(pattern, contract_id))

def validate_function_name(function_name: str) -> bool:
    """Validate function name format"""
    if not function_name or not isinstance(function_name, str):
        return False
    
    # Alphanumeric with underscores
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]{0,255}$'
    return bool(re.match(pattern, function_name))

def validate_gas_limit(gas_limit: int) -> bool:
    """Validate gas limit value"""
    if not isinstance(gas_limit, int):
        return False
    
    return 0 < gas_limit <= 10_000_000  # Reasonable upper limit

def validate_numeric_value(value, min_value: Optional[float] = None, 
                          max_value: Optional[float] = None) -> bool:
    """Validate numeric value with optional range"""
    if not isinstance(value, (int, float)):
        return False
    
    if min_value is not None and value < min_value:
        return False
    
    if max_value is not None and value > max_value:
        return False
    
    return True

def validate_string_length(value: str, min_length: int = 0, 
                          max_length: Optional[int] = None) -> bool:
    """Validate string length"""
    if not isinstance(value, str):
        return False
    
    if len(value) < min_length:
        return False
    
    if max_length is not None and len(value) > max_length:
        return False
    
    return True

def validate_json_structure(json_data: str, required_fields: list = None) -> bool:
    """Validate JSON structure"""
    try:
        import json
        data = json.loads(json_data)
        
        if required_fields:
            for field in required_fields:
                if field not in data:
                    return False
        
        return True
    except:
        return False

def validate_timestamp(timestamp: int) -> bool:
    """Validate timestamp (seconds since epoch)"""
    if not isinstance(timestamp, int):
        return False
    
    # Reasonable range: from 2000 to 2100
    return 946684800 <= timestamp <= 4102444800

def validate_boolean(value) -> bool:
    """Validate boolean value"""
    return isinstance(value, bool)