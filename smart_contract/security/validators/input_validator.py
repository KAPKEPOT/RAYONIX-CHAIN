# smart_contract/security/validators/input_validator.py
import re
import json
import logging
from typing import Any, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger("SmartContract.InputValidator")

class InputValidator:
    """Advanced input validation system with multiple validation strategies"""
    
    def __init__(self):
        # Common patterns for validation
        self.patterns = {
            'eth_address': re.compile(r'^0x[a-fA-F0-9]{40}$'),
            'hex_string': re.compile(r'^0x[a-fA-F0-9]+$'),
            'numeric': re.compile(r'^-?\d+(\.\d+)?$'),
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'ip_address': re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'),
            'domain': re.compile(r'^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        }
        
        # Validation rules by context
        self.validation_rules = {
            'transfer': {
                'amount': {'type': 'numeric', 'min': 0, 'required': True},
                'to': {'type': 'eth_address', 'required': True},
                'memo': {'type': 'string', 'max_length': 256, 'required': False}
            },
            'storage': {
                'key': {'type': 'string', 'max_length': 256, 'required': True},
                'value': {'type': 'any', 'required': True},
                'ttl': {'type': 'numeric', 'min': 0, 'required': False}
            },
            'authentication': {
                'username': {'type': 'string', 'min_length': 3, 'max_length': 32, 'required': True},
                'password': {'type': 'string', 'min_length': 8, 'required': True},
                'token': {'type': 'string', 'required': False}
            }
        }
        
        logger.info("InputValidator initialized")
    
    def validate(self, input_data: Any, context: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate input data with context-aware validation
        Returns: (is_valid, error_message)
        """
        try:
            # Determine validation context
            operation = context.get('operation', '')
            contract_id = context.get('contract', '')
            caller = context.get('caller', '')
            
            # Apply context-specific validation rules
            if operation in self.validation_rules:
                return self._validate_with_rules(input_data, self.validation_rules[operation], context)
            
            # Default validation based on input type
            return self._validate_by_type(input_data, context)
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _validate_with_rules(self, input_data: Any, rules: Dict, context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate input data against specific rules"""
        if not isinstance(input_data, dict):
            return False, "Input must be a dictionary for rule-based validation"
        
        # Check required fields
        for field, rule in rules.items():
            if rule.get('required', False) and field not in input_data:
                return False, f"Missing required field: {field}"
        
        # Validate each field
        for field, value in input_data.items():
            if field in rules:
                rule = rules[field]
                is_valid, error = self._validate_field(value, rule, context)
                if not is_valid:
                    return False, f"Field '{field}': {error}"
        
        return True, None
    
    def _validate_field(self, value: Any, rule: Dict, context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate a single field against its rule"""
        # Type validation
        expected_type = rule.get('type', 'any')
        if expected_type != 'any':
            type_valid, type_error = self._validate_type(value, expected_type)
            if not type_valid:
                return False, type_error
        
        # Value constraints
        if 'min' in rule and isinstance(value, (int, float)):
            if value < rule['min']:
                return False, f"Value must be at least {rule['min']}"
        
        if 'max' in rule and isinstance(value, (int, float)):
            if value > rule['max']:
                return False, f"Value must be at most {rule['max']}"
        
        if 'min_length' in rule and isinstance(value, str):
            if len(value) < rule['min_length']:
                return False, f"Length must be at least {rule['min_length']} characters"
        
        if 'max_length' in rule and isinstance(value, str):
            if len(value) > rule['max_length']:
                return False, f"Length must be at most {rule['max_length']} characters"
        
        # Pattern validation
        if 'pattern' in rule and isinstance(value, str):
            if not re.match(rule['pattern'], value):
                return False, "Value does not match required pattern"
        
        # Custom validation
        if 'validator' in rule and callable(rule['validator']):
            is_valid, error = rule['validator'](value, context)
            if not is_valid:
                return False, error
        
        return True, None
    
    def _validate_type(self, value: Any, expected_type: str) -> Tuple[bool, Optional[str]]:
        """Validate value type"""
        type_map = {
            'string': str,
            'numeric': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
            'eth_address': str,
            'hex_string': str,
            'email': str,
            'url': str,
            'ip_address': str,
            'domain': str
        }
        
        if expected_type in type_map:
            expected = type_map[expected_type]
            if not isinstance(value, expected):
                return False, f"Expected type {expected_type}, got {type(value).__name__}"
            
            # Additional pattern validation for specific types
            if expected_type in self.patterns:
                if not self.patterns[expected_type].match(str(value)):
                    return False, f"Invalid {expected_type} format"
        
        return True, None
    
    def _validate_by_type(self, input_data: Any, context: Dict) -> Tuple[bool, Optional[str]]:
        """Default validation based on input data type"""
        if input_data is None:
            return False, "Input cannot be null"
        
        if isinstance(input_data, dict):
            return self._validate_dict(input_data, context)
        
        if isinstance(input_data, list):
            return self._validate_array(input_data, context)
        
        if isinstance(input_data, str):
            return self._validate_string(input_data, context)
        
        if isinstance(input_data, (int, float)):
            return self._validate_number(input_data, context)
        
        if isinstance(input_data, bool):
            return True, None
        
        return False, f"Unsupported input type: {type(input_data).__name__}"
    
    def _validate_dict(self, data: Dict, context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate dictionary input"""
        # Check for excessive size
        if len(data) > 1000:
            return False, "Dictionary too large (max 1000 items)"
        
        # Validate each key-value pair
        for key, value in data.items():
            if not isinstance(key, str):
                return False, "Dictionary keys must be strings"
            
            if len(key) > 256:
                return False, f"Key '{key}' too long (max 256 characters)"
            
            is_valid, error = self._validate_by_type(value, context)
            if not is_valid:
                return False, f"Value for key '{key}': {error}"
        
        return True, None
    
    def _validate_array(self, data: list, context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate array input"""
        # Check for excessive size
        if len(data) > 10000:
            return False, "Array too large (max 10000 items)"
        
        # Validate each item
        for i, item in enumerate(data):
            is_valid, error = self._validate_by_type(item, context)
            if not is_valid:
                return False, f"Item {i}: {error}"
        
        return True, None
    
    def _validate_string(self, data: str, context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate string input"""
        # Check length
        if len(data) > 1024 * 1024:  # 1MB
            return False, "String too long (max 1MB)"
        
        # Check for suspicious patterns
        if self._contains_suspicious_pattern(data):
            return False, "String contains suspicious patterns"
        
        return True, None
    
    def _validate_number(self, data: (int, float), context: Dict) -> Tuple[bool, Optional[str]]:
        """Validate number input"""
        # Check for extreme values
        if abs(data) > 1e18:  # Very large numbers
            return False, "Number too large"
        
        # Check for NaN or infinity
        if isinstance(data, float) and (data != data or abs(data) == float('inf')):
            return False, "Invalid number value"
        
        return True, None
    
    def _contains_suspicious_pattern(self, data: str) -> bool:
        """Check for suspicious patterns in string"""
        suspicious_patterns = [
            r'<script.*?>',  # Script tags
            r'javascript:',   # JavaScript protocol
            r'on\w+=',        # Event handlers
            r'\.\./',         # Directory traversal
            r'\\x[0-9a-fA-F]{2}',  # Hex encoded characters
            r'%[0-9a-fA-F]{2}',    # URL encoded characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return True
        
        return False
    
    def add_validation_rule(self, context: str, rules: Dict) -> None:
        """Add custom validation rules for a context"""
        self.validation_rules[context] = rules
        logger.info(f"Added validation rules for context: {context}")
    
    def remove_validation_rule(self, context: str) -> bool:
        """Remove validation rules for a context"""
        if context in self.validation_rules:
            del self.validation_rules[context]
            logger.info(f"Removed validation rules for context: {context}")
            return True
        return False
    
    def get_validation_rules(self, context: str) -> Optional[Dict]:
        """Get validation rules for a context"""
        return self.validation_rules.get(context)