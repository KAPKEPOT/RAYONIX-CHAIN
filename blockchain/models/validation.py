# blockchain/models/validation.py
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum, auto
import time

class ValidationLevel(Enum):
    BASIC = auto()
    STANDARD = auto()
    FULL = auto()
    CONSENSUS = auto()
    
    def get_required_validations(self) -> List[str]:
        """Get list of required validation types for this level"""
        if self == ValidationLevel.BASIC:
            return ['structure', 'hash', 'previous_block']
        elif self == ValidationLevel.STANDARD:
            return ['structure', 'hash', 'previous_block', 'merkle_root', 'timestamp', 'difficulty', 'signature']
        elif self == ValidationLevel.FULL:
            return ['structure', 'hash', 'previous_block', 'merkle_root', 'timestamp', 'difficulty', 'signature', 
                   'transactions_basic', 'gas_limit', 'block_size']
        elif self == ValidationLevel.CONSENSUS:
            return ['structure', 'hash', 'previous_block', 'merkle_root', 'timestamp', 'difficulty', 'signature',
                   'transactions_basic', 'gas_limit', 'block_size', 'transactions_full', 'state_transition', 'consensus_rules']
        else:
            return ['structure', 'hash', 'previous_block']

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    execution_time: float
    validation_level: ValidationLevel
    validated_components: List[str] = field(default_factory=list)
    block_hash: Optional[str] = None
    block_height: Optional[int] = None
    validator_address: Optional[str] = None
    signature_valid: Optional[bool] = None
    state_transition_valid: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary"""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'execution_time': self.execution_time,
            'validation_level': self.validation_level.name,
            'validated_components': self.validated_components,
            'block_hash': self.block_hash,
            'block_height': self.block_height,
            'validator_address': self.validator_address,
            'signature_valid': self.signature_valid,
            'state_transition_valid': self.state_transition_valid,
            'timestamp': time.time()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationResult':
        """Create validation result from dictionary"""
        return cls(
            is_valid=data['is_valid'],
            errors=data['errors'],
            warnings=data['warnings'],
            execution_time=data['execution_time'],
            validation_level=ValidationLevel[data['validation_level']],
            validated_components=data.get('validated_components', []),
            block_hash=data.get('block_hash'),
            block_height=data.get('block_height'),
            validator_address=data.get('validator_address'),
            signature_valid=data.get('signature_valid'),
            state_transition_valid=data.get('state_transition_valid')
        )
    
    def add_error(self, error: str, component: str = "unknown"):
        """Add an error to the validation result"""
        self.errors.append(error)
        if component not in self.validated_components:
            self.validated_components.append(component)
        self.is_valid = False
    
    def add_warning(self, warning: str, component: str = "unknown"):
        """Add a warning to the validation result"""
        self.warnings.append(warning)
        if component not in self.validated_components:
            self.validated_components.append(component)
    
    def add_component_validation(self, component: str, is_valid: bool, message: Optional[str] = None):
        """Add component validation result"""
        if component not in self.validated_components:
            self.validated_components.append(component)
        
        if not is_valid and message:
            self.add_error(message, component)
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.validated_components.extend([c for c in other.validated_components if c not in self.validated_components])
        
        # Keep the most restrictive validation level
        if other.validation_level.value > self.validation_level.value:
            self.validation_level = other.validation_level
        
        # Update validity
        if not other.is_valid:
            self.is_valid = False
        
        return self