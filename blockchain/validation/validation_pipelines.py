# blockchain/validation/validation_pipelines.py
from typing import Dict, List, Any, Callable, Tuple
from enum import Enum

from blockchain.models.validation import ValidationLevel
from blockchain.models.block import Block

class PipelineType(Enum):
    BLOCK_VALIDATION = "block_validation"
    TRANSACTION_VALIDATION = "transaction_validation"
    CONSENSUS_VALIDATION = "consensus_validation"

class ValidationPipelineFactory:
    """Factory for creating validation pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline_registry = self._initialize_registry()
    
    def _initialize_registry(self) -> Dict[PipelineType, Dict[ValidationLevel, List[Tuple[str, Callable]]]]:
        """Initialize the pipeline registry"""
        return {
            PipelineType.BLOCK_VALIDATION: {
                ValidationLevel.BASIC: self._create_basic_block_pipeline(),
                ValidationLevel.STANDARD: self._create_standard_block_pipeline(),
                ValidationLevel.FULL: self._create_full_block_pipeline(),
                ValidationLevel.CONSENSUS: self._create_consensus_block_pipeline()
            },
            PipelineType.TRANSACTION_VALIDATION: {
                ValidationLevel.BASIC: self._create_basic_transaction_pipeline(),
                ValidationLevel.STANDARD: self._create_standard_transaction_pipeline(),
                ValidationLevel.FULL: self._create_full_transaction_pipeline(),
                ValidationLevel.CONSENSUS: self._create_consensus_transaction_pipeline()
            }
        }
    
    def get_pipeline(self, pipeline_type: PipelineType, level: ValidationLevel) -> List[Tuple[str, Callable]]:
        """Get a validation pipeline for the specified type and level"""
        return self.pipeline_registry.get(pipeline_type, {}).get(level, [])
    
    def _create_basic_block_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create basic block validation pipeline"""
        return [
            ('block_structure', self._validate_block_structure),
            ('block_hash', self._validate_block_hash),
            ('previous_block', self._validate_previous_block)
        ]
    
    def _create_standard_block_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create standard block validation pipeline"""
        pipeline = self._create_basic_block_pipeline()
        pipeline.extend([
            ('merkle_root', self._validate_merkle_root),
            ('timestamp', self._validate_timestamp),
            ('difficulty', self._validate_difficulty),
            ('signature', self._validate_signature)
        ])
        return pipeline
    
    def _create_full_block_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create full block validation pipeline"""
        pipeline = self._create_standard_block_pipeline()
        pipeline.extend([
            ('transactions_basic', self._validate_transactions_basic),
            ('gas_limit', self._validate_gas_limit),
            ('block_size', self._validate_block_size)
        ])
        return pipeline
    
    def _create_consensus_block_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create consensus-level block validation pipeline"""
        pipeline = self._create_full_block_pipeline()
        pipeline.extend([
            ('transactions_full', self._validate_transactions_full),
            ('state_transition', self._validate_state_transition),
            ('consensus_rules', self._validate_consensus_rules)
        ])
        return pipeline
    
    def _create_basic_transaction_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create basic transaction validation pipeline"""
        return [
            ('transaction_structure', self._validate_transaction_structure),
            ('signature_basic', self._validate_signature_basic)
        ]
    
    def _create_standard_transaction_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create standard transaction validation pipeline"""
        pipeline = self._create_basic_transaction_pipeline()
        pipeline.extend([
            ('signature_full', self._validate_signature_full),
            ('inputs_outputs', self._validate_inputs_outputs)
        ])
        return pipeline
    
    def _create_full_transaction_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create full transaction validation pipeline"""
        pipeline = self._create_standard_transaction_pipeline()
        pipeline.extend([
            ('fee_validation', self._validate_fee),
            ('double_spend_check', self._validate_double_spend)
        ])
        return pipeline
    
    def _create_consensus_transaction_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create consensus-level transaction validation pipeline"""
        pipeline = self._create_full_transaction_pipeline()
        pipeline.extend([
            ('state_dependent_checks', self._validate_state_dependent),
            ('contract_validation', self._validate_contract)
        ])
        return pipeline
    
    # Placeholder validator functions - these would be implemented in validators.py
    def _validate_block_structure(self, block: Block) -> Dict[str, Any]:
        return {'valid': True, 'errors': [], 'warnings': []}
    
    def _validate_block_hash(self, block: Block) -> Dict[str, Any]:
        return {'valid': True, 'errors': [], 'warnings': []}
    
    # ... other placeholder validator functions
    
    def register_validator(self, pipeline_type: PipelineType, level: ValidationLevel, 
                          validator_name: str, validator_func: Callable):
        """Register a custom validator function"""
        if pipeline_type not in self.pipeline_registry:
            self.pipeline_registry[pipeline_type] = {}
        
        if level not in self.pipeline_registry[pipeline_type]:
            self.pipeline_registry[pipeline_type][level] = []
        
        self.pipeline_registry[pipeline_type][level].append((validator_name, validator_func))
    
    def unregister_validator(self, pipeline_type: PipelineType, level: ValidationLevel, 
                            validator_name: str):
        """Unregister a validator function"""
        if (pipeline_type in self.pipeline_registry and 
            level in self.pipeline_registry[pipeline_type]):
            
            pipeline = self.pipeline_registry[pipeline_type][level]
            self.pipeline_registry[pipeline_type][level] = [
                (name, func) for name, func in pipeline if name != validator_name
            ]
    
    def create_custom_pipeline(self, validators: List[Tuple[str, Callable]]) -> List[Tuple[str, Callable]]:
        """Create a custom validation pipeline"""
        return validators
    
    def validate_with_pipeline(self, pipeline: List[Tuple[str, Callable]], 
                              target: Any, *args, **kwargs) -> Dict[str, Any]:
        """Execute a validation pipeline on a target"""
        errors = []
        warnings = []
        
        for validator_name, validator_func in pipeline:
            try:
                result = validator_func(target, *args, **kwargs)
                if not result['valid']:
                    errors.extend(result['errors'])
                if result['warnings']:
                    warnings.extend(result['warnings'])
            except Exception as e:
                errors.append(f"Validator {validator_name} failed: {str(e)}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }