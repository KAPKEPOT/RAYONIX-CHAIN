# blockchain/validation/validation_manager.py
import time
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict

from blockchain.models.validation import ValidationResult, ValidationLevel
from blockchain.models.block import Block
from blockchain.models.transaction import Transaction

logger = logging.getLogger(__name__)

class ValidationManager:
    """Modular validation system with validation pipelines"""
    
    def __init__(self, state_manager: Any, config: Dict[str, Any]):
        self.state_manager = state_manager
        self.config = config
        self.validation_pipelines = self._initialize_pipelines()
        self.cache = {}  # Simple cache for validation results
        self.validation_stats = defaultdict(int)
        self.error_stats = defaultdict(int)
        self.performance_stats = defaultdict(list)
    
    def _initialize_pipelines(self) -> Dict[ValidationLevel, List[Tuple[str, Callable]]]:
        """Initialize all validation pipelines"""
        return {
            ValidationLevel.BASIC: self._create_basic_pipeline(),
            ValidationLevel.STANDARD: self._create_standard_pipeline(),
            ValidationLevel.FULL: self._create_full_pipeline(),
            ValidationLevel.CONSENSUS: self._create_consensus_pipeline()
        }
    
    def validate_block(self, block: Block, level: ValidationLevel = ValidationLevel.FULL) -> ValidationResult:
        """Validate a block with specified validation level"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"block_{block.hash}_{level.name}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if not cached_result.is_stale(300):  # 5 minute cache
                return cached_result
        
        errors = []
        warnings = []
        validated_components = []
        
        try:
            pipeline = self.validation_pipelines[level]
            
            for validator_name, validator_func in pipeline:
                try:
                    component_start = time.time()
                    result = validator_func(block)
                    component_time = time.time() - component_start
                    
                    self.performance_stats[validator_name].append(component_time)
                    
                    if not result['valid']:
                        errors.extend(result['errors'])
                    if result['warnings']:
                        warnings.extend(result['warnings'])
                    
                    validated_components.append(validator_name)
                    
                    if errors and level != ValidationLevel.CONSENSUS:
                        break  # Early exit for non-consensus validation
                        
                except Exception as e:
                    error_msg = f"{validator_name} failed: {str(e)}"
                    errors.append(error_msg)
                    self.error_stats[validator_name] += 1
                    logger.error(error_msg, exc_info=True)
                    break
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            error_msg = f"Validation pipeline failed: {str(e)}"
            errors.append(error_msg)
            is_valid = False
            logger.error(error_msg, exc_info=True)
        
        execution_time = time.time() - start_time
        self.validation_stats[level.name] += 1
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            execution_time=execution_time,
            validation_level=level,
            validated_components=validated_components,
            block_hash=block.hash,
            block_height=block.header.height,
            validator_address=block.header.validator
        )
        
        # Cache result
        self.cache[cache_key] = result
        if len(self.cache) > 10000:  # Limit cache size
            self._clean_cache()
        
        return result
    
    def validate_transaction(self, transaction: Transaction, level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationResult:
        """Validate a transaction with specified validation level"""
        start_time = time.time()
        
        # Check cache first
        cache_key = f"tx_{transaction.hash}_{level.name}"
        if cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if not cached_result.is_stale(180):  # 3 minute cache for transactions
                return cached_result
        
        errors = []
        warnings = []
        validated_components = []
        
        try:
            # Basic validation
            basic_result = self._validate_transaction_structure(transaction)
            if not basic_result['valid']:
                errors.extend(basic_result['errors'])
            if basic_result['warnings']:
                warnings.extend(basic_result['warnings'])
            validated_components.append('structure')
            
            if errors:
                is_valid = False
            else:
                # Signature validation
                sig_result = self._validate_transaction_signature(transaction)
                if not sig_result['valid']:
                    errors.extend(sig_result['errors'])
                validated_components.append('signature')
                
                if level in [ValidationLevel.FULL, ValidationLevel.CONSENSUS]:
                    # State-dependent validation
                    state_result = self._validate_transaction_state(transaction)
                    if not state_result['valid']:
                        errors.extend(state_result['errors'])
                    if state_result['warnings']:
                        warnings.extend(state_result['warnings'])
                    validated_components.append('state')
            
            is_valid = len(errors) == 0
            
        except Exception as e:
            error_msg = f"Transaction validation failed: {str(e)}"
            errors.append(error_msg)
            is_valid = False
            logger.error(error_msg, exc_info=True)
        
        execution_time = time.time() - start_time
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            execution_time=execution_time,
            validation_level=level,
            validated_components=validated_components
        )
        
        # Cache result
        self.cache[cache_key] = result
        if len(self.cache) > 10000:
            self._clean_cache()
        
        return result
    
    def _create_basic_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create basic validation pipeline"""
        return [
            ('block_structure', self._validate_block_structure),
            ('block_hash', self._validate_block_hash),
            ('previous_block', self._validate_previous_block)
        ]
    
    def _create_standard_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create standard validation pipeline"""
        pipeline = self._create_basic_pipeline()
        pipeline.extend([
            ('merkle_root', self._validate_merkle_root),
            ('timestamp', self._validate_timestamp),
            ('difficulty', self._validate_difficulty),
            ('signature', self._validate_signature)
        ])
        return pipeline
    
    def _create_full_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create full validation pipeline"""
        pipeline = self._create_standard_pipeline()
        pipeline.extend([
            ('transactions_basic', self._validate_transactions_basic),
            ('gas_limit', self._validate_gas_limit),
            ('block_size', self._validate_block_size)
        ])
        return pipeline
    
    def _create_consensus_pipeline(self) -> List[Tuple[str, Callable]]:
        """Create consensus-level validation pipeline"""
        pipeline = self._create_full_pipeline()
        pipeline.extend([
            ('transactions_full', self._validate_transactions_full),
            ('state_transition', self._validate_state_transition),
            ('consensus_rules', self._validate_consensus_rules)
        ])
        return pipeline
    
    def _validate_block_structure(self, block: Block) -> Dict[str, Any]:
        """Validate block structure"""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = ['version', 'height', 'previous_hash', 'merkle_root', 
                          'timestamp', 'difficulty', 'nonce', 'validator']
        
        for field in required_fields:
            if not hasattr(block.header, field) or getattr(block.header, field) is None:
                errors.append(f"Missing required field: {field}")
        
        # Check block size
        if block.size > self.config['max_block_size']:
            errors.append(f"Block size {block.size} exceeds maximum {self.config['max_block_size']}")
        
        # Check version compatibility
        if block.header.version not in [1, 2, 3]:
            warnings.append(f"Unusual block version: {block.header.version}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_block_hash(self, block: Block) -> Dict[str, Any]:
        """Validate block hash"""
        errors = []
        
        if not block.verify_hash():
            errors.append("Block hash verification failed")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_previous_block(self, block: Block) -> Dict[str, Any]:
        """Validate previous block reference"""
        errors = []
        
        try:
            previous_block = self.state_manager.database.get_block(block.header.previous_hash)
            if not previous_block:
                errors.append(f"Previous block not found: {block.header.previous_hash}")
            elif previous_block.header.height != block.header.height - 1:
                errors.append(f"Height mismatch with previous block")
                
        except Exception as e:
            errors.append(f"Error validating previous block: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_merkle_root(self, block: Block) -> Dict[str, Any]:
        """Validate merkle root"""
        errors = []
        
        try:
            from blockchain.utils.merkle import MerkleTree
            
            tx_hashes = [tx.hash for tx in block.transactions]
            calculated_root = MerkleTree(tx_hashes).get_root_hash()
            
            if calculated_root != block.header.merkle_root:
                errors.append(f"Invalid merkle root. Calculated: {calculated_root}, Expected: {block.header.merkle_root}")
                
        except Exception as e:
            errors.append(f"Error calculating merkle root: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_timestamp(self, block: Block) -> Dict[str, Any]:
        """Validate block timestamp"""
        errors = []
        warnings = []
        
        current_time = time.time()
        max_future_time = current_time + self.config['max_future_block_time']
        
        if block.header.timestamp > max_future_time:
            errors.append(f"Block timestamp is too far in the future")
        elif block.header.timestamp < current_time - self.config['max_past_block_time']:
            warnings.append("Block timestamp is very old")
        
        # Check if timestamp is reasonable compared to previous block
        try:
            previous_block = self.state_manager.database.get_block(block.header.previous_hash)
            if previous_block and block.header.timestamp < previous_block.header.timestamp:
                errors.append("Block timestamp is earlier than previous block")
        except:
            pass  # Skip if previous block not available
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_difficulty(self, block: Block) -> Dict[str, Any]:
        """Validate block difficulty"""
        errors = []
        
        try:
            expected_difficulty = self.state_manager.consensus.calculate_difficulty(block.header.height)
            if block.header.difficulty != expected_difficulty:
                errors.append(f"Invalid difficulty. Expected: {expected_difficulty}, Got: {block.header.difficulty}")
                
        except Exception as e:
            errors.append(f"Error calculating difficulty: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_signature(self, block: Block) -> Dict[str, Any]:
        """Validate block signature"""
        errors = []
        
        try:
            if not self.state_manager.consensus.validate_block_signature(block):
                errors.append("Invalid block signature")
                
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_transactions_basic(self, block: Block) -> Dict[str, Any]:
        """Basic transaction validation"""
        errors = []
        warnings = []
        
        for tx in block.transactions:
            result = self.validate_transaction(tx, ValidationLevel.BASIC)
            if not result.is_valid:
                errors.extend([f"Transaction {tx.hash}: {e}" for e in result.errors])
            if result.warnings:
                warnings.extend([f"Transaction {tx.hash}: {w}" for w in result.warnings])
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_transactions_full(self, block: Block) -> Dict[str, Any]:
        """Full transaction validation including state checks"""
        errors = []
        warnings = []
        
        for tx in block.transactions:
            result = self.validate_transaction(tx, ValidationLevel.FULL)
            if not result.is_valid:
                errors.extend([f"Transaction {tx.hash}: {e}" for e in result.errors])
            if result.warnings:
                warnings.extend([f"Transaction {tx.hash}: {w}" for w in result.warnings])
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_state_transition(self, block: Block) -> Dict[str, Any]:
        """Validate state transition caused by block"""
        errors = []
        
        try:
            # Create temporary state manager for validation
            temp_state = self.state_manager.__class__(
                self.state_manager.database,
                self.state_manager.utxo_set.__class__(),
                self.state_manager.consensus.__class__(),
                self.state_manager.contract_manager.__class__()
            )
            
            # Copy current state
            temp_state.utxo_set.restore(self.state_manager.utxo_set.snapshot())
            temp_state.consensus.restore(self.state_manager.consensus.snapshot())
            temp_state.contract_manager.restore(self.state_manager.contract_manager.snapshot())
            
            # Try to apply block
            if not temp_state.apply_block(block):
                errors.append("State transition validation failed")
                
        except Exception as e:
            errors.append(f"State transition error: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_consensus_rules(self, block: Block) -> Dict[str, Any]:
        """Validate consensus-specific rules"""
        errors = []
        
        try:
            if not self.state_manager.consensus.validate_block_consensus(block):
                errors.append("Block violates consensus rules")
                
        except Exception as e:
            errors.append(f"Consensus validation error: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_transaction_structure(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate transaction structure"""
        errors = []
        warnings = []
        
        # Check required fields
        if not transaction.inputs:
            errors.append("Transaction has no inputs")
        if not transaction.outputs:
            errors.append("Transaction has no outputs")
        if not transaction.hash:
            errors.append("Transaction has no hash")
        
        # Check size limits
        tx_size = len(transaction.to_bytes())
        if tx_size > self.config['max_transaction_size']:
            errors.append(f"Transaction size {tx_size} exceeds maximum {self.config['max_transaction_size']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_transaction_signature(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate transaction signature"""
        errors = []
        
        try:
            if not transaction.verify_signature():
                errors.append("Invalid transaction signature")
                
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _validate_transaction_state(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate transaction against current state"""
        errors = []
        warnings = []
        
        try:
            # Check if inputs are available and valid
            for tx_input in transaction.inputs:
                utxo = self.state_manager.utxo_set.get_utxo(tx_input.tx_hash, tx_input.output_index)
                if not utxo:
                    errors.append(f"Input not found: {tx_input.tx_hash}:{tx_input.output_index}")
                    continue
                
                # Check if UTXO is spent
                if utxo.spent:
                    errors.append(f"Input already spent: {tx_input.tx_hash}:{tx_input.output_index}")
            
            # Check sufficient funds
            input_sum = sum(inp.amount for inp in transaction.inputs)
            output_sum = sum(out.amount for out in transaction.outputs)
            fee = input_sum - output_sum
            
            if fee < self.config['min_transaction_fee']:
                errors.append(f"Insufficient fee: {fee} < {self.config['min_transaction_fee']}")
            
        except Exception as e:
            errors.append(f"Error validating transaction state: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_gas_limit(self, block: Block) -> Dict[str, Any]:
        """Validate block gas limit"""
        errors = []
        warnings = []
        
        try:
            total_gas = sum(tx.gas_limit for tx in block.transactions if hasattr(tx, 'gas_limit'))
            if total_gas > self.config['max_block_gas']:
                errors.append(f"Block gas limit exceeded: {total_gas} > {self.config['max_block_gas']}")
                
        except Exception as e:
            errors.append(f"Error calculating gas limit: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_block_size(self, block: Block) -> Dict[str, Any]:
        """Validate block size"""
        errors = []
        
        if block.size > self.config['max_block_size']:
            errors.append(f"Block size {block.size} exceeds maximum {self.config['max_block_size']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def _clean_cache(self):
        """Clean up the validation cache"""
        # Remove stale entries and limit size
        current_time = time.time()
        new_cache = {}
        
        for key, result in self.cache.items():
            if not result.is_stale(300):  # Keep only non-stale entries
                new_cache[key] = result
            
            if len(new_cache) >= 5000:  # Limit size
                break
        
        self.cache = new_cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        avg_performance = {}
        for validator, times in self.performance_stats.items():
            if times:
                avg_performance[validator] = sum(times) / len(times)
        
        return {
            'validation_counts': dict(self.validation_stats),
            'error_counts': dict(self.error_stats),
            'average_performance': avg_performance,
            'cache_size': len(self.cache)
        }
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = defaultdict(int)
        self.error_stats = defaultdict(int)
        self.performance_stats = defaultdict(list)