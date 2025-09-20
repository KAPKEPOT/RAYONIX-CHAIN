# blockchain/validation/block_validators.py
import time
from typing import Dict, Any, List

from blockchain.models.block import Block
from blockchain.utils.merkle import MerkleTree

class BlockValidators:
    """Collection of block validation functions"""
    
    def __init__(self, config: Dict[str, Any], database: Any, consensus: Any):
        self.config = config
        self.database = database
        self.consensus = consensus
    
    def validate_structure(self, block: Block) -> Dict[str, Any]:
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
    
    def validate_hash(self, block: Block) -> Dict[str, Any]:
        """Validate block hash"""
        errors = []
        
        if not block.verify_hash():
            errors.append("Block hash verification failed")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_previous_block(self, block: Block) -> Dict[str, Any]:
        """Validate previous block reference"""
        errors = []
        
        try:
            previous_block = self.database.get_block(block.header.previous_hash)
            if not previous_block:
                errors.append(f"Previous block not found: {block.header.previous_hash}")
            elif previous_block.header.height != block.header.height - 1:
                errors.append(f"Height mismatch with previous block")
                
        except Exception as e:
            errors.append(f"Error validating previous block: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_merkle_root(self, block: Block) -> Dict[str, Any]:
        """Validate merkle root"""
        errors = []
        
        try:
            tx_hashes = [tx.hash for tx in block.transactions]
            calculated_root = MerkleTree(tx_hashes).get_root_hash()
            
            if calculated_root != block.header.merkle_root:
                errors.append(f"Invalid merkle root. Calculated: {calculated_root}, Expected: {block.header.merkle_root}")
                
        except Exception as e:
            errors.append(f"Error calculating merkle root: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_timestamp(self, block: Block) -> Dict[str, Any]:
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
            previous_block = self.database.get_block(block.header.previous_hash)
            if previous_block and block.header.timestamp < previous_block.header.timestamp:
                errors.append("Block timestamp is earlier than previous block")
        except:
            pass  # Skip if previous block not available
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def validate_difficulty(self, block: Block) -> Dict[str, Any]:
        """Validate block difficulty"""
        errors = []
        
        try:
            expected_difficulty = self.consensus.calculate_difficulty(block.header.height)
            if block.header.difficulty != expected_difficulty:
                errors.append(f"Invalid difficulty. Expected: {expected_difficulty}, Got: {block.header.difficulty}")
                
        except Exception as e:
            errors.append(f"Error calculating difficulty: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_signature(self, block: Block) -> Dict[str, Any]:
        """Validate block signature"""
        errors = []
        
        try:
            if not self.consensus.validate_block_signature(block):
                errors.append("Invalid block signature")
                
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_gas_limit(self, block: Block) -> Dict[str, Any]:
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
    
    def validate_block_size(self, block: Block) -> Dict[str, Any]:
        """Validate block size"""
        errors = []
        
        if block.size > self.config['max_block_size']:
            errors.append(f"Block size {block.size} exceeds maximum {self.config['max_block_size']}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_consensus_rules(self, block: Block) -> Dict[str, Any]:
        """Validate consensus-specific rules"""
        errors = []
        
        try:
            if not self.consensus.validate_block_consensus(block):
                errors.append("Block violates consensus rules")
                
        except Exception as e:
            errors.append(f"Consensus validation error: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}