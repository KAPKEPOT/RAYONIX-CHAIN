# blockchain/validation/transaction_validators.py
from typing import Dict, Any, List

from utxo_system.models.transaction import Transaction

class TransactionValidators:
    """Collection of transaction validation functions"""
    
    def __init__(self, config: Dict[str, Any], database: Any, utxo_set: Any):
        self.config = config
        self.database = database
        self.utxo_set = utxo_set
    
    def validate_structure(self, transaction: Transaction) -> Dict[str, Any]:
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
    
    def validate_signature_basic(self, transaction: Transaction) -> Dict[str, Any]:
        """Basic signature validation"""
        errors = []
        
        try:
            if not transaction.verify_signature():
                errors.append("Invalid transaction signature")
                
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_signature_full(self, transaction: Transaction) -> Dict[str, Any]:
        """Full signature validation with additional checks"""
        errors = []
        warnings = []
        
        try:
            # Basic signature check
            if not transaction.verify_signature():
                errors.append("Invalid transaction signature")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            # Additional signature checks
            if hasattr(transaction, 'signature_type'):
                if transaction.signature_type not in ['ECDSA', 'Schnorr']:
                    warnings.append(f"Unusual signature type: {transaction.signature_type}")
            
            # Check signature replay protection
            if hasattr(transaction, 'replay_protection'):
                if not transaction.replay_protection:
                    warnings.append("Transaction lacks replay protection")
            
        except Exception as e:
            errors.append(f"Error validating signature: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def validate_inputs_outputs(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate transaction inputs and outputs"""
        errors = []
        warnings = []
        
        try:
            # Check input and output counts
            if len(transaction.inputs) == 0:
                errors.append("Transaction has no inputs")
            if len(transaction.outputs) == 0:
                errors.append("Transaction has no outputs")
            
            if errors:
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            # Check for dust outputs
            for i, output in enumerate(transaction.outputs):
                if output.amount < self.config.get('dust_threshold', 546):
                    warnings.append(f"Output {i} is below dust threshold: {output.amount}")
            
            # Check input amounts
            input_sum = sum(inp.amount for inp in transaction.inputs)
            output_sum = sum(out.amount for out in transaction.outputs)
            
            if input_sum < output_sum:
                errors.append(f"Insufficient inputs: {input_sum} < {output_sum}")
            
        except Exception as e:
            errors.append(f"Error validating inputs/outputs: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def validate_fee(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate transaction fee"""
        errors = []
        warnings = []
        
        try:
            input_sum = sum(inp.amount for inp in transaction.inputs)
            output_sum = sum(out.amount for out in transaction.outputs)
            fee = input_sum - output_sum
            
            if fee < 0:
                errors.append("Negative fee calculated")
                return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            min_fee = self.config['min_transaction_fee']
            if fee < min_fee:
                errors.append(f"Insufficient fee: {fee} < {min_fee}")
            
            # Check fee rate
            tx_size = len(transaction.to_bytes())
            fee_rate = fee / tx_size if tx_size > 0 else 0
            
            min_fee_rate = self.config.get('min_fee_rate', 1)
            if fee_rate < min_fee_rate:
                warnings.append(f"Low fee rate: {fee_rate:.2f} sat/byte < {min_fee_rate} sat/byte")
            
        except Exception as e:
            errors.append(f"Error validating fee: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def validate_double_spend(self, transaction: Transaction) -> Dict[str, Any]:
        """Check for double spends"""
        errors = []
        
        try:
            for tx_input in transaction.inputs:
                utxo = self.utxo_set.get_utxo(tx_input.tx_hash, tx_input.output_index)
                if utxo and utxo.spent:
                    errors.append(f"Input already spent: {tx_input.tx_hash}:{tx_input.output_index}")
        
        except Exception as e:
            errors.append(f"Error checking double spend: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': []}
    
    def validate_state_dependent(self, transaction: Transaction) -> Dict[str, Any]:
        """State-dependent transaction validation"""
        errors = []
        warnings = []
        
        try:
            # Check if inputs are available
            for tx_input in transaction.inputs:
                utxo = self.utxo_set.get_utxo(tx_input.tx_hash, tx_input.output_index)
                if not utxo:
                    errors.append(f"Input not found: {tx_input.tx_hash}:{tx_input.output_index}")
                    continue
                
                # Check if UTXO is locked
                if utxo.locktime > 0 and utxo.locktime > time.time():
                    errors.append(f"Input locked until {utxo.locktime}: {tx_input.tx_hash}:{tx_input.output_index}")
            
            # Check output scripts
            for i, output in enumerate(transaction.outputs):
                if not output.script_pubkey:
                    warnings.append(f"Output {i} has no script pubkey")
            
        except Exception as e:
            errors.append(f"Error in state-dependent validation: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def validate_contract(self, transaction: Transaction) -> Dict[str, Any]:
        """Validate contract-related transactions"""
        errors = []
        warnings = []
        
        try:
            if transaction.is_contract_call():
                # Check contract existence
                contract_address = transaction.get_contract_address()
                if not contract_address:
                    errors.append("Contract call without contract address")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}
                
                # Check contract code (would be implemented with actual contract system)
                if not self._validate_contract_code(contract_address):
                    warnings.append(f"Contract validation skipped for {contract_address}")
                
                # Check gas limits
                if hasattr(transaction, 'gas_limit'):
                    if transaction.gas_limit > self.config.get('max_transaction_gas', 10000000):
                        errors.append(f"Gas limit too high: {transaction.gas_limit}")
            
        except Exception as e:
            errors.append(f"Error validating contract: {str(e)}")
        
        return {'valid': len(errors) == 0, 'errors': errors, 'warnings': warnings}
    
    def _validate_contract_code(self, contract_address: str) -> bool:
        """Validate contract code (placeholder implementation)"""
        # This would be implemented with actual contract validation logic
        return True