# blockchain/utils/block_calculations.py
import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import time
from dataclasses import asdict
from decimal import Decimal, ROUND_DOWN
import secrets


class BlockCalculations:
    """Advanced blockchain calculations with comprehensive error handling and optimization"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = self.config.get('cache_ttl', 300)  # 5 minutes default
        
    def calculate_block_hash(self, header_data: Dict[str, Any], 
                           algorithm: str = 'sha256') -> str:
        """Calculate block hash with multiple algorithm support and caching"""
        cache_key = f"block_hash_{hash(frozenset(header_data.items()))}_{algorithm}"
        
        if cache_key in self._cache:
            cached_value, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_value
        
        try:
            # Normalize and validate header data
            validated_data = self._validate_header_data(header_data)
            
            # Use compact serialization for performance
            header_str = json.dumps(validated_data, sort_keys=True, separators=(',', ':'))
            
            if algorithm == 'sha256':
                hash_result = hashlib.sha256(header_str.encode()).hexdigest()
            elif algorithm == 'sha3_256':
                hash_result = hashlib.sha3_256(header_str.encode()).hexdigest()
            elif algorithm == 'blake2s':
                hash_result = hashlib.blake2s(header_str.encode()).hexdigest()
            elif algorithm == 'double_sha256':
                first_hash = hashlib.sha256(header_str.encode()).digest()
                hash_result = hashlib.sha256(first_hash).hexdigest()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            # Cache the result
            self._cache[cache_key] = (hash_result, time.time())
            
            return hash_result
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate block hash: {e}") from e
    
    def calculate_merkle_root(self, transaction_hashes: List[str], 
                            algorithm: str = 'double_sha256',
                            enable_optimization: bool = True) -> str:
        """Advanced merkle root calculation with multiple algorithms and optimizations"""
        if not transaction_hashes:
            return '0' * 64
        
        cache_key = f"merkle_root_{hash(tuple(transaction_hashes))}_{algorithm}"
        if enable_optimization and cache_key in self._cache:
            cached_value, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return cached_value
        
        try:
            # Validate all hashes are valid hex
            for tx_hash in transaction_hashes:
                if not self._is_valid_hex_hash(tx_hash):
                    raise ValueError(f"Invalid transaction hash: {tx_hash}")
            
            hashes_bytes = [bytes.fromhex(h) for h in transaction_hashes]
            
            # Use optimized algorithm for large sets
            if len(hashes_bytes) > 1000 and enable_optimization:
                merkle_root = self._calculate_merkle_root_optimized(hashes_bytes, algorithm)
            else:
                merkle_root = self._calculate_merkle_root_standard(hashes_bytes, algorithm)
            
            if enable_optimization:
                self._cache[cache_key] = (merkle_root, time.time())
            
            return merkle_root
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate merkle root: {e}") from e
    
    def _calculate_merkle_root_standard(self, hashes_bytes: List[bytes], 
                                      algorithm: str) -> str:
        """Standard merkle root calculation algorithm"""
        current_level = hashes_bytes
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else current_level[i]
                
                combined = left + right
                next_level.append(self._hash_bytes(combined, algorithm))
            
            current_level = next_level
        
        return current_level[0].hex() if current_level else '0' * 64
    
    def _calculate_merkle_root_optimized(self, hashes_bytes: List[bytes], 
                                       algorithm: str) -> str:
        """Optimized merkle root calculation for large datasets"""
        import itertools
        
        # Process in chunks for memory efficiency
        chunk_size = 1000
        chunks = [hashes_bytes[i:i + chunk_size] for i in range(0, len(hashes_bytes), chunk_size)]
        
        # Calculate merkle root for each chunk
        chunk_roots = []
        for chunk in chunks:
            while len(chunk) > 1:
                paired = [chunk[i] + chunk[i + 1] for i in range(0, len(chunk) - 1, 2)]
                if len(chunk) % 2 == 1:
                    paired.append(chunk[-1] + chunk[-1])  # Duplicate last if odd
                chunk = [self._hash_bytes(pair, algorithm) for pair in paired]
            chunk_roots.extend(chunk)
        
        # Final merkle root from chunk roots
        return self._calculate_merkle_root_standard(chunk_roots, algorithm)
    
    def _hash_bytes(self, data: bytes, algorithm: str) -> bytes:
        """Hash bytes with specified algorithm"""
        if algorithm == 'double_sha256':
            return hashlib.sha256(hashlib.sha256(data).digest()).digest()
        elif algorithm == 'sha256':
            return hashlib.sha256(data).digest()
        elif algorithm == 'sha3_256':
            return hashlib.sha3_256(data).digest()
        elif algorithm == 'blake2s':
            return hashlib.blake2s(data).digest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def calculate_block_work(self, difficulty: int, 
                           algorithm: str = 'bitcoin') -> Decimal:
        """Calculate block work value with multiple algorithm support"""
        try:
            difficulty = max(1, difficulty)  # Ensure positive difficulty
            
            if algorithm == 'bitcoin':
                # Bitcoin-style work calculation
                max_target = Decimal('0x00000000FFFF0000000000000000000000000000000000000000000000000000')
                target = max_target / Decimal(difficulty)
                
                if target > 0:
                    work = (Decimal(2) ** 256) / (target + 1)
                else:
                    work = Decimal(2) ** 256
                    
            elif algorithm == 'simple':
                # Simple proportional work calculation
                work = Decimal(difficulty) * Decimal('1000000000000')
                
            elif algorithm == 'logarithmic':
                # Logarithmic work calculation for better scaling
                work = Decimal(2) ** (Decimal(difficulty).ln() * Decimal('1.4426950408889634'))
                
            else:
                raise ValueError(f"Unsupported work algorithm: {algorithm}")
            
            return work.quantize(Decimal('1.00000000'), rounding=ROUND_DOWN)
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate block work: {e}") from e
    
    def calculate_difficulty_adjustment(self, 
                                      previous_difficulty: int, 
                                      actual_block_times: List[int],
                                      target_block_time: int,
                                      adjustment_factor: float = 4.0,
                                      algorithm: str = 'median') -> Dict[str, Any]:
        """Advanced difficulty adjustment with multiple algorithms and statistics"""
        try:
            if not actual_block_times:
                return {
                    'new_difficulty': previous_difficulty,
                    'adjustment_factor': 1.0,
                    'algorithm_used': algorithm,
                    'statistics': {}
                }
            
            # Calculate central tendency based on algorithm
            if algorithm == 'median':
                sorted_times = sorted(actual_block_times)
                middle = len(sorted_times) // 2
                if len(sorted_times) % 2 == 0:
                    typical_time = (sorted_times[middle - 1] + sorted_times[middle]) / 2
                else:
                    typical_time = sorted_times[middle]
                    
            elif algorithm == 'mean':
                typical_time = sum(actual_block_times) / len(actual_block_times)
                
            elif algorithm == 'trimmed_mean':
                # Remove outliers (top and bottom 10%)
                sorted_times = sorted(actual_block_times)
                trim_count = max(1, len(sorted_times) // 10)
                trimmed_times = sorted_times[trim_count:-trim_count] if len(sorted_times) > 20 else sorted_times
                typical_time = sum(trimmed_times) / len(trimmed_times)
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Apply safety limits
            min_time = target_block_time / adjustment_factor
            max_time = target_block_time * adjustment_factor
            typical_time = max(min_time, min(max_time, typical_time))
            
            # Calculate adjustment
            adjustment = target_block_time / typical_time
            new_difficulty = int(previous_difficulty * adjustment)
            
            # Ensure minimum difficulty
            min_difficulty = self.config.get('min_difficulty', 1)
            new_difficulty = max(min_difficulty, new_difficulty)
            
            # Calculate statistics
            stats = {
                'sample_size': len(actual_block_times),
                'min_block_time': min(actual_block_times),
                'max_block_time': max(actual_block_times),
                'typical_block_time': typical_time,
                'adjustment_percentage': (adjustment - 1) * 100
            }
            
            return {
                'new_difficulty': new_difficulty,
                'adjustment_factor': adjustment,
                'algorithm_used': algorithm,
                'statistics': stats
            }
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate difficulty adjustment: {e}") from e
    
    def estimate_block_size(self, transactions: List[Any], 
                          header_size: int = 80,
                          include_overhead: bool = True) -> Dict[str, int]:
        """Comprehensive block size estimation with detailed breakdown"""
        try:
            base_header_size = header_size
            transaction_sizes = []
            
            for tx in transactions:
                tx_size = self.estimate_transaction_size(tx, detailed=False)
                transaction_sizes.append(tx_size)
            
            total_tx_size = sum(transaction_sizes)
            
            # Calculate overhead (segregated witness, etc.)
            overhead = 0
            if include_overhead:
                overhead = self._calculate_block_overhead(len(transactions))
            
            total_size = base_header_size + total_tx_size + overhead
            
            return {
                'total_size': total_size,
                'header_size': base_header_size,
                'transactions_size': total_tx_size,
                'overhead_size': overhead,
                'transaction_count': len(transactions),
                'average_tx_size': total_tx_size // len(transactions) if transactions else 0
            }
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to estimate block size: {e}") from e
    
    def estimate_transaction_size(self, transaction: Any, 
                                detailed: bool = True) -> Union[int, Dict[str, int]]:
        """Advanced transaction size estimation with detailed breakdown"""
        try:
            # Base transaction components
            version_size = 4
            locktime_size = 4
            segwit_marker_size = 0
            segwit_flag_size = 0
            
            # Input estimation
            input_count = len(transaction.inputs) if hasattr(transaction, 'inputs') else 0
            input_size = input_count * self._estimate_input_size(transaction)
            
            # Output estimation
            output_count = len(transaction.outputs) if hasattr(transaction, 'outputs') else 0
            output_size = output_count * self._estimate_output_size(transaction)
            
            # Witness data estimation
            witness_size = 0
            if hasattr(transaction, 'witness') and transaction.witness:
                witness_size = self._estimate_witness_size(transaction)
                segwit_marker_size = 1
                segwit_flag_size = 1
            
            total_size = (version_size + segwit_marker_size + segwit_flag_size + 
                         input_size + output_size + witness_size + locktime_size)
            
            if not detailed:
                return total_size
            
            return {
                'total_size': total_size,
                'version_size': version_size,
                'input_count': input_count,
                'input_size': input_size,
                'output_count': output_count,
                'output_size': output_size,
                'witness_size': witness_size,
                'locktime_size': locktime_size,
                'segwit_overhead': segwit_marker_size + segwit_flag_size
            }
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to estimate transaction size: {e}") from e
    
    def _estimate_input_size(self, transaction: Any) -> int:
        """Estimate size of a single transaction input"""
        # Previous tx hash + output index + scriptSig length + sequence
        return 32 + 4 + 107 + 4  # Conservative estimate
    
    def _estimate_output_size(self, transaction: Any) -> int:
        """Estimate size of a single transaction output"""
        # Value + scriptPubKey length
        return 8 + 25  # Standard P2PKH output
    
    def _estimate_witness_size(self, transaction: Any) -> int:
        """Estimate witness data size"""
        if not hasattr(transaction, 'witness') or not transaction.witness:
            return 0
        
        # Simplified witness estimation
        witness_count = len(transaction.inputs) if hasattr(transaction, 'inputs') else 0
        return witness_count * 72  # Average signature size
    
    def _calculate_block_overhead(self, transaction_count: int) -> int:
        """Calculate block overhead including segregated witness data"""
        base_overhead = 10  # Block metadata
        tx_count_overhead = (transaction_count.bit_length() + 7) // 8  # Varint size
        return base_overhead + tx_count_overhead
    
    def calculate_transaction_fee(self, inputs_value: int, outputs_value: int, 
                                fee_rate: Optional[int] = None) -> Dict[str, Any]:
        """Calculate transaction fee with multiple fee calculation methods"""
        try:
            base_fee = max(0, inputs_value - outputs_value)
            
            result = {
                'base_fee': base_fee,
                'inputs_value': inputs_value,
                'outputs_value': outputs_value,
                'fee_calculation_method': 'difference'
            }
            
            if fee_rate is not None:
                # Calculate fee based on fee rate and estimated size
                estimated_size = self.estimate_transaction_size(self, detailed=False)
                rate_based_fee = fee_rate * estimated_size
                result['rate_based_fee'] = rate_based_fee
                result['fee_calculation_method'] = 'rate_based'
                result['effective_fee_rate'] = base_fee / estimated_size if estimated_size > 0 else 0
            
            return result
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate transaction fee: {e}") from e
    
    def validate_block_header(self, header: Dict[str, Any], 
                            strict_mode: bool = True) -> Dict[str, Any]:
        """Comprehensive block header validation with detailed results"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'timestamp': time.time()
        }
        
        try:
            required_fields = ['version', 'height', 'previous_hash', 'merkle_root', 
                             'timestamp', 'difficulty', 'nonce', 'validator']
            
            # Check required fields
            for field in required_fields:
                if field not in header:
                    validation_results['is_valid'] = False
                    validation_results['errors'].append(f"Missing required field: {field}")
            
            # Validate hash formats
            hash_fields = ['previous_hash', 'merkle_root']
            for field in hash_fields:
                if field in header:
                    if not self._is_valid_hex_hash(header[field]):
                        validation_results['is_valid'] = False
                        validation_results['errors'].append(f"Invalid {field} format")
            
            # Validate value ranges
            if 'version' in header and header['version'] not in [1, 2, 3, 4]:
                validation_results['warnings'].append(f"Unusual version: {header['version']}")
            
            if 'height' in header and header['height'] < 0:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Height cannot be negative")
            
            if 'timestamp' in header:
                current_time = int(time.time())
                if header['timestamp'] > current_time + 7200:  # 2 hours in future
                    validation_results['warnings'].append("Timestamp is too far in future")
                if header['timestamp'] < current_time - 63072000:  # 2 years in past
                    validation_results['warnings'].append("Timestamp is too far in past")
            
            if 'difficulty' in header and header['difficulty'] < 1:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Difficulty must be positive")
            
            # Strict mode validations
            if strict_mode:
                if 'nonce' in header and header['nonce'] < 0:
                    validation_results['warnings'].append("Nonce is negative")
                
                if 'validator' in header and not header['validator']:
                    validation_results['warnings'].append("Empty validator field")
            
            return validation_results
            
        except Exception as e:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation error: {str(e)}")
            return validation_results
    
    def calculate_network_hashrate(self, difficulty: int, block_time: int,
                                 algorithm: str = 'standard') -> Dict[str, Any]:
        """Calculate network hashrate with multiple algorithms and statistics"""
        try:
            if block_time <= 0:
                return {
                    'hashrate': 0.0,
                    'algorithm': algorithm,
                    'error': 'Invalid block time'
                }
            
            if algorithm == 'standard':
                # Standard Bitcoin-style calculation
                hashrate = (difficulty * (2 ** 32)) / block_time
            elif algorithm == 'precise':
                # More precise floating-point calculation
                hashrate = (difficulty * (2.0 ** 32)) / block_time
            elif algorithm == 'logarithmic':
                # Logarithmic scaling for very large values
                hashrate = math.exp(math.log(difficulty) + 32 * math.log(2) - math.log(block_time))
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            formatted = self.format_hashrate(hashrate)
            
            return {
                'hashrate': hashrate,
                'formatted': formatted,
                'algorithm': algorithm,
                'difficulty': difficulty,
                'block_time': block_time
            }
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to calculate network hashrate: {e}") from e
    
    def format_hashrate(self, hashrate: float, precision: int = 2) -> str:
        """Format hashrate into human-readable string with precision control"""
        try:
            units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s', 'EH/s', 'ZH/s']
            unit_index = 0
            current_rate = float(hashrate)
            
            while current_rate >= 1000.0 and unit_index < len(units) - 1:
                current_rate /= 1000.0
                unit_index += 1
            
            format_string = f"{{:.{precision}f}} {{}}"
            return format_string.format(current_rate, units[unit_index])
            
        except Exception as e:
            raise BlockCalculationError(f"Failed to format hashrate: {e}") from e
    
    def _validate_header_data(self, header_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize header data for hashing"""
        validated = header_data.copy()
        
        # Remove non-essential fields that shouldn't affect hash
        non_hash_fields = ['signature', 'extra_data', 'validation_status']
        for field in non_hash_fields:
            validated.pop(field, None)
        
        return validated
    
    def _is_valid_hex_hash(self, hash_string: str, length: int = 64) -> bool:
        """Validate if string is a valid hex hash of specified length"""
        if not isinstance(hash_string, str):
            return False
        
        if len(hash_string) != length:
            return False
        
        try:
            bytes.fromhex(hash_string)
            return True
        except ValueError:
            return False
    
    def clear_cache(self) -> None:
        """Clear calculation cache"""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = time.time()
        expired_count = sum(1 for _, (_, timestamp) in self._cache.items() 
                           if current_time - timestamp > self._cache_ttl)
        
        return {
            'total_entries': len(self._cache),
            'expired_entries': expired_count,
            'cache_ttl': self._cache_ttl,
            'memory_usage_estimate': len(str(self._cache))  # Rough estimate
        }


class BlockCalculationError(Exception):
    """Exception raised for block calculation errors"""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging"""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'original_error': str(self.original_error) if self.original_error else None,
            'timestamp': self.timestamp
        }


# Legacy function wrappers for backward compatibility
def calculate_block_hash(header_data: Dict[str, Any]) -> str:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    return calculator.calculate_block_hash(header_data)

def calculate_merkle_root(transaction_hashes: List[str]) -> str:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    return calculator.calculate_merkle_root(transaction_hashes)

def calculate_block_work(difficulty: int) -> int:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    return int(calculator.calculate_block_work(difficulty))

def calculate_difficulty(previous_difficulty: int, actual_block_time: int, 
                        target_block_time: int, adjustment_factor: float = 4.0) -> int:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    result = calculator.calculate_difficulty_adjustment(
        previous_difficulty, [actual_block_time], target_block_time, adjustment_factor
    )
    return result['new_difficulty']

def estimate_block_size(transactions: List[Any], header_size: int = 80) -> int:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    result = calculator.estimate_block_size(transactions, header_size)
    return result['total_size']

def estimate_transaction_size(transaction: Any) -> int:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    return calculator.estimate_transaction_size(transaction, detailed=False)

def calculate_transaction_fee(inputs_value: int, outputs_value: int) -> int:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    result = calculator.calculate_transaction_fee(inputs_value, outputs_value)
    return result['base_fee']

def validate_block_header(header: Dict[str, Any]) -> bool:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    result = calculator.validate_block_header(header, strict_mode=False)
    return result['is_valid']

def calculate_network_hashrate(difficulty: int, block_time: int) -> float:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    result = calculator.calculate_network_hashrate(difficulty, block_time)
    return result['hashrate']

def format_hashrate(hashrate: float) -> str:
    """Legacy function wrapper"""
    calculator = BlockCalculations()
    return calculator.format_hashrate(hashrate)