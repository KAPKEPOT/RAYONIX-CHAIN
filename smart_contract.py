# smart_contract.py
import hashlib
import json
import re
import time
import pickle
import ast
import inspect
import zlib
import base64
from typing import Dict, List, Any, Optional, Callable, Set, Tuple, Union
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa, padding
from cryptography.hazmat.primitives.serialization import load_pem_public_key, load_pem_private_key
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from enum import Enum, auto
from dataclasses import dataclass, field
import plyvel
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import asyncio
from contextlib import contextmanager
import sys
import traceback
from collections import defaultdict
from decimal import Decimal, getcontext
import secrets
import hmac
from functools import wraps
import logging
import math
import requests
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Tuple
import re
import time
import logging
import json
from urllib.parse import urlparse
import dns.resolver
import whois
from abc import ABC, abstractmethod
import resource
import wasmtime
#import wasmtime.wat
import psutil
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartContract")

# Set precision for financial calculations
getcontext().prec = 36

class ContractType(Enum):
    """Types of smart contracts with versioning"""
    ERC20 = ("ERC-20", "1.0.0")
    ERC721 = ("ERC-721", "1.0.0")
    ERC1155 = ("ERC-1155", "1.0.0")
    GOVERNANCE = ("GOVERNANCE", "2.1.0")
    DEX = ("DEX", "3.0.0")
    LENDING = ("LENDING", "2.0.0")
    ORACLE = ("ORACLE", "1.5.0")
    BRIDGE = ("BRIDGE", "1.2.0")
    CUSTOM = ("CUSTOM", "1.0.0")
    
    def __init__(self, display_name, version):
        self.display_name = display_name
        self.version = version

class ContractState(Enum):
    """Contract lifecycle states with timestamps"""
    ACTIVE = auto()
    PAUSED = auto()
    DESTROYED = auto()
    UPGRADING = auto()
    FROZEN = auto()
    AUDITING = auto()
    MIGRATING = auto()

class ContractSecurityLevel(Enum):
    """Contract security levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class ExecutionResult:
    """Enhanced result of contract execution with detailed metrics"""
    def __init__(self, success: bool, return_value: Any = None, gas_used: int = 0,
                 error: Optional[str] = None, events: List[Dict] = None,
                 execution_time: float = 0, memory_used: int = 0):
        self.success = success
        self.return_value = return_value
        self.gas_used = gas_used
        self.error = error
        self.events = events or []
        self.execution_time = execution_time
        self.memory_used = memory_used
        self.timestamp = time.time()
        self.transaction_hash = None
    
    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'return_value': self.return_value,
            'gas_used': self.gas_used,
            'error': self.error,
            'events': self.events,
            'execution_time': self.execution_time,
            'memory_used': self.memory_used,
            'timestamp': self.timestamp,
            'transaction_hash': self.transaction_hash
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

@dataclass
class ContractStorage:
    """Enhanced contract storage with encryption, compression, and access control"""
    storage: Dict[str, Any] = field(default_factory=dict)
    allowed_writers: Set[str] = field(default_factory=set)
    encryption_key: Optional[bytes] = None
    compression_enabled: bool = True
    versioning_enabled: bool = True
    audit_log: List[Dict] = field(default_factory=list)
    
    def _encrypt_value(self, value: Any) -> Any:
        """Encrypt storage value if encryption is enabled"""
        if not self.encryption_key or not isinstance(value, (str, bytes)):
            return value
        
        try:
            if isinstance(value, str):
                value = value.encode()
            # Validate encryption key length (AES-256 requires 32 bytes)
            if len(self.encryption_key) != 32:
                # Derive proper key using HKDF if needed
                hkdf = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b'contract_storage_encryption',
                backend=default_backend()
                )
                key = hkdf.derive(self.encryption_key)
            else:
                key = self.encryption_key
                
            # Generate a random 12-byte nonce for GCM
            nonce = secrets.token_bytes(12)
            
            # Create AES-GCM cipher
            aesgcm = AESGCM(key)
            
            # Encrypt the data (nonce is prepended to ciphertext)
            ciphertext = aesgcm.encrypt(nonce, value, None)
            
            # Combine nonce and ciphertext for storage
            encrypted_data = nonce + ciphertext
            
             # Return base64 encoded for storage
            return base64.b64encode(encrypted_data).decode('ascii')
             
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            # In production, you might want to raise or handle this differently
            # For now, return original value to avoid data loss
            return value       
                
              
            
            
    def _decrypt_value(self, value: Any) -> Any:
        """Decrypt storage value if encrypted"""
        if not self.encryption_key or not isinstance(value, str):
            return value
        
        try:
            encrypted = base64.b64decode(value)
            
            # Extract nonce (first 12 bytes) and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            
            # Validate/derive key (same logic as encryption)
            if len(self.encryption_key) != 32:
            	hkdf = HKDF(
            	    algorithm=hashes.SHA256(),
            	    length=32,
            	    salt=None,
            	    info=b'contract_storage_encryption',
                backend=default_backend()
            	)
            	key = hkdf.derive(self.encryption_key)
            else:
            	key = self.encryption_key
            	
            # Create AES-GCM cipher and decrypt
            aesgcm = AESGCM(key)
            plaintext = aesgcm.decrypt(nonce, ciphertext, None)
            
            # Try to decode back to string if it was originally a string
            try:
            	return plaintext.decode('utf-8')
            	
            except UnicodeDecodeError:
            	
            	return plaintext
            	
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            # Return original value on decryption failure
            return value
            
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if compression is enabled"""
        if not self.compression_enabled or not isinstance(value, (str, bytes)):
            return value
        
        try:
            if isinstance(value, str):
                value = value.encode()
            compressed = zlib.compress(value)
            return base64.b64encode(compressed).decode()
        except Exception:
            return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if compressed"""
        if not self.compression_enabled or not isinstance(value, str):
            return value
        
        try:
            compressed = base64.b64decode(value)
            decompressed = zlib.decompress(compressed)
            return decompressed.decode()
        except Exception:
            return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from storage with decryption and decompression"""
        value = self.storage.get(key, default)
        value = self._decompress_value(value)
        value = self._decrypt_value(value)
        return value
    
    def set(self, key: str, value: Any, caller: str, reason: str = "") -> bool:
        """Set value in storage with encryption, compression, and access control"""
        if caller not in self.allowed_writers:
            return False
        
        # Process value
        processed_value = self._encrypt_value(value)
        processed_value = self._compress_value(processed_value)
        
        old_value = self.storage.get(key)
        self.storage[key] = processed_value
        
        # Log the change
        self.audit_log.append({
            'timestamp': time.time(),
            'caller': caller,
            'key': key,
            'old_value': old_value,
            'new_value': processed_value,
            'reason': reason
        })
        
        # Keep audit log manageable
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]
        
        return True
    
    def delete(self, key: str, caller: str, reason: str = "") -> bool:
        """Delete key from storage with access control"""
        if caller not in self.allowed_writers:
            return False
        
        if key in self.storage:
            old_value = self.storage[key]
            del self.storage[key]
            
            self.audit_log.append({
                'timestamp': time.time(),
                'caller': caller,
                'key': key,
                'old_value': old_value,
                'new_value': None,
                'reason': reason,
                'action': 'delete'
            })
            
            return True
        return False
    
    def to_dict(self) -> Dict:
        return {
            'storage': self.storage,
            'allowed_writers': list(self.allowed_writers),
            'compression_enabled': self.compression_enabled,
            'versioning_enabled': self.versioning_enabled,
            'audit_log_size': len(self.audit_log)
        }

class GasMeter:
    """Production-ready gas meter with dynamic pricing, limits, and comprehensive operation tracking"""
    
    def __init__(self, gas_limit: int, base_gas_price: int = 1, dynamic_pricing: bool = True):
        self.gas_limit = gas_limit
        self.base_gas_price = base_gas_price
        self.dynamic_pricing = dynamic_pricing
        self.gas_used = 0
        self.operations_count = 0
        self.start_time = time.time()
        self.operation_history = []
        
        # Comprehensive gas cost table with operation-specific pricing
        self.operation_costs = {
            # Storage operations
            'storage_read': 100,
            'storage_write': 500,
            'storage_update': 300,
            'storage_delete': 250,
            'storage_clear': 1000,
            
            # Computation operations
            'arithmetic_op': 5,
            'comparison_op': 3,
            'logical_op': 2,
            'bitwise_op': 4,
            'math_function': 10,
            
            # Cryptographic operations
            'hash_operation': 800,
            'signature_verify': 2000,
            'encryption': 1500,
            'decryption': 1500,
            'key_generation': 3000,
            
            # Contract operations
            'contract_call': 2500,
            'contract_deploy': 10000,
            'contract_destroy': 5000,
            'external_call': 5000,
            
            # Event operations
            'event_emit': 100,
            'log_emit': 50,
            
            # Memory operations
            'memory_alloc': 10,
            'memory_copy': 2,
            'memory_clear': 5,
            
            # Control flow
            'function_call': 50,
            'loop_iteration': 2,
            'conditional_check': 3,
            
            # Data operations
            'data_serialize': 20,
            'data_deserialize': 25,
            'data_compress': 100,
            'data_decompress': 80,
            
            # System operations
            'timestamp_access': 5,
            'random_generation': 15,
            'address_validation': 10
        }
        
        # Dynamic pricing factors based on network conditions
        self.dynamic_factors = {
            'network_congestion': 1.0,
            'storage_utilization': 1.0,
            'computation_intensity': 1.0,
            'time_of_day': 1.0
        }
    
    def consume_gas(self, operation: str, complexity: int = 1, data_size: int = 0, 
                   urgency: float = 1.0) -> None:
        """
        Consume gas for an operation with comprehensive pricing considerations
        
        Args:
            operation: Type of operation being performed
            complexity: Complexity multiplier for the operation
            data_size: Size of data being processed (bytes)
            urgency: Urgency factor for priority operations (1.0 = normal)
        
        Raises:
            OutOfGasError: If operation would exceed gas limit
        """
        # Get base cost for operation
        base_cost = self.operation_costs.get(operation, 100)
        
        # Apply complexity multiplier
        cost = base_cost * complexity
        
        # Apply data size scaling for data-intensive operations
        if data_size > 0 and operation in ['storage_write', 'storage_read', 'data_serialize', 
                                         'data_deserialize', 'memory_copy']:
            cost += max(1, data_size // 1024) * 2  # 2 gas per KB
        
        # Apply dynamic pricing if enabled
        if self.dynamic_pricing:
            dynamic_factor = self._calculate_dynamic_factor(operation, urgency)
            cost = int(cost * dynamic_factor)
        
        # Check if operation would exceed gas limit
        if self.gas_used + cost > self.gas_limit:
            raise OutOfGasError(
                f"Out of gas. Used: {self.gas_used}, Required: {cost}, Limit: {self.gas_limit}, "
                f"Operation: {operation}, Remaining: {self.gas_limit - self.gas_used}"
            )
        
        # Update gas usage and counters
        self.gas_used += cost
        self.operations_count += 1
        
        # Record operation in history
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': operation,
            'cost': cost,
            'complexity': complexity,
            'data_size': data_size,
            'remaining_gas': self.gas_limit - self.gas_used,
            'call_stack': self._get_call_stack()
        })
        
        # Keep history manageable
        if len(self.operation_history) > 1000:
            self.operation_history = self.operation_history[-1000:]
    
    def _calculate_dynamic_factor(self, operation: str, urgency: float) -> float:
        """
        Calculate dynamic pricing factor based on operation type and current conditions
        """
        base_factor = 1.0
        
        # Operation-specific factors
        if operation.startswith('storage_'):
            base_factor *= self.dynamic_factors['storage_utilization']
        elif operation.startswith(('contract_', 'external_')):
            base_factor *= self.dynamic_factors['network_congestion']
        elif operation in ['hash_operation', 'encryption', 'decryption']:
            base_factor *= self.dynamic_factors['computation_intensity']
        
        # Time-based pricing (higher during peak hours)
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour <= 17:  # Business hours
            base_factor *= 1.2
        
        # Apply urgency multiplier
        base_factor *= max(0.5, min(2.0, urgency))
        
        return base_factor
    
    def _get_call_stack(self) -> List[str]:
        """Get current call stack for debugging purposes"""
        try:
            # Get the current frame and walk up the call stack
            frame = inspect.currentframe()
            stack = []
            while frame:
                func_name = frame.f_code.co_name
                filename = frame.f_code.co_filename
                line_no = frame.f_lineno
                stack.append(f"{filename}:{line_no} in {func_name}")
                frame = frame.f_back
            return stack[-5:]  # Return last 5 frames only
        except:
            return ["call_stack_unavailable"]
    
    def refund_gas(self, amount: int, reason: str = "") -> None:
        """
        Refund gas to the meter (for operations that free resources)
        
        Args:
            amount: Amount of gas to refund
            reason: Reason for the refund
        """
        if amount <= 0:
            return
            
        # Cap refund at what was actually used
        refund_amount = min(amount, self.gas_used)
        self.gas_used -= refund_amount
        
        # Record refund in history
        self.operation_history.append({
            'timestamp': time.time(),
            'operation': 'gas_refund',
            'cost': -refund_amount,
            'reason': reason,
            'remaining_gas': self.gas_limit - self.gas_used
        })
    
    def get_remaining_gas(self) -> int:
        """Get remaining gas amount"""
        return self.gas_limit - self.gas_used
    
    def get_execution_cost(self) -> int:
        """Get total execution cost in base gas units"""
        return self.gas_used * self.base_gas_price
    
    def get_execution_cost_wei(self) -> int:
        """Get execution cost in wei (for Ethereum compatibility)"""
        return self.get_execution_cost() * (10 ** 9)  # Convert gwei to wei
    
    def get_operation_stats(self) -> Dict[str, Any]:
        """Get statistics about gas usage by operation type"""
        stats = {
            'total_operations': self.operations_count,
            'total_gas_used': self.gas_used,
            'remaining_gas': self.get_remaining_gas(),
            'execution_time': time.time() - self.start_time,
            'operations_by_type': defaultdict(int),
            'gas_by_operation': defaultdict(int)
        }
        
        for op in self.operation_history:
            if op['cost'] > 0:  # Only count consumption, not refunds
                stats['operations_by_type'][op['operation']] += 1
                stats['gas_by_operation'][op['operation']] += op['cost']
        
        return stats
    
    def set_dynamic_factor(self, factor_name: str, value: float) -> None:
        """Update a dynamic pricing factor"""
        if factor_name in self.dynamic_factors:
            self.dynamic_factors[factor_name] = max(0.1, min(5.0, value))
    
    def estimate_operation_cost(self, operation: str, complexity: int = 1, 
                              data_size: int = 0) -> int:
        """Estimate the cost of an operation without consuming gas"""
        base_cost = self.operation_costs.get(operation, 100)
        cost = base_cost * complexity
        
        if data_size > 0 and operation in ['storage_write', 'storage_read', 'data_serialize', 
                                         'data_deserialize', 'memory_copy']:
            cost += max(1, data_size // 1024) * 2
        
        if self.dynamic_pricing:
            cost = int(cost * self._calculate_dynamic_factor(operation, 1.0))
        
        return cost
    
    def can_execute(self, operation: str, complexity: int = 1, 
                   data_size: int = 0) -> bool:
        """Check if an operation can be executed with remaining gas"""
        estimated_cost = self.estimate_operation_cost(operation, complexity, data_size)
        return self.get_remaining_gas() >= estimated_cost


class OutOfGasError(Exception):
    """Custom exception for out of gas scenarios with detailed information"""
    
    def __init__(self, message: str, gas_used: int = 0, gas_limit: int = 0, 
                 operation: str = "", remaining_gas: int = 0):
        super().__init__(message)
        self.gas_used = gas_used
        self.gas_limit = gas_limit
        self.operation = operation
        self.remaining_gas = remaining_gas
        self.timestamp = time.time()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message': str(self),
            'gas_used': self.gas_used,
            'gas_limit': self.gas_limit,
            'operation': self.operation,
            'remaining_gas': self.remaining_gas,
            'timestamp': self.timestamp
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class GasOptimizer:
    """Advanced gas optimization system with pattern analysis and recommendations"""
    
    def __init__(self):
        self.patterns = self._load_optimization_patterns()
        self.history = []
        self.recommendations = []
    
    def _load_optimization_patterns(self) -> List[Dict]:
        """Load known gas optimization patterns"""
        return [
            {
                'pattern': 'multiple_storage_writes',
                'description': 'Multiple storage writes in quick succession',
                'detection': self._detect_multiple_storage_writes,
                'optimization': 'Use batch storage operations',
                'savings_estimate': 0.3  # 30% savings
            },
            {
                'pattern': 'redundant_computations',
                'description': 'Same computation performed multiple times',
                'detection': self._detect_redundant_computations,
                'optimization': 'Cache computation results',
                'savings_estimate': 0.4
            },
            {
                'pattern': 'inefficient_loops',
                'description': 'Loops with high gas cost operations',
                'detection': self._detect_inefficient_loops,
                'optimization': 'Optimize loop operations or use different algorithm',
                'savings_estimate': 0.5
            },
            {
                'pattern': 'expensive_external_calls',
                'description': 'Frequent expensive external contract calls',
                'detection': self._detect_expensive_external_calls,
                'optimization': 'Batch external calls or use different architecture',
                'savings_estimate': 0.6
            }
        ]
    
    def analyze_operations(self, operation_history: List[Dict]) -> List[Dict]:
        """Analyze operation history for optimization opportunities"""
        optimizations = []
        
        for pattern in self.patterns:
            detection_result = pattern['detection'](operation_history)
            if detection_result['found']:
                optimization = {
                    'pattern': pattern['pattern'],
                    'description': pattern['description'],
                    'recommendation': pattern['optimization'],
                    'estimated_savings': pattern['savings_estimate'],
                    'details': detection_result['details']
                }
                optimizations.append(optimization)
        
        return optimizations
    
    def _detect_multiple_storage_writes(self, history: List[Dict]) -> Dict:
        """Detect multiple storage writes to the same location"""
        storage_writes = [op for op in history if op['operation'].startswith('storage_')]
        write_patterns = {}
        
        for write in storage_writes:
            # Extract key from call stack or operation details
            key = self._extract_storage_key(write)
            if key:
                if key not in write_patterns:
                    write_patterns[key] = []
                write_patterns[key].append(write)
        
        # Find keys with multiple writes in short time
        results = []
        for key, writes in write_patterns.items():
            if len(writes) > 3:  # More than 3 writes to same key
                time_range = writes[-1]['timestamp'] - writes[0]['timestamp']
                if time_range < 10:  # Within 10 seconds
                    results.append({
                        'key': key,
                        'write_count': len(writes),
                        'time_range': time_range,
                        'total_cost': sum(w['cost'] for w in writes)
                    })
        
        return {
            'found': len(results) > 0,
            'details': results
        }
    
    def _extract_storage_key(self, operation: Dict) -> Optional[str]:
        """Extract storage key from operation details"""
        try:
        	# Method 1: Extract from call stack if available
        	call_stack = operation.get('call_stack', [])
        	for frame in reversed(call_stack):
        		if 'storage' in frame.lower() and ('get' in frame.lower() or 'set' in frame.lower()):
        			# Extract key from typical storage access patterns
        			patterns = [
        			    r'storage\[["\']([^"\']+)["\']\]',
        			    r'storage\.get\(["\']([^"\']+)["\']\)',
        			    
        			    r'storage\.set\(["\']([^"\']+)["\']',
        			    r'key=([^,\s\)]+)',
        			    r'get\(["\']([^"\']+)["\']\)',
        			    r'set\(["\']([^"\']+)["\']'
        			]
        			for pattern in patterns:
        				match = re.search(pattern, frame, re.IGNORECASE)
        				if match:
        					key = match.group(1)
        					if len(key) <= 256:
        						return key
        	# Method 2: Extract from operation metadata
        	details = operation.get('details', {})
        	if 'key' in details:
        		return str(details['key'])
        		
        	# Method 3: Look for key in operation description
        	operation_desc = operation.get('operation', '')
        	if 'storage_' in operation_desc:
        		# For storage operations, the key might be in additional metadata
        		metadata = operation.get('metadata', {})
        		if 'key' in metadata:
        			return str(metadata['key'])
        			
        	return None
        	
        except Exception as e:
        	logger.debug(f"Failed to extract storage key from operation: {e}")
        	return None
    
    def _detect_redundant_computations(self, history: List[Dict]) -> Dict:
        """Detect redundant computation operations"""
        try:
        	redundant_ops = []
        	computation_ops = [op for op in history if op['operation'] in [
        	    'math_function', 'arithmetic_op', 'hash_operation', 
        	    'crypto_operation', 'data_serialize', 'data_deserialize'
        	]]
        	
        	# Group operations by type and input parameters
        	operation_groups = defaultdict(list)
        	
        	for op in computation_ops:
        		# Create a signature based on operation type and input characteristics
        		signature = self._create_computation_signature(op)
        		operation_groups[signature].append(op)
        		
        	# Find groups with multiple identical computations
        	for signature, ops in operation_groups.items():
        		if len(ops) > 1:
        			time_range = max(op['timestamp'] for op in ops) - min(op['timestamp'] for op in ops)
        			if time_range < 30: # Within 30 seconds
        				total_cost = sum(op['cost'] for op in ops)
        				potential_savings = total_cost * 0.4 # 40% savings estimate
        				redundant_ops.append({
        				    'signature': signature,
        				    'count': len(ops),
        				    'time_range': time_range,
        				    'total_cost': total_cost,
        				    'potential_savings': potential_savings,
        				    'operations': [op['operation'] for op in ops[:5]]  # Sample operations
        				})
        				
        	return {
        	    'found': len(redundant_ops) > 0,
        	    'details': redundant_ops
        	}
        	
        except Exception as e:
        	logger.error(f"Error detecting redundant computations: {e}")
        	return {'found': False, 'details': []}
        	
    def _create_computation_signature(self, operation: Dict) -> str:
    	try:
    		# Base signature from operation type
    		signature_parts = [operation['operation']]
    		
    		# Add input size if available
    		data_size = operation.get('data_size', 0)
    		
    		if data_size > 0:
    			size_bucket = (data_size // 1024) * 1024  # Bucket by KB
    			signature_parts.append(f"size_{size_bucket}")
    			
    		# Add complexity if available
    		complexity = operation.get('complexity', 1)
    		if complexity > 1:
    			signature_parts.append(f"complexity_{complexity}")
    			
    		# Add call stack pattern (simplified)
    		call_stack = operation.get('call_stack', [])
    		
    		if call_stack:
    			# Use the first few frames to identify the computation context
    			context_frames = []
    			for frame in call_stack[:3]:
    				# Extract function name from frame
    				if ' in ' in frame:
    					func_name = frame.split(' in ')[-1]
    					context_frames.append(func_name.split('(')[0])
    					
    			if context_frames:
    				signature_parts.append('_'.join(context_frames))
    				
    		return '|'.join(signature_parts)
    		
    	except Exception:
    		return operation['operation']  # Fallback to just operation type

    def _detect_inefficient_loops(self, history: List[Dict]) -> Dict:
        """Detect inefficient loop patterns"""
        try:
        	inefficient_loops = []
        	
        	# Find loop operations
        	loop_ops = [op for op in history if 'loop' in op['operation'].lower() or op['operation'] == 'loop_iteration']
        	
        	# Group loop operations by context
        	loop_contexts = defaultdict(list)
        	
        	for op in loop_ops:
        		context = self._extract_loop_context(op)
        		
        		loop_contexts[context].append(op)
        		
        	# Analyze each loop context
        	for context, ops in loop_contexts.items():
        		if len(ops) >= 3:
        			total_iterations = len(ops)
        			total_cost = sum(op['cost'] for op in ops)
        			avg_cost_per_iteration = total_cost / total_iterations
        			
        			# Check for expensive operations inside loops
        			expensive_ops_in_loop = any(
        			    op['cost'] > 1000 for op in ops  # Arbitrary threshold
        			)
        			# Check for high iteration counts
        			high_iteration_count = total_iterations > 100
        			
        			# Check for nested expensive operations
        			nested_expensive = self._check_nested_expensive_operations(ops, history)
        			if expensive_ops_in_loop or high_iteration_count or nested_expensive:
        			        inefficient_loops.append({
        			            'context': context,
        			            'iterations': total_iterations,
        			            'total_cost': total_cost,
        			            'avg_cost_per_iteration': avg_cost_per_iteration,
        			            'expensive_ops_detected': expensive_ops_in_loop,
        			            'high_iteration_count': high_iteration_count,
        			            'nested_expensive_ops': nested_expensive,
        			            'potential_savings': total_cost * 0.5  # 50% savings estimate
        			        })
        			        
        	return {
        	    'found': len(inefficient_loops) > 0,
        	    'details': inefficient_loops
        	}
        	
        except Exception as e:
        	logger.error(f"Error detecting inefficient loops: {e}")
        	return {'found': False, 'details': []}
        	
    def _extract_loop_context(self, operation: Dict) -> str:
    	try:
    		call_stack = operation.get('call_stack', [])
    		if not call_stack:
    			return "unknown_context"
    			
    		# Use the function containing the loop as context
    		for frame in call_stack:
    			if ' in ' in frame:
    				func_part = frame.split(' in ')[-1]
    				if 'loop' in func_part.lower() or 'for' in func_part.lower() or 'while' in func_part.lower():
    					if 'loop' in func_part.lower() or 'for' in func_part.lower() or 'while' in func_part.lower():
    						# Fallback: use the calling function
    						return call_stack[0].split(' in ')[-1].split('(')[0] if call_stack else "unknown"
    	except Exception:
    		return "unknown_context"    
    		
    def _check_nested_expensive_operations(self, loop_ops: List[Dict], full_history: List[Dict]) -> bool:
    	try:
    		if not loop_ops:
    			return False
    			
    		# Get timestamps of loop operations
    		loop_timestamps = [op['timestamp'] for op in loop_ops]
    		min_time = min(loop_timestamps)
    		max_time = max(loop_timestamps)
    		
    		# Find expensive operations within the loop time range
    		expensive_ops = [
    		    op for op in full_history 
    		    if min_time <= op['timestamp'] <= max_time
    		    and op['cost'] > 500  # Expensive operation threshold
    		    and op not in loop_ops  # Exclude the loop operations themselves
    		]
    		
    		return len(expensive_ops) > 0
    		
    	except Exception:
    		return False    		    	    	
  
    def _detect_expensive_external_calls(self, history: List[Dict]) -> Dict:
        """Detect expensive external contract calls"""
        external_calls = [op for op in history if op['operation'] in ['external_call', 'contract_call']]
        
        if len(external_calls) > 5:  # More than 5 external calls
            total_cost = sum(op['cost'] for op in external_calls)
            return {
                'found': True,
                'details': [{
                    'call_count': len(external_calls),
                    'total_cost': total_cost,
                    'average_cost': total_cost / len(external_calls)
                }]
            }
        
        return {'found': False, 'details': []}

class ContractSecurity:
    """Comprehensive security system for contracts with advanced threat detection"""
    
    def __init__(self):
        self.blacklisted_addresses: Set[str] = set()
        self.rate_limits: Dict[str, Dict] = {}
        self.max_execution_time = 30  # seconds
        self.max_memory_usage = 100 * 1024 * 1024  # 100MB
        self.suspicious_activity_log: List[Dict] = []
        self.security_policies: Dict[str, Any] = self._initialize_security_policies()
        self.behavioral_analysis = BehavioralAnalyzer()
        self.threat_intelligence = ThreatIntelligenceFeed()
        
    def _initialize_security_policies(self) -> Dict[str, Any]:
        """Initialize comprehensive security policies"""
        return {
            'rate_limiting': {
                'storage_write': {'max_operations': 100, 'time_window': 60},
                'external_call': {'max_operations': 50, 'time_window': 60},
                'event_emit': {'max_operations': 200, 'time_window': 60},
                'contract_call': {'max_operations': 150, 'time_window': 60},
                'crypto_operation': {'max_operations': 30, 'time_window': 60},
                'default': {'max_operations': 1000, 'time_window': 60}
            },
            'input_validation': {
                'max_string_length': 1024 * 1024,  # 1MB
                'max_array_size': 10000,
                'max_nesting_depth': 20,
                'allowed_characters': r'[\x20-\x7E]',  # Printable ASCII
                'blocked_patterns': [
                    r'__.*__',
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'open\s*\(',
                    r'file\s*\(',
                    r'subprocess',
                    r'os\.',
                    r'sys\.',
                    r'importlib',
                    r'pickle',
                    r'marshal',
                    r'__import__',
                    r'getattr',
                    r'setattr',
                    r'delattr',
                    r'globals',
                    r'locals',
                    r'compile',
                    r'execfile',
                    r'reload',
                    r'input',
                    r'raw_input',
                    r'apply',
                    r'buffer',
                    r'memoryview',
                    r'super',
                    r'property',
                    r'staticmethod',
                    r'classmethod',
                    r'__getattribute__',
                    r'__setattr__',
                    r'__delattr__',
                    r'__getitem__',
                    r'__setitem__',
                    r'__delitem__',
                    r'__call__',
                    r'__get__',
                    r'__set__',
                    r'__delete__',
                    r'__getslice__',
                    r'__setslice__',
                    r'__delslice__'
                ]
            },
            'resource_limits': {
                'max_execution_time': 30,
                'max_memory_usage': 100 * 1024 * 1024,
                'max_gas_per_transaction': 10 * 1000 * 1000,
                'max_storage_usage': 10 * 1024 * 1024 * 1024  # 10GB
            },
            'behavioral_rules': {
                'consecutive_failures_threshold': 5,
                'unusual_time_activity': True,
                'geographic_anomalies': True,
                'transaction_pattern_analysis': True
            }
        }
    
    def check_rate_limit(self, caller: str, operation: str, complexity: int = 1) -> bool:
        """
        Check if caller is rate limited for an operation with complexity consideration
        
        Args:
            caller: Address of the caller
            operation: Type of operation being performed
            complexity: Complexity factor of the operation
            
        Returns:
            bool: True if operation is allowed, False if rate limited
        """
        key = f"{caller}:{operation}"
        now = time.time()
        
        # Get rate limiting policy
        policy = self.security_policies['rate_limiting'].get(
            operation, 
            self.security_policies['rate_limiting']['default']
        )
        
        if key not in self.rate_limits:
            self.rate_limits[key] = {
                'count': 0,
                'last_reset': now,
                'first_operation': now,
                'complexity_total': 0
            }
            return True
        
        limit_info = self.rate_limits[key]
        
        # Reset counter if time window has passed
        if now - limit_info['last_reset'] > policy['time_window']:
            limit_info['count'] = 0
            limit_info['complexity_total'] = 0
            limit_info['last_reset'] = now
            limit_info['first_operation'] = now
        
        # Calculate effective operation count with complexity
        effective_count = limit_info['count'] + complexity
        complexity_total = limit_info['complexity_total'] + complexity
        
        # Check if operation would exceed limits
        max_operations = policy['max_operations']
        if effective_count >= max_operations or complexity_total >= max_operations * 2:
            # Log suspicious activity
            self._log_suspicious_activity(
                caller=caller,
                operation=operation,
                reason="rate_limit_exceeded",
                details={
                    'effective_count': effective_count,
                    'complexity_total': complexity_total,
                    'max_allowed': max_operations,
                    'time_window': policy['time_window']
                }
            )
            return False
        
        # Update counters
        limit_info['count'] += 1
        limit_info['complexity_total'] += complexity
        
        # Behavioral analysis
        if not self.behavioral_analysis.analyze_operation_pattern(caller, operation, now):
            self._log_suspicious_activity(
                caller=caller,
                operation=operation,
                reason="behavioral_anomaly",
                details={'analysis_result': 'unusual_operation_pattern'}
            )
            return False
        
        return True
    
    def validate_input(self, input_data: Any, expected_type: type = None, 
                      context: Dict[str, Any] = None) -> Tuple[bool, Optional[str]]:
        """
        Comprehensive input validation with context awareness
        
        Args:
            input_data: Data to validate
            expected_type: Expected data type
            context: Additional context for validation
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if input_data is None:
            return False, "Input data cannot be None"
        
        context = context or {}
        validation_policy = self.security_policies['input_validation']
        
        # Type validation
        if expected_type and not isinstance(input_data, expected_type):
            return False, f"Expected type {expected_type}, got {type(input_data)}"
        
        # String validation
        if isinstance(input_data, str):
            # Length check
            if len(input_data) > validation_policy['max_string_length']:
                return False, f"String too long: {len(input_data)} characters"
            
            # Character set validation
            if not re.fullmatch(validation_policy['allowed_characters'] + '*', input_data):
                return False, "Contains invalid characters"
            
            # Pattern matching for dangerous constructs
            for pattern in validation_policy['blocked_patterns']:
                if re.search(pattern, input_data, re.IGNORECASE):
                    self._log_suspicious_activity(
                        caller=context.get('caller', 'unknown'),
                        operation=context.get('operation', 'input_validation'),
                        reason="dangerous_pattern_detected",
                        details={
                            'pattern': pattern,
                            'input_sample': input_data[:100] + '...' if len(input_data) > 100 else input_data
                        }
                    )
                    return False, f"Dangerous pattern detected: {pattern}"
        
        # Collection validation
        elif isinstance(input_data, (list, tuple, set)):
            if len(input_data) > validation_policy['max_array_size']:
                return False, f"Collection too large: {len(input_data)} items"
            
            # Recursively validate collection items
            for i, item in enumerate(input_data):
                is_valid, error = self.validate_input(item, None, context)
                if not is_valid:
                    return False, f"Item {i}: {error}"
        
        # Dictionary validation
        elif isinstance(input_data, dict):
            if len(input_data) > validation_policy['max_array_size']:
                return False, f"Dictionary too large: {len(input_data)} items"
            
            # Check nesting depth
            current_depth = context.get('nesting_depth', 0) + 1
            if current_depth > validation_policy['max_nesting_depth']:
                return False, f"Nesting depth exceeded: {current_depth}"
            
            # Recursively validate dictionary values
            new_context = context.copy()
            new_context['nesting_depth'] = current_depth
            
            for key, value in input_data.items():
                # Validate key
                is_valid, error = self.validate_input(key, None, new_context)
                if not is_valid:
                    return False, f"Key '{key}': {error}"
                
                # Validate value
                is_valid, error = self.validate_input(value, None, new_context)
                if not is_valid:
                    return False, f"Value for key '{key}': {error}"
        
        # Binary data validation
        elif isinstance(input_data, bytes):
            if len(input_data) > validation_policy['max_string_length']:
                return False, f"Binary data too large: {len(input_data)} bytes"
            
            # Check for executable code patterns
            if self._contains_executable_patterns(input_data):
                self._log_suspicious_activity(
                    caller=context.get('caller', 'unknown'),
                    operation=context.get('operation', 'input_validation'),
                    reason="executable_code_detected",
                    details={'data_size': len(input_data)}
                )
                return False, "Binary data contains executable patterns"
        
        # Threat intelligence check
        if self.threat_intelligence.is_malicious(input_data, context.get('caller')):
            return False, "Input matches known threat patterns"
        
        return True, None
    
    def _contains_executable_patterns(self, data: bytes) -> bool:
        """Check if binary data contains patterns indicative of executable code"""
        # Common executable patterns (simplified for example)
        executable_patterns = [
            b'\x4D\x5A',  # MZ header (Windows PE)
            b'\x7F\x45\x4C\x46',  # ELF header
            b'\xCA\xFE\xBA\xBE',  # Java class
            b'\xCE\xFA\xED\xFE',  # Mach-O
            b'\xFE\xED\xFA\xCE',  # Mach-O (alternative)
            b'\xFE\xED\xFA\xCF',  # Mach-O 64-bit
            b'\xCF\xFA\xED\xFE'   # Mach-O 64-bit (alternative)
        ]
        
        for pattern in executable_patterns:
            if data.startswith(pattern):
                return True
        
        # Check for high entropy (potential packed/encrypted code)
        if len(data) > 100 and self._calculate_entropy(data) > 7.0:
            return True
        
        return False
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of binary data"""
        if not data:
            return 0.0
        
        entropy = 0.0
        for x in range(256):
            p_x = float(data.count(x)) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        
        return entropy
    
    def _log_suspicious_activity(self, caller: str, operation: str, 
                               reason: str, details: Dict[str, Any]) -> None:
        """Log suspicious activity for security monitoring"""
        log_entry = {
            'timestamp': time.time(),
            'caller': caller,
            'operation': operation,
            'reason': reason,
            'details': details,
            'severity': self._determine_severity(reason, details)
        }
        
        self.suspicious_activity_log.append(log_entry)
        
        # Keep log manageable
        if len(self.suspicious_activity_log) > 10000:
            self.suspicious_activity_log = self.suspicious_activity_log[-10000:]
        
        # Real-time alert for high severity events
        if log_entry['severity'] >= 8:
            self._send_security_alert(log_entry)
    
    def _determine_severity(self, reason: str, details: Dict[str, Any]) -> int:
        """Determine severity level for suspicious activity"""
        severity_map = {
            'rate_limit_exceeded': 5,
            'behavioral_anomaly': 6,
            'dangerous_pattern_detected': 7,
            'executable_code_detected': 9,
            'known_threat_pattern': 8
        }
        
        base_severity = severity_map.get(reason, 4)
        
        # Adjust based on context
        if 'complexity_total' in details and details['complexity_total'] > 100:
            base_severity += 1
        
        return min(base_severity, 10)
    
    def _send_security_alert(self, log_entry: Dict[str, Any]) -> None:
        """Send real-time security alert"""
        # Implementation would integrate with alerting system
        logger.warning(f"SECURITY ALERT: {json.dumps(log_entry)}")
    
    def is_blacklisted(self, address: str) -> bool:
        """Check if address is blacklisted"""
        return (address in self.blacklisted_addresses or 
                self.threat_intelligence.is_address_blacklisted(address))
    
    def check_resource_limits(self, execution_time: float, memory_usage: int, 
                            gas_used: int, storage_usage: int) -> Tuple[bool, Optional[str]]:
        """Check if resource usage exceeds limits"""
        limits = self.security_policies['resource_limits']
        
        if execution_time > limits['max_execution_time']:
            return False, f"Execution time exceeded: {execution_time}s > {limits['max_execution_time']}s"
        
        if memory_usage > limits['max_memory_usage']:
            return False, f"Memory usage exceeded: {memory_usage} > {limits['max_memory_usage']}"
        
        if gas_used > limits['max_gas_per_transaction']:
            return False, f"Gas usage exceeded: {gas_used} > {limits['max_gas_per_transaction']}"
        
        if storage_usage > limits['max_storage_usage']:
            return False, f"Storage usage exceeded: {storage_usage} > {limits['max_storage_usage']}"
        
        return True, None
    
    def update_from_threat_intelligence(self) -> None:
        """Update security settings from threat intelligence feed"""
        # Get latest threat data
        threats = self.threat_intelligence.get_latest_threats()
        
        # Update blacklists
        for address in threats.get('malicious_addresses', []):
            self.blacklisted_addresses.add(address)
        
        # Update security policies based on threat level
        threat_level = self.threat_intelligence.get_current_threat_level()
        if threat_level > 7:  # High threat level
            # Tighten security policies
            for policy in self.security_policies['rate_limiting'].values():
                if isinstance(policy, dict) and 'max_operations' in policy:
                    policy['max_operations'] = max(1, policy['max_operations'] // 2)
            
            self.security_policies['resource_limits']['max_execution_time'] = max(
                5, self.security_policies['resource_limits']['max_execution_time'] // 2
            )

class BehavioralAnalyzer:
    """Advanced behavioral analysis for anomaly detection"""
    
    def __init__(self):
        self.behavior_baselines: Dict[str, Dict] = {}
        self.activity_log: List[Dict] = []
        self.anomaly_threshold = 3.0  # Standard deviations for anomaly detection
        
    def record_activity(self, address: str, operation: str, timestamp: float, 
                      details: Dict[str, Any]) -> None:
        """Record activity for behavioral analysis"""
        log_entry = {
            'address': address,
            'operation': operation,
            'timestamp': timestamp,
            'details': details
        }
        self.activity_log.append(log_entry)
        
        # Update behavior baseline
        self._update_baseline(address, operation, timestamp, details)
    
    def _update_baseline(self, address: str, operation: str, timestamp: float, 
                        details: Dict[str, Any]) -> None:
        """Update behavioral baseline for address and operation"""
        if address not in self.behavior_baselines:
            self.behavior_baselines[address] = {}
        
        if operation not in self.behavior_baselines[address]:
            self.behavior_baselines[address][operation] = {
                'count': 0,
                'timestamps': [],
                'complexities': [],
                'time_of_day': [],
                'day_of_week': []
            }
        
        baseline = self.behavior_baselines[address][operation]
        baseline['count'] += 1
        baseline['timestamps'].append(timestamp)
        
        # Record complexity if available
        if 'complexity' in details:
            baseline['complexities'].append(details['complexity'])
        
        # Record temporal patterns
        dt = datetime.fromtimestamp(timestamp)
        baseline['time_of_day'].append(dt.hour * 3600 + dt.minute * 60 + dt.second)
        baseline['day_of_week'].append(dt.weekday())
        
        # Keep baselines manageable
        for key in ['timestamps', 'complexities', 'time_of_day', 'day_of_week']:
            if len(baseline[key]) > 1000:
                baseline[key] = baseline[key][-1000:]
    
    def analyze_operation_pattern(self, address: str, operation: str, 
                                timestamp: float) -> bool:
        """Analyze if operation pattern is normal"""
        if address not in self.behavior_baselines or operation not in self.behavior_baselines[address]:
            # No baseline yet, consider it normal
            return True
        
        baseline = self.behavior_baselines[address][operation]
        
        if baseline['count'] < 10:  # Need minimum data points
            return True
        
        # Check temporal patterns
        dt = datetime.fromtimestamp(timestamp)
        current_time_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
        current_day_of_week = dt.weekday()
        
        # Calculate z-scores for anomaly detection
        time_zscore = self._calculate_zscore(
            current_time_of_day, 
            baseline['time_of_day']
        )
        
        day_zscore = self._calculate_zscore(
            current_day_of_week,
            baseline['day_of_week']
        )
        
        # Check if patterns are anomalous
        if (abs(time_zscore) > self.anomaly_threshold or 
            abs(day_zscore) > self.anomaly_threshold):
            return False
        
        return True
    
    def _calculate_zscore(self, value: float, data: List[float]) -> float:
        """Calculate z-score for anomaly detection"""
        if not data:
            return 0.0
        
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std_dev = math.sqrt(variance) if variance > 0 else 1.0
        
        return (value - mean) / std_dev if std_dev > 0 else 0.0
    
    def get_behavior_summary(self, address: str) -> Dict[str, Any]:
        """Get behavioral summary for address"""
        if address not in self.behavior_baselines:
            return {'has_baseline': False}
        
        summary = {
            'has_baseline': True,
            'operations': {},
            'total_operations': 0,
            'first_seen': float('inf'),
            'last_seen': 0
        }
        
        for operation, baseline in self.behavior_baselines[address].items():
            summary['operations'][operation] = {
                'count': baseline['count'],
                'avg_complexity': sum(baseline['complexities']) / len(baseline['complexities']) 
                                if baseline['complexities'] else 0,
                'time_pattern': self._analyze_time_pattern(baseline['time_of_day']),
                'day_pattern': self._analyze_day_pattern(baseline['day_of_week'])
            }
            summary['total_operations'] += baseline['count']
            summary['first_seen'] = min(summary['first_seen'], min(baseline['timestamps']))
            summary['last_seen'] = max(summary['last_seen'], max(baseline['timestamps']))
        
        return summary
    
    def _analyze_time_pattern(self, times: List[float]) -> Dict[str, Any]:
        """Analyze time-of-day patterns"""
        if not times:
            return {'pattern': 'unknown'}
        
        # Convert to hours for analysis
        hours = [t / 3600 for t in times]
        mean_hour = sum(hours) / len(hours)
        
        if mean_hour < 6:
            pattern = 'night'
        elif mean_hour < 12:
            pattern = 'morning'
        elif mean_hour < 18:
            pattern = 'afternoon'
        else:
            pattern = 'evening'
        
        return {
            'pattern': pattern,
            'mean_hour': mean_hour,
            'consistency': self._calculate_consistency(hours)
        }
    
    def _analyze_day_pattern(self, days: List[int]) -> Dict[str, Any]:
        """Analyze day-of-week patterns"""
        if not days:
            return {'pattern': 'unknown'}
        
        day_counts = [0] * 7
        for day in days:
            day_counts[day] += 1
        
        max_day = day_counts.index(max(day_counts))
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return {
            'most_active_day': day_names[max_day],
            'day_distribution': day_counts,
            'weekend_ratio': (day_counts[5] + day_counts[6]) / sum(day_counts) if sum(day_counts) > 0 else 0
        }
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate consistency score (1.0 = perfectly consistent)"""
        if len(values) < 2:
            return 1.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Normalize to 0-1 range (lower std_dev = higher consistency)
        max_expected_std_dev = 6.0  # Maximum expected standard deviation for time of day
        return max(0.0, 1.0 - (std_dev / max_expected_std_dev))

class ThreatIntelligenceFeed:
    """Real-time threat intelligence integration"""
    
    def __init__(self):
        self.malicious_ips: Set[str] = set()
        self.malicious_domains: Set[str] = set()
        self.malicious_addresses: Set[str] = set()
        self.known_exploits: Set[str] = set()
        self.last_update: float = 0
        self.update_interval: int = 3600  # 1 hour
        self.threat_level: int = 0  # 0-10 scale
        self.threat_sources: List[Dict] = [
            {
                'name': 'PhishTank',
                'url': 'http://data.phishtank.com/data/online-valid.json',
                'type': 'json',
                'enabled': True
            },
            {
                'name': 'MalwareDomains',
                'url': 'https://mirror.cedia.org.ec/malwaredomains/justdomains',
                'type': 'text',
                'enabled': True
            },
            {
                'name': 'Blockchain Threat Intel',
                'url': 'https://raw.githubusercontent.com/stamparm/maltrail/master/trails/static/suspicious/blockchain.txt',
                'type': 'text',
                'enabled': True
            }
        ]
    
    def update_threat_database(self) -> bool:
        """Update threat database from all sources"""
        if time.time() - self.last_update < self.update_interval:
            return False
        
        success_count = 0
        for source in self.threat_sources:
            if source['enabled']:
                try:
                    if source['type'] == 'json':
                        self._process_json_feed(source['url'])
                    elif source['type'] == 'text':
                        self._process_text_feed(source['url'])
                    elif source['type'] == 'csv':
                        self._process_csv_feed(source['url'])
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to process threat source {source['name']}: {e}")
        
        self.last_update = time.time()
        self._calculate_threat_level()
        return success_count > 0
    
    def _process_json_feed(self, url: str) -> None:
        """Process JSON-based threat feed"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Process PhishTank format
        if 'url' in data and isinstance(data, list):
            for entry in data:
                if 'url' in entry:
                    domain = urlparse(entry['url']).hostname
                    if domain:
                        self.malicious_domains.add(domain)
        
        # Process other JSON formats as needed
        elif 'malicious_ips' in data:
            self.malicious_ips.update(data['malicious_ips'])
        elif 'malicious_domains' in data:
            self.malicious_domains.update(data['malicious_domains'])
        elif 'malicious_addresses' in data:
            self.malicious_addresses.update(data['malicious_addresses'])
    
    def _process_text_feed(self, url: str) -> None:
        """Process text-based threat feed (one entry per line)"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        for line in response.text.split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Try to determine what type of threat this is
                if self._is_ip_address(line):
                    self.malicious_ips.add(line)
                elif self._is_domain(line):
                    self.malicious_domains.add(line)
                elif self._is_blockchain_address(line):
                    self.malicious_addresses.add(line)
                elif self._is_exploit_pattern(line):
                    self.known_exploits.add(line)
    
    def _process_csv_feed(self, url: str) -> None:
        """Process CSV-based threat feed"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Simple CSV parsing
        for line in response.text.split('\n'):
            if line.strip() and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 2:
                    threat_type = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    if threat_type == 'ip' and self._is_ip_address(value):
                        self.malicious_ips.add(value)
                    elif threat_type == 'domain' and self._is_domain(value):
                        self.malicious_domains.add(value)
                    elif threat_type == 'address' and self._is_blockchain_address(value):
                        self.malicious_addresses.add(value)
                    elif threat_type == 'exploit':
                        self.known_exploits.add(value)
    
    def _is_ip_address(self, value: str) -> bool:
        """Check if value is a valid IP address"""
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False
    
    def _is_domain(self, value: str) -> bool:
        """Check if value is a valid domain name"""
        # Simple domain validation
        if not isinstance(value, str) or not value:
            return False
            
        # Remove any protocol prefix and extract domain
        domain = value.strip().lower()
        
        # Remove protocol prefixes
        if domain.startswith(('http://', 'https://', 'ftp://', 'ftps://')):
        	domain = domain.split('://', 1)[1]
        	
        # Remove path, query parameters, and fragments
        domain = domain.split('/')[0].split('?')[0].split('#')[0]
        
        # Remove port numbers
        if ':' in domain:
        	domain = domain.split(':', 1)[0]
        	
        # Basic domain pattern validation (RFC 1034/1123 compliant)
        domain_pattern = r'^([a-z0-9]([a-z0-9\-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$'
        
        if not re.match(domain_pattern, domain):
        	return False
        	
        # Additional domain-specific validation
        if not self._validate_domain_specific_rules(domain):
        	return False
        	
        # DNS resolution with timeout and multiple record types
        try:
        	return self._resolve_domain(domain)
        	
        except Exception as e:
        	logger.debug(f"Domain validation failed for '{domain}': {e}")
        	return False
        	
    def _validate_domain_specific_rules(self, domain: str) -> bool:
    	"""Validate domain-specific rules and restrictions"""
    	# Check for invalid patterns
    	invalid_patterns = [
    	    r'^[0-9]+\.',  # IP address-like
    	    r'\.local$',   # Local domain
    	    r'\.localhost$',
    	    r'\.test$',
    	    r'\.example$',
    	    r'\.invalid$',
    	    r'^.*\d{3,}\.',  # Suspicious number sequences
    	    r'^xn--',        # Internationalized domain (IDN) - handle with care
    	]
    	for pattern in invalid_patterns:
    		if re.match(pattern, domain):
    			
    			return False
    			
    	# Check for suspicious domain characteristics
    	parts = domain.split('.')
    	# Domain should have at least 2 parts
    	if len(parts) < 2:
    		return False
    		
    	# TLD validation
    	tld = parts[-1]
    	valid_tlds = {
    	    'com', 'org', 'net', 'edu', 'gov', 'mil', 'int',
        'io', 'ai', 'co', 'uk', 'de', 'fr', 'jp', 'cn', 
        'ru', 'br', 'in', 'au', 'ca', 'mx', 'es', 'it'
        # Add more TLDs as needed
    	}
    	# Allow unknown TLDs but log them
    	if tld not in valid_tlds:
    		logger.info(f"Domain with uncommon TLD: {domain}")
    		
    	# Subdomain length and content validation
    	for part in parts:
    		# Each part should be between 1 and 63 characters
    		if len(part) < 1 or len(part) > 63:
    			return False
    			
    		# Should not start or end with hyphen
    		if part.startswith('-') or part.endswith('-'):
    			return False
    			
    		# Should not contain consecutive hyphens
    		if '--' in part:
    			
    			return False
    			
    	return True
    	
    def _resolve_domain(self, domain: str) -> bool:
    	record_types = ['A', 'AAAA', 'CNAME', 'MX', 'NS']
    	resolved = False
    	
    	for record_type in record_types:
    		try:
    			# Set timeout for DNS resolution
    			resolver = dns.resolver.Resolver()
    			resolver.timeout = 2.0  # 2 second timeout
    			resolver.lifetime = 3.0  # 3 second total timeout
    			
    			# Query DNS    	
    			answers = resolver.resolve(domain, record_type, raise_on_no_answer=False)
    			# Query DNS 
    			if answers:
    				resolved = True
    				logger.debug(f"Domain '{domain}' resolved with {record_type} records")
    				break
    		except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
    			continue  # Try next record type
    		except dns.resolver.Timeout:
    			logger.warning(f"DNS timeout for domain: {domain}")
    			continue
    		except dns.resolver.NoNameservers:
    			logger.warning(f"No nameservers available for domain: {domain}")
    			continue
    		except dns.exception.DNSException as e:
    			logger.debug(f"DNS exception for {domain}: {e}")
    			continue
    	# Additional validation for resolved domains
    	if resolved:
    		return self._validate_resolved_domain(domain)
    		return False   	
  
    def _validate_resolved_domain(self, domain: str) -> bool:
    	# Check for suspicious IP ranges
    	try:
    		# Get A records
    		resolver = dns.resolver.Resolver()
    		resolver.timeout = 1.0
    		a_records = resolver.resolve(domain, 'A')
    		 		
    		for answer in a_records:
    		    ip = answer.address
    		    
    		    # Check for private/reserved IP ranges
    		    if ipaddress.ip_address(ip).is_private:
    		    	logger.warning(f"Domain '{domain}' resolves to private IP: {ip}")
    		    	return False
    		    	
    		    # Check for known malicious IP ranges
    		    if self._is_malicious_ip(ip):
    		    	logger.warning(f"Domain '{domain}' resolves to malicious IP: {ip}")
    		    	return False
    		    	
    	except Exception:
    		 # Continue even if IP validation fails
    		 pass
    	# WHOIS validation (optional, can be expensive)
    	if self._enable_whois_validation:
    		 try:
    		 	whois_info = whois.whois(domain)
    		 	# Check domain creation date (too new might be suspicious)
    		 	if whois_info.creation_date:
    		 		if isinstance(whois_info.creation_date, list):
    		 			creation_date = whois_info.creation_date[0]
    		 		else:
    		 			creation_date = whois_info.creation_date
    		 		if (datetime.now() - creation_date).days < 7:
    		 			logger.info(f"Recently registered domain: {domain}")
    		 			
    		 except Exception as e:
    		 	logger.debug(f"WHOIS lookup failed for {domain}: {e}")
 	    	
    	return True
    	
    def _is_malicious_ip(self, ip: str) -> bool:
         malicious_ranges = [
             
         ]
         try:
         	ip_obj = ipaddress.ip_address(ip)
         	for range_str in malicious_ranges:
         		if ip_obj in ipaddress.ip_network(range_str):
         			return True
         except ValueError:
         	pass
         return False
         
    def _is_blockchain_address(self, value: str) -> bool:
        """Check if value is a blockchain address"""
        # Ethereum address pattern
        if re.match(r'^0x[a-fA-F0-9]{40}$', value):
            return True
        
        # Bitcoin address patterns
        if (re.match(r'^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$', value) or
            re.match(r'^bc1[ac-hj-np-z02-9]{11,71}$', value)):
            return True
        
        return False
    
    def _is_exploit_pattern(self, value: str) -> bool:
        """Check if value looks like an exploit pattern"""
        exploit_indicators = [
            'eval(',
            'exec(',
            'shell_exec',
            'system(',
            'passthru(',
            'base64_decode',
            'gzinflate',
            'str_rot13',
            'assert(',
            'create_function',
            'preg_replace',
            'include(',
            'require(',
            'file_get_contents',
            'curl_exec',
            'fsockopen',
            'pfsockopen'
        ]
        
        return any(indicator in value.lower() for indicator in exploit_indicators)
    
    def _calculate_threat_level(self) -> None:
        """Calculate current threat level based on threat intelligence"""
        threat_score = 0
        
        # Score based on number of threats
        threat_count = (len(self.malicious_ips) + 
                       len(self.malicious_domains) + 
                       len(self.malicious_addresses) + 
                       len(self.known_exploits))
        
        if threat_count > 10000:
            threat_score += 8
        elif threat_count > 5000:
            threat_score += 6
        elif threat_count > 1000:
            threat_score += 4
        elif threat_count > 100:
            threat_score += 2
        
        # Recent updates increase threat level
        if time.time() - self.last_update < 3600:  # Updated in last hour
            threat_score += 2
        
        # Cap at 10
        self.threat_level = min(10, threat_score)
    
    def is_malicious(self, data: Any, context: Any = None) -> bool:
        """Check if data matches known threat patterns"""
        if isinstance(data, str):
            # Check IP addresses
            if self._is_ip_address(data) and data in self.malicious_ips:
                return True
            
            # Check domains
            if self._is_domain(data) and data in self.malicious_domains:
                return True
            
            # Check blockchain addresses
            if self._is_blockchain_address(data) and data in self.malicious_addresses:
                return True
            
            # Check for exploit patterns
            if any(exploit in data for exploit in self.known_exploits):
                return True
        
        elif isinstance(data, dict):
            # Recursively check dictionary values
            for value in data.values():
                if self.is_malicious(value, context):
                    return True
        
        elif isinstance(data, (list, tuple)):
            # Check list items
            for item in data:
                if self.is_malicious(item, context):
                    return True
        
        return False
    
    def is_address_blacklisted(self, address: str) -> bool:
        """Check if address is in threat intelligence database"""
        return address in self.malicious_addresses
    
    def get_current_threat_level(self) -> int:
        """Get current threat level (0-10)"""
        return self.threat_level
    
    def get_threat_stats(self) -> Dict[str, Any]:
        """Get threat statistics"""
        return {
            'malicious_ips': len(self.malicious_ips),
            'malicious_domains': len(self.malicious_domains),
            'malicious_addresses': len(self.malicious_addresses),
            'known_exploits': len(self.known_exploits),
            'threat_level': self.threat_level,
            'last_update': self.last_update,
            'next_update': self.last_update + self.update_interval
        }

class WASMHostFunctions:
    """Host functions that WebAssembly contracts can call to interact with the blockchain"""
    
    def __init__(self, contract_manager, contract_address: str, caller: str, gas_meter: GasMeter):
        self.contract_manager = contract_manager
        self.contract_address = contract_address
        self.caller = caller
        self.gas_meter = gas_meter
        self.linkings = {}
        
    def register_functions(self, store: wasmtime.Store, instance: wasmtime.Instance) -> None:
        """Register host functions with the WASM instance"""
        # Storage functions
        self.linkings['storage_get'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32()], 
                                               [wasmtime.ValType.i32()]), self.storage_get)
        self.linkings['storage_set'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32(), 
                                               wasmtime.ValType.i32(), wasmtime.ValType.i32()], 
                                               [wasmtime.ValType.i32()]), self.storage_set)
        
        # Contract interaction functions
        self.linkings['call_contract'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32(), 
                                                 wasmtime.ValType.i32(), wasmtime.ValType.i32()], 
                                                 [wasmtime.ValType.i32()]), self.call_contract)
        
        # Cryptographic functions
        self.linkings['keccak256'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32()], 
                                            [wasmtime.ValType.i32()]), self.keccak256)
        
        # System functions
        self.linkings['get_caller'] = wasmtime.Func(store, wasmtime.FuncType([], [wasmtime.ValType.i32()]), self.get_caller)
        self.linkings['get_balance'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32()], 
                                            [wasmtime.ValType.i64()]), self.get_balance)
        
        # Event emitting
        self.linkings['emit_event'] = wasmtime.Func(store, wasmtime.FuncType([wasmtime.ValType.i32(), wasmtime.ValType.i32()], 
                                          [wasmtime.ValType.i32()]), self.emit_event)
        
        # Link all functions to the instance
        for name, func in self.linkings.items():
            instance.exports(store)[name] = func
    
    def storage_get(self, caller: wasmtime.Caller, key_ptr: int, key_len: int) -> int:
        """Host function: Get value from contract storage"""
        try:
            self.gas_meter.consume_gas('storage_read', data_size=key_len)
            
            # Read key from WASM memory
            key = self._read_string_from_memory(caller, key_ptr, key_len)
            
            # Get value from contract storage
            contract = self.contract_manager.get_contract(self.contract_address)
            value = contract.storage.get(key, "")
            
            # Write value to WASM memory and return pointer
            return self._write_string_to_memory(caller, str(value))
            
        except Exception as e:
            logger.error(f"storage_get failed: {e}")
            return 0  # Return null pointer on error
    
    def storage_set(self, caller: wasmtime.Caller, key_ptr: int, key_len: int, 
                   value_ptr: int, value_len: int) -> int:
        """Host function: Set value in contract storage"""
        try:
            self.gas_meter.consume_gas('storage_write', data_size=key_len + value_len)
            
            # Read key and value from WASM memory
            key = self._read_string_from_memory(caller, key_ptr, key_len)
            value = self._read_string_from_memory(caller, value_ptr, value_len)
            
            # Set value in contract storage
            contract = self.contract_manager.get_contract(self.contract_address)
            success = contract.storage.set(key, value, self.caller, "WASM storage_set")
            
            return 1 if success else 0
            
        except Exception as e:
            logger.error(f"storage_set failed: {e}")
            return 0
    
    def call_contract(self, caller: wasmtime.Caller, address_ptr: int, address_len: int,
                     function_ptr: int, function_len: int) -> int:
        """Host function: Call another contract"""
        try:
            self.gas_meter.consume_gas('contract_call')
            
            # Read address and function name from WASM memory
            address = self._read_string_from_memory(caller, address_ptr, address_len)
            function_name = self._read_string_from_memory(caller, function_ptr, function_len)
            
            # Execute the contract call
            result = self.contract_manager.execute_function(
                address, function_name, self.caller, {}, self.gas_meter
            )
            
            # Write result to WASM memory
            return self._write_string_to_memory(caller, str(result.return_value))
            
        except Exception as e:
            logger.error(f"call_contract failed: {e}")
            return 0
    
    def keccak256(self, caller: wasmtime.Caller, data_ptr: int, data_len: int) -> int:
        """Host function: Compute Keccak-256 hash"""
        try:
            self.gas_meter.consume_gas('hash_operation', data_size=data_len)
            
            # Read data from WASM memory
            data = self._read_bytes_from_memory(caller, data_ptr, data_len)
            
            # Compute hash
            hash_result = hashlib.sha3_256(data).hexdigest()
            
            # Write hash to WASM memory
            return self._write_string_to_memory(caller, hash_result)
            
        except Exception as e:
            logger.error(f"keccak256 failed: {e}")
            return 0
    
    def get_caller(self, caller: wasmtime.Caller) -> int:
        """Host function: Get caller address"""
        try:
            self.gas_meter.consume_gas('address_validation')
            return self._write_string_to_memory(caller, self.caller)
        except Exception as e:
            logger.error(f"get_caller failed: {e}")
            return 0
    
    def get_balance(self, caller: wasmtime.Caller, address_ptr: int) -> int:
        """Host function: Get contract balance"""
        try:
            self.gas_meter.consume_gas('address_validation')
            
            # Read address from WASM memory
            address = self._read_string_from_memory(caller, address_ptr, 42)  # Ethereum addresses are 42 chars
            
            # Get balance (simplified - in real implementation, this would query the blockchain state)
            balance = 0  # Placeholder
            return balance
            
        except Exception as e:
            logger.error(f"get_balance failed: {e}")
            return 0
    
    def emit_event(self, caller: wasmtime.Caller, event_name_ptr: int, event_data_ptr: int) -> int:
        """Host function: Emit an event"""
        try:
            self.gas_meter.consume_gas('event_emit')
            
            # Read event name and data from WASM memory
            event_name = self._read_string_from_memory(caller, event_name_ptr, 100)  # Max 100 chars
            event_data = self._read_string_from_memory(caller, event_data_ptr, 1024)  # Max 1KB
            
            # Emit event (would be stored in contract manager)
            logger.info(f"Event emitted: {event_name} - {event_data}")
            return 1
            
        except Exception as e:
            logger.error(f"emit_event failed: {e}")
            return 0
    
    def _read_string_from_memory(self, caller: wasmtime.Caller, ptr: int, length: int) -> str:
        """Read a string from WASM memory"""
        memory = caller.get("memory")
        if not memory:
            raise ValueError("No memory available")
        
        data = memory.data_ptr(caller)[ptr:ptr+length]
        return data.decode('utf-8')
    
    def _read_bytes_from_memory(self, caller: wasmtime.Caller, ptr: int, length: int) -> bytes:
        """Read bytes from WASM memory"""
        memory = caller.get("memory")
        if not memory:
            raise ValueError("No memory available")
        
        return bytes(memory.data_ptr(caller)[ptr:ptr+length])
    
    def _write_string_to_memory(self, caller: wasmtime.Caller, data: str) -> int:
        """Write a string to WASM memory and return pointer"""
        memory = caller.get("memory")
        if not memory:
            raise ValueError("No memory available")
        
        # Convert string to bytes
        data_bytes = data.encode('utf-8')
        length = len(data_bytes)
        
        # Allocate memory
        alloc_func = caller.get("alloc")
        if not alloc_func:
            raise ValueError("No alloc function available")
        
        ptr = alloc_func(caller, length)
        if ptr == 0:
            raise ValueError("Memory allocation failed")
        
        # Write data to memory
        memory.write(caller, ptr, data_bytes)
        return ptr

class SmartContract:
    """Production-ready smart contract implementation with WebAssembly execution"""
    
    def __init__(self, contract_id: str, owner: str, contract_type: ContractType, 
                 wasm_bytecode: bytes, initial_balance: int = 0):
        self.contract_id = contract_id
        self.owner = owner
        self.contract_type = contract_type
        self.wasm_bytecode = wasm_bytecode
        self.balance = initial_balance
        self.storage = ContractStorage()
        self.storage.allowed_writers.add(owner)
        self.state = ContractState.ACTIVE
        self.created_at = time.time()
        self.last_modified = time.time()
        self.version = "1.0.0"
        self.security_level = ContractSecurityLevel.MEDIUM
        self.gas_optimizer = GasOptimizer()
        self.wasm_engine = wasmtime.Engine()
        self.wasm_store = wasmtime.Store(self.wasm_engine)
        self.wasm_module = None
        self.wasm_instance = None
        
        # Initialize WASM module
        try:
            self.wasm_module = wasmtime.Module(self.wasm_engine, wasm_bytecode)
            self._instantiate_wasm_module()
        except Exception as e:
            logger.error(f"Failed to initialize WASM module for contract {contract_id}: {e}")
            raise
    
    def _instantiate_wasm_module(self) -> None:
        """Instantiate the WASM module with host functions"""
        if not self.wasm_module:
            raise ValueError("WASM module not initialized")
        
        # Create linker and add host functions
        linker = wasmtime.Linker(self.wasm_engine)
        
        # Define memory
        memory = wasmtime.Memory(self.wasm_store, wasmtime.MemoryType(wasmtime.Limits(1, None)))
        linker.define(self.wasm_store, "env", "memory", memory)
        
        # Define alloc function (simple implementation)
        def alloc_func(caller: wasmtime.Caller, size: int) -> int:
            # Simple memory allocation - in real implementation, this would manage memory properly
            return 1024  # Fixed offset for demo
        
        linker.define_func(self.wasm_store, "env", "alloc", 
                          wasmtime.FuncType([wasmtime.ValType.i32()], [wasmtime.ValType.i32()]), 
                          alloc_func)
        
        # Instantiate the module
        self.wasm_instance = linker.instantiate(self.wasm_store, self.wasm_module)
    
    def execute_function(self, function_name: str, caller: str, args: Dict[str, Any], 
                        gas_meter: GasMeter, contract_manager: Any) -> ExecutionResult:
        """
        Execute a contract function using the WebAssembly virtual machine
        
        Args:
            function_name: Name of the function to execute
            caller: Address of the caller
            args: Function arguments
            gas_meter: Gas meter for tracking execution cost
            contract_manager: Contract manager for inter-contract calls
        
        Returns:
            ExecutionResult: Result of the execution
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Check if contract is active
            if self.state != ContractState.ACTIVE:
                return ExecutionResult(
                    success=False,
                    error=f"Contract is not active. Current state: {self.state.name}",
                    gas_used=gas_meter.gas_used
                )
            
            # Initialize host functions
            host_functions = WASMHostFunctions(contract_manager, self.contract_id, caller, gas_meter)
            
            # Get the function from WASM instance
            if not self.wasm_instance:
                return ExecutionResult(
                    success=False,
                    error="WASM instance not initialized",
                    gas_used=gas_meter.gas_used
                )
            
            func = self.wasm_instance.exports(self.wasm_store).get(function_name)
            if not func:
                return ExecutionResult(
                    success=False,
                    error=f"Function {function_name} not found in contract",
                    gas_used=gas_meter.gas_used
                )
            
            # Prepare arguments
            wasm_args = []
            for arg_name, arg_value in args.items():
                # Convert argument to WASM compatible format
                if isinstance(arg_value, (int, float)):
                    wasm_args.append(wasmtime.Val.i32(int(arg_value)))
                elif isinstance(arg_value, str):
                    # Write string to memory and pass pointer
                    ptr = host_functions._write_string_to_memory(self.wasm_store, arg_value)
                    wasm_args.append(wasmtime.Val.i32(ptr))
                else:
                    # Serialize complex types to JSON
                    json_str = json.dumps(arg_value)
                    ptr = host_functions._write_string_to_memory(self.wasm_store, json_str)
                    wasm_args.append(wasmtime.Val.i32(ptr))
            
            # Execute the function
            result = func(self.wasm_store, *wasm_args)
            
            # Process result
            if result:
                # Read return value from memory if needed
                return_value = result[0].value if hasattr(result[0], 'value') else result[0]
            else:
                return_value = None
            
            execution_time = time.time() - start_time
            memory_used = self._get_memory_usage() - start_memory
            
            return ExecutionResult(
                success=True,
                return_value=return_value,
                gas_used=gas_meter.gas_used,
                execution_time=execution_time,
                memory_used=memory_used
            )
            
        except OutOfGasError as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"Out of gas: {e}",
                gas_used=gas_meter.gas_used,
                execution_time=execution_time,
                memory_used=self._get_memory_usage() - start_memory
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing function {function_name}: {e}")
            return ExecutionResult(
                success=False,
                error=str(e),
                gas_used=gas_meter.gas_used,
                execution_time=execution_time,
                memory_used=self._get_memory_usage() - start_memory
            )
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage of the process"""
        process = psutil.Process()
        return process.memory_info().rss
    
    def upgrade_contract(self, new_wasm_bytecode: bytes, upgrade_reason: str, 
                        caller: str) -> bool:
        """Upgrade contract with new WASM bytecode"""
        if caller != self.owner:
            return False
        
        if self.state != ContractState.ACTIVE:
            return False
        
        try:
            # Create backup of current state
            backup_bytecode = self.wasm_bytecode
            backup_module = self.wasm_module
            backup_instance = self.wasm_instance
            
            # Update to new bytecode
            self.wasm_bytecode = new_wasm_bytecode
            self.wasm_module = wasmtime.Module(self.wasm_engine, new_wasm_bytecode)
            self._instantiate_wasm_module()
            
            self.version = self._increment_version(self.version)
            self.last_modified = time.time()
            
            logger.info(f"Contract {self.contract_id} upgraded by {caller}. Reason: {upgrade_reason}")
            return True
            
        except Exception as e:
            # Restore backup on failure
            self.wasm_bytecode = backup_bytecode
            self.wasm_module = backup_module
            self.wasm_instance = backup_instance
            logger.error(f"Contract upgrade failed: {e}")
            return False
    
    def _increment_version(self, current_version: str) -> str:
        """Increment semantic version"""
        parts = current_version.split('.')
        if len(parts) == 3:
            try:
                major, minor, patch = map(int, parts)
                return f"{major}.{minor}.{patch + 1}"
            except ValueError:
                pass
        return current_version
    
    def transfer_ownership(self, new_owner: str, caller: str) -> bool:
        """Transfer contract ownership"""
        if caller != self.owner:
            return False
        
        old_owner = self.owner
        self.owner = new_owner
        self.storage.allowed_writers.discard(old_owner)
        self.storage.allowed_writers.add(new_owner)
        self.last_modified = time.time()
        
        logger.info(f"Contract {self.contract_id} ownership transferred from {old_owner} to {new_owner}")
        return True
    
    def destroy_contract(self, caller: str, reason: str) -> bool:
        """Destroy the contract and clean up resources"""
        if caller != self.owner:
            return False
        
        self.state = ContractState.DESTROYED
        self.storage = ContractStorage()  # Clear storage
        self.wasm_instance = None
        self.wasm_module = None
        self.wasm_bytecode = b''
        self.last_modified = time.time()
        
        logger.info(f"Contract {self.contract_id} destroyed by {caller}. Reason: {reason}")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert contract to dictionary for serialization"""
        return {
            'contract_id': self.contract_id,
            'owner': self.owner,
            'contract_type': self.contract_type.name,
            'balance': self.balance,
            'state': self.state.name,
            'created_at': self.created_at,
            'last_modified': self.last_modified,
            'version': self.version,
            'security_level': self.security_level.name,
            'storage': self.storage.to_dict(),
            'wasm_bytecode_size': len(self.wasm_bytecode)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get contract statistics"""
        return {
            'contract_id': self.contract_id,
            'owner': self.owner,
            'balance': self.balance,
            'state': self.state.name,
            'storage_size': len(str(self.storage.storage)),
            'storage_entries': len(self.storage.storage),
            'audit_log_entries': len(self.storage.audit_log),
            'version': self.version,
            'age_days': (time.time() - self.created_at) / 86400,
            'last_modified_days': (time.time() - self.last_modified) / 86400
        }

class ContractManager:
    """Advanced contract manager with atomic state transitions and inter-contract calls"""
    
    def __init__(self, db_path: str = "contracts_db"):
        self.contracts: Dict[str, SmartContract] = {}
        self.db = plyvel.DB(db_path, create_if_missing=True)
        self.security = ContractSecurity()
        self.call_stack: List[Dict] = []
        self.execution_cache: Dict[str, Any] = {}
        self.state_journal: Dict[str, Dict] = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.lock = threading.RLock()
        self.gas_price = 1  # Base gas price in gwei
        
        # Load contracts from database
        self._load_contracts_from_db()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _load_contracts_from_db(self) -> None:
        """Load contracts from LevelDB"""
        try:
            for key, value in self.db:
                if key.startswith(b'contract_'):
                    try:
                        contract_data = pickle.loads(value)
                        contract = SmartContract(
                            contract_data['contract_id'],
                            contract_data['owner'],
                            ContractType[contract_data['contract_type']],
                            contract_data['wasm_bytecode'],
                            contract_data['balance']
                        )
                        contract.storage = contract_data.get('storage', ContractStorage())
                        contract.state = ContractState[contract_data.get('state', 'ACTIVE')]
                        contract.version = contract_data.get('version', '1.0.0')
                        self.contracts[contract.contract_id] = contract
                    except Exception as e:
                        logger.error(f"Failed to load contract from DB: {e}")
        except Exception as e:
            logger.error(f"Error loading contracts from database: {e}")
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Threat intelligence updates
        def threat_intel_updater():
            while True:
                try:
                    self.security.threat_intelligence.update_threat_database()
                    self.security.update_from_threat_intelligence()
                    time.sleep(3600)  # Update every hour
                except Exception as e:
                    logger.error(f"Threat intelligence update failed: {e}")
                    time.sleep(300)  # Retry after 5 minutes
        
        # Gas price updater
        def gas_price_updater(self):
            """Production-ready gas price updater for RAYONIX blockchain with network-based calculation"""
            # Configuration for gas price calculation
            GAS_PRICE_CONFIG = {
                'min_gas_price': 1,  # Minimum gas price in RXY
                'max_gas_price': 100,  # Maximum gas price in RXY
                'base_gas_price': 5,  # Base gas price in normal conditions
                'adjustment_sensitivity': 0.2,  # How quickly gas price adjusts to network conditions
                'update_interval': 30,  # Update every 30 seconds in normal conditions
                'emergency_update_interval': 5,  # Update every 5 seconds during high congestion
                'max_mempool_size': 10000,  # Maximum mempool size for congestion calculation
                'target_block_utilization': 0.7,  # Target block space utilization (70%)
    
            }
            
    async def fetch_network_conditions(self):
        """Fetch current network conditions from peers"""
        network_stats = {
            'mempool_size': 0,
            'pending_transactions': 0,
            'average_fee_rate': 0,
            'block_utilization': 0,
            'network_latency': 0,
            'validator_count': 0
        }
        
        try:
            # Get active peers
            peers = self.network.get_peers()
            if not peers:
                return network_stats
            
            # Query multiple peers for network statistics
            peer_responses = []
            async with aiohttp.ClientSession() as session:
                for peer in peers[:5]:  # Query first 5 peers
                    try:
                        async with session.get(
                            f"http://{peer}/network/stats",
                            timeout=5
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                peer_responses.append(data)
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        continue
            
            if not peer_responses:
                return network_stats
            
            # Calculate averages from peer responses
            network_stats['mempool_size'] = statistics.median(
                [resp.get('mempool_size', 0) for resp in peer_responses]
            )
            network_stats['pending_transactions'] = statistics.median(
                [resp.get('pending_transactions', 0) for resp in peer_responses]
            )
            network_stats['average_fee_rate'] = statistics.median(
                [resp.get('average_fee_rate', GAS_PRICE_CONFIG['base_gas_price']) for resp in peer_responses]
            )
            network_stats['block_utilization'] = statistics.median(
                [resp.get('block_utilization', 0) for resp in peer_responses]
            )
            network_stats['validator_count'] = statistics.median(
                [resp.get('active_validators', 1) for resp in peer_responses]
            )
            
        except Exception as e:
            logger.warning(f"Failed to fetch network conditions: {e}")
        
        return network_stats
    
    def calculate_dynamic_gas_price(self, network_stats, local_stats):
        """Calculate dynamic gas price based on network and local conditions"""
        # Base gas price
        new_price = GAS_PRICE_CONFIG['base_gas_price']
        
        # Factor 1: Mempool congestion (most important)
        mempool_congestion = min(1.0, network_stats['mempool_size'] / GAS_PRICE_CONFIG['max_mempool_size'])
        congestion_factor = 1.0 + (mempool_congestion * 2.0)  # 1.0 to 3.0
        
        # Factor 2: Block utilization
        utilization_factor = 1.0
        if network_stats['block_utilization'] > GAS_PRICE_CONFIG['target_block_utilization']:
            over_utilization = (network_stats['block_utilization'] - GAS_PRICE_CONFIG['target_block_utilization']) / (1.0 - GAS_PRICE_CONFIG['target_block_utilization'])
            utilization_factor = 1.0 + (over_utilization * 1.5)  # 1.0 to 2.5
        
        # Factor 3: Validator count (more validators = lower fees)
        validator_factor = max(0.5, min(2.0, 10.0 / max(1, network_stats['validator_count'])))
        
        # Factor 4: Time of day (higher during business hours)
        current_time = time.time()
        hour = (current_time % 86400) / 3600
        time_factor = 1.2 if 9 <= hour <= 17 else 0.9
        
        # Factor 5: Network latency (higher latency = higher fees)
        latency_factor = 1.0 + min(1.0, network_stats['network_latency'] / 1000)  # 1.0 to 2.0
        
        # Calculate final price
        new_price = new_price * congestion_factor * utilization_factor * validator_factor * time_factor * latency_factor
        
        # Apply bounds
        new_price = max(GAS_PRICE_CONFIG['min_gas_price'], min(GAS_PRICE_CONFIG['max_gas_price'], new_price))
        
        # Smooth adjustment to prevent rapid fluctuations
        if hasattr(self, 'gas_price'):
            max_change = GAS_PRICE_CONFIG['base_gas_price'] * GAS_PRICE_CONFIG['adjustment_sensitivity']
            new_price = max(self.gas_price - max_change, min(self.gas_price + max_change, new_price))
        
        return int(new_price)
    
    def get_local_stats(self):
        """Get local node statistics"""
        return {
            'mempool_size': len(self.transaction_manager.mempool),
            'pending_transactions': sum(1 for tx in self.transaction_manager.mempool.values()),
            'local_fee_estimate': self._calculate_local_fee_estimate(),
            'block_production_rate': self._get_block_production_rate(),
            'node_connectivity': self.network.get_connection_count()
        }
    
    def _calculate_local_fee_estimate(self):
        """Calculate fee estimate based on local mempool"""
        if not self.transaction_manager.mempool:
            return GAS_PRICE_CONFIG['base_gas_price']
        
        # Calculate average fee rate from recent transactions
        fee_rates = []
        for tx_data in list(self.transaction_manager.mempool.values())[-100:]:  # Last 100 transactions
            if len(tx_data) >= 3:
                fee_rate = tx_data[2]  # fee_rate is the third element
                fee_rates.append(fee_rate)
        
        if not fee_rates:
            return GAS_PRICE_CONFIG['base_gas_price']
        
        return statistics.median(fee_rates)
    
    def _get_block_production_rate(self):
        """Get recent block production rate"""
        try:
            # Get last 10 blocks
            recent_blocks = []
            current_hash = self.chain_head
            for _ in range(10):
                block = self.database.get_block(current_hash)
                if not block:
                    break
                recent_blocks.append(block)
                current_hash = block.header.previous_hash
            
            if len(recent_blocks) < 2:
                return 0
            
            # Calculate average block time
            block_times = []
            for i in range(1, len(recent_blocks)):
                time_diff = recent_blocks[i-1].header.timestamp - recent_blocks[i].header.timestamp
                block_times.append(time_diff)
            
            if not block_times:
                return 0
            
            avg_block_time = statistics.mean(block_times)
            return 1 / avg_block_time if avg_block_time > 0 else 0
            
        except Exception:
            return 0
    
    def determine_update_interval(self, network_stats):
        """Determine appropriate update interval based on network conditions"""
        base_interval = GAS_PRICE_CONFIG['update_interval']
        
        # Reduce interval during high congestion
        congestion = network_stats['mempool_size'] / GAS_PRICE_CONFIG['max_mempool_size']
        if congestion > 0.8:
            return GAS_PRICE_CONFIG['emergency_update_interval']
        elif congestion > 0.6:
            return base_interval / 2
        
        return base_interval
    
    async def update_gas_price_async(self):
        """Async gas price update routine"""
        self.update_errors = 0
        
        while getattr(self, 'running', True):
            try:
                # Fetch network conditions from peers
                network_stats = await self.fetch_network_conditions()
                
                # Get local statistics
                local_stats = self.get_local_stats()
                
                # Calculate new gas price
                new_price = self.calculate_dynamic_gas_price(network_stats, local_stats)
                
                # Update gas price atomically
                with self.lock:
                    old_price = getattr(self, 'gas_price', GAS_PRICE_CONFIG['base_gas_price'])
                    self.gas_price = new_price
                
                # Log price change
                if old_price != new_price:
                    logger.info(
                        f"Gas price updated: {old_price} → {new_price} RXY | "
                        f"Mempool: {network_stats['mempool_size']} | "
                        f"Utilization: {network_stats['block_utilization']:.1%} | "
                        f"Validators: {network_stats['validator_count']}"
                    )
                
                # Update metrics
                self._update_gas_metrics(new_price, network_stats, local_stats)
                
                # Determine next update interval
                update_interval = self.determine_update_interval(network_stats)
                
                # Reset error counter on successful update
                self.update_errors = 0
                
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                self.update_errors += 1
                logger.error(f"Gas price update failed (attempt {self.update_errors}): {e}")
                
                # Exponential backoff with cap
                backoff_time = min(300, 2 ** min(self.update_errors, 8))
                await asyncio.sleep(backoff_time)
                
                # If we have too many errors, reset to base price
                if self.update_errors > 10:
                    with self.lock:
                        self.gas_price = GAS_PRICE_CONFIG['base_gas_price']
                    logger.warning("Reset gas price to base due to persistent errors")
    
    def _update_gas_metrics(self, new_price, network_stats, local_stats):
        """Update gas price metrics for monitoring"""
        metrics = {
            'current_gas_price': new_price,
            'timestamp': time.time(),
            'network_mempool_size': network_stats['mempool_size'],
            'network_block_utilization': network_stats['block_utilization'],
            'network_validator_count': network_stats['validator_count'],
            'local_mempool_size': local_stats['mempool_size'],
            'block_production_rate': local_stats['block_production_rate'],
            'update_errors': self.update_errors
        }
        
        # Store metrics for historical analysis
        if not hasattr(self, 'gas_price_history'):
            self.gas_price_history = deque(maxlen=1000)
        
        self.gas_price_history.append(metrics)
        
        # Update performance metrics
        if hasattr(self, 'performance_metrics'):
            self.performance_metrics['gas_price'] = new_price
            self.performance_metrics['gas_price_last_updated'] = time.time()
    
    def get_gas_price_stats(self):
        """Get current gas price statistics"""
        if not hasattr(self, 'gas_price'):
            return {
                'current_price': GAS_PRICE_CONFIG['base_gas_price'],
                'status': 'not_initialized',
                'last_update': 0
            }
        
        return {
            'current_price': self.gas_price,
            'min_price': GAS_PRICE_CONFIG['min_gas_price'],
            'max_price': GAS_PRICE_CONFIG['max_gas_price'],
            'base_price': GAS_PRICE_CONFIG['base_gas_price'],
            'last_updated': getattr(self, '_last_gas_price_update', 0),
            'update_errors': getattr(self, 'update_errors', 0),
            'history_size': len(getattr(self, 'gas_price_history', [])),
            'status': 'active'
        }
    
    # Start the gas price updater when the node starts
    def start_gas_price_updater(self):
        """Start the gas price updater task"""
        if hasattr(self, '_gas_price_task') and not self._gas_price_task.done():
            return
        
        self._gas_price_task = asyncio.create_task(self.update_gas_price_async())
        logger.info("Gas price updater started")
    
    def stop_gas_price_updater(self):
        """Stop the gas price updater task"""
        if hasattr(self, '_gas_price_task'):
            self._gas_price_task.cancel()
            logger.info("Gas price updater stopped")
    			                      
                            
    def deploy_contract(self, contract_id: str, owner: str, contract_type: ContractType,
                       wasm_bytecode: bytes, initial_balance: int = 0) -> bool:
        """
        Deploy a new contract with WASM bytecode (not Python source code)
        
        Args:
            contract_id: Unique contract identifier
            owner: Owner's address
            contract_type: Type of contract
            wasm_bytecode: Pre-compiled WebAssembly bytecode
            initial_balance: Initial contract balance
        
        Returns:
            bool: True if deployment successful
        """
        with self.lock:
            if contract_id in self.contracts:
                logger.error(f"Contract {contract_id} already exists")
                return False
            
            # Validate WASM bytecode
            if not self._validate_wasm_bytecode(wasm_bytecode):
                logger.error(f"Invalid WASM bytecode for contract {contract_id}")
                return False
            
            # Security check
            if self.security.is_blacklisted(owner):
                logger.error(f"Owner {owner} is blacklisted")
                return False
            
            try:
                # Create new contract
                contract = SmartContract(
                    contract_id, owner, contract_type, wasm_bytecode, initial_balance
                )
                
                # Store in memory and database
                self.contracts[contract_id] = contract
                self._save_contract_to_db(contract)
                
                logger.info(f"Contract {contract_id} deployed by {owner}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to deploy contract {contract_id}: {e}")
                return False
    
    def _validate_wasm_bytecode(self, wasm_bytecode: bytes) -> bool:
        """Validate WASM bytecode for safety and compatibility"""
        if not wasm_bytecode or len(wasm_bytecode) < 8:
            return False
        
        # Check WASM magic number
        if wasm_bytecode[:4] != b'\x00asm':
            return False
        
        # Check version
        if wasm_bytecode[4:8] != b'\x01\x00\x00\x00':
            return False
        
        # Additional validation could include:
        # - Maximum size limits
        # - Forbidden opcodes
        # - Resource usage estimation
        # - Known vulnerability patterns
        
        return True
    
    def execute_function(self, contract_id: str, function_name: str, caller: str,
                       args: Dict[str, Any], gas_limit: int = 1000000) -> ExecutionResult:
        """
        Execute a contract function with atomic state transitions
        
        Args:
            contract_id: Contract identifier
            function_name: Function to execute
            caller: Caller's address
            args: Function arguments
            gas_limit: Maximum gas allowed
        
        Returns:
            ExecutionResult: Execution result with detailed metrics
        """
        # Create state journal for atomic execution
        journal_id = self._create_state_journal(contract_id)
        
        try:
            with self.lock:
                # Security checks
                if not self._pre_execution_checks(contract_id, caller, function_name, args):
                    return ExecutionResult(
                        success=False,
                        error="Security check failed",
                        gas_used=0
                    )
                
                # Get contract
                contract = self.contracts.get(contract_id)
                if not contract:
                    return ExecutionResult(
                        success=False,
                        error=f"Contract {contract_id} not found",
                        gas_used=0
                    )
                
                # Initialize gas meter
                gas_meter = GasMeter(gas_limit, self.gas_price)
                
                # Execute the function
                result = contract.execute_function(
                    function_name, caller, args, gas_meter, self
                )
                
                # Post-execution checks
                if result.success:
                    security_ok, error_msg = self.security.check_resource_limits(
                        result.execution_time,
                        result.memory_used,
                        result.gas_used,
                        len(str(contract.storage.storage))
                    )
                    
                    if not security_ok:
                        result.success = False
                        result.error = f"Resource limit exceeded: {error_msg}"
                
                # Commit or revert state changes
                if result.success:
                    self._commit_state_journal(journal_id)
                else:
                    self._revert_state_journal(journal_id)
                
                return result
                
        except Exception as e:
            # Always revert on unexpected errors
            self._revert_state_journal(journal_id)
            logger.error(f"Unexpected error executing {function_name} on {contract_id}: {e}")
            return ExecutionResult(
                success=False,
                error=f"Unexpected error: {e}",
                gas_used=0
            )
    
    def _pre_execution_checks(self, contract_id: str, caller: str, 
                            function_name: str, args: Dict[str, Any]) -> bool:
        """Perform comprehensive pre-execution security checks"""
        # Blacklist check
        if self.security.is_blacklisted(caller):
            logger.warning(f"Blacklisted caller {caller} attempted to execute {function_name}")
            return False
        
        # Input validation
        for arg_name, arg_value in args.items():
            is_valid, error = self.security.validate_input(
                arg_value, 
                context={'caller': caller, 'operation': function_name, 'contract': contract_id}
            )
            if not is_valid:
                logger.warning(f"Invalid input {arg_name} from {caller}: {error}")
                return False
        
        # Rate limiting
        if not self.security.check_rate_limit(caller, f"execute_{function_name}", 1):
            logger.warning(f"Rate limit exceeded for {caller} executing {function_name}")
            return False
        
        return True
    
    def _create_state_journal(self, contract_id: str) -> str:
        """Create a journal for atomic state transitions"""
        journal_id = f"journal_{contract_id}_{time.time()}_{secrets.token_hex(4)}"
        
        with self.lock:
            contract = self.contracts.get(contract_id)
            if contract:
                # Create deep copy of contract state for journal
                journal_state = {
                    'storage': pickle.dumps(contract.storage),
                    'balance': contract.balance,
                    'state': contract.state,
                    'version': contract.version
                }
                self.state_journal[journal_id] = journal_state
        
        return journal_id
    
    def _commit_state_journal(self, journal_id: str) -> None:
        """Commit state changes from journal"""
        # In our implementation, since we're using copy-on-write at the contract level,
        # the commit is essentially just removing the journal as changes are already applied
        with self.lock:
            if journal_id in self.state_journal:
                del self.state_journal[journal_id]
    
    def _revert_state_journal(self, journal_id: str) -> None:
        """Revert state changes using journal"""
        with self.lock:
            if journal_id in self.state_journal:
                journal = self.state_journal[journal_id]
                contract_id = journal_id.split('_')[1]
                
                if contract_id in self.contracts:
                    contract = self.contracts[contract_id]
                    # Restore state from journal
                    contract.storage = pickle.loads(journal['storage'])
                    contract.balance = journal['balance']
                    contract.state = journal['state']
                    contract.version = journal['version']
                
                del self.state_journal[journal_id]
    
    def call_contract(self, from_contract: str, to_contract: str, function_name: str,
                     args: Dict[str, Any], gas_limit: int) -> ExecutionResult:
        """
        Secure inter-contract call implementation with proper call stack management
        
        Args:
            from_contract: Calling contract ID
            to_contract: Target contract ID
            function_name: Function to call
            args: Function arguments
            gas_limit: Gas limit for the call
        
        Returns:
            ExecutionResult: Result of the inter-contract call
        """
        # Push call to call stack
        call_id = self._push_call_stack(from_contract, to_contract, function_name, gas_limit)
        
        try:
            # Execute the call
            result = self.execute_function(to_contract, function_name, from_contract, args, gas_limit)
            
            # Update call stack with result
            self._update_call_stack(call_id, result)
            
            return result
            
        except Exception as e:
            # Pop call from stack on error
            self._pop_call_stack(call_id)
            logger.error(f"Inter-contract call failed: {e}")
            return ExecutionResult(
                success=False,
                error=f"Inter-contract call failed: {e}",
                gas_used=0
            )
    
    def _push_call_stack(self, from_contract: str, to_contract: str, 
                        function_name: str, gas_limit: int) -> str:
        """Push a call to the call stack"""
        call_id = f"call_{from_contract}_{to_contract}_{time.time()}_{secrets.token_hex(4)}"
        
        call_info = {
            'id': call_id,
            'from': from_contract,
            'to': to_contract,
            'function': function_name,
            'gas_limit': gas_limit,
            'start_time': time.time(),
            'status': 'executing'
        }
        
        with self.lock:
            self.call_stack.append(call_info)
        
        return call_id
    
    def _update_call_stack(self, call_id: str, result: ExecutionResult) -> None:
        """Update call stack with execution result"""
        with self.lock:
            for call in self.call_stack:
                if call['id'] == call_id:
                    call['end_time'] = time.time()
                    call['status'] = 'completed' if result.success else 'failed'
                    call['gas_used'] = result.gas_used
                    call['error'] = result.error
                    break
    
    def _pop_call_stack(self, call_id: str) -> None:
        """Remove a call from the call stack"""
        with self.lock:
            self.call_stack = [call for call in self.call_stack if call['id'] != call_id]
    
    def get_contract(self, contract_id: str) -> Optional[SmartContract]:
        """Get contract by ID"""
        return self.contracts.get(contract_id)
    
    def _save_contract_to_db(self, contract: SmartContract) -> None:
        """Save contract to LevelDB"""
        try:
            contract_data = {
                'contract_id': contract.contract_id,
                'owner': contract.owner,
                'contract_type': contract.contract_type.name,
                'wasm_bytecode': contract.wasm_bytecode,
                'balance': contract.balance,
                'storage': contract.storage,
                'state': contract.state.name,
                'version': contract.version
            }
            
            key = f"contract_{contract.contract_id}".encode()
            value = pickle.dumps(contract_data)
            self.db.put(key, value)
            
        except Exception as e:
            logger.error(f"Failed to save contract {contract.contract_id} to DB: {e}")
    
    def get_contract_stats(self, contract_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a contract"""
        contract = self.contracts.get(contract_id)
        if not contract:
            return None
        
        return contract.get_stats()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        with self.lock:
            total_contracts = len(self.contracts)
            total_balance = sum(contract.balance for contract in self.contracts.values())
            total_storage = sum(len(str(contract.storage.storage)) for contract in self.contracts.values())
            
            return {
                'total_contracts': total_contracts,
                'total_balance': total_balance,
                'total_storage_bytes': total_storage,
                'active_calls': len(self.call_stack),
                'pending_journals': len(self.state_journal),
                'threat_level': self.security.threat_intelligence.get_current_threat_level(),
                'gas_price': self.gas_price,
                'blacklisted_addresses': len(self.security.blacklisted_addresses)
            }
    
    def cleanup_old_journals(self) -> None:
        """Clean up old state journals"""
        with self.lock:
            current_time = time.time()
            journals_to_remove = []
            
            for journal_id, journal in self.state_journal.items():
                # Journals older than 1 hour should be cleaned up
                if current_time - float(journal_id.split('_')[3]) > 3600:
                    journals_to_remove.append(journal_id)
            
            for journal_id in journals_to_remove:
                del self.state_journal[journal_id]
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self.thread_pool.shutdown()
            self.db.close()
        except:
            pass
# Add class-level configuration
@property
def _enable_whois_validation(self) -> bool:
    """Whether to enable WHOIS validation (can be performance intensive)"""
    return getattr(self, '_whois_validation_enabled', False)

@_enable_whois_validation.setter
def _enable_whois_validation(self, value: bool):
    self._whois_validation_enabled = value              