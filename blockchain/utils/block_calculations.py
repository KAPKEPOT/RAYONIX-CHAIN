# blockchain/utils/block_calculations.py
import hashlib
import math
from typing import List, Dict, Any

def calculate_block_hash(header_data: Dict[str, Any]) -> str:
    """Calculate block hash from header data"""
    import json
    header_str = json.dumps(header_data, sort_keys=True)
    return hashlib.sha256(header_str.encode()).hexdigest()

def calculate_merkle_root(transaction_hashes: List[str]) -> str:
    """Calculate merkle root from transaction hashes"""
    if not transaction_hashes:
        return '0' * 64
    
    # Convert all hashes to bytes
    hashes_bytes = [bytes.fromhex(h) for h in transaction_hashes]
    
    while len(hashes_bytes) > 1:
        # If odd number, duplicate the last hash
        if len(hashes_bytes) % 2 == 1:
            hashes_bytes.append(hashes_bytes[-1])
        
        new_level = []
        for i in range(0, len(hashes_bytes), 2):
            # Concatenate and hash pairs
            pair = hashes_bytes[i] + hashes_bytes[i+1]
            new_hash = hashlib.sha256(hashlib.sha256(pair).digest()).digest()
            new_level.append(new_hash)
        
        hashes_bytes = new_level
    
    return hashes_bytes[0].hex() if hashes_bytes else '0' * 64

def calculate_block_work(difficulty: int) -> int:
    """Calculate block work value based on difficulty"""
    # This implements the standard Bitcoin-style work calculation
    max_target = 0x00000000FFFF0000000000000000000000000000000000000000000000000000
    target = max_target / difficulty
    
    # Work is proportional to 2^256 / (target + 1)
    if target > 0:
        work = (2 ** 256) // (target + 1)
    else:
        work = 2 ** 256  # Maximum work for minimum difficulty
    
    return work

def calculate_difficulty(previous_difficulty: int, actual_block_time: int, 
                        target_block_time: int, adjustment_factor: float = 4.0) -> int:
    """Calculate new difficulty based on actual block time"""
    # Limit adjustment to avoid extreme changes
    actual_block_time = max(actual_block_time, target_block_time / adjustment_factor)
    actual_block_time = min(actual_block_time, target_block_time * adjustment_factor)
    
    # Calculate difficulty adjustment
    adjustment = target_block_time / actual_block_time
    new_difficulty = int(previous_difficulty * adjustment)
    
    # Apply minimum difficulty
    min_difficulty = 1
    return max(min_difficulty, new_difficulty)

def estimate_block_size(transactions: List[Any], header_size: int = 80) -> int:
    """Estimate block size in bytes"""
    transaction_sizes = sum(estimate_transaction_size(tx) for tx in transactions)
    return header_size + transaction_sizes

def estimate_transaction_size(transaction: Any) -> int:
    """Estimate transaction size in bytes"""
    # Base transaction size (version, locktime, etc.)
    base_size = 10
    
    # Input sizes (approx 150 bytes per input)
    input_size = len(transaction.inputs) * 150 if hasattr(transaction, 'inputs') else 0
    
    # Output sizes (approx 34 bytes per output)
    output_size = len(transaction.outputs) * 34 if hasattr(transaction, 'outputs') else 0
    
    return base_size + input_size + output_size

def calculate_transaction_fee(inputs_value: int, outputs_value: int) -> int:
    """Calculate transaction fee"""
    return max(0, inputs_value - outputs_value)

def calculate_fee_rate(transaction: Any) -> float:
    """Calculate fee rate in satoshis per byte"""
    fee = calculate_transaction_fee(
        sum(inp.amount for inp in transaction.inputs),
        sum(out.amount for out in transaction.outputs)
    )
    size = estimate_transaction_size(transaction)
    
    return fee / size if size > 0 else 0.0

def validate_block_header(header: Dict[str, Any]) -> bool:
    """Validate block header structure and values"""
    required_fields = ['version', 'height', 'previous_hash', 'merkle_root', 
                      'timestamp', 'difficulty', 'nonce', 'validator']
    
    # Check required fields
    for field in required_fields:
        if field not in header:
            return False
    
    # Check hash values are valid hex
    try:
        bytes.fromhex(header['previous_hash'])
        bytes.fromhex(header['merkle_root'])
    except:
        return False
    
    # Check values are within reasonable ranges
    if header['version'] not in [1, 2, 3]:
        return False
    
    if header['height'] < 0:
        return False
    
    if header['timestamp'] < 0:
        return False
    
    if header['difficulty'] < 1:
        return False
    
    return True

def calculate_network_hashrate(difficulty: int, block_time: int) -> float:
    """Calculate network hashrate based on difficulty and block time"""
    if block_time <= 0:
        return 0.0
    
    # Hashrate = difficulty * 2^32 / block_time
    return (difficulty * (2 ** 32)) / block_time

def format_hashrate(hashrate: float) -> str:
    """Format hashrate into human-readable string"""
    units = ['H/s', 'KH/s', 'MH/s', 'GH/s', 'TH/s', 'PH/s']
    unit_index = 0
    
    while hashrate >= 1000 and unit_index < len(units) - 1:
        hashrate /= 1000
        unit_index += 1
    
    return f"{hashrate:.2f} {units[unit_index]}"