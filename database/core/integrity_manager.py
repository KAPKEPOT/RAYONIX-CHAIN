import hashlib
import time
import threading
import logging
import struct
import asyncio
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.hmac import HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import os
from merkle_system.merkle import SparseMerkleTree, HashAlgorithm, MerkleTree, MerkleTreeStats, ProofFormat
from config.merkle_config import MerkleTreeConfig

logger = logging.getLogger(__name__)

class IntegrityManager:
    def __init__(self, db_path: str, config: MerkleTreeConfig):
        """Initialize Integrity Manager"""
        self.config = config
        self.db_path = db_path
        self.merkle_tree: Optional[SparseMerkleTree] = None
        self.merkle_config = MerkleTreeConfig(
            hash_algorithm=config.hash_algorithm,
            double_hash=config.double_hash
        )
        self.key_to_index: Dict[bytes, int] = {}
        self.index_to_key: Dict[int, bytes] = {}
        self.next_index = 0
        self.integrity_verified = False
        self.last_integrity_check = 0
        self.corrupted_keys: Set[bytes] = set()
        self.recovery_attempts: Dict[bytes, int] = {}
        self._lock = threading.RLock()
        self.stats = MerkleTreeStats()
        
        self._initialize_merkle_tree()
    
    def _initialize_merkle_tree(self):
        """Initialize the Merkle tree and load existing state"""
        if not self.config.enabled:
            logger.info("Merkle integrity protection disabled")
            return
            
        try:
            self.merkle_tree = SparseMerkleTree(
                depth=self.config.merkle_tree_depth,
                config=self.merkle_config
            )
            
            # Load existing key-index mappings
            self._load_merkle_state()
            self.integrity_verified = True
            logger.info("Merkle integrity manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Merkle integrity manager: {e}")
            self.merkle_tree = None
    
    def _load_merkle_state(self):
        """Load existing Merkle tree state from disk"""
        state_file = Path(self.db_path) / "merkle_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                
                self.key_to_index = {bytes.fromhex(k): v for k, v in state.get('key_to_index', {}).items()}
                self.index_to_key = {int(k): bytes.fromhex(v) for k, v in state.get('index_to_key', {}).items()}
                self.next_index = state.get('next_index', 0)
                self.corrupted_keys = {bytes.fromhex(k) for k in state.get('corrupted_keys', [])}
                
                logger.info(f"Loaded Merkle state: {len(self.key_to_index)} keys, {len(self.corrupted_keys)} corrupted")
                
            except Exception as e:
                logger.warning(f"Failed to load Merkle state: {e}")
                # Continue with empty state
    
    def _save_merkle_state(self):
        """Save Merkle tree state to disk"""
        if not self.config.enabled:
            return
            
        try:
            state_file = Path(self.db_path) / "merkle_state.json"
            state = {
                'key_to_index': {k.hex(): v for k, v in self.key_to_index.items()},
                'index_to_key': {k: v.hex() for k, v in self.index_to_key.items()},
                'next_index': self.next_index,
                'corrupted_keys': [k.hex() for k in self.corrupted_keys],
                'timestamp': time.time()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save Merkle state: {e}")
    
    def _get_leaf_index(self, key: bytes) -> int:
        """Get or create leaf index for a key"""
        with self._lock:
            if key in self.key_to_index:
                return self.key_to_index[key]
            
            # Assign new index
            index = self.next_index
            self.key_to_index[key] = index
            self.index_to_key[index] = key
            self.next_index += 1
            
            # Save state periodically
            if self.next_index % 1000 == 0:
                self._save_merkle_state()
                
            return index
    
    def _calculate_leaf_data(self, key: bytes, value: Any) -> str:
        """Calculate leaf data for Merkle tree from key-value pair"""
        try:
            # Create deterministic representation for hashing
            key_hash = hashlib.sha256(key).hexdigest()
            
            if isinstance(value, bytes):
                value_hash = hashlib.sha256(value).hexdigest()
            else:
                value_str = str(value)
                value_hash = hashlib.sha256(value_str.encode('utf-8')).hexdigest()
            
            return key_hash + value_hash
            
        except Exception as e:
            logger.error(f"Failed to calculate leaf data for key {key.hex()}: {e}")
            # Fallback: use key only
            return hashlib.sha256(key).hexdigest() * 2
    
    def register_put_operation(self, key: bytes, value: Any) -> bool:
        """Register a put operation in the Merkle tree"""
        if not self.config.enabled or not self.merkle_tree:
            return True
            
        try:
            with self._lock:
                leaf_index = self._get_leaf_index(key)
                leaf_data = self._calculate_leaf_data(key, value)
                
                self.merkle_tree.update_leaf(leaf_index, leaf_data)
                
                # Remove from corrupted keys if it was there
                if key in self.corrupted_keys:
                    self.corrupted_keys.remove(key)
                    self.recovery_attempts.pop(key, None)
                #Use _get_data_size instead of len()
                data_size = self._get_data_size(value)
                self.stats.record_operation('register_put', data_size, 0.0, True)
                return True
                
        except Exception as e:
            data_size = self._get_data_size(value)
            self.stats.record_operation('register_put', data_size, 0.0, False)
            logger.error(f"Failed to register put operation for key {key.hex()}: {e}")
            return False
    
    def verify_data_integrity(self, key: bytes, value: Any) -> Tuple[bool, Optional[str]]:
        """Verify data integrity using Merkle tree"""
        if not self.config.enabled or not self.merkle_tree:
            return True, "Merkle verification disabled"
            
        if key in self.corrupted_keys:
            return False, "Key marked as corrupted"
            
        try:
            with self._lock:
                if key not in self.key_to_index:
                    return True, "Key not in Merkle tree (new data)"
                
                leaf_index = self.key_to_index[key]
                leaf_data = self._calculate_leaf_data(key, value)
                
                start_time = time.time()
                is_valid = self.merkle_tree.verify_leaf(leaf_index, leaf_data)
                duration = time.time() - start_time
                
                self.stats.record_operation(
                    'verify_integrity', 
                    len(value), 
                    duration, 
                    is_valid,
                    {'key': key.hex()}
                )
                
                if not is_valid:
                    logger.warning(f"Data integrity violation detected for key: {key.hex()}")
                    self.corrupted_keys.add(key)
                    return False, "Merkle integrity check failed"
                
                return True, "Integrity verified"
                
        except Exception as e:
            logger.error(f"Integrity verification failed for key {key.hex()}: {e}")
            return False, f"Verification error: {e}"
    
    def get_proof(self, key: bytes, value: Any) -> Optional[bytes]:
        """Get Merkle proof for key-value pair"""
        if not self.config.enabled or not self.merkle_tree:
            return None
            
        try:
            with self._lock:
                if key not in self.key_to_index:
                    return None
                
                leaf_index = self.key_to_index[key]
                leaf_data = self._calculate_leaf_data(key, value)
                
                if hasattr(self.config, 'merkle_proof_format'):
                	proof_format = getattr(ProofFormat, self.config.merkle_proof_format.upper())
                else:
                	proof_format = ProofFormat.BINARY
                proof = self.merkle_tree.get_proof(leaf_index, proof_format)
                return proof
                
        except Exception as e:
            logger.error(f"Failed to get proof for key {key.hex()}: {e}")
            return None
    
    def register_delete_operation(self, key: bytes):
        """Register a delete operation in the Merkle tree"""
        if not self.config.enabled or not self.merkle_tree:
            return
            
        try:
            with self._lock:
                if key in self.key_to_index:
                    leaf_index = self.key_to_index[key]
                    # Set to default value (effectively removing from active tree)
                    default_value = "0" * 64  # Default hash for sparse tree
                    self.merkle_tree.update_leaf(leaf_index, default_value)
                    
                    # Clean up mappings (optional - we might want to keep for audit)
                    # del self.index_to_key[leaf_index]
                    # del self.key_to_index[key]
                
                if key in self.corrupted_keys:
                    self.corrupted_keys.remove(key)
                    self.recovery_attempts.pop(key, None)
                    
        except Exception as e:
            logger.error(f"Failed to register delete operation for key {key.hex()}: {e}")
    
    def attempt_recovery(self, key: bytes, original_value: Any = None) -> bool:
        """Attempt to recover a corrupted key"""
        if not self.config.auto_recover:
            return False
            
        with self._lock:
            if key not in self.corrupted_keys:
                return True
                
            # Limit recovery attempts
            attempts = self.recovery_attempts.get(key, 0)
            if attempts >= 3:
                logger.error(f"Max recovery attempts exceeded for key: {key.hex()}")
                return False
                
            self.recovery_attempts[key] = attempts + 1
            
            # If we have the original value, re-register it
            if original_value is not None:
                success = self.register_put_operation(key, original_value)
                if success:
                    logger.info(f"Successfully recovered key: {key.hex()}")
                    return True
            
            logger.warning(f"Recovery attempt {attempts + 1} failed for key: {key.hex()}")
            return False
    
    def get_integrity_root(self) -> Optional[str]:
        """Get current Merkle root hash"""
        if not self.config.enabled or not self.merkle_tree:
            return None
        return self.merkle_tree.get_root()
    
    def get_corrupted_keys(self) -> List[bytes]:
        """Get list of currently corrupted keys"""
        with self._lock:
            return list(self.corrupted_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get integrity manager statistics"""
        with self._lock:
            merkle_stats = self.stats.get_stats() if hasattr(self.stats, 'get_stats') else {}
            return {
                'enabled': self.config.enabled,
                'total_keys': len(self.key_to_index),
                'corrupted_keys': list(self.corrupted_keys),
                #'corrupted_keys': len(self.corrupted_keys),
                'integrity_root': self.get_integrity_root(),
                'recovery_attempts': dict(self.recovery_attempts),
                'merkle_stats': merkle_stats
            }
    
    def run_integrity_check(self, database) -> Dict[str, Any]:
        """Run comprehensive integrity check on database"""
        if not self.config.enabled:
            return {'status': 'disabled'}
            
        start_time = time.time()
        results = {
            'status': 'completed',
            'checked_keys': 0,
            'corrupted_keys': 0,
            'recovered_keys': 0,
            'duration': 0
        }
        
        try:
            # Check all keys in Merkle tree
            with self._lock:
                for key_bytes, index in self.key_to_index.items():
                    if key_bytes in self.corrupted_keys:
                        continue  # Already known to be corrupted
                        
                    try:
                        # Try to read the value
                        value = database.get(key_bytes, use_cache=False, check_ttl=False)
                        valid, reason = self.verify_data_integrity(key_bytes, value)
                        
                        if not valid:
                            results['corrupted_keys'] += 1
                            logger.warning(f"Integrity check failed for key: {key_bytes.hex()}")
                            
                            # Attempt recovery
                            if self.attempt_recovery(key_bytes, value):
                                results['recovered_keys'] += 1
                        
                    except (KeyNotFoundError, DatabaseError):
                        # Key doesn't exist in database but is in Merkle tree
                        self.corrupted_keys.add(key_bytes)
                        results['corrupted_keys'] += 1
                    
                    results['checked_keys'] += 1
                    
                    # Yield periodically for large databases
                    if results['checked_keys'] % 1000 == 0:
                        time.sleep(0.01)  # Prevent blocking
                
                self.last_integrity_check = time.time()
                
        except Exception as e:
            results['status'] = f'failed: {e}'
            logger.error(f"Integrity check failed: {e}")
        
        results['duration'] = time.time() - start_time
        return results

    def _get_data_size(self, value: Any) -> int:
        """Calculate approximate data size for statistics"""
        try:
            if value is None:
                return 0
            elif isinstance(value, (str, bytes, bytearray)):
                return len(value)
            elif isinstance(value, (int, float, bool)):
                return 8  # Approximate size for primitive types
            else:
                # For complex objects, use string representation length
                return len(str(value))
        except:
            return 0  # Fallback              