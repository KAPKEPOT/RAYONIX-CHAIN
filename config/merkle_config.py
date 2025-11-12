from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union, Set, Callable
from enum import Enum

class HashAlgorithm(Enum):
    SHA256 = "sha256"
    SHA512 = "sha512"
    BLAKE2B = "blake2b"

class ProofFormat(Enum):
    BINARY = "binary"
    JSON = "json"
    HEX = "hex"
    
@dataclass
class MerkleTreeConfig:
    """Configuration for Merkle tree integrity protection"""
    enabled: bool = True
    merkle_tree_depth: int = 256
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    double_hash: bool = True
    verify_on_read: bool = True
    verify_on_write: bool = True
    auto_recover: bool = True
    store_proofs: bool = False
    proof_format: ProofFormat = ProofFormat.BINARY
    integrity_check_interval: int = 3600  # seconds