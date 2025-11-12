# merkle.py
#merkle_system/merkle.py
import hashlib
import json
import msgpack
import struct
import asyncio
import concurrent.futures
from typing import List, Dict, Optional, Tuple, Set, Iterator, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum
import threading
import time
from contextlib import contextmanager
import os
import zlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.exceptions import InvalidSignature

# Constants
MAX_BATCH_SIZE = 1000  # Maximum items to process in a single batch
DEFAULT_HASH_ALGORITHM = 'sha256'
PROOF_VERSION = 1

class HashAlgorithm(Enum):
    SHA256 = 'sha256'
    SHA3_256 = 'sha3_256'
    BLAKE2B = 'blake2b'
    BLAKE2S = 'blake2s'
    SHA512 = 'sha512'

class ProofFormat(Enum):
    JSON = 'json'
    MSGPACK = 'msgpack'
    BINARY = 'binary'

@dataclass
class MerkleTreeConfig:
    hash_algorithm: HashAlgorithm = HashAlgorithm.SHA256
    double_hash: bool = False
    use_encoding: bool = True
    enable_parallel_processing: bool = True
    batch_size: int = MAX_BATCH_SIZE
    cache_enabled: bool = True
    cache_max_size: int = 10000

class MerkleNode:
    __slots__ = ['hash', 'left', 'right', 'is_leaf', 'depth', 'index', 'size']
    
    def __init__(self, hash_value: str, left: Optional['MerkleNode'] = None, 
                 right: Optional['MerkleNode'] = None, is_leaf: bool = False, 
                 index: int = -1, size: int = 1):
        self.hash = hash_value
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.depth = 0
        self.index = index
        self.size = size  # Number of leaf nodes in this subtree
        
        if left and right:
            self.depth = max(left.depth, right.depth) + 1
            self.size = left.size + right.size

    def to_dict(self) -> Dict:
        return {
            'hash': self.hash,
            'is_leaf': self.is_leaf,
            'depth': self.depth,
            'index': self.index,
            'size': self.size,
            'left': self.left.to_dict() if self.left else None,
            'right': self.right.to_dict() if self.right else None
        }

class MerkleTree:
  
    def __init__(self, data_items: List[str], config: Optional[MerkleTreeConfig] = None):
        """
        Initialize Merkle tree with data items
        
        Args:
            data_items: List of data strings to include in the tree
            config: Configuration object for tree behavior
        """
        self.config = config or MerkleTreeConfig()
        self.leaves: List[MerkleNode] = []
        self.root: Optional[MerkleNode] = None
        self.levels: List[List[MerkleNode]] = []
        self._node_cache: Dict[str, MerkleNode] = {}
        self._build_lock = threading.RLock()
        self.build_time: float = 0.0
        
        self.build_tree(data_items)
        
    def _get_hash_function(self):
        """Get the appropriate hash function based on configuration"""
        if self.config.hash_algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256
        elif self.config.hash_algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256
        elif self.config.hash_algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b
        elif self.config.hash_algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s
        elif self.config.hash_algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512
        else:
            raise ValueError(f"Unsupported hash algorithm: {self.config.hash_algorithm}")
    
    def _hash_data(self, data: str) -> str:
        """Hash data using configured algorithm and options"""
        hash_func = self._get_hash_function()
        
        if self.config.use_encoding:
            data_bytes = data.encode('utf-8')
        else:
            data_bytes = data if isinstance(data, bytes) else data.encode('utf-8')
        
        hash_result = hash_func(data_bytes).hexdigest()
        
        if self.config.double_hash:
            hash_result = hash_func(hash_result.encode('utf-8')).hexdigest()
            
        return hash_result
    
    def _calculate_node_hash(self, left_hash: str, right_hash: str) -> str:
        """Calculate parent node hash from child hashes"""
        combined = left_hash + right_hash
        return self._hash_data(combined)
    
    def _hash_batch(self, batch: List[str]) -> List[str]:
        """Hash a batch of data items efficiently"""
        return [self._hash_data(item) for item in batch]
    
    async def _hash_batch_async(self, batch: List[str]) -> List[str]:
        """Asynchronously hash a batch of data items"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._hash_batch, batch)
    
    def _build_tree_level(self, nodes: List[MerkleNode]) -> List[MerkleNode]:
        """Build a single level of the Merkle tree"""
        next_level = []
        
        # Process nodes in pairs
        for i in range(0, len(nodes), 2):
            left_node = nodes[i]
            right_node = nodes[i + 1] if i + 1 < len(nodes) else left_node
            
            parent_hash = self._calculate_node_hash(left_node.hash, right_node.hash)
            parent_node = MerkleNode(parent_hash, left_node, right_node)
            parent_node.depth = left_node.depth + 1
            parent_node.index = i // 2
            parent_node.size = left_node.size + (right_node.size if i + 1 < len(nodes) else 0)
            
            next_level.append(parent_node)
        
        return next_level
    
    async def _build_tree_level_async(self, nodes: List[MerkleNode]) -> List[MerkleNode]:
        """Asynchronously build a single level of the Merkle tree"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._build_tree_level, nodes)
    
    def build_tree(self, data_items: List[str]):
        """Build the complete Merkle tree from data items with optimizations"""
        start_time = time.time()
        
        with self._build_lock:
            if not data_items:
                self.root = MerkleNode(self._hash_data(""))
                return
                
            # Create leaf nodes efficiently
            if self.config.enable_parallel_processing and len(data_items) > self.config.batch_size:
                self._build_tree_parallel(data_items)
            else:
                self._build_tree_sequential(data_items)
            
            self.build_time = time.time() - start_time
    
    def _build_tree_sequential(self, data_items: List[str]):
        """Build tree sequentially (memory efficient)"""
        # Create leaf nodes
        self.leaves = [
            MerkleNode(self._hash_data(item), is_leaf=True, index=i, size=1)
            for i, item in enumerate(data_items)
        ]
        
        current_level = self.leaves
        self.levels = [current_level]
        
        # Build tree levels until we reach the root
        while len(current_level) > 1:
            next_level = self._build_tree_level(current_level)
            self.levels.append(next_level)
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
    
    def _build_tree_parallel(self, data_items: List[str]):
        """Build tree using parallel processing for large datasets"""
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        num_processes = min(multiprocessing.cpu_count(), 8)
        batch_size = (len(data_items) + num_processes - 1) // num_processes
        
        # Hash leaves in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            for i in range(0, len(data_items), batch_size):
                batch = data_items[i:i + batch_size]
                futures.append(executor.submit(self._hash_batch, batch))
            
            hashed_batches = []
            for future in as_completed(futures):
                hashed_batches.extend(future.result())
            
            # Create leaf nodes
            self.leaves = [
                MerkleNode(hash_val, is_leaf=True, index=i, size=1)
                for i, hash_val in enumerate(hashed_batches)
            ]
        
        # Build upper levels (still sequential but optimized)
        current_level = self.leaves
        self.levels = [current_level]
        
        while len(current_level) > 1:
            next_level = self._build_tree_level(current_level)
            self.levels.append(next_level)
            current_level = next_level
        
        self.root = current_level[0] if current_level else None
    
    async def build_tree_async(self, data_items: List[str]):
        """Asynchronously build the Merkle tree"""
        start_time = time.time()
        
        with self._build_lock:
            if not data_items:
                self.root = MerkleNode(self._hash_data(""))
                return
                
            # Create leaf nodes asynchronously
            if len(data_items) > self.config.batch_size:
                # Process in batches
                batches = [data_items[i:i + self.config.batch_size] 
                          for i in range(0, len(data_items), self.config.batch_size)]
                
                hashed_batches = await asyncio.gather(
                    *[self._hash_batch_async(batch) for batch in batches]
                )
                
                # Flatten results
                hashed_items = []
                for batch in hashed_batches:
                    hashed_items.extend(batch)
            else:
                hashed_items = self._hash_batch(data_items)
            
            # Create leaf nodes
            self.leaves = [
                MerkleNode(hash_val, is_leaf=True, index=i, size=1)
                for i, hash_val in enumerate(hashed_items)
            ]
            
            # Build upper levels
            current_level = self.leaves
            self.levels = [current_level]
            
            while len(current_level) > 1:
                next_level = await self._build_tree_level_async(current_level)
                self.levels.append(next_level)
                current_level = next_level
            
            self.root = current_level[0] if current_level else None
        
        self.build_time = time.time() - start_time
    
    def get_root_hash(self) -> str:
        """Get the Merkle root hash"""
        return self.root.hash if self.root else ""
    
    def get_proof(self, data_item: str, format: ProofFormat = ProofFormat.BINARY) -> Optional[bytes]:
        """
        Get Merkle proof for a data item in specified format
        
        Returns:
            Serialized proof bytes or None if item not found
        """
        target_hash = self._hash_data(data_item)
        proof_dict = self._get_proof_dict_by_hash(target_hash)
        
        if not proof_dict:
            return None
        
        return self._serialize_proof(proof_dict, format)
    
    def _get_proof_dict_by_hash(self, target_hash: str) -> Optional[Dict]:
        """Get Merkle proof as dictionary for a specific hash"""
        # Find the leaf node
        leaf_index = -1
        for i, leaf in enumerate(self.leaves):
            if leaf.hash == target_hash:
                leaf_index = i
                break
        
        if leaf_index == -1:
            return None
        
        proof = {
            'version': PROOF_VERSION,
            'algorithm': self.config.hash_algorithm.value,
            'leaf_hash': target_hash,
            'leaf_index': leaf_index,
            'sibling_hashes': [],
            'path_indices': [],      # 0 for left, 1 for right at each level
            'tree_depth': len(self.levels) - 1,
            'total_leaves': len(self.leaves),
            'root_hash': self.get_root_hash()
        }
        
        current_index = leaf_index
        current_level = 0
        
        # Traverse up the tree to collect proof
        while current_level < len(self.levels) - 1:
            current_nodes = self.levels[current_level]
            
            # Determine if current node is left or right child
            is_left = current_index % 2 == 0
            sibling_index = current_index + 1 if is_left else current_index - 1
            
            if sibling_index < len(current_nodes):
                sibling_node = current_nodes[sibling_index]
                proof['sibling_hashes'].append(sibling_node.hash)
                proof['path_indices'].append(0 if is_left else 1)
            
            current_index //= 2
            current_level += 1
        
        return proof
    
    def _serialize_proof(self, proof: Dict, format: ProofFormat) -> bytes:
        """Serialize proof to specified format"""
        if format == ProofFormat.JSON:
            return json.dumps(proof).encode('utf-8')
        elif format == ProofFormat.MSGPACK:
            return msgpack.packb(proof)
        elif format == ProofFormat.BINARY:
            return self._serialize_proof_binary(proof)
        else:
            raise ValueError(f"Unsupported proof format: {format}")
    
    def _serialize_proof_binary(self, proof: Dict) -> bytes:
        """Serialize proof to compact binary format"""
        # Header: version(1) + algorithm(1) + depth(1) + total_leaves(4) + path_length(1)
        version = PROOF_VERSION
        algorithm_code = self._get_algorithm_code(proof['algorithm'])
        depth = proof['tree_depth']
        total_leaves = proof['total_leaves']
        path_length = len(proof['path_indices'])
        
        header = struct.pack('!BBBIB', version, algorithm_code, depth, total_leaves, path_length)
        
        # Leaf hash (32 bytes)
        leaf_hash = bytes.fromhex(proof['leaf_hash'])
        
        # Root hash (32 bytes)
        root_hash = bytes.fromhex(proof['root_hash'])
        
        # Path indices (packed as bits)
        path_bits = 0
        for i, index in enumerate(proof['path_indices']):
            if index == 1:
                path_bits |= (1 << i)
        
        path_bytes = struct.pack('!B', path_bits)
        
        # Sibling hashes (each 32 bytes)
        sibling_hashes = b''.join([bytes.fromhex(h) for h in proof['sibling_hashes']])
        
        # Combine all parts
        return header + leaf_hash + root_hash + path_bytes + sibling_hashes
    
    def _get_algorithm_code(self, algorithm: str) -> int:
        """Get numeric code for hash algorithm"""
        algorithms = {
            'sha256': 0,
            'sha3_256': 1,
            'blake2b': 2,
            'blake2s': 3,
            'sha512': 4
        }
        return algorithms.get(algorithm, 0)
    
    def get_proof_by_index(self, leaf_index: int, format: ProofFormat = ProofFormat.BINARY) -> Optional[bytes]:
        """Get Merkle proof for a leaf by its index in specified format"""
        if leaf_index < 0 or leaf_index >= len(self.leaves):
            return None
        
        target_hash = self.leaves[leaf_index].hash
        proof_dict = self._get_proof_dict_by_hash(target_hash)
        
        if not proof_dict:
            return None
        
        return self._serialize_proof(proof_dict, format)
    
    @classmethod
    def deserialize_proof(cls, proof_data: bytes, format: ProofFormat = ProofFormat.BINARY) -> Optional[Dict]:
        """Deserialize proof from specified format"""
        try:
            if format == ProofFormat.JSON:
                return json.loads(proof_data.decode('utf-8'))
            elif format == ProofFormat.MSGPACK:
                return msgpack.unpackb(proof_data)
            elif format == ProofFormat.BINARY:
                return cls._deserialize_proof_binary(proof_data)
            else:
                return None
        except (json.JSONDecodeError, msgpack.UnpackException, struct.error):
            return None
    
    @classmethod
    def _deserialize_proof_binary(cls, proof_data: bytes) -> Optional[Dict]:
        """Deserialize proof from compact binary format"""
        try:
            # Parse header
            header = proof_data[:9]
            version, algorithm_code, depth, total_leaves, path_length = struct.unpack('!BBBIB', header)
            
            if version != PROOF_VERSION:
                return None
            
            # Parse leaf hash (32 bytes)
            leaf_hash_start = 9
            leaf_hash = proof_data[leaf_hash_start:leaf_hash_start + 32].hex()
            
            # Parse root hash (32 bytes)
            root_hash_start = leaf_hash_start + 32
            root_hash = proof_data[root_hash_start:root_hash_start + 32].hex()
            
            # Parse path bits (1 byte)
            path_byte_start = root_hash_start + 32
            path_bits = proof_data[path_byte_start]
            
            # Parse sibling hashes
            sibling_start = path_byte_start + 1
            sibling_hashes = []
            for i in range(path_length):
                hash_start = sibling_start + i * 32
                sibling_hash = proof_data[hash_start:hash_start + 32].hex()
                sibling_hashes.append(sibling_hash)
            
            # Reconstruct path indices from bits
            path_indices = []
            for i in range(path_length):
                path_indices.append(1 if (path_bits >> i) & 1 else 0)
            
            # Get algorithm name from code
            algorithm_codes = {
                0: 'sha256',
                1: 'sha3_256',
                2: 'blake2b',
                3: 'blake2s',
                4: 'sha512'
            }
            algorithm = algorithm_codes.get(algorithm_code, 'sha256')
            
            return {
                'version': version,
                'algorithm': algorithm,
                'leaf_hash': leaf_hash,
                'leaf_index': -1,  # Not stored in binary format
                'sibling_hashes': sibling_hashes,
                'path_indices': path_indices,
                'tree_depth': depth,
                'total_leaves': total_leaves,
                'root_hash': root_hash
            }
        except (IndexError, struct.error):
            return None
    
    @classmethod
    def verify_proof(cls, proof_data: bytes, target_hash: str, root_hash: str, 
                    format: ProofFormat = ProofFormat.BINARY) -> bool:
        """
        Verify a Merkle proof
        
        Args:
            proof_data: Serialized proof data
            target_hash: Hash of the data item to verify
            root_hash: Expected root hash of the tree
            format: Format of the proof data
            
        Returns:
            True if proof is valid, False otherwise
        """
        proof = cls.deserialize_proof(proof_data, format)
        if not proof or 'sibling_hashes' not in proof:
            return False
        
        current_hash = target_hash
        sibling_hashes = proof['sibling_hashes']
        path_indices = proof.get('path_indices', [])
        
        # Determine hash function from proof
        algorithm = proof.get('algorithm', 'sha256')
        hash_func = getattr(hashlib, algorithm)
        
        # Reconstruct the root hash
        for i, sibling_hash in enumerate(sibling_hashes):
            if i < len(path_indices):
                is_right_child = path_indices[i] == 1
            else:
                # Fallback: assume alternating positions
                is_right_child = i % 2 == 0
            
            if is_right_child:
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            
            # Use the appropriate hash function
            current_hash = hash_func(combined.encode('utf-8')).hexdigest()
        
        return current_hash == root_hash
    
    def verify_leaf(self, data_item: str) -> bool:
        """Verify that a data item is in the tree"""
        proof_data = self.get_proof(data_item, ProofFormat.BINARY)
        if not proof_data:
            return False
        return self.verify_proof(proof_data, self._hash_data(data_item), self.get_root_hash())
    
    def get_leaf_count(self) -> int:
        """Get number of leaf nodes"""
        return len(self.leaves)
    
    def get_tree_depth(self) -> int:
        """Get depth of the tree"""
        return len(self.levels) - 1 if self.levels else 0
    
    def get_level_hashes(self, level: int) -> List[str]:
        """Get all hashes at a specific level"""
        if level < 0 or level >= len(self.levels):
            return []
        return [node.hash for node in self.levels[level]]
    
    def find_leaf_by_hash(self, target_hash: str) -> Optional[int]:
        """Find leaf index by hash, returns index or None if not found"""
        for i, leaf in enumerate(self.leaves):
            if leaf.hash == target_hash:
                return i
        return None
    
    def to_dict(self) -> Dict:
        """Convert tree to dictionary representation"""
        return {
            'root_hash': self.get_root_hash(),
            'hash_algorithm': self.config.hash_algorithm.value,
            'double_hash': self.config.double_hash,
            'use_encoding': self.config.use_encoding,
            'leaf_count': self.get_leaf_count(),
            'tree_depth': self.get_tree_depth(),
            'build_time': self.build_time,
            'levels': [
                [node.hash for node in level]
                for level in self.levels
            ],
            'leaves': [leaf.hash for leaf in self.leaves]
        }
    
    def save_to_file(self, filename: str, compress: bool = True):
        """Save tree representation to file"""
        data = self.to_dict()
        
        if compress:
            with open(filename, 'wb') as f:
                compressed = zlib.compress(json.dumps(data).encode('utf-8'))
                f.write(compressed)
        else:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str, compressed: bool = True) -> Optional['MerkleTree']:
        """Load tree representation from file"""
        try:
            if compressed:
                with open(filename, 'rb') as f:
                    decompressed = zlib.decompress(f.read())
                    data = json.loads(decompressed.decode('utf-8'))
            else:
                with open(filename, 'r') as f:
                    data = json.load(f)
            
            # Create a mock tree with the saved data
            # Note: This doesn't reconstruct the full tree structure, just the metadata
            tree = cls([])
            tree.root = MerkleNode(data['root_hash']) if data['root_hash'] else None
            tree.config.hash_algorithm = HashAlgorithm(data['hash_algorithm'])
            tree.config.double_hash = data['double_hash']
            tree.config.use_encoding = data['use_encoding']
            tree.build_time = data.get('build_time', 0)
            
            return tree
        except (FileNotFoundError, json.JSONDecodeError, zlib.error):
            return None

class CompactMerkleTree(MerkleTree):
    """Merkle tree with compact representation for storage efficiency"""
    
    def __init__(self, data_items: List[str], config: Optional[MerkleTreeConfig] = None):
        super().__init__(data_items, config)
        self._build_compact_representation()
    
    def _build_compact_representation(self):
        """Build compact representation of the tree"""
        self.compact_nodes = {}
        self._traverse_and_store(self.root)
    
    def _traverse_and_store(self, node: MerkleNode, path: str = ""):
        """Recursively traverse tree and store nodes in compact form"""
        if not node:
            return
        
        self.compact_nodes[path] = node.hash
        
        if node.left:
            self._traverse_and_store(node.left, path + "0")
        if node.right:
            self._traverse_and_store(node.right, path + "1")
    
    def get_compact_proof(self, data_item: str, format: ProofFormat = ProofFormat.BINARY) -> Optional[bytes]:
        """Get compact Merkle proof"""
        proof_dict = self._get_proof_dict_by_hash(self._hash_data(data_item))
        if not proof_dict:
            return None
        
        # Convert to compact form
        compact_proof = {
            'version': proof_dict['version'],
            'algorithm': proof_dict['algorithm'],
            'leaf_hash': proof_dict['leaf_hash'],
            'leaf_index': proof_dict['leaf_index'],
            'sibling_hashes': proof_dict['sibling_hashes'],
            'bitmask': self._create_bitmask(proof_dict['path_indices']),
            'root_hash': proof_dict['root_hash']
        }
        
        return self._serialize_proof(compact_proof, format)
    
    def _create_bitmask(self, path_indices: List[int]) -> int:
        """Create bitmask from path indices"""
        bitmask = 0
        for i, index in enumerate(path_indices):
            if index == 1:  # right child
                bitmask |= (1 << i)
        return bitmask

class MerkleTreeFactory:
    """Factory for creating and managing multiple Merkle trees"""
    
    def __init__(self, config: Optional[MerkleTreeConfig] = None):
        self.config = config or MerkleTreeConfig()
        self.trees: Dict[str, MerkleTree] = {}
        self._lock = threading.RLock()
    
    def create_tree(self, tree_id: str, data_items: List[str]) -> MerkleTree:
        """Create a new Merkle tree and store it"""
        with self._lock:
            tree = MerkleTree(data_items, self.config)
            self.trees[tree_id] = tree
            return tree
    
    async def create_tree_async(self, tree_id: str, data_items: List[str]) -> MerkleTree:
        """Asynchronously create a new Merkle tree and store it"""
        with self._lock:
            tree = MerkleTree([], self.config)  # Create empty tree
            self.trees[tree_id] = tree
            await tree.build_tree_async(data_items)
            return tree
    
    def get_tree(self, tree_id: str) -> Optional[MerkleTree]:
        """Get a Merkle tree by ID"""
        with self._lock:
            return self.trees.get(tree_id)
    
    def remove_tree(self, tree_id: str):
        """Remove a Merkle tree"""
        with self._lock:
            if tree_id in self.trees:
                del self.trees[tree_id]
    
    def clear(self):
        """Clear all trees"""
        with self._lock:
            self.trees.clear()

# Advanced utility functions with production optimizations
async def create_merkle_tree_from_file(filename: str, config: Optional[MerkleTreeConfig] = None) -> MerkleTree:
    """Create Merkle tree from file content asynchronously"""
    loop = asyncio.get_event_loop()
    
    def read_file():
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    data_items = await loop.run_in_executor(None, read_file)
    return MerkleTree(data_items, config)

async def create_merkle_tree_from_large_file(filename: str, config: Optional[MerkleTreeConfig] = None, 
                                           batch_size: int = MAX_BATCH_SIZE) -> MerkleTree:
    """Create Merkle tree from large file using streaming"""
    tree = MerkleTree([], config)
    batch = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                batch.append(stripped)
                
                if len(batch) >= batch_size:
                    await tree.build_tree_async(batch)
                    batch = []
    
    if batch:
        await tree.build_tree_async(batch)
    
    return tree

def batch_verify_proofs(proofs: List[bytes], target_hashes: List[str], root_hash: str, 
                       format: ProofFormat = ProofFormat.BINARY) -> List[bool]:
    """Batch verify multiple Merkle proofs efficiently"""
    results = []
    for proof, target_hash in zip(proofs, target_hashes):
        results.append(MerkleTree.verify_proof(proof, target_hash, root_hash, format))
    return results

async def batch_verify_proofs_async(proofs: List[bytes], target_hashes: List[str], root_hash: str,
                                  format: ProofFormat = ProofFormat.BINARY) -> List[bool]:
    """Asynchronously batch verify multiple Merkle proofs"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, batch_verify_proofs, proofs, target_hashes, root_hash, format)

def create_merkle_mountain_range(blocks: List[str], range_size: int = 10, 
                               config: Optional[MerkleTreeConfig] = None) -> List[MerkleTree]:
    """Create Merkle Mountain Range for efficient append operations"""
    trees = []
    current_range = []
    
    for block in blocks:
        current_range.append(block)
        if len(current_range) >= range_size:
            trees.append(MerkleTree(current_range, config))
            current_range = []
    
    if current_range:
        trees.append(MerkleTree(current_range, config))
    
    return trees

# Performance monitoring and statistics
class MerkleTreeStats:
    """Collect and report statistics about Merkle tree operations"""
    
    def __init__(self):
        self.operations: List[Dict] = []
        self.lock = threading.RLock()
    
    def record_operation(self, operation_type: str, size: int, duration: float, 
                       success: bool = True, metadata: Optional[Dict] = None):
        """Record an operation with timing and metadata"""
        with self.lock:
            self.operations.append({
                'type': operation_type,
                'size': size,
                'duration': duration,
                'timestamp': time.time(),
                'success': success,
                'metadata': metadata or {}
            })
    
    def get_stats(self, window_seconds: int = 3600) -> Dict:
        """Get statistics for the specified time window"""
        with self.lock:
            now = time.time()
            recent_ops = [op for op in self.operations 
                         if op['timestamp'] > now - window_seconds]
            
            if not recent_ops:
                return {}
            
            build_ops = [op for op in recent_ops if op['type'] == 'build']
            verify_ops = [op for op in recent_ops if op['type'] == 'verify']
            
            return {
                'total_operations': len(recent_ops),
                'build_operations': len(build_ops),
                'verify_operations': len(verify_ops),
                'avg_build_time': sum(op['duration'] for op in build_ops) / len(build_ops) if build_ops else 0,
                'avg_verify_time': sum(op['duration'] for op in verify_ops) / len(verify_ops) if verify_ops else 0,
                'success_rate': sum(1 for op in recent_ops if op['success']) / len(recent_ops),
                'throughput': len(recent_ops) / window_seconds
            }

# Global statistics collector
global_stats = MerkleTreeStats()

class SparseMerkleTree:
    """Production-ready Sparse Merkle Tree for efficient sparse datasets"""
    
    def __init__(self, depth: int = 256, default_value: str = "0" * 64, 
                 config: Optional[MerkleTreeConfig] = None):
        """
        Initialize sparse Merkle tree
        
        Args:
            depth: Depth of the tree (determines capacity: 2^depth leaves)
            default_value: Default hash value for empty nodes
            config: Configuration for hash algorithm and other settings
        """
        self.depth = depth
        self.default_value = default_value
        self.config = config or MerkleTreeConfig()
        self.leaves: Dict[int, str] = {}  # index -> hash
        self.nodes: Dict[str, str] = {}   # path -> hash
        self._lock = threading.RLock()
        self._hash_func = self._get_hash_function()
        self._initialize_tree()
    
    def _get_hash_function(self):
        """Get the appropriate hash function based on configuration"""
        if self.config.hash_algorithm == HashAlgorithm.SHA256:
            return hashlib.sha256
        elif self.config.hash_algorithm == HashAlgorithm.SHA3_256:
            return hashlib.sha3_256
        elif self.config.hash_algorithm == HashAlgorithm.BLAKE2B:
            return hashlib.blake2b
        elif self.config.hash_algorithm == HashAlgorithm.BLAKE2S:
            return hashlib.blake2s
        elif self.config.hash_algorithm == HashAlgorithm.SHA512:
            return hashlib.sha512
        else:
            return hashlib.sha256
    
    def _hash_data(self, data: str) -> str:
        """Hash data using configured algorithm"""
        data_bytes = data.encode('utf-8')
        hash_result = self._hash_func(data_bytes).hexdigest()
        
        if self.config.double_hash:
            hash_result = self._hash_func(hash_result.encode('utf-8')).hexdigest()
            
        return hash_result
    
    def _initialize_tree(self):
        """Initialize tree with default values and precompute default hashes"""
        # Precompute default values for each level
        default_hashes = [self.default_value]
        for i in range(self.depth):
            combined = default_hashes[-1] + default_hashes[-1]
            default_hashes.append(self._hash_data(combined))
        
        self.default_hashes = default_hashes
        self._default_root = default_hashes[-1]
    
    def _get_path(self, index: int) -> str:
        """Get binary path for given index (left-padded with zeros)"""
        if index < 0 or index >= (1 << self.depth):
            raise ValueError(f"Index {index} out of range for tree depth {self.depth}")
        return bin(index)[2:].zfill(self.depth)
    
    def _get_default_hash(self, level: int) -> str:
        """Get default hash for a specific level"""
        return self.default_hashes[self.depth - level - 1]
    
    def update_leaf(self, index: int, value: str):
        """
        Update leaf value and propagate changes efficiently
        
        Args:
            index: Leaf index to update
            value: New hash value for the leaf
        """
        with self._lock:
            path = self._get_path(index)
            old_value = self.leaves.get(index, self.default_hashes[0])
            
            # If value is the same as default and leaf doesn't exist, no update needed
            if value == self.default_hashes[0] and index not in self.leaves:
                return
            
            self.leaves[index] = value
            
            # Update the leaf node
            current_hash = value
            self.nodes[path] = current_hash
            
            # Propagate changes up the tree
            for level in range(self.depth - 1, -1, -1):
                node_path = path[:level + 1]
                sibling_path = path[:level] + ('1' if path[level] == '0' else '0')
                
                # Get sibling hash (from storage or default)
                sibling_hash = self.nodes.get(sibling_path, self._get_default_hash(level))
                
                # Combine based on position
                if path[level] == '0':
                    combined = current_hash + sibling_hash
                else:
                    combined = sibling_hash + current_hash
                
                current_hash = self._hash_data(combined)
                parent_path = path[:level]
                self.nodes[parent_path] = current_hash
            
            # Update root
            self.nodes[""] = current_hash
    
    def delete_leaf(self, index: int):
        """
        Delete a leaf (set it to default value)
        
        Args:
            index: Leaf index to delete
        """
        self.update_leaf(index, self.default_hashes[0])
        
        with self._lock:
            # Clean up storage if this was the last reference
            path = self._get_path(index)
            if index in self.leaves:
                del self.leaves[index]
            
            # Remove leaf node from storage if it exists
            if path in self.nodes:
                del self.nodes[path]
            
            # Clean up orphaned internal nodes (optional optimization)
            self._cleanup_orphaned_nodes()
    
    def _cleanup_orphaned_nodes(self):
        """Remove internal nodes that are no longer needed"""
        # This is an optimization to save memory
        # In production, you might want to keep this disabled for performance
        paths_to_remove = []
        
        for path in self.nodes:
            if path == "" or len(path) == self.depth:
                continue  # Skip root and leaves
            
            # Check if both children are default
            left_child = path + "0"
            right_child = path + "1"
            
            left_hash = self.nodes.get(left_child, self._get_default_hash(self.depth - len(left_child) - 1))
            right_hash = self.nodes.get(right_child, self._get_default_hash(self.depth - len(right_child) - 1))
            
            if (left_hash == self._get_default_hash(self.depth - len(left_child) - 1) and
                right_hash == self._get_default_hash(self.depth - len(right_child) - 1)):
                paths_to_remove.append(path)
        
        for path in paths_to_remove:
            del self.nodes[path]
    
    def get_leaf(self, index: int) -> str:
        """
        Get leaf value at specific index
        
        Args:
            index: Leaf index
            
        Returns:
            Leaf hash value (default if not set)
        """
        return self.leaves.get(index, self.default_hashes[0])
    
    def get_root(self) -> str:
        """Get current root hash"""
        return self.nodes.get("", self._default_root)
    
    def get_proof(self, index: int, format: ProofFormat = ProofFormat.BINARY) -> bytes:
        """
        Get inclusion proof for leaf
        
        Args:
            index: Leaf index to prove
            format: Proof serialization format
            
        Returns:
            Serialized proof data
        """
        with self._lock:
            proof_dict = self._get_proof_dict(index)
            return self._serialize_proof(proof_dict, format)
    
    def _get_proof_dict(self, index: int) -> Dict:
        """Get proof as dictionary"""
        path = self._get_path(index)
        leaf_hash = self.leaves.get(index, self.default_hashes[0])
        
        proof = {
            'version': PROOF_VERSION,
            'algorithm': self.config.hash_algorithm.value,
            'leaf_hash': leaf_hash,
            'leaf_index': index,
            'sibling_hashes': [],
            'path': path,
            'tree_depth': self.depth,
            'root_hash': self.get_root()
        }
        
        current_path = path
        for level in range(self.depth - 1, -1, -1):
            sibling_path = current_path[:level] + ('1' if current_path[level] == '0' else '0')
            sibling_hash = self.nodes.get(sibling_path, self._get_default_hash(level))
            proof['sibling_hashes'].append(sibling_hash)
            current_path = current_path[:level]
        
        return proof
    
    def _serialize_proof(self, proof: Dict, format: ProofFormat) -> bytes:
        """Serialize proof to specified format"""
        if format == ProofFormat.JSON:
            return json.dumps(proof).encode('utf-8')
        elif format == ProofFormat.MSGPACK:
            return msgpack.packb(proof)
        elif format == ProofFormat.BINARY:
            return self._serialize_sparse_proof_binary(proof)
        else:
            raise ValueError(f"Unsupported proof format: {format}")
    
    def _serialize_sparse_proof_binary(self, proof: Dict) -> bytes:
        """Serialize sparse Merkle proof to compact binary format"""
        # Header: version(1) + algorithm(1) + depth(1) + path_length(1) + leaf_index(8)
        version = PROOF_VERSION
        algorithm_code = self._get_algorithm_code(proof['algorithm'])
        depth = proof['tree_depth']
        path_length = len(proof['path'])
        leaf_index = proof['leaf_index']
        
        header = struct.pack('!BBBBI', version, algorithm_code, depth, path_length, leaf_index)
        
        # Leaf hash (32 bytes)
        leaf_hash = bytes.fromhex(proof['leaf_hash'])
        
        # Root hash (32 bytes)
        root_hash = bytes.fromhex(proof['root_hash'])
        
        # Path (as ASCII bytes)
        path_bytes = proof['path'].encode('ascii')
        
        # Sibling hashes (each 32 bytes)
        sibling_hashes = b''.join([bytes.fromhex(h) for h in proof['sibling_hashes']])
        
        # Combine all parts
        return header + leaf_hash + root_hash + path_bytes + sibling_hashes
    
    def _get_algorithm_code(self, algorithm: str) -> int:
        """Get numeric code for hash algorithm"""
        algorithms = {
            'sha256': 0,
            'sha3_256': 1,
            'blake2b': 2,
            'blake2s': 3,
            'sha512': 4
        }
        return algorithms.get(algorithm, 0)
    
    @classmethod
    def deserialize_proof(cls, proof_data: bytes, format: ProofFormat = ProofFormat.BINARY) -> Optional[Dict]:
        """Deserialize sparse Merkle proof"""
        try:
            if format == ProofFormat.JSON:
                return json.loads(proof_data.decode('utf-8'))
            elif format == ProofFormat.MSGPACK:
                return msgpack.unpackb(proof_data)
            elif format == ProofFormat.BINARY:
                return cls._deserialize_sparse_proof_binary(proof_data)
            else:
                return None
        except (json.JSONDecodeError, msgpack.UnpackException, struct.error):
            return None
    
    @classmethod
    def _deserialize_sparse_proof_binary(cls, proof_data: bytes) -> Optional[Dict]:
        """Deserialize sparse Merkle proof from binary format"""
        try:
            # Parse header
            header = proof_data[:12]
            version, algorithm_code, depth, path_length, leaf_index = struct.unpack('!BBBBI', header)
            
            if version != PROOF_VERSION:
                return None
            
            # Parse leaf hash (32 bytes)
            leaf_hash_start = 12
            leaf_hash = proof_data[leaf_hash_start:leaf_hash_start + 32].hex()
            
            # Parse root hash (32 bytes)
            root_hash_start = leaf_hash_start + 32
            root_hash = proof_data[root_hash_start:root_hash_start + 32].hex()
            
            # Parse path
            path_start = root_hash_start + 32
            path = proof_data[path_start:path_start + path_length].decode('ascii')
            
            # Parse sibling hashes
            sibling_start = path_start + path_length
            sibling_hashes = []
            for i in range(depth):
                hash_start = sibling_start + i * 32
                sibling_hash = proof_data[hash_start:hash_start + 32].hex()
                sibling_hashes.append(sibling_hash)
            
            # Get algorithm name from code
            algorithm_codes = {
                0: 'sha256',
                1: 'sha3_256',
                2: 'blake2b',
                3: 'blake2s',
                4: 'sha512'
            }
            algorithm = algorithm_codes.get(algorithm_code, 'sha256')
            
            return {
                'version': version,
                'algorithm': algorithm,
                'leaf_hash': leaf_hash,
                'leaf_index': leaf_index,
                'sibling_hashes': sibling_hashes,
                'path': path,
                'tree_depth': depth,
                'root_hash': root_hash
            }
        except (IndexError, struct.error, UnicodeDecodeError):
            return None
    
    @classmethod
    def verify_proof(cls, proof_data: bytes, target_hash: str, root_hash: str, 
                    format: ProofFormat = ProofFormat.BINARY) -> bool:
        """
        Verify a sparse Merkle proof
        
        Args:
            proof_data: Serialized proof data
            target_hash: Hash of the data item to verify
            root_hash: Expected root hash of the tree
            format: Format of the proof data
            
        Returns:
            True if proof is valid, False otherwise
        """
        proof = cls.deserialize_proof(proof_data, format)
        if not proof or 'sibling_hashes' not in proof:
            return False
        
        current_hash = target_hash
        sibling_hashes = proof['sibling_hashes']
        path = proof.get('path', '')
        
        # Determine hash function from proof
        algorithm = proof.get('algorithm', 'sha256')
        hash_func = getattr(hashlib, algorithm)
        
        # Reconstruct the root hash
        for level, sibling_hash in enumerate(sibling_hashes):
            if level < len(path):
                bit = path[level]
            else:
                bit = '0'  # Default to left if path is shorter than depth
            
            if bit == '0':
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            
            # Use the appropriate hash function
            current_hash = hash_func(combined.encode('utf-8')).hexdigest()
        
        return current_hash == root_hash
    
    def verify_leaf(self, index: int, value: str) -> bool:
        """Verify that a leaf value is in the tree"""
        proof_data = self.get_proof(index, ProofFormat.BINARY)
        if not proof_data:
            return False
        return self.verify_proof(proof_data, value, self.get_root())
    
    def get_leaf_count(self) -> int:
        """Get number of non-default leaf nodes"""
        return len(self.leaves)
    
    def get_capacity(self) -> int:
        """Get total capacity of the tree (2^depth)"""
        return 1 << self.depth
    
    def to_dict(self) -> Dict:
        """Convert tree to dictionary representation"""
        return {
            'root_hash': self.get_root(),
            'hash_algorithm': self.config.hash_algorithm.value,
            'double_hash': self.config.double_hash,
            'depth': self.depth,
            'leaf_count': self.get_leaf_count(),
            'capacity': self.get_capacity(),
            'default_value': self.default_value
        }
    
    def batch_update(self, updates: Dict[int, str]):
        """
        Batch update multiple leaves efficiently
        
        Args:
            updates: Dictionary of index -> value updates
        """
        with self._lock:
            for index, value in updates.items():
                self.update_leaf(index, value)
    
    async def batch_update_async(self, updates: Dict[int, str]):
        """
        Asynchronously batch update multiple leaves
        
        Args:
            updates: Dictionary of index -> value updates
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.batch_update, updates)
        
class MerkleTreeStats:
    """Collect and report statistics about Merkle tree operations"""
    
    def __init__(self):
        self.operations: List[Dict] = []
        self.lock = threading.RLock()
    
    def record_operation(self, operation_type: str, size: int, duration: float, 
                       success: bool = True, metadata: Optional[Dict] = None):
        """Record an operation with timing and metadata"""
        with self.lock:
            self.operations.append({
                'type': operation_type,
                'size': size,
                'duration': duration,
                'timestamp': time.time(),
                'success': success,
                'metadata': metadata or {}
            })
    
    def get_stats(self, window_seconds: int = 3600) -> Dict:
        """Get statistics for the specified time window"""
        with self.lock:
            now = time.time()
            recent_ops = [op for op in self.operations 
                         if op['timestamp'] > now - window_seconds]
            
            if not recent_ops:
                return {}
            
            build_ops = [op for op in recent_ops if op['type'] == 'build']
            verify_ops = [op for op in recent_ops if op['type'] == 'verify']
            
            return {
                'total_operations': len(recent_ops),
                'build_operations': len(build_ops),
                'verify_operations': len(verify_ops),
                'avg_build_time': sum(op['duration'] for op in build_ops) / len(build_ops) if build_ops else 0,
                'avg_verify_time': sum(op['duration'] for op in verify_ops) / len(verify_ops) if verify_ops else 0,
                'success_rate': sum(1 for op in recent_ops if op['success']) / len(recent_ops),
                'throughput': len(recent_ops) / window_seconds
            }       

# Add these utility functions at the end of the file

def create_sparse_merkle_tree(depth: int = 256, config: Optional[MerkleTreeConfig] = None) -> SparseMerkleTree:
    """Create a new sparse Merkle tree"""
    return SparseMerkleTree(depth=depth, config=config)

async def create_sparse_merkle_tree_from_dict(data_dict: Dict[int, str], 
                                            depth: int = 256,
                                            config: Optional[MerkleTreeConfig] = None) -> SparseMerkleTree:
    """Create sparse Merkle tree from dictionary of index -> value pairs"""
    tree = SparseMerkleTree(depth=depth, config=config)
    await tree.batch_update_async(data_dict)
    return tree

def sparse_merkle_proof_to_json(proof_data: bytes) -> Optional[str]:
    """Convert sparse Merkle proof to JSON string for readability"""
    proof = SparseMerkleTree.deserialize_proof(proof_data, ProofFormat.BINARY)
    if proof:
        return json.dumps(proof, indent=2)
    return None