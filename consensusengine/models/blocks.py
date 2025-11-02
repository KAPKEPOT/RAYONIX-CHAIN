# consensus/models/blocks.py
import time
import hashlib
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend


try:
    from merkle import ProofFormat
except ImportError:
    # Fallback if merkle module is not available
    from enum import Enum
    
    class ProofFormat(Enum):
        JSON = 'json'
        MSGPACK = 'msgpack'
        BINARY = 'binary'

@dataclass
class BlockHeader:
    """Block header structure"""
    version: int = 1
    height: int = 0
    previous_hash: str = "0" * 64
    merkle_root: str = "0" * 64
    timestamp: float = field(default_factory=time.time)
    difficulty: int = 1
    nonce: int = 0
    validator: str = ""
    signature: str = ""
    
    def calculate_hash(self) -> str:
        """Calculate block header hash"""
        header_data = {
            'version': self.version,
            'height': self.height,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'difficulty': self.difficulty,
            'nonce': self.nonce,
            'validator': self.validator
        }
        header_json = json.dumps(header_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(header_json.encode()).hexdigest()
    
    def get_signing_data(self) -> bytes:
        """Get data that should be signed for the block header"""
        signing_data = {
            'version': self.version,
            'height': self.height,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'difficulty': self.difficulty,
            'nonce': self.nonce,
            'validator': self.validator
        }
        return json.dumps(signing_data, sort_keys=True).encode()

@dataclass
class Transaction:
    """Transaction structure"""
    tx_hash: str
    sender: str
    receiver: str
    amount: int
    fee: int
    timestamp: float
    signature: str
    data: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate transaction structure"""
        if not self.tx_hash or len(self.tx_hash) != 64:
            return False
        
        if not self.sender or not self.receiver:
            return False
        
        if self.amount <= 0:
            return False
        
        if self.fee < 0:
            return False
        
        return True

@dataclass
class Block:
    """Complete block structure"""
    header: BlockHeader
    transactions: List[Transaction] = field(default_factory=list)
    
    @property
    def hash(self) -> str:
        """Get block hash"""
        return self.header.calculate_hash()
    
    def calculate_merkle_root(self) -> str:
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            return "0" * 64
        
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        
        # Create proper Merkle tree with blockchain-optimized configuration
        from merkle_system.merkle import MerkleTree, MerkleTreeConfig, HashAlgorithm
        
        config = MerkleTreeConfig(
            hash_algorithm=HashAlgorithm.SHA256,
            double_hash=True,  # Double hashing for extra security
            use_encoding=True,
            enable_parallel_processing=len(tx_hashes) > 100,  # Parallel for large blocks
            batch_size=1000,
            cache_enabled=True
        )
        
        merkle_tree = MerkleTree(tx_hashes, config)
        return merkle_tree.get_root_hash()

    def get_merkle_proof(self, tx_hash: str, proof_format: ProofFormat = ProofFormat.BINARY) -> Optional[bytes]:
    	"""Get Merkle proof for a transaction in specified format"""
    	if not self.transactions:
    		return None
    		
    	from merkle_system.merkle import MerkleTree, MerkleTreeConfig, HashAlgorithm, ProofFormat
    	tx_hashes = [tx.tx_hash for tx in self.transactions]
    	
    	# Recreate the Merkle tree (should match the one used for root calculation)
    	
    	config = MerkleTreeConfig(
    	    
    	    hash_algorithm=HashAlgorithm.SHA256,
    	    double_hash=True,
    	    use_encoding=True
    	)
    	
    	merkle_tree = MerkleTree(tx_hashes, config)
    	
    	# Verify we're using the same root
    	if merkle_tree.get_root_hash() != self.header.merkle_root:
    		logger.error("Merkle tree recreation mismatch - cannot generate valid proof")
    		return None
    	
    	return merkle_tree.get_proof(tx_hash, proof_format)
    	
    def validate_merkle_proof(self, tx_hash: str, proof_data: bytes, proof_format: ProofFormat = ProofFormat.BINARY) -> bool:
    	"""Validate Merkle proof for transaction inclusion"""
    	from merkle_system.merkle import MerkleTree
    	
    	return MerkleTree.verify_proof(
    	    proof_data=proof_data,
    	    target_hash=tx_hash,
    	    root_hash=self.header.merkle_root,
    	    format=proof_format
    	)
    	
    def get_merkle_proof_by_index(self, tx_index: int, proof_format: ProofFormat = ProofFormat.BINARY) -> Optional[bytes]:
    	"""Get Merkle proof for a transaction by its index in the block"""
    	if tx_index < 0 or tx_index >= len(self.transactions):
    		return None
    		
    	tx_hash = self.transactions[tx_index].tx_hash
    	return self.get_merkle_proof(tx_hash, proof_format)
    	
    def validate(self, previous_block: Optional['Block'] = None) -> bool:
        """Validate block structure and consistency with proper Merkle validation"""
        # Validate header
        if self.header.height < 0:
        	logger.error("Invalid block height")
        	return False
        
        if previous_block and self.header.previous_hash != previous_block.hash:
        	logger.error("Previous hash mismatch")
        	return False
        	
        # Validate Merkle root using proper calculation
        try:
        	calculated_root = self.calculate_merkle_root()
        	if calculated_root != self.header.merkle_root:
        		logger.error(f"Merkle root mismatch: calculated {calculated_root}, header {self.header.merkle_root}")
        		return False
        except Exception as e:
        	logger.error(f"Merkle root calculation failed: {e}")
        	return False
        	
        # Validate transaction hashes are unique
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        if len(tx_hashes) != len(set(tx_hashes)):
        	logger.error("Duplicate transaction hashes in block")
        	return False
        
        # Validate individual transactions
        for tx in self.transactions:
        	if not tx.validate():
        		logger.error(f"Invalid transaction: {tx.tx_hash}")
        		return False
        
        # Additional Merkle tree integrity checks for large blocks
        if len(self.transactions) > 1000:
        	if not self._validate_merkle_tree_integrity():
        		return False
        
        return True
       
    def _validate_merkle_tree_integrity(self) -> bool:
    	"""Perform additional Merkle tree integrity checks for large blocks"""
    	try:
    		from merkle_system.merkle import MerkleTree, MerkleTreeConfig, HashAlgorithm
    		
    		tx_hashes = [tx.tx_hash for tx in self.transactions]
    		config = MerkleTreeConfig(
    		    hash_algorithm=HashAlgorithm.SHA256,
    		    double_hash=True
    		)
    		
    		merkle_tree = MerkleTree(tx_hashes, config)
    		
    		# Verify tree depth is reasonable
    		tree_depth = merkle_tree.get_tree_depth()
    		max_expected_depth = 20  # For 1M transactions, depth ~= log2(1M) ≈ 20
    		if tree_depth > max_expected_depth:
    			logger.error(f"Suspicious Merkle tree depth: {tree_depth}")
    			return False
    			
    		# Verify we can generate and validate proofs for sample transactions
    		sample_indices = [0, len(tx_hashes) // 2, len(tx_hashes) - 1]
    		for idx in sample_indices:
    			if idx < len(tx_hashes):
    				proof = merkle_tree.get_proof_by_index(idx)
    				if not proof:
    					logger.error(f"Failed to generate proof for transaction at index {idx}")
    					return False
    				
    				if not MerkleTree.verify_proof(proof, tx_hashes[idx], self.header.merkle_root):
    					logger.error(f"Generated proof failed verification for transaction at index {idx}")
    					return False
    					
    		return True
    	
    	except Exception as e:
    		logger.error(f"Merkle tree integrity validation failed: {e}")
    		return False
    		
    def get_light_client_header(self) -> Dict[str, Any]:
    	"""Get block header information for light clients"""
    	return {
    	    'height': self.header.height,
    	    'hash': self.hash,
    	    'previous_hash': self.header.previous_hash,
    	    'merkle_root': self.header.merkle_root,
    	    'timestamp': self.header.timestamp,
    	    'difficulty': self.header.difficulty,
    	    'validator': self.header.validator,
    	    'transaction_count': len(self.transactions)
    	}
    
    def verify_transaction_inclusion(self, tx_hash: str, proof_data: bytes, proof_format: ProofFormat = ProofFormat.BINARY) -> bool:
    	"""Verify transaction inclusion for light clients"""
    	return self.validate_merkle_proof(tx_hash, proof_data, proof_format)
    	
    def get_transaction_with_proof(self, tx_hash: str) -> Optional[Dict[str, Any]]:
    	"""Get transaction with its Merkle proof for light clients"""
    	tx = None
    	for transaction in self.transactions:
    		if transaction.tx_hash == tx_hash:
    			tx = transaction
    			break
    		
    	if not tx:
    		return None
    	
    	proof = self.get_merkle_proof(tx_hash)    	
    	if not proof:
    		return None
    		
    	return {
    	    'transaction': tx.to_dict() if hasattr(tx, 'to_dict') else tx.__dict__,
    	    'merkle_proof': proof.hex() if isinstance(proof, bytes) else proof,
    	    'block_header': self.get_light_client_header(),
    	    'tx_index': self.transactions.index(tx) if tx in self.transactions else -1
    	}
   
@dataclass
class BlockProposal:
    """Block proposal structure for consensus"""
    height: int
    block_hash: str
    validator_address: str
    timestamp: float
    signature: str
    view_number: int
    round_number: int
    parent_hash: str
    tx_hashes: List[str] = field(default_factory=list)
    justification: Optional[Dict] = None
    block_data: Optional[Dict] = None  # Full block data if available
    
    def to_dict(self) -> Dict:
        """Serialize block proposal to dictionary"""
        return {
            'height': self.height,
            'block_hash': self.block_hash,
            'validator_address': self.validator_address,
            'timestamp': self.timestamp,
            'signature': self.signature,
            'view_number': self.view_number,
            'round_number': self.round_number,
            'parent_hash': self.parent_hash,
            'tx_hashes': self.tx_hashes.copy(),
            'justification': self.justification.copy() if self.justification else None,
            'block_data': self.block_data.copy() if self.block_data else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BlockProposal':
        """Deserialize block proposal from dictionary"""
        return cls(
            height=data['height'],
            block_hash=data['block_hash'],
            validator_address=data['validator_address'],
            timestamp=data['timestamp'],
            signature=data['signature'],
            view_number=data['view_number'],
            round_number=data['round_number'],
            parent_hash=data['parent_hash'],
            tx_hashes=data.get('tx_hashes', []),
            justification=data.get('justification'),
            block_data=data.get('block_data')
        )
    
    def get_signing_data(self) -> bytes:
        """Get data that should be signed for the proposal"""
        data = f"{self.height}|{self.round_number}|{self.block_hash}|{self.parent_hash}|{','.join(self.tx_hashes)}"
        return data.encode('utf-8')
    
    def verify_signature(self, public_key: str) -> bool:
        """Verify proposal signature"""
        try:
            if isinstance(self.signature, str):
                signature_bytes = bytes.fromhex(self.signature)
            else:
                signature_bytes = self.signature
            
            verifying_key = serialization.load_der_public_key(
                bytes.fromhex(public_key),
                backend=default_backend()
            )
            
            signing_data = self.get_signing_data()
            verifying_key.verify(
                signature_bytes,
                signing_data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
            
        except (InvalidSignature, ValueError, Exception) as e:
            return False
    
    def validate(self, expected_validator: str, current_height: int, current_round: int) -> bool:
        """Validate block proposal"""
        if self.validator_address != expected_validator:
            return False
        
        if self.height != current_height:
            return False
        
        if self.round_number != current_round:
            return False
        
        if not self.block_hash or len(self.block_hash) != 64:
            return False
        
        if not self.parent_hash or len(self.parent_hash) != 64:
            return False
        
        # Check timestamp is not too far in future (2 hours)
        if self.timestamp > time.time() + 7200:
            return False
        
        return True