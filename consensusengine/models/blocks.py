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
        
        while len(tx_hashes) > 1:
            new_hashes = []
            
            for i in range(0, len(tx_hashes), 2):
                if i + 1 < len(tx_hashes):
                    combined = tx_hashes[i] + tx_hashes[i + 1]
                    new_hash = hashlib.sha256(combined.encode()).hexdigest()
                else:
                    new_hash = tx_hashes[i]
                
                new_hashes.append(new_hash)
            
            tx_hashes = new_hashes
        
        return tx_hashes[0] if tx_hashes else "0" * 64
    
    def validate(self, previous_block: Optional['Block'] = None) -> bool:
        """Validate block structure and consistency"""
        # Validate header
        if self.header.height < 0:
            return False
        
        if previous_block and self.header.previous_hash != previous_block.hash:
            return False
        
        # Validate Merkle root
        if self.calculate_merkle_root() != self.header.merkle_root:
            return False
        
        # Validate transactions
        for tx in self.transactions:
            if not tx.validate():
                return False
        
        return True

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