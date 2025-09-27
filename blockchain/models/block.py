# blockchain/models/block.py
import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from enum import Enum, auto
import msgpack

class BlockVersion(Enum):
    GENESIS = 1
    STANDARD = 2
    ENHANCED = 3

@dataclass
class BlockHeader:
    version: int
    height: int
    previous_hash: str
    merkle_root: str
    timestamp: int
    difficulty: int
    nonce: int
    validator: str
    signature: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert header to dictionary for serialization"""
        return asdict(self)
    
    def to_bytes(self) -> bytes:
        """Serialize header to bytes for hashing"""
        data = {
            'version': self.version,
            'height': self.height,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'difficulty': self.difficulty,
            'nonce': self.nonce,
            'validator': self.validator
        }
        return msgpack.packb(data, use_bin_type=True)
    
    def calculate_hash(self) -> str:
        """Calculate block header hash"""
        header_bytes = self.to_bytes()
        return hashlib.sha256(header_bytes).hexdigest()

@dataclass
class Block:
    header: BlockHeader
    transactions: List[Any]  # Changed from Transaction to Any to avoid circular import
    hash: str
    chainwork: int
    size: int
    received_time: float = field(default_factory=time.time)
    weight: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for serialization"""
        # Import Transaction locally to avoid circular imports
        from blockchain.models.transaction import Transaction
        
        return {
            'header': self.header.to_dict(),
            'transactions': [tx.to_dict() if hasattr(tx, 'to_dict') else tx for tx in self.transactions],
            'hash': self.hash,
            'chainwork': self.chainwork,
            'size': self.size,
            'received_time': self.received_time
        }
    
    def to_bytes(self) -> bytes:
        """Serialize block to bytes"""
        return msgpack.packb(self.to_dict(), use_bin_type=True)
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'Block':
        """Deserialize block from bytes"""
        # Import Transaction locally to avoid circular imports
        from blockchain.models.transaction import Transaction
        
        block_dict = msgpack.unpackb(data, raw=False)
        header_data = block_dict['header']
        
        header = BlockHeader(
            version=header_data['version'],
            height=header_data['height'],
            previous_hash=header_data['previous_hash'],
            merkle_root=header_data['merkle_root'],
            timestamp=header_data['timestamp'],
            difficulty=header_data['difficulty'],
            nonce=header_data['nonce'],
            validator=header_data['validator'],
            signature=header_data.get('signature'),
            extra_data=header_data.get('extra_data', {})
        )
        
        # Handle transactions - they might be dicts or Transaction objects
        transactions = []
        for tx_data in block_dict['transactions']:
            if isinstance(tx_data, dict) and 'version' in tx_data:
                # It's a transaction dict, convert to Transaction object
                transactions.append(Transaction.from_dict(tx_data))
            else:
                # It's already a Transaction object or other type
                transactions.append(tx_data)
        
        return cls(
            header=header,
            transactions=transactions,
            hash=block_dict['hash'],
            chainwork=block_dict['chainwork'],
            size=block_dict['size'],
            received_time=block_dict.get('received_time', time.time())
        )
    
    def verify_hash(self) -> bool:
        """Verify that the block hash is correct"""
        return self.hash == self.header.calculate_hash()
        
    def __post_init__(self):
        if not self.hash:
            self.hash = self.header.calculate_hash()
        # Calculate size if not provided
        if not self.size:
            self.size = len(self.to_bytes())