# utxo_system/models/utxo.py
import struct
from typing import Dict, Optional
from dataclasses import dataclass, field
from utxo_system.utils.helpers import current_timestamp

@dataclass
class UTXO:
    tx_hash: str
    output_index: int
    address: str
    amount: int
    spent: bool = False
    locktime: int = 0
    created_at_block: int = 0
    created_at: int = field(default_factory=current_timestamp)
    
    @property
    def id(self) -> str:
        return f"{self.tx_hash}:{self.output_index}"
    
    def is_spendable(self, current_block_height: int, current_time: int) -> bool:
        if self.spent:
            return False
        
        if self.locktime > 0:
            if self.locktime < 500000000:  # Block height
                return current_block_height >= self.locktime
            else:  # Timestamp
                return current_time >= self.locktime
        
        return True

    def serialize(self) -> bytes:
        """Serialize UTXO to bytes for efficient storage"""
        spent_byte = b'\x01' if self.spent else b'\x00'
        locktime_bytes = struct.pack('>I', self.locktime)
        created_at_block_bytes = struct.pack('>I', self.created_at_block)
        amount_bytes = struct.pack('>Q', self.amount)
        address_bytes = self.address.encode('utf-8')
        address_len = struct.pack('B', len(address_bytes))
        tx_hash_bytes = bytes.fromhex(self.tx_hash)
        output_index_bytes = struct.pack('>I', self.output_index)
        created_at_bytes = struct.pack('>Q', self.created_at)
        
        return (spent_byte + locktime_bytes + created_at_block_bytes + amount_bytes + 
                address_len + address_bytes + tx_hash_bytes + output_index_bytes + created_at_bytes)
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'UTXO':
        """Deserialize UTXO from bytes"""
        spent = data[0] == 1
        locktime = struct.unpack('>I', data[1:5])[0]
        created_at_block = struct.unpack('>I', data[5:9])[0]
        amount = struct.unpack('>Q', data[9:17])[0]
        address_len = data[17]
        address = data[18:18+address_len].decode('utf-8')
        tx_hash = data[18+address_len:18+address_len+32].hex()
        output_index = struct.unpack('>I', data[18+address_len+32:18+address_len+36])[0]
        created_at = struct.unpack('>Q', data[18+address_len+36:18+address_len+44])[0]
        
        return cls(
            tx_hash=tx_hash,
            output_index=output_index,
            address=address,
            amount=amount,
            spent=spent,
            locktime=locktime,
            created_at_block=created_at_block,
            created_at=created_at
        )
    
    def to_dict(self) -> Dict:
        return {
            'tx_hash': self.tx_hash,
            'output_index': self.output_index,
            'address': self.address,
            'amount': self.amount,
            'spent': self.spent,
            'locktime': self.locktime,
            'created_at_block': self.created_at_block,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UTXO':
        return cls(
            tx_hash=data['tx_hash'],
            output_index=data['output_index'],
            address=data['address'],
            amount=data['amount'],
            spent=data['spent'],
            locktime=data.get('locktime', 0),
            created_at_block=data.get('created_at_block', 0),
            created_at=data.get('created_at', current_timestamp())
        )