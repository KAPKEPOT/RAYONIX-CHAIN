# utxo_system/database/serialization.py
import struct
from typing import Optional
from ..models.utxo import UTXO

def serialize_utxo(utxo: UTXO) -> bytes:
    """Serialize UTXO to bytes for efficient storage"""
    spent_byte = b'\x01' if utxo.spent else b'\x00'
    locktime_bytes = struct.pack('>I', utxo.locktime)
    created_at_block_bytes = struct.pack('>I', utxo.created_at_block)
    amount_bytes = struct.pack('>Q', utxo.amount)
    address_bytes = utxo.address.encode('utf-8')
    address_len = struct.pack('B', len(address_bytes))
    tx_hash_bytes = bytes.fromhex(utxo.tx_hash)
    output_index_bytes = struct.pack('>I', utxo.output_index)
    created_at_bytes = struct.pack('>Q', utxo.created_at)
    
    return (spent_byte + locktime_bytes + created_at_block_bytes + amount_bytes + 
            address_len + address_bytes + tx_hash_bytes + output_index_bytes + created_at_bytes)

def deserialize_utxo(data: bytes) -> UTXO:
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
    
    return UTXO(
        tx_hash=tx_hash,
        output_index=output_index,
        address=address,
        amount=amount,
        spent=spent,
        locktime=locktime,
        created_at_block=created_at_block,
        created_at=created_at
    )