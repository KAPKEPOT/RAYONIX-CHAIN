# utxo_system/models/transaction.py
import hashlib
import json
import zlib
import msgpack
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from utxo_system.crypto.signatures import sign_transaction_input, verify_transaction_signature
from utxo_system.exceptions import SerializationError, DeserializationError
from utxo_system.utils.logging_config import logger

@dataclass
class TransactionInput:
    tx_hash: str
    output_index: int
    signature: Optional[str] = None
    public_key: Optional[str] = None
    address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'tx_hash': self.tx_hash,
            'output_index': self.output_index,
            'signature': self.signature,
            'public_key': self.public_key,
            'address': self.address
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransactionInput':
        return cls(
            tx_hash=data['tx_hash'],
            output_index=data['output_index'],
            signature=data.get('signature'),
            public_key=data.get('public_key'),
            address=data.get('address')
        )

@dataclass
class TransactionOutput:
    address: str
    amount: int
    locktime: int = 0
    script_pubkey: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'amount': self.amount,
            'locktime': self.locktime,
            'script_pubkey': self.script_pubkey
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TransactionOutput':
        return cls(
            address=data['address'],
            amount=data['amount'],
            locktime=data.get('locktime', 0),
            script_pubkey=data.get('script_pubkey')
        )

class Transaction:
    def __init__(self, inputs: List[TransactionInput], outputs: List[TransactionOutput], 
                 locktime: int = 0, version: int = 1):
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.locktime = locktime
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        tx_data = json.dumps({
            'version': self.version,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'locktime': self.locktime
        }, sort_keys=True)
        return hashlib.sha256(tx_data.encode()).hexdigest()
    
    def sign_input(self, input_index: int, private_key, utxo, sighash_type: int = 1):
        sign_transaction_input(self, input_index, private_key, utxo, sighash_type)
    
    def verify_input_signature(self, input_index: int, utxo_set) -> bool:
        return verify_transaction_signature(self, input_index, utxo_set)
    
    def to_bytes(self) -> bytes:
        """Production-ready transaction serialization"""
        try:
            tx_data = {
                'version': self.version,
                'inputs': [inp.to_dict() for inp in self.inputs],
                'outputs': [out.to_dict() for out in self.outputs],
                'locktime': self.locktime,
                'hash': self.hash
            }
            
            serialized = msgpack.packb(tx_data, use_bin_type=True)
            return zlib.compress(serialized)
            
        except Exception as e:
            logger.error(f"Transaction serialization error: {e}")
            try:
                tx_dict = {
                    'version': self.version,
                    'inputs': [inp.to_dict() for inp in self.inputs],
                    'outputs': [out.to_dict() for out in self.outputs],
                    'locktime': self.locktime,
                    'hash': self.hash
                }
                return json.dumps(tx_dict, sort_keys=True).encode('utf-8')
            except Exception as fallback_error:
                logger.critical(f"Critical: Transaction serialization completely failed: {fallback_error}")
                raise SerializationError(f"Transaction cannot be serialized: {fallback_error}")

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Transaction':
        """Deserialize transaction from bytes"""
        try:
            try:
                data = zlib.decompress(data)
                tx_data = msgpack.unpackb(data, raw=False)
            except:
                tx_data = json.loads(data.decode('utf-8'))
            
            inputs = [TransactionInput.from_dict(inp_dict) for inp_dict in tx_data.get('inputs', [])]
            outputs = [TransactionOutput.from_dict(out_dict) for out_dict in tx_data.get('outputs', [])]
            
            tx = cls(
                inputs=inputs,
                outputs=outputs,
                locktime=tx_data.get('locktime', 0),
                version=tx_data.get('version', 1)
            )
            tx.hash = tx_data.get('hash', tx.calculate_hash())
            return tx
            
        except Exception as e:
            logger.error(f"Transaction deserialization error: {e}")
            raise DeserializationError(f"Failed to deserialize transaction: {e}")
    
    def get_related_addresses(self) -> List[str]:
        addresses = set()
        
        for inp in self.inputs:
            if inp.address:
                addresses.add(inp.address)
        
        for output in self.outputs:
            addresses.add(output.address)
        
        return list(addresses)
    
    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'hash': self.hash,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'locktime': self.locktime
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        inputs = [TransactionInput.from_dict(inp_dict) for inp_dict in data['inputs']]
        outputs = [TransactionOutput.from_dict(out_dict) for out_dict in data['outputs']]
        
        tx = cls(
            inputs=inputs,
            outputs=outputs,
            locktime=data['locktime'],
            version=data['version']
        )
        tx.hash = data['hash']
        return tx
    
    def calculate_fee(self, utxo_set) -> int:
        total_input = 0
        total_output = sum(output.amount for output in self.outputs)
        
        for tx_input in self.inputs:
            utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
            utxo = utxo_set.get_utxo(utxo_id)
            if utxo:
                total_input += utxo.amount
        
        return total_input - total_output