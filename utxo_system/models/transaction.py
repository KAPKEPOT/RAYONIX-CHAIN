# utxo_system/models/transaction.py
import hashlib
import json
import zlib
import msgpack
import secrets
from typing import List, Dict, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from utxo_system.crypto.signatures import sign_transaction_input, verify_transaction_signature
from utxo_system.exceptions import SerializationError, DeserializationError, ValidationError
from utxo_system.utils.logging_config import logger


class TransactionStatus(Enum):
    """Transaction status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    INVALID = "invalid"


class SighashType(Enum):
    """Signature hash types for transaction signing"""
    ALL = 1
    NONE = 2
    SINGLE = 3
    ANYONECANPAY_ALL = 0x81
    ANYONECANPAY_NONE = 0x82
    ANYONECANPAY_SINGLE = 0x83


class AddressType(Enum):
    """Address types supported by the system"""
    P2PKH = "p2pkh"
    P2SH = "p2sh"
    P2WPKH = "p2wpkh"
    P2WSH = "p2wsh"
    P2TR = "p2tr"
    RAYONIX = "rayonix"


@dataclass
class TransactionInput:
    """Enhanced transaction input with witness support and validation"""
    tx_hash: str
    output_index: int
    sequence: int = 0xFFFFFFFF
    signature: Optional[str] = None
    public_key: Optional[str] = None
    address: Optional[str] = None
    witness: Optional[List[str]] = field(default_factory=list)
    script_sig: Optional[str] = None
    redeem_script: Optional[str] = None
    
    def __post_init__(self):
        """Validate input parameters"""
        if not self.tx_hash or len(self.tx_hash) != 64:
            raise ValidationError("Invalid transaction hash: must be 64-character hex string")
        if self.output_index < 0:
            raise ValidationError("Output index cannot be negative")
        if self.sequence < 0:
            raise ValidationError("Sequence cannot be negative")
        if self.tx_hash != "0" * 64:  # Not coinbase
            if not self.address:
                raise ValidationError("Address is required for non-coinbase inputs")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper serialization"""
        return {
            'tx_hash': self.tx_hash,
            'output_index': self.output_index,
            'sequence': self.sequence,
            'signature': self.signature,
            'public_key': self.public_key,
            'address': self.address,
            'witness': self.witness,
            'script_sig': self.script_sig,
            'redeem_script': self.redeem_script
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionInput':
        """Create from dictionary with validation"""
        return cls(
            tx_hash=data['tx_hash'],
            output_index=data['output_index'],
            sequence=data.get('sequence', 0xFFFFFFFF),
            signature=data.get('signature'),
            public_key=data.get('public_key'),
            address=data.get('address'),
            witness=data.get('witness', []),
            script_sig=data.get('script_sig'),
            redeem_script=data.get('redeem_script')
        )
    
    def is_final(self) -> bool:
        """Check if input is final (cannot be replaced)"""
        return self.sequence == 0xFFFFFFFF
    
    def is_coinbase(self) -> bool:
        """Check if this is a coinbase input"""
        return self.tx_hash == "0" * 64 and self.output_index == -1
    
    def get_outpoint(self) -> str:
        """Get unique outpoint identifier"""
        return f"{self.tx_hash}:{self.output_index}"
    
    def validate(self) -> bool:
        """Validate input structure and content"""
        try:
            self.__post_init__()
            return True
        except ValidationError:
            return False


@dataclass
class TransactionOutput:
    """Enhanced transaction output with scripting support"""
    address: str
    amount: int
    locktime: int = 0
    script_pubkey: Optional[str] = None
    script_type: str = "p2pkh"
    is_spent: bool = False
    spent_by: Optional[str] = None
    
    def __post_init__(self):
        """Validate output parameters"""
        if self.amount <= 0:
            raise ValidationError("Output amount must be positive")
        if self.locktime < 0:
            raise ValidationError("Locktime cannot be negative")
        if not self.address:
            raise ValidationError("Output address is required")
        if len(self.address) < 20:  # Basic address length validation
            raise ValidationError("Invalid address format")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'address': self.address,
            'amount': self.amount,
            'locktime': self.locktime,
            'script_pubkey': self.script_pubkey,
            'script_type': self.script_type,
            'is_spent': self.is_spent,
            'spent_by': self.spent_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransactionOutput':
        """Create from dictionary"""
        return cls(
            address=data['address'],
            amount=data['amount'],
            locktime=data.get('locktime', 0),
            script_pubkey=data.get('script_pubkey'),
            script_type=data.get('script_type', 'p2pkh'),
            is_spent=data.get('is_spent', False),
            spent_by=data.get('spent_by')
        )
    
    def is_locked(self, current_block_height: int = 0, current_timestamp: int = 0) -> bool:
        """Check if output is locked by time or block height"""
        if self.locktime == 0:
            return False
        
        if self.locktime < 500000000:  # Block height lock
            return current_block_height < self.locktime
        else:  # Timestamp lock
            return current_timestamp < self.locktime
    
    def can_spend(self, current_block_height: int = 0, current_timestamp: int = 0) -> bool:
        """Check if output can be spent"""
        return not self.is_spent and not self.is_locked(current_block_height, current_timestamp)
    
    def mark_spent(self, spending_txid: str) -> None:
        """Mark output as spent"""
        self.is_spent = True
        self.spent_by = spending_txid
    
    def validate(self) -> bool:
        """Validate output structure"""
        try:
            self.__post_init__()
            return True
        except ValidationError:
            return False


class Transaction:
    """Production-ready transaction implementation with enhanced features"""
    
    VERSION = 1
    WITNESS_MARKER = 0x00
    WITNESS_FLAG = 0x01
    
    def __init__(self, inputs: List[TransactionInput], outputs: List[TransactionOutput], 
                 locktime: int = 0, version: int = VERSION, witness_data: Optional[Dict] = None,
                 metadata: Optional[Dict] = None):
        self.version = version
        self.inputs = inputs
        self.outputs = outputs
        self.locktime = locktime
        self.witness_data = witness_data or {}
        self.metadata = metadata or {}
        self.timestamp = int(__import__('time').time())
        self.nonce = secrets.randbits(64)
        
        # Calculate hashes
        self.hash = self.calculate_hash()
        self.witness_hash = self.calculate_witness_hash()
        
        # Transaction state
        self.status = TransactionStatus.PENDING
        self.confirmations: int = 0
        self.block_height: Optional[int] = None
        self.block_hash: Optional[str] = None
        
        # Validation cache
        self._validation_cache: Dict[str, Any] = {}
        
        # Validate transaction structure
        self._validate_transaction()
    
    def _validate_transaction(self) -> None:
        """Validate transaction structure and basic rules"""
        if not self.inputs:
            raise ValidationError("Transaction must have at least one input")
        
        if not self.outputs:
            raise ValidationError("Transaction must have at least one output")
        
        if self.locktime < 0:
            raise ValidationError("Locktime cannot be negative")
        
        if self.version < 1:
            raise ValidationError("Invalid transaction version")
        
        # Check for duplicate inputs
        input_outpoints = set()
        for inp in self.inputs:
            if not inp.validate():
                raise ValidationError(f"Invalid transaction input: {inp.get_outpoint()}")
            
            outpoint = inp.get_outpoint()
            if outpoint in input_outpoints:
                raise ValidationError(f"Duplicate input: {outpoint}")
            input_outpoints.add(outpoint)
        
        # Validate outputs
        for i, output in enumerate(self.outputs):
            if not output.validate():
                raise ValidationError(f"Invalid transaction output at index {i}")
    
    def calculate_hash(self) -> str:
        """Calculate transaction hash (excluding witness data) using double SHA256"""
        tx_data = {
            'version': self.version,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'locktime': self.locktime,
            'nonce': self.nonce
        }
        
        # Remove witness-related fields for hash calculation
        for inp_dict in tx_data['inputs']:
            inp_dict.pop('witness', None)
            inp_dict.pop('signature', None)
            inp_dict.pop('public_key', None)
            inp_dict.pop('script_sig', None)
        
        # Use deterministic serialization
        serialized = msgpack.packb(tx_data, use_bin_type=True, strict_types=True)
        return hashlib.sha256(hashlib.sha256(serialized).digest()).hexdigest()
    
    def calculate_witness_hash(self) -> str:
        """Calculate witness transaction hash (including witness data)"""
        tx_data = {
            'version': self.version,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'locktime': self.locktime,
            'witness_data': self.witness_data,
            'nonce': self.nonce,
            'timestamp': self.timestamp
        }
        
        serialized = msgpack.packb(tx_data, use_bin_type=True, strict_types=True)
        return hashlib.sha256(hashlib.sha256(serialized).digest()).hexdigest()
    
    def sign_input(self, input_index: int, private_key: Any, utxo: Any, 
                   sighash_type: SighashType = SighashType.ALL) -> None:
        """Sign a specific transaction input with enhanced error handling"""
        if input_index < 0 or input_index >= len(self.inputs):
            raise ValidationError(f"Invalid input index: {input_index}")
        
        if not utxo:
            raise ValidationError("UTXO reference is required for signing")
        
        try:
            sign_transaction_input(self, input_index, private_key, utxo, sighash_type)
            # Invalidate cache after signing
            self._validation_cache.clear()
        except Exception as e:
            logger.error(f"Failed to sign input {input_index}: {e}")
            raise
    
    def verify_input_signature(self, input_index: int, utxo_set: Any) -> bool:
        """Verify signature for a specific input with caching"""
        cache_key = f"verify_input_{input_index}"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        if input_index < 0 or input_index >= len(self.inputs):
            return False
        
        try:
            result = verify_transaction_signature(self, input_index, utxo_set)
            self._validation_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f"Signature verification failed for input {input_index}: {e}")
            self._validation_cache[cache_key] = False
            return False
    
    def verify_all_signatures(self, utxo_set: Any) -> bool:
        """Verify all input signatures with comprehensive checking"""
        cache_key = "verify_all_signatures"
        if cache_key in self._validation_cache:
            return self._validation_cache[cache_key]
        
        if self.is_coinbase():
            self._validation_cache[cache_key] = True
            return True
        
        for i in range(len(self.inputs)):
            if not self.verify_input_signature(i, utxo_set):
                self._validation_cache[cache_key] = False
                return False
        
        self._validation_cache[cache_key] = True
        return True
    
    def is_coinbase(self) -> bool:
        """Check if this is a coinbase transaction"""
        return (len(self.inputs) == 1 and 
                self.inputs[0].is_coinbase())
    
    def is_final(self, block_height: int = 0, median_time_past: int = 0) -> bool:
        """Check if transaction is final considering locktime and sequence numbers"""
        if self.locktime == 0:
            return True
        
        if any(not inp.is_final() for inp in self.inputs):
            return False
        
        if self.locktime < 500000000:  # Block height
            return block_height >= self.locktime
        else:  # Timestamp
            return median_time_past >= self.locktime
    
    def to_bytes(self) -> bytes:
        """Advanced transaction serialization with multiple formats and compression"""
        try:
            # Format 1: Optimized binary format with compression
            tx_data = {
                'version': self.version,
                'inputs': [inp.to_dict() for inp in self.inputs],
                'outputs': [out.to_dict() for out in self.outputs],
                'locktime': self.locktime,
                'witness_data': self.witness_data,
                'metadata': self.metadata,
                'hash': self.hash,
                'witness_hash': self.witness_hash,
                'timestamp': self.timestamp,
                'nonce': self.nonce,
                'status': self.status.value,
                'confirmations': self.confirmations,
                'block_height': self.block_height,
                'block_hash': self.block_hash
            }
            
            # Use msgpack for efficient binary serialization
            serialized = msgpack.packb(tx_data, use_bin_type=True, strict_types=True)
            
            # Add compression with intelligent fallback
            try:
                compressed = zlib.compress(serialized, level=zlib.Z_BEST_COMPRESSION)
                if len(compressed) < len(serialized) * 0.9:  # Only use if significant compression
                    return b'\x01' + compressed  # Marker for compressed data
            except Exception as compression_error:
                logger.warning(f"Compression failed, using uncompressed: {compression_error}")
            
            return b'\x00' + serialized  # Marker for uncompressed data
            
        except Exception as e:
            logger.error(f"Transaction serialization error: {e}")
            
            # Fallback to JSON format
            try:
                tx_dict = self.to_dict()
                json_data = json.dumps(tx_dict, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
                return b'\x02' + json_data.encode('utf-8')  # Marker for JSON format
            except Exception as fallback_error:
                logger.critical(f"Critical: Transaction serialization completely failed: {fallback_error}")
                raise SerializationError(f"Transaction cannot be serialized: {fallback_error}")

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Transaction':
        """Robust deserialization with format auto-detection and validation"""
        if len(data) < 1:
            raise DeserializationError("Empty transaction data")
        
        format_marker = data[0]
        payload = data[1:]
        
        try:
            if format_marker == 0x01:  # Compressed msgpack
                payload = zlib.decompress(payload)
                tx_data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            elif format_marker == 0x00:  # Uncompressed msgpack
                tx_data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            elif format_marker == 0x02:  # JSON format
                tx_data = json.loads(payload.decode('utf-8'))
            else:
                # Auto-detect format
                try:
                    payload = zlib.decompress(data)
                    tx_data = msgpack.unpackb(payload, raw=False, strict_map_key=False)
                except:
                    tx_data = json.loads(data.decode('utf-8'))
            
            # Reconstruct inputs and outputs
            inputs = [TransactionInput.from_dict(inp_dict) for inp_dict in tx_data.get('inputs', [])]
            outputs = [TransactionOutput.from_dict(out_dict) for out_dict in tx_data.get('outputs', [])]
            
            # Create transaction instance
            tx = cls(
                inputs=inputs,
                outputs=outputs,
                locktime=tx_data.get('locktime', 0),
                version=tx_data.get('version', cls.VERSION),
                witness_data=tx_data.get('witness_data', {}),
                metadata=tx_data.get('metadata', {})
            )
            
            # Restore additional fields
            tx.hash = tx_data.get('hash', tx.calculate_hash())
            tx.witness_hash = tx_data.get('witness_hash', tx.calculate_witness_hash())
            tx.timestamp = tx_data.get('timestamp', tx.timestamp)
            tx.nonce = tx_data.get('nonce', tx.nonce)
            tx.status = TransactionStatus(tx_data.get('status', 'pending'))
            tx.confirmations = tx_data.get('confirmations', 0)
            tx.block_height = tx_data.get('block_height')
            tx.block_hash = tx_data.get('block_hash')
            
            return tx
            
        except Exception as e:
            logger.error(f"Transaction deserialization error: {e}")
            raise DeserializationError(f"Failed to deserialize transaction: {e}")
    
    def get_related_addresses(self) -> List[str]:
        """Get all addresses related to this transaction with deduplication"""
        addresses: Set[str] = set()
        
        for inp in self.inputs:
            if inp.address:
                addresses.add(inp.address)
        
        for output in self.outputs:
            addresses.add(output.address)
        
        return sorted(list(addresses))
    
    def get_input_amounts(self, utxo_set: Any) -> Dict[str, int]:
        """Get amounts for all inputs with error handling"""
        input_amounts = {}
        for inp in self.inputs:
            if inp.is_coinbase():
                continue  # Coinbase inputs have no previous output
                
            utxo_id = inp.get_outpoint()
            try:
                utxo = utxo_set.get_utxo(utxo_id)
                if utxo:
                    input_amounts[utxo_id] = getattr(utxo, 'amount', 0)
            except Exception as e:
                logger.warning(f"Could not get amount for UTXO {utxo_id}: {e}")
        
        return input_amounts
    
    def get_output_amounts(self) -> Dict[str, int]:
        """Get amounts for all outputs with indexing"""
        return {f"{self.hash}:{i}": out.amount for i, out in enumerate(self.outputs)}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to comprehensive dictionary representation"""
        return {
            'version': self.version,
            'hash': self.hash,
            'witness_hash': self.witness_hash,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs],
            'locktime': self.locktime,
            'witness_data': self.witness_data,
            'metadata': self.metadata,
            'status': self.status.value,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'confirmations': self.confirmations,
            'block_height': self.block_height,
            'block_hash': self.block_hash,
            'is_coinbase': self.is_coinbase(),
            'size': self.calculate_size(),
            'vsize': self.calculate_vsize()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary with validation"""
        inputs = [TransactionInput.from_dict(inp_dict) for inp_dict in data['inputs']]
        outputs = [TransactionOutput.from_dict(out_dict) for out_dict in data['outputs']]
        
        tx = cls(
            inputs=inputs,
            outputs=outputs,
            locktime=data['locktime'],
            version=data['version'],
            witness_data=data.get('witness_data', {}),
            metadata=data.get('metadata', {})
        )
        
        # Restore additional fields
        tx.hash = data.get('hash', tx.calculate_hash())
        tx.witness_hash = data.get('witness_hash', tx.calculate_witness_hash())
        tx.status = TransactionStatus(data.get('status', 'pending'))
        tx.timestamp = data.get('timestamp', tx.timestamp)
        tx.nonce = data.get('nonce', tx.nonce)
        tx.confirmations = data.get('confirmations', 0)
        tx.block_height = data.get('block_height')
        tx.block_hash = data.get('block_hash')
        
        return tx
    
    def calculate_fee(self, utxo_set: Any) -> int:
        """Calculate transaction fee with comprehensive error handling"""
        if self.is_coinbase():
            return 0  # Coinbase transactions have no fee
        
        total_input = 0
        total_output = sum(output.amount for output in self.outputs)
        
        for tx_input in self.inputs:
            if tx_input.is_coinbase():
                continue
                
            utxo_id = tx_input.get_outpoint()
            try:
                utxo = utxo_set.get_utxo(utxo_id)
                if utxo:
                    total_input += getattr(utxo, 'amount', 0)
                else:
                    logger.warning(f"UTXO not found: {utxo_id}")
            except Exception as e:
                logger.error(f"Error getting UTXO {utxo_id}: {e}")
                continue
        
        fee = total_input - total_output
        if fee < 0:
            logger.warning(f"Negative fee calculated: {fee}. Inputs: {total_input}, Outputs: {total_output}")
            return 0
        
        return fee
    
    def calculate_size(self) -> int:
        """Calculate approximate transaction size in bytes"""
        return len(self.to_bytes())
    
    def calculate_vsize(self) -> int:
        """Calculate virtual size (for segwit transactions)"""
        base_size = self.calculate_size()
        if not self.witness_data:
            return base_size
        
        # Simplified vsize calculation: (base_size * 3 + total_size) / 4
        witness_size = sum(len(str(witness)) for witness_list in self.witness_data.values() 
                          for witness in witness_list) if isinstance(self.witness_data, dict) else 0
        total_size = base_size + witness_size
        return (base_size * 3 + total_size) // 4
    
    def get_output(self, index: int) -> Optional[TransactionOutput]:
        """Get output by index with bounds checking"""
        if 0 <= index < len(self.outputs):
            return self.outputs[index]
        return None
    
    def add_witness_data(self, input_index: int, witness: List[str]) -> None:
        """Add witness data for segwit transaction"""
        if input_index < 0 or input_index >= len(self.inputs):
            raise ValidationError(f"Invalid input index: {input_index}")
        
        witness_key = f"input_{input_index}"
        self.witness_data[witness_key] = witness
        self.witness_hash = self.calculate_witness_hash()
        self._validation_cache.clear()
    
    def validate_structure(self) -> bool:
        """Comprehensive transaction structure validation"""
        try:
            self._validate_transaction()
            
            # Additional validation checks
            if self.calculate_hash() != self.hash:
                logger.warning("Transaction hash mismatch")
                return False
            
            # Check output amounts are positive
            for output in self.outputs:
                if output.amount <= 0:
                    return False
            
            return True
        except ValidationError:
            return False
    
    def update_status(self, confirmations: int, block_height: Optional[int] = None, 
                     block_hash: Optional[str] = None) -> None:
        """Update transaction status and confirmation information"""
        self.confirmations = confirmations
        self.block_height = block_height
        self.block_hash = block_hash
        
        if confirmations >= 6:  # Consider confirmed after 6 blocks
            self.status = TransactionStatus.CONFIRMED
        elif confirmations > 0:
            self.status = TransactionStatus.PENDING
        else:
            self.status = TransactionStatus.PENDING
    
    def __eq__(self, other: Any) -> bool:
        """Equality comparison based on transaction hash"""
        if not isinstance(other, Transaction):
            return False
        return self.hash == other.hash
    
    def __hash__(self) -> int:
        """Hash implementation for use in sets and dictionaries"""
        return int(self.hash[:16], 16) if self.hash else 0
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return (f"Transaction(hash={self.hash[:16]}..., inputs={len(self.inputs)}, "
                f"outputs={len(self.outputs)}, amount={sum(out.amount for out in self.outputs)})")
    
    def __str__(self) -> str:
        """User-friendly string representation"""
        total_amount = sum(output.amount for output in self.outputs)
        return (f"Transaction {self.hash[:16]}... ({self.status.value}): "
                f"{len(self.inputs)} inputs, {len(self.outputs)} outputs, "
                f"amount: {total_amount}")
                
class TransactionFactory:
    """Factory class for creating different types of transactions"""
    
    @staticmethod
    def create_coinbase_transaction(recipient_address: str, amount: int, 
                                  block_height: int, extra_data: str = "") -> Transaction:
        """Create a coinbase transaction"""
        coinbase_input = TransactionInput(
            tx_hash="0" * 64,
            output_index=-1,
            sequence=0xFFFFFFFF,
            signature=None,
            public_key=None,
            address=None
        )
        
        # Include block height and extra data in output
        output = TransactionOutput(
            address=recipient_address,
            amount=amount,
            script_pubkey=f"Coinbase_{block_height}_{extra_data}"[:100]  # Limit length
        )
        
        return Transaction(
            inputs=[coinbase_input],
            outputs=[output],
            locktime=0,
            metadata={
                'coinbase': True,
                'block_height': block_height,
                'extra_data': extra_data
            }
        )
    
    @staticmethod
    def create_transfer_transaction(inputs: List[TransactionInput], 
                                  outputs: List[TransactionOutput],
                                  locktime: int = 0) -> Transaction:
        """Create a standard transfer transaction"""
        return Transaction(
            inputs=inputs,
            outputs=outputs,
            locktime=locktime
        )
    
    @staticmethod
    def create_multisig_transaction(inputs: List[TransactionInput], 
                                  outputs: List[TransactionOutput],
                                  required_signatures: int, 
                                  public_keys: List[str]) -> Transaction:
        """Create a multisignature transaction"""
        tx = Transaction(inputs, outputs)
        
        # Store multisig info in witness data
        tx.witness_data['multisig_info'] = {
            'required_signatures': required_signatures,
            'public_keys': public_keys,
            'total_keys': len(public_keys)
        }
        
        return tx
    
    @staticmethod
    def create_contract_transaction(inputs: List[TransactionInput],
                                  contract_address: str,
                                  contract_data: str,
                                  amount: int = 0) -> Transaction:
        """Create a smart contract transaction"""
        output = TransactionOutput(
            address=contract_address,
            amount=amount,
            script_type="contract"
        )
        
        return Transaction(
            inputs=inputs,
            outputs=[output],
            metadata={
                'contract_data': contract_data,
                'contract_address': contract_address
            }
        )                