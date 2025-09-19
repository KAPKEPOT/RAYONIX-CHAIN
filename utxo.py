# utxo.py
import hashlib
import json
import plyvel
import struct
import threading
from typing import List, Dict, Set, Tuple, Optional, Iterator
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from contextlib import contextmanager

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database key prefixes
UTXO_PREFIX = b'u:'  # u:tx_hash:output_index -> UTXO data
ADDRESS_INDEX_PREFIX = b'a:'  # a:address -> set of UTXO IDs
SPENT_UTXO_PREFIX = b's:'  # s:tx_hash:output_index -> spent UTXO data
METADATA_PREFIX = b'm:'  # m:key -> metadata values
LAST_BLOCK_HEIGHT_KEY = b'm:last_block_height'

class UTXO:
    def __init__(self, tx_hash: str, output_index: int, address: str, amount: int):
        self.tx_hash = tx_hash
        self.output_index = output_index
        self.address = address
        self.amount = amount
        self.spent = False
        self.locktime = 0  # Block height or timestamp when spendable
        self.created_at_block = 0  # Block height when this UTXO was created
        
    def to_dict(self) -> Dict:
        return {
            'tx_hash': self.tx_hash,
            'output_index': self.output_index,
            'address': self.address,
            'amount': self.amount,
            'spent': self.spent,
            'locktime': self.locktime,
            'created_at_block': self.created_at_block
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UTXO':
        utxo = cls(
            data['tx_hash'],
            data['output_index'],
            data['address'],
            data['amount']
        )
        utxo.spent = data['spent']
        utxo.locktime = data.get('locktime', 0)
        utxo.created_at_block = data.get('created_at_block', 0)
        return utxo
    
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
        # Format: spent(1) + locktime(4) + created_at_block(4) + amount(8) + address_len(1) + address + tx_hash(32) + output_index(4)
        spent_byte = b'\x01' if self.spent else b'\x00'
        locktime_bytes = struct.pack('>I', self.locktime)
        created_at_bytes = struct.pack('>I', self.created_at_block)
        amount_bytes = struct.pack('>Q', self.amount)
        address_bytes = self.address.encode('utf-8')
        address_len = struct.pack('B', len(address_bytes))
        tx_hash_bytes = bytes.fromhex(self.tx_hash)
        output_index_bytes = struct.pack('>I', self.output_index)
        
        return (spent_byte + locktime_bytes + created_at_bytes + amount_bytes + 
                address_len + address_bytes + tx_hash_bytes + output_index_bytes)
    
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
        
        utxo = cls(tx_hash, output_index, address, amount)
        utxo.spent = spent
        utxo.locktime = locktime
        utxo.created_at_block = created_at_block
        return utxo

class Transaction:
    def __init__(self, inputs: List[Dict], outputs: List[Dict], locktime: int = 0, version: int = 1):
        self.version = version
        self.inputs = inputs  # [{ 'tx_hash', 'output_index', 'signature', 'public_key', 'address' }]
        self.outputs = outputs  # [{ 'address', 'amount', 'locktime' }]
        self.locktime = locktime
        self.hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        serializable_inputs = []
        for inp in self.inputs:
        	if isinstance(inp, dict):
        		serializable_inputs.append(inp)
        	else:
        		# Convert TransactionInput object to dict if needed
        		serializable_inputs.append({
        		    'tx_hash': getattr(inp, 'tx_hash', ''),
        		    'output_index': getattr(inp, 'output_index', 0),
        		    'signature': getattr(inp, 'signature', ''),
        		    'public_key': getattr(inp, 'public_key', ''),
        		    'address': getattr(inp, 'address', '')
        		})
        serializable_outputs = []
        for out in self.outputs:
        	if isinstance(out, dict):
        		serializable_outputs.append(out)
        	else:
        		# Convert TransactionOutput object to dict
        		serializable_outputs.append({
        		    'address': getattr(out, 'address', ''),
        		    'amount': getattr(out, 'amount', 0),
        		    'locktime': getattr(out, 'locktime', 0),
        		    'script_pubkey': getattr(out, 'script_pubkey', '')
        		})
        tx_data = json.dumps({
            'version': self.version,
            'inputs': serializable_inputs,
            'outputs': serializable_outputs,
            'locktime': self.locktime
        }, sort_keys=True)
        return hashlib.sha256(tx_data.encode()).hexdigest()
    
    def sign_input(self, input_index: int, private_key: ec.EllipticCurvePrivateKey, 
                  utxo: UTXO, sighash_type: int = 1):
        if input_index >= len(self.inputs):
            raise ValueError("Invalid input index")
        
        # Create signing data
        signing_data = self._get_signing_data(input_index, utxo, sighash_type)
        
        # Sign the data
        signature = private_key.sign(
            signing_data.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        
        # Store signature and public key
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        ).hex()
        
        self.inputs[input_index]['signature'] = signature.hex()
        self.inputs[input_index]['public_key'] = public_key
    
    def _get_signing_data(self, input_index: int, utxo: UTXO, sighash_type: int) -> str:
        # Create copy without signatures for this input
        inputs_copy = []
        for i, inp in enumerate(self.inputs):
            if i == input_index:
                inp_copy = {k: v for k, v in inp.items() if k not in ['signature', 'public_key']}
            else:
                inp_copy = {k: v for k, v in inp.items() if k != 'signature'}
            inputs_copy.append(inp_copy)
        
        # Include referenced UTXO in signing data
        signing_data = json.dumps({
            'version': self.version,
            'inputs': inputs_copy,
            'outputs': self.outputs,
            'locktime': self.locktime,
            'referenced_utxo': utxo.to_dict(),
            'sighash_type': sighash_type
        }, sort_keys=True)
        
        return signing_data
        
    def to_bytes(self) -> bytes:
    	"""Convert transaction to bytes for serialization"""
    	try:
    		# Convert transaction data to JSON and encode to bytes
    		tx_dict = self.to_dict() if hasattr(self, 'to_dict') else asdict(self)
    		return json.dumps(tx_dict, sort_keys=True).encode('utf-8')
    	except Exception as e:
    		logger.error(f"Error converting transaction to bytes: {e}")
    		return b''
    
    def verify_input_signature(self, input_index: int, utxo_set: 'UTXOSet') -> bool:
        if input_index >= len(self.inputs) or 'signature' not in self.inputs[input_index]:
            return False
        
        try:
            signature = bytes.fromhex(self.inputs[input_index]['signature'])
            public_key_bytes = bytes.fromhex(self.inputs[input_index]['public_key'])
            
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), public_key_bytes
            )
            
            # Get the referenced UTXO
            tx_input = self.inputs[input_index]
            utxo_id = f"{tx_input['tx_hash']}:{tx_input['output_index']}"
            utxo = utxo_set.get_utxo(utxo_id)
            
            if not utxo:
                return False
            
            # Reconstruct signing data
            signing_data = self._get_signing_data(input_index, utxo, 1)
            
            public_key.verify(
                signature,
                signing_data.encode(),
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except (InvalidSignature, ValueError, Exception):
            return False
    
    def get_related_addresses(self) -> List[str]:
        addresses = set()
        
        for inp in self.inputs:
            if 'address' in inp:
                addresses.add(inp['address'])
        
        for output in self.outputs:
            addresses.add(output['address'])
        
        return list(addresses)
    
    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'hash': self.hash,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'locktime': self.locktime
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        tx = cls(
            data['inputs'],
            data['outputs'],
            data['locktime'],
            data['version']
        )
        tx.hash = data['hash']
        return tx
    
    def calculate_fee(self, utxo_set: 'UTXOSet') -> int:
        total_input = 0
        total_output = sum(output['amount'] for output in self.outputs)
        
        for tx_input in self.inputs:
            utxo_id = f"{tx_input['tx_hash']}:{tx_input['output_index']}"
            utxo = utxo_set.get_utxo(utxo_id)
            if utxo:
                total_input += utxo.amount
        
        return total_input - total_output

class UTXOSet:
    def __init__(self, db_path: str = './utxo_db'):
        self.db_path = db_path
        self.db = plyvel.DB(db_path, create_if_missing=True)
        self.lock = threading.RLock()
        
    def close(self):
        """Close the database connection"""
        self.db.close()
    
    def _get_utxo_key(self, utxo_id: str) -> bytes:
        """Get database key for a UTXO"""
        return UTXO_PREFIX + utxo_id.encode('utf-8')
    
    def _get_address_index_key(self, address: str) -> bytes:
        """Get database key for address index"""
        return ADDRESS_INDEX_PREFIX + address.encode('utf-8')
    
    def _get_spent_utxo_key(self, utxo_id: str) -> bytes:
        """Get database key for spent UTXO"""
        return SPENT_UTXO_PREFIX + utxo_id.encode('utf-8')
    
    @contextmanager
    def atomic_write(self):
        """Context manager for atomic database operations"""
        batch = self.db.write_batch()
        try:
            yield batch
            batch.write()
        except Exception:
            batch.clear()
            raise
    
    def add_utxo(self, utxo: UTXO, batch=None):
        """Add a UTXO to the database"""
        utxo_id = utxo.id
        utxo_key = self._get_utxo_key(utxo_id)
        address_key = self._get_address_index_key(utxo.address)
        
        write_batch = batch if batch else self.db
        
        # Store the UTXO
        write_batch.put(utxo_key, utxo.serialize())
        
        # Update address index
        existing_utxos = set()
        existing_data = write_batch.get(address_key)
        if existing_data:
            # Deserialize the set of UTXO IDs
            existing_utxos = set(json.loads(existing_data.decode('utf-8')))
        
        existing_utxos.add(utxo_id)
        write_batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
    
    def spend_utxo(self, utxo_id: str, batch=None):
        """Mark a UTXO as spent"""
        with self.lock:
            utxo_key = self._get_utxo_key(utxo_id)
            utxo_data = self.db.get(utxo_key)
            
            if not utxo_data:
                return False
            
            utxo = UTXO.deserialize(utxo_data)
            utxo.spent = True
            
            write_batch = batch if batch else self.db
            
            # Move to spent UTXOs
            spent_key = self._get_spent_utxo_key(utxo_id)
            write_batch.put(spent_key, utxo.serialize())
            write_batch.delete(utxo_key)
            
            # Update address index
            address_key = self._get_address_index_key(utxo.address)
            existing_data = write_batch.get(address_key)
            if existing_data:
                existing_utxos = set(json.loads(existing_data.decode('utf-8')))
                existing_utxos.discard(utxo_id)
                
                if existing_utxos:
                    write_batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
                else:
                    write_batch.delete(address_key)
            
            return True
    
    def get_utxos_for_address(self, address: str, current_block_height: int = 0, 
                             current_time: int = 0) -> List[UTXO]:
        """Get all spendable UTXOs for an address"""
        with self.lock:
            address_key = self._get_address_index_key(address)
            address_data = self.db.get(address_key)
            
            if not address_data:
                return []
            
            utxo_ids = json.loads(address_data.decode('utf-8'))
            utxos = []
            
            for utxo_id in utxo_ids:
                utxo_key = self._get_utxo_key(utxo_id)
                utxo_data = self.db.get(utxo_key)
                
                if utxo_data:
                    utxo = UTXO.deserialize(utxo_data)
                    if utxo.is_spendable(current_block_height, current_time):
                        utxos.append(utxo)
            
            return utxos
    
    def get_balance(self, address: str, current_block_height: int = 0, 
                   current_time: int = 0) -> int:
        """Get the balance for an address"""
        utxos = self.get_utxos_for_address(address, current_block_height, current_time)
        return sum(utxo.amount for utxo in utxos)
    
    def find_spendable_utxos(self, address: str, amount: int, 
                            current_block_height: int = 0, current_time: int = 0) -> Tuple[List[UTXO], int]:
        """Find UTXOs to spend for a given amount"""
        utxos = self.get_utxos_for_address(address, current_block_height, current_time)
        utxos.sort(key=lambda x: x.amount, reverse=True)
        
        total = 0
        selected = []
        
        for utxo in utxos:
            if total >= amount:
                break
            selected.append(utxo)
            total += utxo.amount
        
        return selected, total
    
    def get_utxo(self, utxo_id: str) -> Optional[UTXO]:
        """Get a UTXO by ID"""
        with self.lock:
            # First check unspent UTXOs
            utxo_key = self._get_utxo_key(utxo_id)
            utxo_data = self.db.get(utxo_key)
            
            if utxo_data:
                return UTXO.deserialize(utxo_data)
            
            # Then check spent UTXOs
            spent_key = self._get_spent_utxo_key(utxo_id)
            spent_data = self.db.get(spent_key)
            
            if spent_data:
                return UTXO.deserialize(spent_data)
            
            return None
    
    def validate_transaction(self, transaction: Transaction, current_block_height: int = 0) -> Tuple[bool, str]:
        """
        Validate a transaction before processing it.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        with self.lock:
            # Check all inputs exist and are spendable
            for i, tx_input in enumerate(transaction.inputs):
                utxo_id = f"{tx_input['tx_hash']}:{tx_input['output_index']}"
                utxo = self.get_utxo(utxo_id)
                
                if not utxo:
                    return False, f"Input UTXO {utxo_id} does not exist"
                
                if utxo.spent:
                    return False, f"Input UTXO {utxo_id} is already spent"
                
                if not utxo.is_spendable(current_block_height, 0):  # Using 0 for time as we're using block height
                    return False, f"Input UTXO {utxo_id} is not spendable yet"
                
                # Verify signature
                if not transaction.verify_input_signature(i, self):
                    return False, f"Invalid signature for input {i}"
            
            # Check output amounts are positive
            for output in transaction.outputs:
                if output['amount'] <= 0:
                    return False, "Output amount must be positive"
            
            # Check fee is reasonable (at least non-negative)
            fee = transaction.calculate_fee(self)
            if fee < 0:
                return False, "Transaction has negative fee"
            
            return True, ""
    
    def process_transaction(self, transaction: Transaction, block_height: int = 0):
        """
        Process a transaction by spending inputs and creating new UTXOs from outputs.
        
        Args:
            transaction: The transaction to process
            block_height: Current block height for locktime validation
        """
        with self.atomic_write() as batch:
            # Validate transaction first
            is_valid, error_msg = self.validate_transaction(transaction, block_height)
            if not is_valid:
                raise ValueError(f"Invalid transaction: {error_msg}")
            
            # Spend the inputs
            for tx_input in transaction.inputs:
                utxo_id = f"{tx_input['tx_hash']}:{tx_input['output_index']}"
                self.spend_utxo(utxo_id, batch)
            
            # Create new UTXOs from outputs
            for i, output in enumerate(transaction.outputs):
                utxo = UTXO(
                    tx_hash=transaction.hash,
                    output_index=i,
                    address=output['address'],
                    amount=output['amount']
                )
                
                # Set locktime if specified in output
                if 'locktime' in output:
                    utxo.locktime = output['locktime']
                
                # Record the block height when this UTXO was created
                utxo.created_at_block = block_height
                
                self.add_utxo(utxo, batch)
    
    def process_block_transactions(self, transactions: List[Transaction], block_height: int) -> bool:
        """
        Process all transactions in a block atomically.
        
        Returns:
            True if all transactions were processed successfully, False otherwise
        """
        with self.atomic_write() as batch:
            # First validate all transactions
            for tx in transactions:
                is_valid, error_msg = self.validate_transaction(tx, block_height)
                if not is_valid:
                    return False
            
            # Then process all transactions
            for tx in transactions:
                # Spend the inputs
                for tx_input in tx.inputs:
                    utxo_id = f"{tx_input['tx_hash']}:{tx_input['output_index']}"
                    self.spend_utxo(utxo_id, batch)
                
                # Create new UTXOs from outputs
                for i, output in enumerate(tx.outputs):
                    utxo = UTXO(
                        tx_hash=tx.hash,
                        output_index=i,
                        address=output['address'],
                        amount=output['amount']
                    )
                    
                    # Set locktime if specified in output
                    if 'locktime' in output:
                        utxo.locktime = output['locktime']
                    
                    # Record the block height when this UTXO was created
                    utxo.created_at_block = block_height
                    
                    self.add_utxo(utxo, batch)
            
            # Update last processed block height
            self.db.put(LAST_BLOCK_HEIGHT_KEY, struct.pack('>I', block_height))
            
            return True
    
    def get_last_processed_block_height(self) -> int:
        """Get the last block height that was processed"""
        data = self.db.get(LAST_BLOCK_HEIGHT_KEY)
        if data:
            return struct.unpack('>I', data)[0]
        return 0
    
    def rebuild_indexes(self):
        """Rebuild all indexes from UTXO data (for recovery/maintenance)"""
        with self.lock:
            with self.atomic_write() as batch:
                # Clear existing address indexes
                for key, _ in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                    batch.delete(key)
                
                # Rebuild address indexes from UTXOs
                for key, value in self.db.iterator(prefix=UTXO_PREFIX):
                    utxo = UTXO.deserialize(value)
                    utxo_id = key[len(UTXO_PREFIX):].decode('utf-8')
                    
                    # Update address index
                    address_key = self._get_address_index_key(utxo.address)
                    existing_data = self.db.get(address_key)
                    existing_utxos = set()
                    
                    if existing_data:
                        existing_utxos = set(json.loads(existing_data.decode('utf-8')))
                    
                    existing_utxos.add(utxo_id)
                    batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
    
    def compact_database(self):
        """Compact the database to reclaim space"""
        self.db.compact_range()
    
    def get_stats(self) -> Dict:
        """Get statistics about the UTXO set"""
        with self.lock:
            stats = {
                'total_utxos': 0,
                'total_spent_utxos': 0,
                'total_addresses': 0,
                'total_size_mb': 0
            }
            
            # Count UTXOs
            for _, _ in self.db.iterator(prefix=UTXO_PREFIX):
                stats['total_utxos'] += 1
            
            # Count spent UTXOs
            for _, _ in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                stats['total_spent_utxos'] += 1
            
            # Count addresses
            for _, _ in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                stats['total_addresses'] += 1
            
            # Estimate database size
            stats['total_size_mb'] = sum(1 for _ in self.db.iterator()) * 0.0001  # Rough estimate
            
            return stats
    
    def to_dict(self) -> Dict:
        """Convert UTXO set to dictionary (for debugging, not for production use)"""
        result = {
            'utxos': {},
            'spent_utxos': {},
            'address_utxos': {}
        }
        
        with self.lock:
            # Get all UTXOs
            for key, value in self.db.iterator(prefix=UTXO_PREFIX):
                utxo_id = key[len(UTXO_PREFIX):].decode('utf-8')
                utxo = UTXO.deserialize(value)
                result['utxos'][utxo_id] = utxo.to_dict()
            
            # Get all spent UTXOs
            for key, value in self.db.iterator(prefix=SPENT_UTXO_PREFIX):
                utxo_id = key[len(SPENT_UTXO_PREFIX):].decode('utf-8')
                utxo = UTXO.deserialize(value)
                result['spent_utxos'][utxo_id] = utxo.to_dict()
            
            # Get address indexes
            for key, value in self.db.iterator(prefix=ADDRESS_INDEX_PREFIX):
                address = key[len(ADDRESS_INDEX_PREFIX):].decode('utf-8')
                utxo_ids = json.loads(value.decode('utf-8'))
                result['address_utxos'][address] = utxo_ids
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict, db_path: str = './utxo_db') -> 'UTXOSet':
        """Create UTXO set from dictionary (for debugging, not for production use)"""
        utxo_set = cls(db_path)
        
        with utxo_set.atomic_write() as batch:
            # Add UTXOs
            for utxo_id, utxo_data in data.get('utxos', {}).items():
                utxo = UTXO.from_dict(utxo_data)
                batch.put(utxo_set._get_utxo_key(utxo_id), utxo.serialize())
                
                # Update address index
                address_key = utxo_set._get_address_index_key(utxo.address)
                existing_data = batch.get(address_key)
                existing_utxos = set()
                
                if existing_data:
                    existing_utxos = set(json.loads(existing_data.decode('utf-8')))
                
                existing_utxos.add(utxo_id)
                batch.put(address_key, json.dumps(list(existing_utxos)).encode('utf-8'))
            
            # Add spent UTXOs
            for utxo_id, utxo_data in data.get('spent_utxos', {}).items():
                utxo = UTXO.from_dict(utxo_data)
                batch.put(utxo_set._get_spent_utxo_key(utxo_id), utxo.serialize())
        
        return utxo_set