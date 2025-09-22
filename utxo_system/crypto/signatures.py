# utxo_system/crypto/signatures.py
import json
import hashlib
from typing import Optional
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from utxo_system.models.transaction import Transaction
from utxo_system.models.utxo import UTXO
from utxo_system.utils.logging_config import logger

def sign_transaction_input(transaction: Transaction, input_index: int, private_key, 
                          utxo: UTXO, sighash_type: int = 1) -> None:
    if input_index >= len(transaction.inputs):
        raise ValueError("Invalid input index")
    
    # Create signing data
    signing_data = _get_signing_data(transaction, input_index, utxo, sighash_type)
    
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
    
    transaction.inputs[input_index].signature = signature.hex()
    transaction.inputs[input_index].public_key = public_key

def _get_signing_data(transaction: Transaction, input_index: int, utxo: UTXO, sighash_type: int) -> str:
    # Create copy without signatures for this input
    inputs_copy = []
    for i, inp in enumerate(transaction.inputs):
        if i == input_index:
            inp_copy = {
                'tx_hash': inp.tx_hash,
                'output_index': inp.output_index,
                'address': inp.address
            }
        else:
            inp_copy = {
                'tx_hash': inp.tx_hash,
                'output_index': inp.output_index,
                'address': inp.address,
                'public_key': inp.public_key
            }
        inputs_copy.append(inp_copy)
    
    # Include referenced UTXO in signing data
    signing_data = json.dumps({
        'version': transaction.version,
        'inputs': inputs_copy,
        'outputs': [out.to_dict() for out in transaction.outputs],
        'locktime': transaction.locktime,
        'referenced_utxo': utxo.to_dict(),
        'sighash_type': sighash_type
    }, sort_keys=True)
    
    return signing_data

def verify_transaction_signature(transaction: Transaction, input_index: int, utxo_set) -> bool:
    if input_index >= len(transaction.inputs) or not transaction.inputs[input_index].signature:
        return False
    
    try:
        signature = bytes.fromhex(transaction.inputs[input_index].signature)
        public_key_bytes = bytes.fromhex(transaction.inputs[input_index].public_key)
        
        public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256K1(), public_key_bytes
        )
        
        # Get the referenced UTXO
        tx_input = transaction.inputs[input_index]
        utxo_id = f"{tx_input.tx_hash}:{tx_input.output_index}"
        utxo = utxo_set.get_utxo(utxo_id)
        
        if not utxo:
            return False
        
        # Reconstruct signing data
        signing_data = _get_signing_data(transaction, input_index, utxo, 1)
        
        public_key.verify(
            signature,
            signing_data.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except (InvalidSignature, ValueError, Exception) as e:
        logger.warning(f"Signature verification failed: {e}")
        return False