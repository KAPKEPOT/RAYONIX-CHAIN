import hashlib
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from rayonix_wallet.core.exceptions import CryptoError

class TransactionSigner:
    """Transaction signing utilities"""
    
    def __init__(self, config):
        self.config = config
    
    def sign_data(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data using ECDSA with proper canonical signatures"""
        try:
            private_key_obj = ec.derive_private_key(
                int.from_bytes(private_key, 'big'),
                ec.SECP256K1(),
                default_backend()
            )
            
            signature = private_key_obj.sign(
                data,
                ec.ECDSA(hashes.SHA256())
            )
            
            return self._ensure_canonical_signature(signature)
        except Exception as e:
            raise CryptoError(f"Signing failed: {e}")
    
    def _ensure_canonical_signature(self, signature: bytes) -> bytes:
        """Ensure signature is canonical (low S value)"""
        r_len = signature[3]
        r = signature[4:4+r_len]
        s_start = 4 + r_len + 2
        s_len = signature[s_start-1]
        s = signature[s_start:s_start+s_len]
        
        s_int = int.from_bytes(s, 'big')
        curve_order = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        
        if s_int > curve_order // 2:
            s_int = curve_order - s_int
            s = s_int.to_bytes((s_int.bit_length() + 7) // 8, 'big')
            
            der_sig = b'\x30' + len(signature).to_bytes(1, 'big')
            der_sig += b'\x02' + r_len.to_bytes(1, 'big') + r
            der_sig += b'\x02' + s_len.to_bytes(1, 'big') + s
            
            return der_sig
        
        return signature
    
    def create_signature_hash(self, transaction: dict, input_index: int, script_pubkey: str) -> bytes:
        """Create signature hash following Rayonix protocol"""
        serialized = self._serialize_for_signing(transaction, input_index, script_pubkey)
        return hashlib.sha256(hashlib.sha256(serialized).digest()).digest()
    
    def _serialize_for_signing(self, transaction: dict, input_index: int, script_pubkey: str) -> bytes:
        """Serialize transaction for signing following Rayonix protocol"""
        version = transaction['version'].to_bytes(4, 'little')
        input_count = len(transaction['vin']).to_bytes(1, 'little')
        
        inputs_data = b''
        for i, tx_in in enumerate(transaction['vin']):
            txid = bytes.fromhex(tx_in['txid'])[::-1]
            vout = tx_in['vout'].to_bytes(4, 'little')
            
            if i == input_index:
                script = bytes.fromhex(script_pubkey)
                script_len = len(script).to_bytes(1, 'little')
            else:
                script = b''
                script_len = b'\x00'
            
            sequence = tx_in.get('sequence', 0xffffffff).to_bytes(4, 'little')
            inputs_data += txid + vout + script_len + script + sequence
        
        output_count = len(transaction['vout']).to_bytes(1, 'little')
        outputs_data = b''
        for tx_out in transaction['vout']:
            value = tx_out['value'].to_bytes(8, 'little')
            script_pubkey = bytes.fromhex(tx_out.get('script_pubkey', ''))
            script_len = len(script_pubkey).to_bytes(1, 'little')
            outputs_data += value + script_len + script_pubkey
        
        locktime = transaction.get('locktime', 0).to_bytes(4, 'little')
        sighash_type = b'\x01\x00\x00\x80'
        
        return version + input_count + inputs_data + output_count + outputs_data + locktime + sighash_type
    
    def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify ECDSA signature"""
        try:
            public_key_obj = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(),
                public_key
            )
            
            public_key_obj.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except InvalidSignature:
            return False
        except Exception:
            return False