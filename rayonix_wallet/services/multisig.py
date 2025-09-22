import json
from typing import Dict, List, Optional
from ..core.exceptions import MultisigError
from ..crypto.signing import TransactionSigner

class MultisigManager:
    """Multi-signature wallet support"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.signer = TransactionSigner(wallet.config)
        self.multisig_config = None
    
    def set_multisig(self, required: int, public_keys: List[str]) -> bool:
        """Setup multi-signature wallet"""
        try:
            if len(public_keys) < required:
                raise MultisigError("Required signatures cannot exceed number of public keys")
            
            if self.wallet.wallet_id in public_keys:
                raise MultisigError("Cannot include own public key in multisig setup")
            
            self.multisig_config = {
                'required': required,
                'public_keys': public_keys,
                'total_keys': len(public_keys),
                'address': self._derive_multisig_address(public_keys, required)
            }
            
            # Update wallet type
            self.wallet.config.wallet_type = 'multisig'
            
            logger.info(f"Multisig wallet configured: {required} of {len(public_keys)} signatures required")
            return True
            
        except Exception as e:
            logger.error(f"Multisig setup failed: {e}")
            return False
    
    def _derive_multisig_address(self, public_keys: List[str], required: int) -> str:
        """Derive multisignature address"""
        # Sort public keys for deterministic address generation
        sorted_keys = sorted(public_keys)
        
        # Create multisig script
        script = f"{required}"
        for pub_key in sorted_keys:
            script += f" {pub_key}"
        script += f" {len(sorted_keys)} OP_CHECKMULTISIG"
        
        # Hash the script and create address
        script_hash = hashlib.sha256(script.encode()).digest()
        ripemd_hash = hashlib.new('ripemd160', script_hash).digest()
        
        network_byte = b'\x3C' if self.wallet.config.network == "mainnet" else b'\x6F'
        payload = network_byte + ripemd_hash
        
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        address_bytes = payload + checksum
        
        return base58.b58encode(address_bytes).decode('ascii')
    
    def add_cosigner(self, public_key: str) -> bool:
        """Add cosigner to multisig wallet"""
        if not self.multisig_config:
            raise MultisigError("Multisig not configured")
        
        if public_key in self.multisig_config['public_keys']:
            raise MultisigError("Public key already added")
        
        self.multisig_config['public_keys'].append(public_key)
        self.multisig_config['total_keys'] = len(self.multisig_config['public_keys'])
        
        # Re-derive address with new key set
        self.multisig_config['address'] = self._derive_multisig_address(
            self.multisig_config['public_keys'],
            self.multisig_config['required']
        )
        
        return True
    
    def create_multisig_transaction(self, to_address: str, amount: int, 
                                  fee_rate: Optional[int] = None) -> Dict:
        """Create multisig transaction requiring multiple signatures"""
        if not self.multisig_config:
            raise MultisigError("Multisig not configured")
        
        try:
            # This would be similar to regular transaction creation
            # but would create a partially signed transaction
            
            transaction_data = {
                'version': 1,
                'to_address': to_address,
                'amount': amount,
                'fee_rate': fee_rate or self.wallet.transaction_manager.get_fee_estimate(),
                'multisig_config': self.multisig_config,
                'signatures': [],
                'status': 'draft'
            }
            
            return transaction_data
            
        except Exception as e:
            raise MultisigError(f"Multisig transaction creation failed: {e}")
    
    def sign_multisig_transaction(self, transaction: Dict) -> Dict:
        """Sign multisig transaction"""
        if not self.multisig_config:
            raise MultisigError("Multisig not configured")
        
        try:
            # This would involve creating a signature for the transaction
            # using the wallet's private key
            
            # For demonstration purposes
            signature = {
                'public_key': self.wallet.key_manager.get_public_key().hex(),
                'signature': 'mock_signature_hex',
                'timestamp': int(time.time())
            }
            
            transaction['signatures'].append(signature)
            
            # Check if we have enough signatures
            if len(transaction['signatures']) >= self.multisig_config['required']:
                transaction['status'] = 'ready_to_broadcast'
            
            return transaction
            
        except Exception as e:
            raise MultisigError(f"Multisig signing failed: {e}")
    
    def finalize_multisig_transaction(self, transaction: Dict) -> Optional[str]:
        """Finalize multisig transaction with required signatures"""
        if not self.multisig_config:
            raise MultisigError("Multisig not configured")
        
        if len(transaction['signatures']) < self.multisig_config['required']:
            raise MultisigError("Insufficient signatures")
        
        try:
            # Combine signatures and create final transaction
            # This would vary depending on the cryptocurrency protocol
            
            # For demonstration purposes
            final_tx = json.dumps(transaction).encode()
            txid = hashlib.sha256(final_tx).hexdigest()
            
            # Broadcast transaction
            if hasattr(self.wallet, 'blockchain_interface'):
                success = self.wallet.blockchain_interface.broadcast_transaction(final_tx.hex())
                if success:
                    return txid
            
            raise MultisigError("Failed to broadcast multisig transaction")
            
        except Exception as e:
            raise MultisigError(f"Multisig finalization failed: {e}")
    
    def get_multisig_address(self) -> Optional[str]:
        """Get multisignature address"""
        if self.multisig_config:
            return self.multisig_config['address']
        return None
    
    def get_multisig_config(self) -> Optional[Dict]:
        """Get multisig configuration"""
        return self.multisig_config
    
    def validate_multisig_setup(self) -> bool:
        """Validate multisig configuration"""
        if not self.multisig_config:
            return False
        
        try:
            # Verify the address can be re-derived with current configuration
            expected_address = self._derive_multisig_address(
                self.multisig_config['public_keys'],
                self.multisig_config['required']
            )
            
            return expected_address == self.multisig_config['address']
            
        except Exception:
            return False