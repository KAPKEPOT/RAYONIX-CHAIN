import time
from typing import Dict, List, Optional
from rayonix_wallet.core.types import Transaction
from rayonix_wallet.core.exceptions import TransactionError, InsufficientFundsError
from rayonix_wallet.crypto.signing import TransactionSigner

class TransactionManager:
    """Transaction creation and management"""
    
    def __init__(self, wallet):
        self.wallet = wallet
        self.signer = TransactionSigner(wallet.config)
    
    def send_transaction(self, to_address: str, amount: int, 
                        fee_rate: Optional[int] = None, memo: Optional[str] = None) -> Optional[str]:
        """Create and send transaction"""
        try:
            # Validate recipient address
            if not self.wallet.validate_address(to_address):
                raise TransactionError("Invalid recipient address")
            
            # Check if wallet has sufficient funds
            balance = self.wallet.get_balance()
            if balance.available < amount:
                raise InsufficientFundsError("Insufficient funds")
            
            # Get fee rate if not specified
            if fee_rate is None:
                fee_rate = self.get_fee_estimate()
            
            # Select UTXOs for transaction
            utxos = self._select_utxos(amount, fee_rate)
            if not utxos:
                raise InsufficientFundsError("Insufficient funds after fees")
            
            # Create raw transaction
            raw_tx = self._create_raw_transaction(utxos, to_address, amount, fee_rate, memo)
            
            # Sign transaction
            signed_tx = self._sign_transaction(raw_tx, utxos)
            
            # Broadcast transaction
            txid = self.wallet.blockchain_interface.broadcast_transaction(signed_tx)
            if not txid:
                raise TransactionError("Failed to broadcast transaction")
            
            # Save transaction to database
            transaction = self._create_transaction_record(txid, utxos, to_address, amount, fee_rate, memo)
            self.wallet.db.save_transaction(transaction)
            self.wallet.transactions[txid] = transaction
            
            # Mark UTXOs as spent
            for utxo in utxos:
                self.wallet.db.mark_utxo_spent(utxo['txid'], utxo['vout'], txid)
            
            logger.info(f"Transaction sent: {txid}")
            return txid
            
        except Exception as e:
            logger.error(f"Transaction failed: {e}")
            raise
    
    def _select_utxos(self, amount: int, fee_rate: int) -> List[Dict]:
        """Select UTXOs for transaction using coin selection algorithm"""
        utxos = self.wallet.db.get_utxos(unspent_only=True)
        utxos.sort(key=lambda x: x['amount'], reverse=True)
        
        selected_utxos = []
        total_amount = 0
        estimated_fee = 0
        
        for utxo in utxos:
            selected_utxos.append(utxo)
            total_amount += utxo['amount']
            
            # Estimate fee with current selection
            estimated_fee = self._estimate_fee(len(selected_utxos), 2, fee_rate)
            
            if total_amount >= amount + estimated_fee:
                break
        
        if total_amount < amount + estimated_fee:
            return []
        
        return selected_utxos
    
    def _estimate_fee(self, input_count: int, output_count: int, fee_rate: int) -> int:
        """Estimate transaction fee"""
        # Base transaction size
        base_size = 10
        # Input size (approx 150 bytes per input)
        input_size = input_count * 150
        # Output size (approx 34 bytes per output)
        output_size = output_count * 34
        
        total_size = base_size + input_size + output_size
        return total_size * fee_rate
    
    def _create_raw_transaction(self, utxos: List[Dict], to_address: str, 
                              amount: int, fee_rate: int, memo: Optional[str] = None) -> Dict:
        """Create raw transaction structure"""
        total_input = sum(utxo['amount'] for utxo in utxos)
        fee = self._estimate_fee(len(utxos), 2, fee_rate)
        change_amount = total_input - amount - fee
        
        # Build transaction
        transaction = {
            'version': 1,
            'locktime': 0,
            'vin': [],
            'vout': [
                {
                    'value': amount,
                    'script_pubkey': self._address_to_script(to_address),
                    'address': to_address
                }
            ]
        }
        
        # Add change output if needed
        if change_amount > 0:
            change_address = self.wallet.derive_address(0, True)  # Use change address
            transaction['vout'].append({
                'value': change_amount,
                'script_pubkey': self._address_to_script(change_address),
                'address': change_address
            })
        
        # Add inputs
        for utxo in utxos:
            transaction['vin'].append({
                'txid': utxo['txid'],
                'vout': utxo['vout'],
                'script_sig': '',
                'sequence': 0xffffffff
            })
        
        return transaction
    
    def _address_to_script(self, address: str) -> str:
        """Convert address to scriptPubKey"""
        # This is a simplified version - actual implementation depends on address type
        if address.startswith('1'):
            return f"76a914{hashlib.new('ripemd160', hashlib.sha256(bytes.fromhex(address)).digest()).hex()}88ac"
        elif address.startswith('3'):
            return f"a914{hashlib.new('ripemd160', hashlib.sha256(bytes.fromhex(address)).digest()).hex()}87"
        else:
            return f"0014{hashlib.new('ripemd160', hashlib.sha256(bytes.fromhex(address)).digest()).hex()}"
    
    def _sign_transaction(self, transaction: Dict, utxos: List[Dict]) -> str:
        """Sign transaction inputs"""
        signed_inputs = []
        
        for i, tx_in in enumerate(transaction['vin']):
            utxo = utxos[i]
            script_pubkey = utxo['script_pubkey']
            
            # Create signature hash
            sighash = self.signer.create_signature_hash(transaction, i, script_pubkey)
            
            # Get private key for this UTXO
            private_key = self.wallet.key_manager.export_private_key(
                utxo['address'], 
                self.wallet.addresses[utxo['address']].derivation_path
            )
            
            # Sign the hash
            signature = self.signer.sign_data(sighash, bytes.fromhex(private_key))
            
            # Create scriptSig
            script_sig = self._create_script_sig(signature, private_key)
            tx_in['script_sig'] = script_sig.hex()
            signed_inputs.append(tx_in)
        
        transaction['vin'] = signed_inputs
        return self._serialize_transaction(transaction)
    
    def _create_script_sig(self, signature: bytes, public_key: bytes) -> bytes:
        """Create scriptSig for transaction input"""
        # Push signature
        script_sig = bytes([len(signature)]) + signature
        # Push public key
        script_sig += bytes([len(public_key)]) + public_key
        return script_sig
    
    def _serialize_transaction(self, transaction: Dict) -> str:
        """Serialize transaction to hex"""
        # Simplified serialization - actual implementation would be more complex
        import struct
        tx_data = b''
        
        # Version
        tx_data += struct.pack('<I', transaction['version'])
        
        # Input count
        tx_data += struct.pack('<B', len(transaction['vin']))
        
        # Inputs
        for tx_in in transaction['vin']:
            tx_data += bytes.fromhex(tx_in['txid'])[::-1]  # Little endian
            tx_data += struct.pack('<I', tx_in['vout'])
            tx_data += struct.pack('<B', len(tx_in['script_sig']))
            tx_data += bytes.fromhex(tx_in['script_sig'])
            tx_data += struct.pack('<I', tx_in['sequence'])
        
        # Output count
        tx_data += struct.pack('<B', len(transaction['vout']))
        
        # Outputs
        for tx_out in transaction['vout']:
            tx_data += struct.pack('<Q', tx_out['value'])
            script_pubkey = bytes.fromhex(tx_out['script_pubkey'])
            tx_data += struct.pack('<B', len(script_pubkey))
            tx_data += script_pubkey
        
        # Locktime
        tx_data += struct.pack('<I', transaction['locktime'])
        
        return tx_data.hex()
    
    def _create_transaction_record(self, txid: str, utxos: List[Dict], 
                                 to_address: str, amount: int, fee: int, memo: Optional[str]) -> Transaction:
        """Create transaction record for database"""
        from_addresses = list(set(utxo['address'] for utxo in utxos))
        from_address = from_addresses[0] if len(from_addresses) == 1 else 'multiple'
        
        return Transaction(
            txid=txid,
            amount=amount,
            fee=fee,
            confirmations=0,
            timestamp=int(time.time()),
            block_height=None,
            from_address=from_address,
            to_address=to_address,
            status='pending',
            direction='out',
            memo=memo,
            exchange_rate=None
        )
    
    def get_fee_estimate(self, priority: str = "medium") -> int:
        """Get transaction fee estimate"""
        try:
            if hasattr(self.wallet, 'blockchain_interface'):
                return self.wallet.blockchain_interface.get_fee_estimate(priority)
            else:
                return self.wallet.config.transaction_fees.get(priority, 2)
        except:
            return self.wallet.config.transaction_fees.get(priority, 2)
    
    def sweep_private_key(self, private_key: str, to_address: str, 
                         fee_rate: Optional[int] = None) -> Optional[str]:
        """Sweep funds from private key to wallet address"""
        try:
            # Validate destination address
            if not self.wallet.validate_address(to_address):
                raise TransactionError("Invalid destination address")
            
            # Get balance from private key
            # This would require blockchain interface to get UTXOs for the private key
            # Implementation depends on specific blockchain API
            
            raise NotImplementedError("Private key sweeping not implemented")
            
        except Exception as e:
            logger.error(f"Private key sweep failed: {e}")
            raise
    
    def replace_by_fee(self, txid: str, new_fee_rate: int) -> Optional[str]:
        """Replace transaction with higher fee (RBF)"""
        try:
            if txid not in self.wallet.transactions:
                raise TransactionError("Transaction not found")
            
            transaction = self.wallet.transactions[txid]
            if transaction.confirmations > 0:
                raise TransactionError("Transaction already confirmed")
            
            # Implementation would vary by blockchain
            # Generally involves creating a new transaction with higher fee
            # and signaling RBF in the transaction
            
            raise NotImplementedError("Replace-by-fee not implemented")
            
        except Exception as e:
            logger.error(f"RBF failed: {e}")
            raise