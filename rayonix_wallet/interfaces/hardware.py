from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from rayonix_wallet.core.exceptions import WalletError

class HardwareWalletInterface(ABC):
    """Abstract base class for hardware wallet interfaces"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Connect to hardware wallet"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from hardware wallet"""
        pass
    
    @abstractmethod
    def get_public_key(self, derivation_path: str) -> str:
        """Get public key from derivation path"""
        pass
    
    @abstractmethod
    def sign_transaction(self, transaction: Dict, derivation_paths: List[str]) -> Dict:
        """Sign transaction with hardware wallet"""
        pass
    
    @abstractmethod
    def sign_message(self, message: str, derivation_path: str) -> str:
        """Sign message with hardware wallet"""
        pass
    
    @abstractmethod
    def verify_message(self, message: str, signature: str, public_key: str) -> bool:
        """Verify message signature"""
        pass
    
    @abstractmethod
    def get_device_info(self) -> Dict:
        """Get hardware wallet device information"""
        pass
    
    @abstractmethod
    def setup_device(self, passphrase: Optional[str] = None) -> bool:
        """Setup new hardware wallet device"""
        pass

class LedgerInterface(HardwareWalletInterface):
    """Ledger hardware wallet interface"""
    
    def __init__(self):
        self.connected = False
        self.device = None
    
    def connect(self) -> bool:
        """Connect to Ledger device"""
        try:
            from ledgerblue.comm import getDongle
            self.device = getDongle()
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ledger: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Ledger device"""
        try:
            if self.device:
                self.device.close()
            self.connected = False
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Ledger: {e}")
            return False
    
    def get_public_key(self, derivation_path: str) -> str:
        """Get public key from derivation path"""
        if not self.connected:
            raise WalletError("Not connected to Ledger")
        
        try:
            # Convert derivation path to bytes
            path_components = self._parse_derivation_path(derivation_path)
            apdu = self._create_apdu_for_public_key(path_components)
            
            # Send APDU to device
            response = self.device.exchange(apdu)
            public_key = response[1:1 + response[0]]
            return public_key.hex()
        except Exception as e:
            raise WalletError(f"Failed to get public key: {e}")
    
    def sign_transaction(self, transaction: Dict, derivation_paths: List[str]) -> Dict:
        """Sign transaction with Ledger"""
        if not self.connected:
            raise WalletError("Not connected to Ledger")
        
        try:
            # This would involve sending the transaction data to the device
            # and receiving signatures for each input
            
            # Placeholder implementation
            signatures = {}
            for path in derivation_paths:
                signatures[path] = self._sign_transaction_data(transaction, path)
            
            return signatures
        except Exception as e:
            raise WalletError(f"Failed to sign transaction: {e}")
    
    def _sign_transaction_data(self, transaction: Dict, derivation_path: str) -> str:
        """Sign transaction data at specific derivation path"""
        # Actual implementation would use specific APDU commands
        # for the cryptocurrency being used
        
        # Placeholder
        return "mock_signature"
    
    def sign_message(self, message: str, derivation_path: str) -> str:
        """Sign message with Ledger"""
        if not self.connected:
            raise WalletError("Not connected to Ledger")
        
        try:
            # Convert message to bytes and derivation path to components
            message_bytes = message.encode()
            path_components = self._parse_derivation_path(derivation_path)
            
            # Create APDU for message signing
            apdu = self._create_apdu_for_message_signing(path_components, message_bytes)
            
            # Send to device and get response
            response = self.device.exchange(apdu)
            signature = response[1:1 + response[0]]
            return signature.hex()
        except Exception as e:
            raise WalletError(f"Failed to sign message: {e}")
    
    def verify_message(self, message: str, signature: str, public_key: str) -> bool:
        """Verify message signature"""
        # This would typically be done on the device
        # For now, use software verification
        
        from ..crypto.signing import TransactionSigner
        signer = TransactionSigner()
        
        message_hash = hashlib.sha256(message.encode()).digest()
        return signer.verify_signature(message_hash, bytes.fromhex(signature), bytes.fromhex(public_key))
    
    def get_device_info(self) -> Dict:
        """Get Ledger device information"""
        if not self.connected:
            raise WalletError("Not connected to Ledger")
        
        try:
            # Get device information via APDU
            apdu = bytes.fromhex("E0C4000000")  # GET_DEVICE_INFO
            response = self.device.exchange(apdu)
            
            # Parse response
            return {
                'model': response[0],
                'firmware_version': f"{response[1]}.{response[2]}.{response[3]}",
                'serial_number': response[4:20].hex(),
                'has_pin': bool(response[20]),
                'has_passphrase': bool(response[21])
            }
        except Exception as e:
            raise WalletError(f"Failed to get device info: {e}")
    
    def setup_device(self, passphrase: Optional[str] = None) -> bool:
        """Setup new Ledger device"""
        # This would involve going through the device setup process
        # which varies by model and firmware
        
        raise NotImplementedError("Device setup not implemented")
    
    def _parse_derivation_path(self, path: str) -> List[int]:
        """Parse BIP32 derivation path to components"""
        if not path.startswith('m/'):
            raise WalletError("Invalid derivation path")
        
        components = path.split('/')[1:]
        result = []
        
        for comp in components:
            if comp.endswith("'"):
                result.append(0x80000000 | int(comp[:-1]))
            else:
                result.append(int(comp))
        
        return result
    
    def _create_apdu_for_public_key(self, path_components: List[int]) -> bytes:
        """Create APDU for getting public key"""
        # This is cryptocurrency-specific
        # Placeholder implementation
        
        apdu = bytearray()
        apdu.append(0xE0)  # CLA
        apdu.append(0x02)  # INS (GET_PUBLIC_KEY)
        apdu.append(0x00)  # P1
        apdu.append(0x00)  # P2
        apdu.append(len(path_components) * 4)  # Data length
        
        for comp in path_components:
            apdu.extend(comp.to_bytes(4, 'big'))
        
        return bytes(apdu)

class TrezorInterface(HardwareWalletInterface):
    """Trezor hardware wallet interface"""
    
    def __init__(self):
        self.connected = False
        self.client = None
    
    def connect(self) -> bool:
        """Connect to Trezor device"""
        try:
            import trezorlib
            from trezorlib.transport import get_transport
            from trezorlib.client import TrezorClient
            
            transport = get_transport()
            self.client = TrezorClient(transport)
            self.connected = True
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Trezor: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Trezor device"""
        try:
            if self.client:
                self.client.close()
            self.connected = False
            return True
        except Exception as e:
            logger.error(f"Failed to disconnect from Trezor: {e}")
            return False
    
    # Other methods would be implemented similarly to LedgerInterface
    # but using trezorlib specific APIs