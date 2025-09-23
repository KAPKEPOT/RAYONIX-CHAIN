import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

from rayonix_wallet.core.exceptions import CryptoError
from rayonix_wallet.utils.secure import SecureString

class EncryptionManager:
    """Encryption utilities for wallet data"""
    
    def __init__(self, config):
        self.config = config
    
    def encrypt_data(self, data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Encrypt data using AES-GCM"""
        try:
            iv = os.urandom(12)
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            
            encryptor = cipher.encryptor()
            if associated_data:
                encryptor.authenticate_additional_data(associated_data)
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            return iv + encryptor.tag + ciphertext
            
        except Exception as e:
            raise CryptoError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-GCM"""
        try:
            iv = encrypted_data[:12]
            tag = encrypted_data[12:28]
            ciphertext = encrypted_data[28:]
            
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, tag),
                backend=default_backend()
            )
            
            decryptor = cipher.decryptor()
            if associated_data:
                decryptor.authenticate_additional_data(associated_data)
            
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        except InvalidTag:
            raise CryptoError("Decryption failed: Invalid authentication tag")
        except Exception as e:
            raise CryptoError(f"Decryption failed: {e}")
    
    def encrypt_with_passphrase(self, data: bytes, passphrase: str, salt: Optional[bytes] = None) -> dict:
        """Encrypt data with passphrase using PBKDF2 and AES-GCM"""
        salt = salt or os.urandom(16)
        key = self._derive_key_from_passphrase(passphrase, salt)
        
        encrypted_data = self.encrypt_data(data, key)
        return {
            'data': encrypted_data,
            'salt': salt,
            'algorithm': 'AES-GCM',
            'kdf': 'PBKDF2-HMAC-SHA256',
            'iterations': 100000
        }
    
    def decrypt_with_passphrase(self, encrypted_package: dict, passphrase: str) -> bytes:
        """Decrypt data encrypted with passphrase"""
        try:
            key = self._derive_key_from_passphrase(
                passphrase, 
                encrypted_package['salt'],
                encrypted_package.get('iterations', 300000)
            )
            
            return self.decrypt_data(encrypted_package['data'], key)
            
        except Exception as e:
            raise CryptoError(f"Passphrase decryption failed: {e}")
    
    def _derive_key_from_passphrase(self, passphrase: str, salt: bytes, iterations: int = 300000) -> bytes:
        """Derive encryption key from passphrase using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
            backend=default_backend()
        )
        return kdf.derive(passphrase.encode())
    
    def encrypt_wallet_data(self, wallet_data: dict, encryption_key: SecureString) -> bytes:
        """Encrypt complete wallet data"""
        import json
        try:
            serialized_data = json.dumps(wallet_data).encode('utf-8')
            return self.encrypt_data(serialized_data, encryption_key.get_value())
        except Exception as e:
            raise CryptoError(f"Wallet encryption failed: {e}")
    
    def decrypt_wallet_data(self, encrypted_data: bytes, encryption_key: SecureString) -> dict:
        """Decrypt complete wallet data"""
        import json
        try:
            decrypted_data = self.decrypt_data(encrypted_data, encryption_key.get_value())
            return json.loads(decrypted_data.decode('utf-8'))
        except Exception as e:
            raise CryptoError(f"Wallet decryption failed: {e}")