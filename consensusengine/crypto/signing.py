# consensus/crypto/signing.py
import hashlib
import os
from typing import Optional, Union
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import logging

logger = logging.getLogger('CryptoManager')

class CryptoManager:
    """Production-ready cryptographic operations for consensus"""
    
    def __init__(self, key_type: str = "ecdsa", curve: str = "secp256k1"):
        """
        Initialize cryptographic manager
        
        Args:
            key_type: Type of cryptographic keys ("ecdsa" or "rsa")
            curve: Elliptic curve name for ECDSA
        """
        self.key_type = key_type
        self.curve = curve
        self.private_key = None
        self.public_key = None
        self.backend = default_backend()
        
        # Load or generate keys
        self._initialize_keys()
    
    def _initialize_keys(self):
        """Initialize or load cryptographic keys"""
        try:
            # Try to load existing keys
            if self._load_keys_from_file():
                logger.info("Loaded existing cryptographic keys")
                return
            
            # Generate new keys
            self._generate_keys()
            self._save_keys_to_file()
            logger.info("Generated new cryptographic keys")
            
        except Exception as e:
            logger.error(f"Error initializing cryptographic keys: {e}")
            raise
    
    def _generate_keys(self):
        """Generate new cryptographic key pair"""
        if self.key_type == "ecdsa":
            # Generate ECDSA key pair
            if self.curve == "secp256k1":
                curve = ec.SECP256K1()
            elif self.curve == "p256":
                curve = ec.SECP256R1()
            else:
                raise ValueError(f"Unsupported curve: {self.curve}")
            
            self.private_key = ec.generate_private_key(curve, self.backend)
            self.public_key = self.private_key.public_key()
            
        elif self.key_type == "rsa":
            # Generate RSA key pair
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=self.backend
            )
            self.public_key = self.private_key.public_key()
        else:
            raise ValueError(f"Unsupported key type: {self.key_type}")
    
    def _load_keys_from_file(self) -> bool:
        """Load keys from file storage"""
        try:
            key_dir = os.path.expanduser("~/.consensus/keys")
            os.makedirs(key_dir, exist_ok=True)
            
            private_key_path = os.path.join(key_dir, "private_key.pem")
            public_key_path = os.path.join(key_dir, "public_key.pem")
            
            if not os.path.exists(private_key_path) or not os.path.exists(public_key_path):
                return False
            
            # Load private key
            with open(private_key_path, "rb") as f:
                private_key_data = f.read()
                self.private_key = serialization.load_pem_private_key(
                    private_key_data,
                    password=None,
                    backend=self.backend
                )
            
            # Load public key
            with open(public_key_path, "rb") as f:
                public_key_data = f.read()
                self.public_key = serialization.load_pem_public_key(
                    public_key_data,
                    backend=self.backend
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading keys from file: {e}")
            return False
    
    def _save_keys_to_file(self):
        """Save keys to file storage"""
        try:
            key_dir = os.path.expanduser("~/.consensus/keys")
            os.makedirs(key_dir, exist_ok=True)
            
            private_key_path = os.path.join(key_dir, "private_key.pem")
            public_key_path = os.path.join(key_dir, "public_key.pem")
            
            # Save private key
            private_key_pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            with open(private_key_path, "wb") as f:
                f.write(private_key_pem)
            
            # Save public key
            public_key_pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            with open(public_key_path, "wb") as f:
                f.write(public_key_pem)
            
            # Set secure permissions
            os.chmod(private_key_path, 0o600)
            os.chmod(public_key_path, 0o644)
            
        except Exception as e:
            logger.error(f"Error saving keys to file: {e}")
            raise
    
    def sign_data(self, data: bytes) -> str:
        """
        Sign data using private key
        
        Args:
            data: Data to sign
            
        Returns:
            Hex-encoded signature
        """
        try:
            if not self.private_key:
                raise ValueError("Private key not available")
            
            if self.key_type == "ecdsa":
                signature = self.private_key.sign(data, ec.ECDSA(hashes.SHA256()))
            elif self.key_type == "rsa":
                signature = self.private_key.sign(data, hashes.SHA256())
            else:
                raise ValueError(f"Unsupported key type: {self.key_type}")
            
            return signature.hex()
            
        except Exception as e:
            logger.error(f"Error signing data: {e}")
            raise
    
    def verify_signature(self, public_key: Union[str, bytes], data: bytes, signature: Union[str, bytes]) -> bool:
        """
        Verify signature using public key
        
        Args:
            public_key: Public key as hex string or bytes
            data: Original data that was signed
            signature: Signature to verify as hex string or bytes
            
        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Convert hex strings to bytes if necessary
            if isinstance(public_key, str):
                public_key_bytes = bytes.fromhex(public_key)
            else:
                public_key_bytes = public_key
            
            if isinstance(signature, str):
                signature_bytes = bytes.fromhex(signature)
            else:
                signature_bytes = signature
            
            # Load public key
            verifying_key = serialization.load_der_public_key(public_key_bytes, backend=self.backend)
            
            # Verify signature based on key type
            if isinstance(verifying_key, ec.EllipticCurvePublicKey):
                verifying_key.verify(signature_bytes, data, ec.ECDSA(hashes.SHA256()))
            elif isinstance(verifying_key, rsa.RSAPublicKey):
                verifying_key.verify(signature_bytes, data, hashes.SHA256())
            else:
                logger.error(f"Unsupported public key type: {type(verifying_key)}")
                return False
            
            return True
            
        except InvalidSignature:
            return False
        except Exception as e:
            logger.error(f"Error verifying signature: {e}")
            return False
    
    def get_public_key_der(self) -> bytes:
        """
        Get public key in DER format
        
        Returns:
            Public key as DER-encoded bytes
        """
        if not self.public_key:
            raise ValueError("Public key not available")
        
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    def get_public_key_hex(self) -> str:
        """
        Get public key as hex string
        
        Returns:
            Hex-encoded public key
        """
        return self.get_public_key_der().hex()
    
    def get_address_from_public_key(self, public_key: Optional[bytes] = None) -> str:
        """
        Derive address from public key
        
        Args:
            public_key: Public key (uses own public key if None)
            
        Returns:
            Derived address string
        """
        try:
            if public_key is None:
                public_key = self.get_public_key_der()
            
            # Hash the public key
            if isinstance(public_key, str):
                public_key_bytes = bytes.fromhex(public_key)
            else:
                public_key_bytes = public_key
            
            # Use SHA256 followed by RIPEMD160 (similar to Bitcoin)
            sha256_hash = hashlib.sha256(public_key_bytes).digest()
            ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
            
            # Take first 20 bytes for address (40 hex chars)
            address = ripemd160_hash[:20].hex()
            
            return address
            
        except Exception as e:
            logger.error(f"Error deriving address from public key: {e}")
            raise
    
    def encrypt_data(self, data: bytes, recipient_public_key: bytes) -> bytes:
        """
        Encrypt data for recipient using ECIES
        
        Args:
            data: Data to encrypt
            recipient_public_key: Recipient's public key
            
        Returns:
            Encrypted data
        """
        try:
            # Load recipient's public key
            recipient_key = serialization.load_der_public_key(recipient_public_key, backend=self.backend)
            
            if not isinstance(recipient_key, ec.EllipticCurvePublicKey):
                raise ValueError("Recipient key must be ECDSA public key")
            
            # Generate ephemeral key pair
            ephemeral_private_key = ec.generate_private_key(recipient_key.curve, self.backend)
            ephemeral_public_key = ephemeral_private_key.public_key()
            
            # Perform ECDH key exchange
            shared_secret = ephemeral_private_key.exchange(ec.ECDH(), recipient_key)
            
            # Derive encryption key using HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'consensus_encryption',
                backend=self.backend
            ).derive(shared_secret)
            
            # Encrypt data using AES (simplified - in production use proper AEAD)
            # This is a placeholder implementation
            encrypted_data = self._simple_encrypt(data, derived_key)
            
            # Prepend ephemeral public key
            ephemeral_public_key_bytes = ephemeral_public_key.public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint
            )
            
            return ephemeral_public_key_bytes + encrypted_data
            
        except Exception as e:
            logger.error(f"Error encrypting data: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data using own private key
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            Decrypted data
        """
        try:
            if not self.private_key:
                raise ValueError("Private key not available")
            
            # Extract ephemeral public key (first 65 bytes for uncompressed point)
            ephemeral_public_key_bytes = encrypted_data[:65]
            ciphertext = encrypted_data[65:]
            
            # Load ephemeral public key
            ephemeral_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                self.private_key.curve, ephemeral_public_key_bytes
            )
            
            # Perform ECDH key exchange
            shared_secret = self.private_key.exchange(ec.ECDH(), ephemeral_public_key)
            
            # Derive decryption key using HKDF
            derived_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'consensus_encryption',
                backend=self.backend
            ).derive(shared_secret)
            
            # Decrypt data
            return self._simple_decrypt(ciphertext, derived_key)
            
        except Exception as e:
            logger.error(f"Error decrypting data: {e}")
            raise
    
    def _simple_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple encryption implementation (placeholder)"""
        # In production, use proper authenticated encryption like AES-GCM
        # This is a simplified version for demonstration
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        import os
        
        # Generate random IV
        iv = os.urandom(16)
        
        # Pad data
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(data) + padder.finalize()
        
        # Encrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return iv + ciphertext
    
    def _simple_decrypt(self, ciphertext: bytes, key: bytes) -> bytes:
        """Simple decryption implementation (placeholder)"""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding
        
        # Extract IV
        iv = ciphertext[:16]
        actual_ciphertext = ciphertext[16:]
        
        # Decrypt
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(actual_ciphertext) + decryptor.finalize()
        
        # Unpad
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded_data) + unpadder.finalize()
        
        return data
    
    def create_certificate(self, subject: str, validity_days: int = 365) -> bytes:
        """
        Create self-signed certificate
        
        Args:
            subject: Certificate subject
            validity_days: Certificate validity in days
            
        Returns:
            PEM-encoded certificate
        """
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID
            from datetime import datetime, timedelta
            
            # Create certificate builder
            subject_name = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, subject),
            ])
            
            issuer_name = subject_name  # Self-signed
            
            certificate = x509.CertificateBuilder().subject_name(
                subject_name
            ).issuer_name(
                issuer_name
            ).public_key(
                self.public_key
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=validity_days)
            ).add_extension(
                x509.BasicConstraints(ca=True, path_length=0), critical=True
            ).sign(self.private_key, hashes.SHA256(), self.backend)
            
            return certificate.public_bytes(serialization.Encoding.PEM)
            
        except Exception as e:
            logger.error(f"Error creating certificate: {e}")
            raise
    
    def verify_certificate(self, certificate_pem: bytes) -> bool:
        """
        Verify certificate
        
        Args:
            certificate_pem: PEM-encoded certificate
            
        Returns:
            True if certificate is valid, False otherwise
        """
        try:
            from cryptography import x509
            from datetime import datetime
            
            certificate = x509.load_pem_x509_certificate(certificate_pem, self.backend)
            
            # Check validity period
            now = datetime.utcnow()
            if now < certificate.not_valid_before or now > certificate.not_valid_after:
                return False
            
            # Verify signature
            issuer_public_key = certificate.issuer.public_key()
            certificate.verify_directly_issued_by(issuer_public_key)
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying certificate: {e}")
            return False