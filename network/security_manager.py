import time
import secrets
import json
import hashlib
import logging
from typing import Dict, Optional, Tuple
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature

from .models import NetworkType, ProtocolType
from .exceptions import HandshakeError, SecurityError
from .utils import EncryptionUtils, CryptoUtils, NetworkUtils

logger = logging.getLogger("AdvancedNetwork")

class SecurityManager:
    """Manages cryptographic operations and secure session establishment"""
    
    def __init__(self, config):
        self.config = config
        self.private_key = CryptoUtils.generate_ec_keypair()
        self.session_keys: Dict[str, bytes] = {}
        self.used_nonces: Set[str] = set()
        self.nonce_cleanup_interval = 300  # Clean up nonces every 5 minutes
        
        # Network magic numbers based on network type
        self.network_magic = {
            NetworkType.MAINNET: b'RAYX',
            NetworkType.TESTNET: b'RAYT',
            NetworkType.DEVNET: b'RAYD',
            NetworkType.REGTEST: b'RAYR'
        }
        self.magic = self.network_magic[self.config.network_type]
    
    def get_public_key(self) -> bytes:
        """Get public key in compressed format"""
        public_key = self.private_key.public_key()
        return CryptoUtils.serialize_public_key(public_key)
    
    def sign_data(self, data: bytes) -> bytes:
        """Sign data with private key"""
        return CryptoUtils.sign_data(self.private_key, data)
    
    async def perform_handshake(self, reader, writer, connection_id: str, protocol: ProtocolType):
        """Perform cryptographic handshake with perfect forward secrecy"""
        try:
            # Generate ephemeral key pair for this session
            ephemeral_private_key = CryptoUtils.generate_ec_keypair()
            ephemeral_public_key = CryptoUtils.serialize_public_key(ephemeral_private_key.public_key())
            
            # Create handshake data with nonce to prevent replay attacks
            nonce = NetworkUtils.generate_nonce()
            handshake_data = {
                'node_id': self.config.node_id,
                'public_key': self.get_public_key().hex(),
                'ephemeral_public_key': ephemeral_public_key.hex(),
                'version': '1.0',
                'capabilities': ['block', 'transaction', 'consensus', 'dht'],
                'timestamp': time.time(),
                'nonce': nonce,
                'network': self.config.network_type.name,
                'listen_port': self.config.listen_port
            }
            
            # Sign handshake
            signature = self.sign_data(json.dumps(handshake_data, sort_keys=True).encode())
            handshake_data['signature'] = signature.hex()
            
            # Send handshake
            from .message_manager import MessageManager
            message_manager = MessageManager(self.config)
            
            handshake_message = {
                'message_id': NetworkUtils.generate_node_id(),
                'message_type': 'HANDSHAKE',
                'payload': handshake_data,
                'nonce': NetworkUtils.generate_nonce()
            }
            
            serialized_message = message_manager.serialize_message(handshake_message)
            await NetworkUtils.send_data(writer, serialized_message)
            
            # Receive response with timeout
            try:
                response_data = await asyncio.wait_for(
                    NetworkUtils.receive_data(reader),
                    timeout=self.config.connection_timeout
                )
                response_message = message_manager.deserialize_message(response_data)
                
                if response_message.get('message_type') != 'HANDSHAKE':
                    raise HandshakeError("Expected handshake response")
                
                # Verify response
                if not await self.verify_handshake(response_message['payload'], nonce):
                    raise HandshakeError("Handshake verification failed")
                
                # Extract peer's ephemeral public key
                peer_ephemeral_public_key = bytes.fromhex(response_message['payload']['ephemeral_public_key'])
                
                # Derive session key using ECDH with ephemeral keys
                shared_secret = ephemeral_private_key.exchange(
                    ec.ECDH(),
                    CryptoUtils.deserialize_public_key(peer_ephemeral_public_key)
                )
                
                # Use HKDF to derive session key
                self.session_keys[connection_id] = HKDF(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=None,
                    info=b'session_key_derivation',
                    backend=default_backend()
                ).derive(shared_secret)
                
                logger.info(f"Handshake completed with {connection_id}")
                
            except asyncio.TimeoutError:
                raise HandshakeError("Handshake timeout")
                
        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            raise HandshakeError(f"Handshake failed: {e}")
    
    async def perform_websocket_handshake(self, websocket, connection_id: str):
        """Perform WebSocket handshake with perfect forward secrecy"""
        try:
            # Generate ephemeral key pair for this session
            ephemeral_private_key = CryptoUtils.generate_ec_keypair()
            ephemeral_public_key = CryptoUtils.serialize_public_key(ephemeral_private_key.public_key())
            
            nonce = NetworkUtils.generate_nonce()
            handshake_data = {
                'node_id': self.config.node_id,
                'public_key': self.get_public_key().hex(),
                'ephemeral_public_key': ephemeral_public_key.hex(),
                'version': '1.0',
                'capabilities': ['block', 'transaction', 'consensus', 'dht'],
                'timestamp': time.time(),
                'nonce': nonce,
                'network': self.config.network_type.name,
                'listen_port': self.config.listen_port
            }
            
            signature = self.sign_data(json.dumps(handshake_data, sort_keys=True).encode())
            handshake_data['signature'] = signature.hex()
            
            from .message_manager import MessageManager
            message_manager = MessageManager(self.config)
            
            handshake_message = {
                'message_id': NetworkUtils.generate_node_id(),
                'message_type': 'HANDSHAKE',
                'payload': handshake_data,
                'nonce': NetworkUtils.generate_nonce()
            }
            
            await websocket.send(message_manager.serialize_message(handshake_message))
            
            # Receive response
            response = await asyncio.wait_for(
                websocket.recv(),
                timeout=self.config.connection_timeout
            )
            response_message = message_manager.deserialize_message(response)
            
            if response_message.get('message_type') != 'HANDSHAKE':
                raise HandshakeError("Expected handshake response")
            
            if not await self.verify_handshake(response_message['payload'], nonce):
                raise HandshakeError("WebSocket handshake verification failed")
            
            peer_ephemeral_public_key = bytes.fromhex(response_message['payload']['ephemeral_public_key'])
            
            shared_secret = ephemeral_private_key.exchange(
                ec.ECDH(),
                CryptoUtils.deserialize_public_key(peer_ephemeral_public_key)
            )
            
            self.session_keys[connection_id] = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'session_key_derivation',
                backend=default_backend()
            ).derive(shared_secret)
            
            logger.info(f"WebSocket handshake completed with {connection_id}")
            
        except asyncio.TimeoutError:
            raise HandshakeError("WebSocket handshake timeout")
        except Exception as e:
            logger.error(f"WebSocket handshake failed: {e}")
            raise HandshakeError(f"WebSocket handshake failed: {e}")
    
    async def verify_handshake(self, handshake_data: Dict, expected_nonce: str) -> bool:
        """Verify handshake signature and nonce"""
        try:
            # Verify nonce to prevent replay attacks
            received_nonce = handshake_data.get('nonce', '')
            if received_nonce != expected_nonce:
                logger.warning("Handshake nonce mismatch")
                return False
            
            # Check if nonce was already used (replay attack)
            if received_nonce in self.used_nonces:
                logger.warning("Replay attack detected - duplicate nonce")
                return False
            
            self.used_nonces.add(received_nonce)
            
            # Verify signature
            public_key_bytes = bytes.fromhex(handshake_data['public_key'])
            signature = bytes.fromhex(handshake_data['signature'])
            
            # Create copy without signature for verification
            verify_data = handshake_data.copy()
            del verify_data['signature']
            
            public_key = CryptoUtils.deserialize_public_key(public_key_bytes)
            
            # Verify the signature
            verify_success = CryptoUtils.verify_signature(
                public_key,
                signature,
                json.dumps(verify_data, sort_keys=True).encode()
            )
            
            if not verify_success:
                logger.warning("Handshake signature verification failed")
                return False
            
            # Verify timestamp is recent
            timestamp = handshake_data.get('timestamp', 0)
            if time.time() - timestamp > 300:  # 5 minutes
                logger.warning("Handshake timestamp too old")
                return False
            
            return True
            
        except (InvalidSignature, ValueError, KeyError) as e:
            logger.warning(f"Handshake verification failed: {e}")
            return False
    
    def encrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Encrypt data using session key"""
        if connection_id not in self.session_keys:
            raise SecurityError("No session key for encryption")
        
        return EncryptionUtils.encrypt_data(data, self.session_keys[connection_id])
    
    def decrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Decrypt data using session key"""
        if connection_id not in self.session_keys:
            raise SecurityError("No session key for decryption")
        
        return EncryptionUtils.decrypt_data(data, self.session_keys[connection_id])
    
    def remove_session_key(self, connection_id: str):
        """Remove session key for a connection"""
        if connection_id in self.session_keys:
            del self.session_keys[connection_id]
    
    async def cleanup_used_nonces(self):
        """Clean up old used nonces to prevent memory exhaustion"""
        while True:
            await asyncio.sleep(self.nonce_cleanup_interval)
            current_time = time.time()
            # We'd typically remove nonces older than a certain age
            # For this implementation, we'll just clear periodically
            self.used_nonces.clear()
            logger.debug("Cleaned up used nonces")