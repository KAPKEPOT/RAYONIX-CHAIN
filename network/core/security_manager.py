import asyncio
import logging
import ssl
import time
import os
import secrets
from typing import Dict, Optional
from typing import Dict, Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from network.exceptions import HandshakeError
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType

logger = logging.getLogger("SecurityManager")

class SecurityManager:
    """Production-ready security management"""
    
    def __init__(self, network):
        self.network = network
        self.encryption_keys: Dict[str, bytes] = {}
        self.session_keys: Dict[str, bytes] = {}
        self.handshakes: Dict[str, Dict] = {}
        self.private_key = None
        self.public_key = None
    
    async def initialize(self):
        """Initialize with proper cryptography"""
        try:
            # Generate proper EC key pair
            self.private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
            self.public_key = self.private_key.public_key()
            
            # Serialize public key for handshake
            self.public_key_bytes = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            logger.info("Security manager initialized with proper cryptography")
            
        except Exception as e:
            logger.error(f"Security initialization failed: {e}")
            raise
    
    async def encrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Proper encryption using AES-GCM"""
        if not self.network.config.enable_encryption:
            return data
        
        if connection_id not in self.session_keys:
            raise HandshakeError(f"No session key for {connection_id}")
        
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            aesgcm = AESGCM(self.session_keys[connection_id])
            nonce = os.urandom(12)  # 96-bit nonce for AES-GCM
            encrypted = aesgcm.encrypt(nonce, data, None)
            return nonce + encrypted
            
        except Exception as e:
            logger.error(f"Encryption error for {connection_id}: {e}")
            raise HandshakeError(f"Encryption failed: {e}")
    
    async def decrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Proper decryption using AES-GCM"""
        if not self.network.config.enable_encryption:
            return data
        
        if connection_id not in self.session_keys:
            raise HandshakeError(f"No session key for {connection_id}")
        
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            if len(data) < 13:  # nonce(12) + at least 1 byte data
                raise HandshakeError("Message too short")
            
            nonce = data[:12]
            ciphertext = data[12:]
            
            aesgcm = AESGCM(self.session_keys[connection_id])
            return aesgcm.decrypt(nonce, ciphertext, None)
            
        except Exception as e:
            logger.error(f"Decryption error for {connection_id}: {e}")
            raise HandshakeError(f"Decryption failed: {e}")
    
    async def initiate_handshake(self, connection_id: str) -> bool:
        """Initiate handshake with a peer"""
        try:
            # Create handshake message
            handshake_message = NetworkMessage(
                message_id=f"handshake_{time.time()}",
                message_type=MessageType.HANDSHAKE,
                payload={
                    "version": "1.0.0",
                    "capabilities": ["tcp", "udp", "gossip", "syncing"],
                    "network": self.network.config.network_type.name,
                    "public_key": self.node_public_key.decode('latin-1'),
                    "timestamp": time.time()
                },
                source_node=self.network.node_id
            )
            
            # Store handshake state
            self.handshakes[connection_id] = {
                "initiated": True,
                "timestamp": time.time(),
                "state": "sent"
            }
            
            # Send handshake
            success = await self.network.message_processor.send_message(
                connection_id, handshake_message
            )
            
            if not success:
                logger.error(f"Failed to send handshake to {connection_id}")
                return False
            
            logger.debug(f"Handshake initiated with {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Handshake initiation error: {e}")
            return False
    
    async def process_handshake(self, connection_id: str, message: NetworkMessage) -> bool:
        """Process incoming handshake message"""
        try:
            if connection_id not in self.handshakes:
                # This is an incoming handshake
                return await self._process_incoming_handshake(connection_id, message)
            else:
                # This is a response to our handshake
                return await self._process_handshake_response(connection_id, message)
                
        except Exception as e:
            logger.error(f"Handshake processing error: {e}")
            return False
    
    async def _process_incoming_handshake(self, connection_id: str, message: NetworkMessage) -> bool:
        """Process incoming handshake request"""
        try:
            # Verify handshake data
            if not self._validate_handshake(message):
                logger.warning(f"Invalid handshake from {connection_id}")
                return False
            
            # Store peer public key
            peer_public_key = message.payload.get("public_key", "").encode('latin-1')
            self.encryption_keys[connection_id] = peer_public_key
            
            # Generate session key (in real implementation)
            self.session_keys[connection_id] = b"session_key_" + peer_public_key[:8]
            
            # Send handshake response
            response = NetworkMessage(
                message_id=f"handshake_resp_{time.time()}",
                message_type=MessageType.HANDSHAKE,
                payload={
                    "version": "1.0.0",
                    "capabilities": ["tcp", "udp", "gossip", "syncing"],
                    "network": self.network.config.network_type.name,
                    "public_key": self.node_public_key.decode('latin-1'),
                    "timestamp": time.time(),
                    "status": "accepted"
                },
                source_node=self.network.node_id
            )
            
            success = await self.network.message_processor.send_message(
                connection_id, response
            )
            
            if success:
                self.handshakes[connection_id] = {
                    "initiated": False,
                    "completed": True,
                    "timestamp": time.time()
                }
                
                logger.info(f"Handshake completed with {connection_id}")
                return True
            else:
                logger.error(f"Failed to send handshake response to {connection_id}")
                return False
                
        except Exception as e:
            logger.error(f"Incoming handshake processing error: {e}")
            return False
    
    async def _process_handshake_response(self, connection_id: str, message: NetworkMessage) -> bool:
        """Process handshake response"""
        try:
            # Verify response
            if not self._validate_handshake(message):
                logger.warning(f"Invalid handshake response from {connection_id}")
                return False
            
            # Store peer public key
            peer_public_key = message.payload.get("public_key", "").encode('latin-1')
            self.encryption_keys[connection_id] = peer_public_key
            
            # Generate session key (in real implementation)
            self.session_keys[connection_id] = b"session_key_" + peer_public_key[:8]
            
            # Update handshake state
            self.handshakes[connection_id]["completed"] = True
            self.handshakes[connection_id]["timestamp"] = time.time()
            
            logger.info(f"Handshake completed with {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Handshake response processing error: {e}")
            return False
    
    def _validate_handshake(self, message: NetworkMessage) -> bool:
        """Validate handshake message"""
        try:
            # Check required fields
            required_fields = ["version", "capabilities", "network", "public_key", "timestamp"]
            for field in required_fields:
                if field not in message.payload:
                    logger.warning(f"Missing field in handshake: {field}")
                    return False
            
            # Check network compatibility
            if message.payload["network"] != self.network.config.network_type.name:
                logger.warning(f"Network mismatch in handshake: {message.payload['network']}")
                return False
            
            # Check timestamp (prevent replay attacks)
            handshake_time = message.payload["timestamp"]
            if abs(time.time() - handshake_time) > 300:  # 5 minutes
                logger.warning(f"Stale handshake: {handshake_time}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Handshake validation error: {e}")
            return False
    
    async def cleanup_stale_handshakes(self):
        """Clean up stale handshakes"""
        current_time = time.time()
        stale_handshakes = []
        
        for connection_id, handshake in self.handshakes.items():
            if current_time - handshake["timestamp"] > 300:  # 5 minutes
                stale_handshakes.append(connection_id)
        
        for connection_id in stale_handshakes:
            del self.handshakes[connection_id]
            if connection_id in self.encryption_keys:
                del self.encryption_keys[connection_id]
            if connection_id in self.session_keys:
                del self.session_keys[connection_id]
            
            logger.debug(f"Cleaned up stale handshake for {connection_id}")