import asyncio
import logging
from typing import Dict, Any, Tuple
from ..interfaces.protocol_interface import IProtocolHandler
from ..utils.serialization import deserialize_message
from ..utils.compression import decompress_data
from ..models.message_header import MessageHeader
import struct
import hashlib

logger = logging.getLogger("UDPHandler")

class UDPHandler(IProtocolHandler):
    """UDP protocol implementation"""
    
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.transport = None
        self.protocol = None
    
    async def start_server(self):
        """Start UDP server"""
        try:
            loop = asyncio.get_running_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: self.network.udp_protocol,
                local_addr=(self.config.listen_ip, self.config.listen_port),
                reuse_port=True
            )
            
            self.transport = transport
            self.protocol = protocol
            
            logger.info(f"UDP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            return transport
            
        except Exception as e:
            logger.error(f"UDP server error: {e}")
            raise
    
    async def stop_server(self):
        """Stop UDP server"""
        if self.transport:
            self.transport.close()
    
    async def handle_connection(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming UDP datagram"""
        connection_id = f"udp_{addr[0]}_{addr[1]}"
        
        # Check if peer is banned
        if await self.network.ban_manager.is_peer_banned(addr[0]):
            logger.warning(f"Rejecting UDP from banned peer: {addr[0]}")
            return
        
        try:
            # Parse message header
            header, payload = self.parse_message_header(data)
            
            # Verify magic number
            if header.magic != self.network.magic:
                logger.warning(f"Invalid magic number from {addr}")
                return
            
            # Verify checksum
            expected_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            if header.checksum != expected_checksum:
                logger.warning(f"Invalid checksum from {addr}")
                return
            
            # Check message size
            if len(payload) > self.config.max_message_size:
                logger.warning(f"Oversized message from {addr}")
                return
            
            # Check rate limit
            if not await self.network.rate_limiter.check_rate_limit(connection_id, len(payload)):
                logger.warning(f"Rate limit exceeded for {connection_id}")
                return
            
            # Decrypt if encryption enabled
            if self.config.enable_encryption:
                try:
                    payload = self.network.security_manager.decrypt_data(payload, connection_id)
                except Exception as e:
                    logger.error(f"Decryption failed for {connection_id}: {e}")
                    return
            
            # Decompress if compression enabled
            if self.config.enable_compression:
                try:
                    payload = decompress_data(payload)
                except Exception as e:
                    logger.error(f"Decompression failed for {connection_id}: {e}")
                    return
            
            # Deserialize message
            try:
                message = deserialize_message(payload)
            except Exception as e:
                logger.error(f"Deserialization failed for {connection_id}: {e}")
                return
            
            # Update metrics
            self.network.metrics_collector.update_connection_metrics(
                connection_id, bytes_received=len(data), messages_received=1
            )
            
            # Process message
            await self.network.message_processor.process_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"UDP handling error from {addr}: {e}")
    
    async def send_message(self, connection_id: str, message: Any) -> bool:
        """Send message via UDP"""
        # UDP is connectionless, so we need address info
        # This method would need to be implemented differently for UDP
        # For now, we'll return False as UDP sending is handled differently
        return False
    
    def sendto(self, data: bytes, addr: Tuple[str, int]):
        """Send UDP datagram to address"""
        if self.transport:
            self.transport.sendto(data, addr)
    
    def parse_message_header(self, data: bytes) -> Tuple[MessageHeader, bytes]:
        """Parse message header from data"""
        if len(data) < 24:  # Header size
            raise Exception("Message too short for header")
        
        # Unpack header
        magic, command, length, checksum = struct.unpack('4s12sI4s', data[:24])
        
        # Extract payload
        payload = data[24:24+length]
        
        if len(payload) != length:
            raise Exception("Payload length mismatch")
        
        header = MessageHeader()
        header.magic = magic
        header.command = command
        header.length = length
        header.checksum = checksum
        
        return header, payload