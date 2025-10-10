import asyncio
import ssl
import logging
import time
from typing import Dict, Any, Optional, Tuple
from network.interfaces.protocol_interface import IProtocolHandler
from network.exceptions import ConnectionError
from network.utils.serialization import serialize_message, deserialize_message
from network.utils.compression import compress_data, decompress_data
from network.models.message_header import MessageHeader
import struct
import hashlib

logger = logging.getLogger("TCPHandler")

class TCPHandler(IProtocolHandler):
    """TCP protocol implementation"""
    
    def __init__(self, network, config, ssl_context):
        self.network = network
        self.config = config
        self.ssl_context = ssl_context
        self.server = None
        self.connections: Dict[str, Dict[str, Any]] = {}
    
    async def start_server(self):
        """Start TCP server"""
        try:
            self.server = await asyncio.start_server(
                self.handle_connection,
                self.config.listen_ip,
                self.config.listen_port,
                reuse_address=True,
                reuse_port=True,
                ssl=self.ssl_context if self.config.enable_encryption else None
            )
            
            logger.info(f"TCP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            return self.server
            
        except Exception as e:
            logger.error(f"TCP server error: {e}")
            raise
    
    async def stop_server(self):
        """Stop TCP server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        peer_addr = writer.get_extra_info('peername')
        if not peer_addr:
            return
            
        connection_id = f"tcp_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await self.network.ban_manager.is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting connection from banned peer: {peer_addr[0]}")
            writer.close()
            await writer.wait_closed()
            return
        
        try:
            # Store connection
            self.connections[connection_id] = {
                'reader': reader,
                'writer': writer,
                'address': peer_addr,
                'last_activity': time.time()
            }
            
            # Start message processing
            asyncio.create_task(self.process_messages(connection_id))
            
            logger.info(f"TCP connection established with {peer_addr}")
            
        except Exception as e:
            logger.error(f"TCP connection error with {peer_addr}: {e}")
            writer.close()
            await writer.wait_closed()
    
    async def process_messages(self, connection_id: str):
        """Process incoming messages for a TCP connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        reader = connection['reader']
        
        try:
            while connection_id in self.connections:
                # Read message with header
                data = await self.receive_data(reader)
                if not data:
                    break
                
                # Parse message header
                header, payload = self.parse_message_header(data)
                
                # Verify magic number
                if header.magic != self.network.magic:
                    logger.warning(f"Invalid magic number from {connection_id}")
                    await self.network.penalize_peer(connection_id, -10)
                    continue
                
                # Verify checksum
                expected_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
                if header.checksum != expected_checksum:
                    logger.warning(f"Invalid checksum from {connection_id}")
                    await self.network.penalize_peer(connection_id, -10)
                    continue
                
                # Check message size
                if len(payload) > self.config.max_message_size:
                    logger.warning(f"Oversized message from {connection_id}")
                    await self.network.penalize_peer(connection_id, -20)
                    continue
                
                # Check rate limit
                if not await self.network.rate_limiter.check_rate_limit(connection_id, len(payload)):
                    logger.warning(f"Rate limit exceeded for {connection_id}")
                    await self.network.penalize_peer(connection_id, -5)
                    continue
                
                # Decrypt if encryption enabled
                if self.config.enable_encryption:
                    try:
                        payload = self.network.security_manager.decrypt_data(payload, connection_id)
                    except Exception as e:
                        logger.error(f"Decryption failed for {connection_id}: {e}")
                        await self.network.penalize_peer(connection_id, -15)
                        continue
                
                # Decompress if compression enabled
                if self.config.enable_compression:
                    try:
                        payload = decompress_data(payload)
                    except Exception as e:
                        logger.error(f"Decompression failed for {connection_id}: {e}")
                        await self.network.penalize_peer(connection_id, -5)
                        continue
                
                # Deserialize message
                try:
                    message = deserialize_message(payload)
                except Exception as e:
                    logger.error(f"Deserialization failed for {connection_id}: {e}")
                    await self.network.penalize_peer(connection_id, -10)
                    continue
                
                # Update metrics
                self.network.metrics_collector.update_connection_metrics(
                    connection_id, bytes_received=len(data), messages_received=1
                )
                
                # Process message
                await self.network.message_processor.process_message(connection_id, message)
                
        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by peer: {connection_id}")
        except Exception as e:
            logger.error(f"Message processing error for {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                await self.close_connection(connection_id)
    
    async def send_message(self, connection_id: str, message: Any) -> bool:
        """Send message via TCP"""
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        try:
            # Serialize message
            serialized = serialize_message(message)
            
            # Compress if enabled
            if self.config.enable_compression:
                serialized = compress_data(serialized)
            
            # Encrypt if enabled
            if self.config.enable_encryption:
                serialized = self.network.security_manager.encrypt_data(serialized, connection_id)
            
            # Create message header
            header = self.create_message_header(serialized)
            full_message = header + serialized
            
            # Check rate limit
            if not await self.network.rate_limiter.check_rate_limit(connection_id, len(full_message)):
                logger.warning(f"Rate limit exceeded for sending to {connection_id}")
                return False
            
            # Send message
            writer = self.connections[connection_id]['writer']
            await self.send_data(writer, full_message)
            
            # Update metrics
            self.network.metrics_collector.update_connection_metrics(
                connection_id, bytes_sent=len(full_message), messages_sent=1
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self.close_connection(connection_id)
            return False
    
    async def close_connection(self, connection_id: str):
        """Close TCP connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        
        try:
            writer = connection.get('writer')
            if writer:
                writer.close()
                await writer.wait_closed()
        except Exception as e:
            logger.debug(f"Error closing connection {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                del self.connections[connection_id]
            self.network.metrics_collector.remove_connection_metrics(connection_id)
            self.network.rate_limiter.remove_connection(connection_id)
            
            logger.debug(f"TCP connection {connection_id} closed")
    
    def create_message_header(self, payload: bytes) -> bytes:
        """Create message header with magic, command, length, and checksum"""
        # Calculate checksum (first 4 bytes of double SHA256)
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        # Create header structure
        header = struct.pack(
            '4s12sI4s',
            self.network.magic,  # 4 bytes magic
            b'RAYX_MSG',  # 12 bytes command (padded)
            len(payload),  # 4 bytes length
            checksum  # 4 bytes checksum
        )
        
        return header
    
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
    
    async def send_data(self, writer: asyncio.StreamWriter, data: bytes):
        """Send data with proper error handling"""
        try:
            writer.write(data)
            await writer.drain()
        except Exception as e:
            raise ConnectionError(f"Failed to send data: {e}")
    
    async def receive_data(self, reader: asyncio.StreamReader) -> bytes:
        """Receive data with proper error handling"""
        try:
            # First read the header to know how much data to expect
            header_data = await reader.readexactly(24)
            header, _ = self.parse_message_header(header_data + b'\x00' * 24)  # Pad for parsing
            
            # Read the payload
            payload = await reader.readexactly(header.length)
            
            return header_data + payload
            
        except asyncio.IncompleteReadError:
            raise ConnectionError("Connection closed during data reception")
        except Exception as e:
            raise ConnectionError(f"Failed to receive data: {e}")