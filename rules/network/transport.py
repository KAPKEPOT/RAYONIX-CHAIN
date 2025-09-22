import asyncio
import aiohttp
from typing import Dict, List, Optional, Callable
import logging
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from rules.exceptions import NetworkError
from rules.utils import RateLimiter, Backoff

logger = logging.getLogger('consensus.network')

@dataclass
class Peer:
    """Network peer information"""
    id: str
    address: str
    port: int
    public_key: str
    last_seen: float = 0.0
    score: int = 100
    connected: bool = False

class NetworkTransport:
    """Network transport layer for consensus messages"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 26656, 
                 max_peers: int = 50, peer_discovery: bool = True):
        self.host = host
        self.port = port
        self.max_peers = max_peers
        self.peer_discovery = peer_discovery
        
        self.peers: Dict[str, Peer] = {}
        self.connected_peers: Dict[str, Peer] = {}
        
        self.message_handlers = {}
        self.connection_handlers = {}
        
        self.session = None
        self.server = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        self.rate_limiter = RateLimiter(1000, 1.0)  # 1000 messages per second
        self.backoff = Backoff(base_delay=1.0, max_delay=30.0)
        
        self.running = False
        
    async def start(self) -> None:
        """Start network transport"""
        try:
            self.session = aiohttp.ClientSession()
            self.server = await asyncio.start_server(
                self._handle_connection, self.host, self.port
            )
            
            self.running = True
            asyncio.create_task(self._peer_maintenance())
            asyncio.create_task(self._message_processing())
            
            logger.info(f"Network transport started on {self.host}:{self.port}")
            
        except Exception as e:
            raise NetworkError(f"Failed to start network transport: {e}")
    
    async def stop(self) -> None:
        """Stop network transport"""
        self.running = False
        
        if self.session:
            await self.session.close()
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        
        logger.info("Network transport stopped")
    
    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Handle incoming connection"""
        try:
            peer_addr = writer.get_extra_info('peername')
            peer_id = f"{peer_addr[0]}:{peer_addr[1]}"
            
            # Handshake protocol
            handshake = await self._perform_handshake(reader, writer)
            if not handshake:
                writer.close()
                return
            
            peer = Peer(
                id=peer_id,
                address=peer_addr[0],
                port=peer_addr[1],
                public_key=handshake['public_key'],
                last_seen=time.time(),
                connected=True
            )
            
            self.peers[peer_id] = peer
            self.connected_peers[peer_id] = peer
            
            logger.info(f"Connected to peer: {peer_id}")
            
            # Start message processing for this peer
            asyncio.create_task(self._process_peer_messages(reader, writer, peer))
            
        except Exception as e:
            logger.error(f"Connection handling failed: {e}")
            writer.close()
    
    async def _perform_handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> Optional[Dict]:
        """Perform connection handshake"""
        try:
            # Send our handshake
            handshake = {
                'version': '1.0.0',
                'public_key': 'our_public_key',  # This would be actual public key
                'timestamp': time.time()
            }
            
            handshake_data = json.dumps(handshake).encode()
            writer.write(len(handshake_data).to_bytes(4, 'big'))
            writer.write(handshake_data)
            await writer.drain()
            
            # Receive peer handshake
            length_bytes = await reader.read(4)
            if not length_bytes:
                return None
                
            length = int.from_bytes(length_bytes, 'big')
            handshake_data = await reader.read(length)
            peer_handshake = json.loads(handshake_data.decode())
            
            # Validate handshake
            if not self._validate_handshake(peer_handshake):
                return None
                
            return peer_handshake
            
        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            return None
    
    def _validate_handshake(self, handshake: Dict) -> bool:
        """Validate handshake data"""
        required_fields = {'version', 'public_key', 'timestamp'}
        if not all(field in handshake for field in required_fields):
            return False
        
        # Check version compatibility
        if handshake['version'] != '1.0.0':
            return False
        
        # Check timestamp (within 10 seconds)
        if abs(time.time() - handshake['timestamp']) > 10:
            return False
        
        return True
    
    async def _process_peer_messages(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, peer: Peer) -> None:
        """Process messages from peer"""
        try:
            while self.running:
                # Read message length
                length_bytes = await reader.read(4)
                if not length_bytes:
                    break
                
                length = int.from_bytes(length_bytes, 'big')
                if length > 10 * 1024 * 1024:  # 10MB limit
                    logger.warning(f"Message too large from {peer.id}")
                    break
                
                # Read message data
                message_data = await reader.read(length)
                if not message_data:
                    break
                
                # Parse and handle message
                message = json.loads(message_data.decode())
                await self._handle_message(message, peer)
                
        except Exception as e:
            logger.error(f"Message processing failed for {peer.id}: {e}")
        finally:
            writer.close()
            self.connected_peers.pop(peer.id, None)
            logger.info(f"Disconnected from peer: {peer.id}")
    
    async def _handle_message(self, message: Dict, peer: Peer) -> None:
        """Handle incoming message"""
        try:
            message_type = message.get('type')
            if not message_type or message_type not in self.message_handlers:
                logger.warning(f"Unknown message type from {peer.id}: {message_type}")
                return
            
            # Rate limiting
            if not self.rate_limiter.acquire():
                logger.warning(f"Rate limit exceeded from {peer.id}")
                return
            
            # Call message handler
            for handler in self.message_handlers.get(message_type, []):
                try:
                    await handler(message, peer)
                except Exception as e:
                    logger.error(f"Message handler failed: {e}")
                    
        except Exception as e:
            logger.error(f"Message handling failed: {e}")
    
    async def broadcast_message(self, message: Dict, message_type: str) -> None:
        """Broadcast message to all connected peers"""
        if not self.rate_limiter.acquire():
            return
        
        message_data = json.dumps({
            'type': message_type,
            'data': message,
            'timestamp': time.time()
        }).encode()
        
        length = len(message_data).to_bytes(4, 'big')
        
        for peer_id, peer in list(self.connected_peers.items()):
            try:
                # This would actually send to the peer's writer
                # For now, just log
                logger.debug(f"Broadcasting {message_type} to {peer_id}")
            except Exception as e:
                logger.error(f"Failed to broadcast to {peer_id}: {e}")
    
    async def send_message(self, peer_id: str, message: Dict, message_type: str) -> bool:
        """Send message to specific peer"""
        try:
            if peer_id not in self.connected_peers:
                return False
            
            message_data = json.dumps({
                'type': message_type,
                'data': message,
                'timestamp': time.time()
            }).encode()
            
            length = len(message_data).to_bytes(4, 'big')
            
            # This would actually send to the peer's writer
            logger.debug(f"Sending {message_type} to {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {peer_id}: {e}")
            return False
    
    def register_message_handler(self, message_type: str, handler: Callable) -> None:
        """Register message handler"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def register_connection_handler(self, event_type: str, handler: Callable) -> None:
        """Register connection event handler"""
        if event_type not in self.connection_handlers:
            self.connection_handlers[event_type] = []
        self.connection_handlers[event_type].append(handler)
    
    async def _peer_maintenance(self) -> None:
        """Maintain peer connections"""
        while self.running:
            try:
                # Remove stale peers
                current_time = time.time()
                stale_peers = [
                    peer_id for peer_id, peer in self.peers.items()
                    if current_time - peer.last_seen > 300  # 5 minutes
                ]
                
                for peer_id in stale_peers:
                    self.peers.pop(peer_id, None)
                
                # Peer discovery
                if self.peer_discovery and len(self.connected_peers) < self.max_peers:
                    await self._discover_peers()
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Peer maintenance failed: {e}")
                await asyncio.sleep(60)
    
    async def _discover_peers(self) -> None:
        """Discover new peers"""
        # This would implement actual peer discovery logic
        # For now, just log
        logger.debug("Peer discovery running")
    
    async def _message_processing(self) -> None:
        """Background message processing"""
        while self.running:
            try:
                # Process any queued messages
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Message processing loop failed: {e}")
                await asyncio.sleep(1)
    
    def get_peer_count(self) -> int:
        """Get number of connected peers"""
        return len(self.connected_peers)
    
    def get_peer_info(self, peer_id: str) -> Optional[Dict]:
        """Get information about specific peer"""
        if peer_id in self.peers:
            peer = self.peers[peer_id]
            return {
                'id': peer.id,
                'address': peer.address,
                'port': peer.port,
                'public_key': peer.public_key,
                'last_seen': peer.last_seen,
                'score': peer.score,
                'connected': peer.connected
            }
        return None
    
    def get_connected_peers(self) -> List[Dict]:
        """Get list of all connected peers"""
        return [self.get_peer_info(pid) for pid in self.connected_peers]