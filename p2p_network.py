# advanced_network.py
import asyncio
import aiohttp
import websockets
import socket
import ssl
import threading
import time
import json
import pickle
import zlib
from typing import Dict, List, Optional, Set, Tuple, Callable, Any, Deque
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import uuid
import hashlib
import secrets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import msgpack
import bencode
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from collections import deque, defaultdict
import ipaddress
from urllib.parse import urlparse
import dns.resolver
import random
import select
from contextlib import asynccontextmanager
from asyncio import DatagramProtocol, Transport
import struct
import aiodns
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import async_timeout

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AdvancedNetwork")

class NetworkError(Exception):
    """Base network error"""
    pass

class ConnectionError(NetworkError):
    """Connection-related errors"""
    pass

class HandshakeError(NetworkError):
    """Handshake-related errors"""
    pass

class MessageError(NetworkError):
    """Message-related errors"""
    pass

class PeerBannedError(NetworkError):
    """Peer is banned"""
    pass

class RateLimitError(NetworkError):
    """Rate limit exceeded"""
    pass

class NetworkType(Enum):
    MAINNET = auto()
    TESTNET = auto()
    DEVNET = auto()
    REGTEST = auto()

class ProtocolType(Enum):
    TCP = auto()
    UDP = auto()
    WEBSOCKET = auto()
    HTTP = auto()
    HTTPS = auto()

class ConnectionState(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    AUTHENTICATING = auto()
    READY = auto()
    ERROR = auto()

class MessageType(Enum):
    PING = auto()
    PONG = auto()
    HANDSHAKE = auto()
    PEER_LIST = auto()
    BLOCK = auto()
    TRANSACTION = auto()
    CONSENSUS = auto()
    SYNC_REQUEST = auto()
    SYNC_RESPONSE = auto()
    GOSSIP = auto()
    RPC_REQUEST = auto()
    RPC_RESPONSE = auto()
    GET_BLOCKS = auto()
    BLOCK_HEADERS = auto()
    GET_DATA = auto()
    NOT_FOUND = auto()
    MEMPOOL = auto()
    FILTER_LOAD = auto()
    FILTER_ADD = auto()
    FILTER_CLEAR = auto()
    MERKLE_BLOCK = auto()
    ALERT = auto()
    SEND_HEADERS = auto()
    FEE_FILTER = auto()
    SEND_CMPCT = auto()
    CMPCT_BLOCK = auto()
    GET_BLOCK_TXN = auto()
    BLOCK_TXN = auto()

@dataclass
class NodeConfig:
    """Network node configuration"""
    network_type: NetworkType = NetworkType.MAINNET
    listen_ip: str = "0.0.0.0"
    listen_port: int = 30303
    public_ip: Optional[str] = None
    public_port: Optional[int] = None
    max_connections: int = 50
    max_peers: int = 1000
    connection_timeout: int = 30
    message_timeout: int = 10
    ping_interval: int = 60
    bootstrap_nodes: List[str] = field(default_factory=list)
    enable_nat_traversal: bool = True
    enable_encryption: bool = True
    enable_compression: bool = True
    enable_dht: bool = True
    enable_gossip: bool = True
    enable_syncing: bool = True
    max_message_size: int = 10 * 1024 * 1024  # 10MB
    rate_limit_per_peer: int = 1000  # messages per minute
    ban_threshold: int = -100  # Reputation score for auto-ban
    ban_duration: int = 3600  # 1 hour in seconds
    dht_bootstrap_nodes: List[Tuple[str, int]] = field(default_factory=list)
    dns_seeds: List[str] = field(default_factory=list)

@dataclass
class PeerInfo:
    """Peer information"""
    node_id: str
    address: str
    port: int
    protocol: ProtocolType
    version: str
    capabilities: List[str]
    last_seen: float = field(default_factory=time.time)
    connection_count: int = 0
    failed_attempts: int = 0
    reputation: int = 100
    latency: float = 0.0
    state: ConnectionState = ConnectionState.DISCONNECTED
    public_key: Optional[str] = None
    user_agent: str = ""
    services: int = 0
    last_attempt: float = 0.0
    next_attempt: float = 0.0
    banned_until: Optional[float] = None

@dataclass
class NetworkMessage:
    """Network message structure"""
    message_id: str
    message_type: MessageType
    payload: Any
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time-to-live for gossip
    signature: Optional[str] = None
    source_node: Optional[str] = None
    destination_node: Optional[str] = None
    priority: int = 0  # 0=low, 1=normal, 2=high, 3=critical

@dataclass
class ConnectionMetrics:
    """Connection performance metrics"""
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    connection_time: float = 0.0
    last_activity: float = field(default_factory=time.time)
    latency_history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    success_rate: float = 1.0
    message_rate: float = 0.0  # Messages per second
    bandwidth_rate: float = 0.0  # Bytes per second

@dataclass
class MessageHeader:
    """Network message header structure"""
    magic: bytes = b'RAYX'  # Network magic number
    command: bytes = b'\x00' * 12  # 12-byte command name
    length: int = 0  # Payload length
    checksum: bytes = b'\x00' * 4  # First 4 bytes of sha256(sha256(payload))

class AdvancedP2PNetwork:
    """Advanced P2P network implementation with multiple protocols and security"""
    
    def __init__(self, config: NodeConfig, node_id: Optional[str] = None):
        self.config = config
        self.node_id = node_id or self._generate_node_id()
        self.private_key = self._generate_crypto_keys()
        
        # Network state
        self.peers: Dict[str, PeerInfo] = {}
        self.connections: Dict[str, Any] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # DHT and routing
        self.dht_table: Dict[str, List[PeerInfo]] = defaultdict(list)
        self.routing_table: Dict[str, List[str]] = defaultdict(list)
        
        # Metrics and statistics
        self.metrics = ConnectionMetrics()
        self.message_queue = asyncio.PriorityQueue()
        self.connection_pool: Dict[str, Any] = {}
        
        # Security
        self.session_keys: Dict[str, bytes] = {}
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.banned_peers: Dict[str, float] = {}  # peer_id -> ban_until_timestamp
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # connection_id -> rate limit data
        
        # Threading and async
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)
        self.process_executor = ProcessPoolExecutor(max_workers=4)
        self.running = False
        
        # Message processing
        self.priority_queues: Dict[int, asyncio.Queue] = {
            0: asyncio.Queue(),  # Low priority
            1: asyncio.Queue(),  # Normal priority
            2: asyncio.Queue(),  # High priority
            3: asyncio.Queue()   # Critical priority
        }
        
        # Initialize components
        self._initialize_network()
    
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        return hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    
    def _generate_crypto_keys(self) -> ec.EllipticCurvePrivateKey:
        """Generate cryptographic keys"""
        return ec.generate_private_key(ec.SECP256K1(), default_backend())
    
    def _initialize_network(self):
        """Initialize network components"""
        # Create SSL context for secure connections
        self.ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize DNS resolver
        self.dns_resolver = aiodns.DNSResolver()
        
        # Network magic numbers based on network type
        self.network_magic = {
            NetworkType.MAINNET: b'RAYX',
            NetworkType.TESTNET: b'RAYT',
            NetworkType.DEVNET: b'RAYD',
            NetworkType.REGTEST: b'RAYR'
        }
        
        # Current network magic
        self.magic = self.network_magic[self.config.network_type]
        
        # Initialize protocol handlers
        self.protocol_handlers = {
            ProtocolType.TCP: self._handle_tcp_connection,
            ProtocolType.UDP: self._handle_udp_connection,
            ProtocolType.WEBSOCKET: self._handle_websocket_connection,
            ProtocolType.HTTP: self._handle_http_connection,
            ProtocolType.HTTPS: self._handle_https_connection
        }

    async def start(self):
        """Start the network node"""
        self.running = True
        
        # Start server listeners
        server_tasks = [
            self._start_tcp_server(),
            self._start_udp_server(),
            self._start_websocket_server(),
            self._start_http_server()
        ]
        
        # Start background tasks
        background_tasks = [
            self._message_processor(),
            self._connection_manager(),
            self._peer_discovery(),
            self._metrics_collector(),
            self._gossip_broadcaster(),
            self._nat_traversal(),
            self._rate_limiter(),
            self._ban_manager()
        ]
        
        # Start priority message processors
        for priority in self.priority_queues:
            asyncio.create_task(self._priority_message_processor(priority))
        
        # Bootstrap to network
        await self._bootstrap_network()
        
        # Run all tasks
        try:
            await asyncio.gather(*server_tasks + background_tasks)
        except asyncio.CancelledError:
            logger.info("Network shutdown requested")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the network node"""
        self.running = False
        
        # Close all connections
        for connection_id in list(self.connections.keys()):
            await self._close_connection(connection_id)
        
        # Shutdown executors
        self.executor.shutdown(wait=False)
        self.process_executor.shutdown(wait=False)
        
        logger.info("Network stopped gracefully")
    
    def is_connected(self) -> bool:
        """Check if network has active connections"""
        return len(self.connections) > 0
    
    async def _start_tcp_server(self):
        """Start TCP server"""
        try:
            server = await asyncio.start_server(
                self._handle_tcp_connection,
                self.config.listen_ip,
                self.config.listen_port,
                reuse_address=True,
                reuse_port=True,
                ssl=self.ssl_context if self.config.enable_encryption else None
            )
            
            logger.info(f"TCP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            logger.error(f"TCP server error: {e}")
            raise
    
    async def _start_udp_server(self):
        """Start UDP server"""
        try:
            loop = asyncio.get_running_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: UDPProtocol(self),
                local_addr=(self.config.listen_ip, self.config.listen_port),
                reuse_port=True
            )
            
            self.udp_transport = transport
            self.udp_protocol = protocol
            
            logger.info(f"UDP server listening on {self.config.listen_ip}:{self.config.listen_port}")
            
            # Keep the server running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"UDP server error: {e}")
            raise
    
    async def _start_websocket_server(self):
        """Start WebSocket server"""
        try:
            server = await websockets.serve(
                self._handle_websocket_connection,
                self.config.listen_ip,
                self.config.listen_port + 1,  # Different port for WS
                ssl=self.ssl_context if self.config.enable_encryption else None,
                ping_interval=None,  # We handle our own pings
                max_size=self.config.max_message_size
            )
            
            logger.info(f"WebSocket server listening on {self.config.listen_ip}:{self.config.listen_port + 1}")
            await server.wait_closed()
            
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
    
    async def _start_http_server(self):
        """Start HTTP server"""
        try:
            server = await asyncio.start_server(
                self._handle_http_connection,
                self.config.listen_ip,
                self.config.listen_port + 2,  # Different port for HTTP
                reuse_address=True,
                reuse_port=True,
                ssl=self.ssl_context if self.config.enable_encryption else None
            )
            
            logger.info(f"HTTP server listening on {self.config.listen_ip}:{self.config.listen_port + 2}")
            
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            logger.error(f"HTTP server error: {e}")
            raise
    
    async def _handle_tcp_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming TCP connection"""
        peer_addr = writer.get_extra_info('peername')
        if not peer_addr:
            return
            
        connection_id = f"tcp_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await self._is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting connection from banned peer: {peer_addr[0]}")
            writer.close()
            await writer.wait_closed()
            return
        
        try:
            # Perform handshake with timeout
            async with async_timeout.timeout(self.config.connection_timeout):
                await self._perform_handshake(reader, writer, connection_id)
            
            # Add to connections
            self.connections[connection_id] = {
                'reader': reader,
                'writer': writer,
                'protocol': ProtocolType.TCP,
                'metrics': ConnectionMetrics(),
                'address': peer_addr,
                'state': ConnectionState.READY,
                'last_activity': time.time()
            }
            
            # Initialize rate limiting
            self.rate_limits[connection_id] = {
                'message_count': 0,
                'last_reset': time.time(),
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
            # Start message processing
            asyncio.create_task(self._process_messages(connection_id))
            
            logger.info(f"TCP connection established with {peer_addr}")
            
        except asyncio.TimeoutError:
            logger.warning(f"TCP handshake timeout with {peer_addr}")
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            logger.error(f"TCP connection error with {peer_addr}: {e}")
            writer.close()
            await writer.wait_closed()

    async def _handle_udp_connection(self, data: bytes, addr: tuple):
        """Handle incoming UDP datagram"""
        connection_id = f"udp_{addr[0]}_{addr[1]}"
        
        # Check if peer is banned
        if await self._is_peer_banned(addr[0]):
            logger.warning(f"Rejecting UDP from banned peer: {addr[0]}")
            return
        
        try:
            # Parse message header
            header, payload = self._parse_message_header(data)
            
            # Verify magic number
            if header.magic != self.magic:
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
            
            # Process message
            message = self._deserialize_message(payload)
            await self._handle_message(connection_id, message)
            
        except Exception as e:
            logger.error(f"UDP handling error from {addr}: {e}")

    async def _handle_websocket_connection(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection"""
        peer_addr = websocket.remote_address
        if not peer_addr:
            return
            
        connection_id = f"ws_{peer_addr[0]}_{peer_addr[1]}"
        
        # Check if peer is banned
        if await self._is_peer_banned(peer_addr[0]):
            logger.warning(f"Rejecting WebSocket from banned peer: {peer_addr[0]}")
            await websocket.close()
            return
        
        try:
            # Perform handshake with timeout
            async with async_timeout.timeout(self.config.connection_timeout):
                await self._perform_websocket_handshake(websocket, connection_id)
            
            # Add to connections
            self.connections[connection_id] = {
                'websocket': websocket,
                'protocol': ProtocolType.WEBSOCKET,
                'metrics': ConnectionMetrics(),
                'address': peer_addr,
                'state': ConnectionState.READY,
                'last_activity': time.time()
            }
            
            # Initialize rate limiting
            self.rate_limits[connection_id] = {
                'message_count': 0,
                'last_reset': time.time(),
                'bytes_sent': 0,
                'bytes_received': 0
            }
            
            # Start message processing
            asyncio.create_task(self._process_websocket_messages(connection_id))
            
            logger.info(f"WebSocket connection established with {peer_addr}")
            
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket handshake timeout with {peer_addr}")
            await websocket.close()
        except Exception as e:
            logger.error(f"WebSocket connection error with {peer_addr}: {e}")
            await websocket.close()

    async def _handle_http_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming HTTP connection"""
        peer_addr = writer.get_extra_info('peername')
        if not peer_addr:
            return
            
        connection_id = f"http_{peer_addr[0]}_{peer_addr[1]}"
        
        try:
            request = await reader.read(8192)
            if not request:
                return
            
            # Parse HTTP request
            request_str = request.decode('utf-8')
            lines = request_str.split('\r\n')
            
            if not lines:
                response = self._create_http_response(400, "Bad Request")
                writer.write(response)
                await writer.drain()
                return
            
            # Parse request line
            request_line = lines[0].split()
            if len(request_line) < 3:
                response = self._create_http_response(400, "Bad Request")
                writer.write(response)
                await writer.drain()
                return
            
            method, path, version = request_line
            
            # Handle different endpoints
            if method == 'GET':
                if path == '/peers':
                    peers_data = json.dumps([{
                        'node_id': peer.node_id,
                        'address': peer.address,
                        'port': peer.port,
                        'protocol': peer.protocol.name,
                        'reputation': peer.reputation
                    } for peer in self.peers.values()])
                    response = self._create_http_response(200, "OK", peers_data, 'application/json')
                elif path == '/status':
                    status = {
                        'node_id': self.node_id,
                        'connections': len(self.connections),
                        'peers': len(self.peers),
                        'status': 'running',
                        'network': self.config.network_type.name
                    }
                    response = self._create_http_response(200, "OK", json.dumps(status), 'application/json')
                else:
                    response = self._create_http_response(404, "Not Found")
            else:
                response = self._create_http_response(405, "Method Not Allowed")
            
            writer.write(response)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"HTTP connection error: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _handle_https_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle incoming HTTPS connection"""
        # Similar to HTTP but with encrypted transport
        await self._handle_http_connection(reader, writer)

    async def _perform_handshake(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, connection_id: str):
        """Perform cryptographic handshake with perfect forward secrecy"""
        try:
            # Generate ephemeral key pair for this session
            ephemeral_private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            ephemeral_public_key = ephemeral_private_key.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint
            )
            
            # Create handshake data with nonce to prevent replay attacks
            nonce = secrets.token_bytes(32)
            handshake_data = {
                'node_id': self.node_id,
                'public_key': self._get_public_key().hex(),
                'ephemeral_public_key': ephemeral_public_key.hex(),
                'version': '1.0',
                'capabilities': ['block', 'transaction', 'consensus', 'dht'],
                'timestamp': time.time(),
                'nonce': nonce.hex(),
                'network': self.config.network_type.name
            }
            
            # Sign handshake
            signature = self._sign_data(json.dumps(handshake_data).encode())
            handshake_data['signature'] = signature.hex()
            
            # Send handshake
            handshake_message = NetworkMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HANDSHAKE,
                payload=handshake_data
            )
            
            await self._send_message_internal(writer, handshake_message, ProtocolType.TCP)
            
            # Receive response
            response_data = await self._receive_data(reader)
            response_message = self._deserialize_message(response_data)
            
            if response_message.message_type != MessageType.HANDSHAKE:
                raise HandshakeError("Expected handshake response")
            
            # Verify response
            if not self._verify_handshake(response_message.payload, nonce):
                raise HandshakeError("Handshake verification failed")
            
            # Extract peer's ephemeral public key
            peer_ephemeral_public_key = bytes.fromhex(response_message.payload['ephemeral_public_key'])
            
            # Derive session key using ECDH with ephemeral keys
            shared_secret = ephemeral_private_key.exchange(
                ec.ECDH(),
                ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), peer_ephemeral_public_key)
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
            
        except Exception as e:
            logger.error(f"Handshake failed: {e}")
            raise HandshakeError(f"Handshake failed: {e}")

    async def _perform_websocket_handshake(self, websocket: websockets.WebSocketServerProtocol, connection_id: str):
        """Perform WebSocket handshake with perfect forward secrecy"""
        try:
            # Similar to TCP handshake but over WebSocket
            ephemeral_private_key = ec.generate_private_key(ec.SECP256K1(), default_backend())
            ephemeral_public_key = ephemeral_private_key.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.CompressedPoint
            )
            
            nonce = secrets.token_bytes(32)
            handshake_data = {
                'node_id': self.node_id,
                'public_key': self._get_public_key().hex(),
                'ephemeral_public_key': ephemeral_public_key.hex(),
                'version': '1.0',
                'capabilities': ['block', 'transaction', 'consensus', 'dht'],
                'timestamp': time.time(),
                'nonce': nonce.hex(),
                'network': self.config.network_type.name
            }
            
            signature = self._sign_data(json.dumps(handshake_data).encode())
            handshake_data['signature'] = signature.hex()
            
            handshake_message = NetworkMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.HANDSHAKE,
                payload=handshake_data
            )
            
            await websocket.send(self._serialize_message(handshake_message))
            
            # Receive response
            response = await websocket.recv()
            response_message = self._deserialize_message(response)
            
            if response_message.message_type != MessageType.HANDSHAKE:
                raise HandshakeError("Expected handshake response")
            
            if not self._verify_handshake(response_message.payload, nonce):
                raise HandshakeError("WebSocket handshake verification failed")
            
            peer_ephemeral_public_key = bytes.fromhex(response_message.payload['ephemeral_public_key'])
            
            shared_secret = ephemeral_private_key.exchange(
                ec.ECDH(),
                ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256K1(), peer_ephemeral_public_key)
            )
            
            self.session_keys[connection_id] = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=b'session_key_derivation',
                backend=default_backend()
            ).derive(shared_secret)
            
            logger.info(f"WebSocket handshake completed with {connection_id}")
            
        except Exception as e:
            logger.error(f"WebSocket handshake failed: {e}")
            raise HandshakeError(f"WebSocket handshake failed: {e}")

    def _verify_handshake(self, handshake_data: Dict, expected_nonce: bytes) -> bool:
        """Verify handshake signature and nonce"""
        try:
            # Verify nonce to prevent replay attacks
            received_nonce = bytes.fromhex(handshake_data['nonce'])
            if received_nonce != expected_nonce:
                logger.warning("Handshake nonce mismatch")
                return False
            
            public_key_bytes = bytes.fromhex(handshake_data['public_key'])
            signature = bytes.fromhex(handshake_data['signature'])
            
            # Create copy without signature for verification
            verify_data = handshake_data.copy()
            del verify_data['signature']
            
            public_key = ec.EllipticCurvePublicKey.from_encoded_point(
                ec.SECP256K1(), public_key_bytes
            )
            
            public_key.verify(
                signature,
                json.dumps(verify_data).encode(),
                ec.ECDSA(hashes.SHA256())
            )
            
            return True
            
        except (InvalidSignature, ValueError, KeyError) as e:
            logger.warning(f"Handshake verification failed: {e}")
            return False

    async def _process_messages(self, connection_id: str):
        """Process incoming messages for a TCP connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        reader = connection['reader']
        
        try:
            while self.running and connection_id in self.connections:
                # Read message with header
                data = await self._receive_data(reader)
                if not data:
                    break
                
                # Parse message header
                header, payload = self._parse_message_header(data)
                
                # Verify magic number
                if header.magic != self.magic:
                    logger.warning(f"Invalid magic number from {connection_id}")
                    await self._penalize_peer(connection_id, -10)
                    continue
                
                # Verify checksum
                expected_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
                if header.checksum != expected_checksum:
                    logger.warning(f"Invalid checksum from {connection_id}")
                    await self._penalize_peer(connection_id, -10)
                    continue
                
                # Check message size
                if len(payload) > self.config.max_message_size:
                    logger.warning(f"Oversized message from {connection_id}")
                    await self._penalize_peer(connection_id, -20)
                    continue
                
                # Check rate limit
                if not await self._check_rate_limit(connection_id, len(payload)):
                    logger.warning(f"Rate limit exceeded for {connection_id}")
                    await self._penalize_peer(connection_id, -5)
                    continue
                
                # Decrypt if encryption enabled
                if self.config.enable_encryption:
                    try:
                        payload = self._decrypt_data(payload, connection_id)
                    except Exception as e:
                        logger.error(f"Decryption failed for {connection_id}: {e}")
                        await self._penalize_peer(connection_id, -15)
                        continue
                
                # Decompress if compression enabled
                if self.config.enable_compression:
                    try:
                        payload = self._decompress_data(payload)
                    except Exception as e:
                        logger.error(f"Decompression failed for {connection_id}: {e}")
                        await self._penalize_peer(connection_id, -5)
                        continue
                
                # Deserialize message
                try:
                    message = self._deserialize_message(payload)
                except Exception as e:
                    logger.error(f"Deserialization failed for {connection_id}: {e}")
                    await self._penalize_peer(connection_id, -10)
                    continue
                
                # Update metrics
                connection['metrics'].messages_received += 1
                connection['metrics'].bytes_received += len(data)
                connection['metrics'].last_activity = time.time()
                
                # Add to appropriate priority queue
                await self.priority_queues[message.priority].put((connection_id, message))
                
        except asyncio.IncompleteReadError:
            logger.debug(f"Connection closed by peer: {connection_id}")
        except Exception as e:
            logger.error(f"Message processing error for {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                await self._close_connection(connection_id)

    async def _process_websocket_messages(self, connection_id: str):
        """Process WebSocket messages"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        websocket = connection['websocket']
        
        try:
            async for message_data in websocket:
                if not self.running or connection_id not in self.connections:
                    break
                
                # Check rate limit
                if not await self._check_rate_limit(connection_id, len(message_data)):
                    logger.warning(f"Rate limit exceeded for {connection_id}")
                    await self._penalize_peer(connection_id, -5)
                    continue
                
                # Process encrypted/compressed message
                payload = message_data
                
                if self.config.enable_encryption:
                    try:
                        payload = self._decrypt_data(payload, connection_id)
                    except Exception as e:
                        logger.error(f"WebSocket decryption failed for {connection_id}: {e}")
                        await self._penalize_peer(connection_id, -15)
                        continue
                
                if self.config.enable_compression:
                    try:
                        payload = self._decompress_data(payload)
                    except Exception as e:
                        logger.error(f"WebSocket decompression failed for {connection_id}: {e}")
                        await self._penalize_peer(connection_id, -5)
                        continue
                
                try:
                    message = self._deserialize_message(payload)
                except Exception as e:
                    logger.error(f"WebSocket deserialization failed for {connection_id}: {e}")
                    await self._penalize_peer(connection_id, -10)
                    continue
                
                # Update metrics
                connection['metrics'].messages_received += 1
                connection['metrics'].bytes_received += len(message_data)
                connection['metrics'].last_activity = time.time()
                
                # Add to priority queue
                await self.priority_queues[message.priority].put((connection_id, message))
                
        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket message processing error for {connection_id}: {e}")
        finally:
            if connection_id in self.connections:
                await self._close_connection(connection_id)

    async def _priority_message_processor(self, priority: int):
        """Process messages from a specific priority queue"""
        while self.running:
            try:
                connection_id, message = await self.priority_queues[priority].get()
                await self._handle_message(connection_id, message)
                self.priority_queues[priority].task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Priority {priority} message processor error: {e}")

    async def _handle_message(self, connection_id: str, message: NetworkMessage):
        """Handle incoming message"""
        try:
            # Call registered handlers
            if message.message_type in self.message_handlers:
                for handler in self.message_handlers[message.message_type]:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Message handler error: {e}")
            
            # Handle specific message types
            if message.message_type == MessageType.PING:
                await self._handle_ping(connection_id, message)
            elif message.message_type == MessageType.PONG:
                await self._handle_pong(connection_id, message)
            elif message.message_type == MessageType.PEER_LIST:
                await self._handle_peer_list(connection_id, message)
            elif message.message_type == MessageType.HANDSHAKE:
                await self._handle_handshake(connection_id, message)
            elif message.message_type == MessageType.SYNC_REQUEST:
                await self._handle_sync_request(connection_id, message)
            elif message.message_type == MessageType.SYNC_RESPONSE:
                await self._handle_sync_response(connection_id, message)
            elif message.message_type == MessageType.GOSSIP:
                await self._handle_gossip(connection_id, message)
            elif message.message_type == MessageType.RPC_REQUEST:
                await self._handle_rpc_request(connection_id, message)
            elif message.message_type == MessageType.RPC_RESPONSE:
                await self._handle_rpc_response(connection_id, message)
            
        except Exception as e:
            logger.error(f"Message handling error for {connection_id}: {e}")

    async def _handle_ping(self, connection_id: str, message: NetworkMessage):
        """Handle ping message"""
        pong_message = NetworkMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PONG,
            payload={'timestamp': time.time(), 'original_ping_id': message.message_id}
        )
        
        await self.send_message(connection_id, pong_message)

    async def _handle_pong(self, connection_id: str, message: NetworkMessage):
        """Handle pong message"""
        if connection_id in self.connections:
            # Update latency
            latency = time.time() - message.payload['timestamp']
            self.connections[connection_id]['metrics'].latency_history.append(latency)
            
            # Update peer reputation
            await self._update_peer_reputation(connection_id, 1)

    async def _handle_peer_list(self, connection_id: str, message: NetworkMessage):
        """Handle peer list message"""
        try:
            peers = message.payload.get('peers', [])
            for peer_info in peers:
                peer_id = peer_info.get('node_id')
                if peer_id and peer_id != self.node_id:
                    # Add to peer discovery
                    await self._add_peer_from_discovery(peer_info)
            
            # Update reputation for sharing peers
            await self._update_peer_reputation(connection_id, 5)
            
        except Exception as e:
            logger.error(f"Peer list handling error: {e}")

    async def _handle_handshake(self, connection_id: str, message: NetworkMessage):
        """Handle handshake message"""
        # Already handled during connection establishment
        pass

    async def _handle_sync_request(self, connection_id: str, message: NetworkMessage):
        """Handle sync request"""
        try:
            # Process sync request and prepare response
            # This would typically involve sending block headers or data
            sync_response = NetworkMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SYNC_RESPONSE,
                payload={'status': 'success', 'data': []}
            )
            
            await self.send_message(connection_id, sync_response)
            
        except Exception as e:
            logger.error(f"Sync request handling error: {e}")

    async def _handle_sync_response(self, connection_id: str, message: NetworkMessage):
        """Handle sync response"""
        # Process sync data
        pass

    async def _handle_gossip(self, connection_id: str, message: NetworkMessage):
        """Handle gossip message"""
        try:
            # Check TTL
            if message.ttl <= 0:
                return
                
            # Decrement TTL and rebroadcast
            message.ttl -= 1
            await self.broadcast_message(message, exclude=[connection_id])
            
        except Exception as e:
            logger.error(f"Gossip handling error: {e}")

    async def _handle_rpc_request(self, connection_id: str, message: NetworkMessage):
        """Handle RPC request"""
        try:
            # Process RPC and send response
            response = NetworkMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.RPC_RESPONSE,
                payload={'request_id': message.message_id, 'result': None}
            )
            
            await self.send_message(connection_id, response)
            
        except Exception as e:
            logger.error(f"RPC request handling error: {e}")

    async def _handle_rpc_response(self, connection_id: str, message: NetworkMessage):
        """Handle RPC response"""
        # Complete pending request
        request_id = message.payload.get('request_id')
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            if not future.done():
                future.set_result(message.payload.get('result'))
            del self.pending_requests[request_id]

    async def send_message(self, connection_id: str, message: NetworkMessage, priority: int = 1) -> bool:
        """Send message to specific connection"""
        if connection_id not in self.connections:
            logger.warning(f"Connection {connection_id} not found")
            return False
        
        try:
            connection = self.connections[connection_id]
            protocol = connection['protocol']
            
            # Serialize message
            serialized = self._serialize_message(message)
            
            # Compress if enabled
            if self.config.enable_compression:
                serialized = self._compress_data(serialized)
            
            # Encrypt if enabled
            if self.config.enable_encryption:
                serialized = self._encrypt_data(serialized, connection_id)
            
            # Create message header
            header = self._create_message_header(serialized)
            full_message = header + serialized
            
            # Check rate limit
            if not await self._check_rate_limit(connection_id, len(full_message)):
                logger.warning(f"Rate limit exceeded for sending to {connection_id}")
                return False
            
            # Send based on protocol
            if protocol == ProtocolType.TCP:
                writer = connection['writer']
                await self._send_data(writer, full_message)
            elif protocol == ProtocolType.WEBSOCKET:
                websocket = connection['websocket']
                await websocket.send(full_message)
            elif protocol == ProtocolType.UDP:
                addr = connection['address']
                if hasattr(self, 'udp_transport'):
                    self.udp_transport.sendto(full_message, addr)
            
            # Update metrics
            connection['metrics'].messages_sent += 1
            connection['metrics'].bytes_sent += len(full_message)
            connection['metrics'].last_activity = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            await self._close_connection(connection_id)
            return False

    async def broadcast_message(self, message: NetworkMessage, exclude: List[str] = None, priority: int = 1):
        """Broadcast message to all connected peers"""
        exclude = exclude or []
        tasks = []
        
        for connection_id in list(self.connections.keys()):
            if connection_id not in exclude:
                task = asyncio.create_task(self.send_message(connection_id, message, priority))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def connect_to_peer(self, address: str, port: int, protocol: ProtocolType = ProtocolType.TCP) -> Optional[str]:
        """Connect to a peer"""
        try:
            connection_id = f"{protocol.name.lower()}_{address}_{port}"
            
            # Check if already connected
            if connection_id in self.connections:
                return connection_id
            
            # Check if banned
            if await self._is_peer_banned(address):
                logger.warning(f"Cannot connect to banned peer: {address}")
                return None
            
            # Connect based on protocol
            if protocol == ProtocolType.TCP:
                reader, writer = await asyncio.open_connection(
                    address, port,
                    ssl=self.ssl_context if self.config.enable_encryption else None
                )
                
                # Perform handshake
                await self._perform_handshake(reader, writer, connection_id)
                
                # Store connection
                self.connections[connection_id] = {
                    'reader': reader,
                    'writer': writer,
                    'protocol': protocol,
                    'metrics': ConnectionMetrics(),
                    'address': (address, port),
                    'state': ConnectionState.READY,
                    'last_activity': time.time()
                }
                
                # Start message processing
                asyncio.create_task(self._process_messages(connection_id))
                
            elif protocol == ProtocolType.WEBSOCKET:
                ssl_context = self.ssl_context if self.config.enable_encryption else None
                websocket = await websockets.connect(
                    f"ws://{address}:{port}",
                    ssl=ssl_context,
                    ping_interval=None
                )
                
                # Perform handshake
                await self._perform_websocket_handshake(websocket, connection_id)
                
                # Store connection
                self.connections[connection_id] = {
                    'websocket': websocket,
                    'protocol': protocol,
                    'metrics': ConnectionMetrics(),
                    'address': (address, port),
                    'state': ConnectionState.READY,
                    'last_activity': time.time()
                }
                
                # Start message processing
                asyncio.create_task(self._process_websocket_messages(connection_id))
            
            logger.info(f"Connected to peer {address}:{port} via {protocol.name}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to connect to {address}:{port}: {e}")
            await self._update_peer_reputation_by_address(address, -5)
            return None

    async def _close_connection(self, connection_id: str):
        """Close a connection"""
        if connection_id not in self.connections:
            return
            
        connection = self.connections[connection_id]
        
        try:
            if connection['protocol'] == ProtocolType.TCP:
                writer = connection.get('writer')
                if writer:
                    writer.close()
                    await writer.wait_closed()
            elif connection['protocol'] == ProtocolType.WEBSOCKET:
                websocket = connection.get('websocket')
                if websocket:
                    await websocket.close()
        except Exception as e:
            logger.debug(f"Error closing connection {connection_id}: {e}")
        finally:
            # Clean up
            if connection_id in self.connections:
                del self.connections[connection_id]
            if connection_id in self.session_keys:
                del self.session_keys[connection_id]
            if connection_id in self.rate_limits:
                del self.rate_limits[connection_id]
            
            logger.debug(f"Connection {connection_id} closed")

    async def _connection_manager(self):
        """Manage connections and peer health"""
        while self.running:
            try:
                # Check connection health
                current_time = time.time()
                for connection_id in list(self.connections.keys()):
                    connection = self.connections[connection_id]
                    
                    # Check for stale connections
                    if current_time - connection['metrics'].last_activity > self.config.connection_timeout * 2:
                        logger.warning(f"Closing stale connection: {connection_id}")
                        await self._close_connection(connection_id)
                        continue
                    
                    # Send periodic ping
                    if current_time - connection['metrics'].last_activity > self.config.ping_interval:
                        ping_message = NetworkMessage(
                            message_id=str(uuid.uuid4()),
                            message_type=MessageType.PING,
                            payload={'timestamp': current_time}
                        )
                        await self.send_message(connection_id, ping_message)
                
                # Maintain target number of connections
                await self._maintain_connections()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Connection manager error: {e}")
                await asyncio.sleep(10)

    async def _maintain_connections(self):
        """Maintain target number of connections"""
        current_count = len(self.connections)
        target_count = min(self.config.max_connections, len(self.peers))
        
        if current_count < target_count // 2:
            # Need more connections
            await self._discover_and_connect_peers(target_count - current_count)
        elif current_count > target_count * 1.2:
            # Too many connections, close some
            await self._prune_connections(current_count - target_count)

    async def _discover_and_connect_peers(self, count: int):
        """Discover and connect to new peers"""
        try:
            # Get candidate peers sorted by reputation
            candidate_peers = sorted(
                self.peers.values(),
                key=lambda p: p.reputation,
                reverse=True
            )
            
            # Filter out already connected and banned peers
            connected_addresses = {
                conn['address'][0] for conn in self.connections.values()
            }
            
            candidates = [
                peer for peer in candidate_peers
                if peer.address not in connected_addresses
                and not await self._is_peer_banned(peer.address)
                and peer.next_attempt <= time.time()
            ]
            
            # Connect to top candidates
            connection_tasks = []
            for peer in candidates[:min(count, len(candidates))]:
                task = asyncio.create_task(
                    self.connect_to_peer(peer.address, peer.port, peer.protocol)
                )
                connection_tasks.append(task)
            
            if connection_tasks:
                await asyncio.gather(*connection_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Peer discovery and connection error: {e}")

    async def _prune_connections(self, count: int):
        """Prune excess connections"""
        # Sort connections by reputation and activity
        connections_to_prune = sorted(
            self.connections.items(),
            key=lambda x: (
                self.peers.get(x[0], PeerInfo(node_id='', address='', port=0, protocol=ProtocolType.TCP, version='', capabilities=[])).reputation,
                x[1]['metrics'].last_activity
            )
        )[:count]
        
        for connection_id, _ in connections_to_prune:
            await self._close_connection(connection_id)

    async def _peer_discovery(self):
        """Discover new peers through various methods"""
        while self.running:
            try:
                # DNS-based discovery
                if self.config.dns_seeds:
                    await self._dns_discovery()
                
                # DHT-based discovery
                if self.config.enable_dht:
                    await self._dht_discovery()
                
                # Request peer lists from connected peers
                await self._request_peer_lists()
                
                # Clean up old peers
                await self._cleanup_peers()
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Peer discovery error: {e}")
                await asyncio.sleep(60)

    async def _dns_discovery(self):
        """Discover peers through DNS seeds"""
        for dns_seed in self.config.dns_seeds:
            try:
                answers = await self.dns_resolver.query(dns_seed, 'A')
                for answer in answers:
                    await self._add_peer_from_discovery({
                        'address': answer.host,
                        'port': self.config.listen_port,
                        'protocol': ProtocolType.TCP.name
                    })
            except Exception as e:
                logger.error(f"DNS discovery failed for {dns_seed}: {e}")

    async def _dht_discovery(self):
        """Discover peers through Distributed Hash Table"""
        # This would implement Kademlia DHT protocol
        # For now, we'll use a simplified approach
        try:
            # Query bootstrap nodes
            for bootstrap_node in self.config.dht_bootstrap_nodes:
                address, port = bootstrap_node
                # Simulate DHT lookup - in real implementation, this would use Kademlia protocol
                discovered_peers = await self._simulate_dht_lookup(address, port)
                for peer_info in discovered_peers:
                    await self._add_peer_from_discovery(peer_info)
                    
        except Exception as e:
            logger.error(f"DHT discovery error: {e}")

    async def _simulate_dht_lookup(self, address: str, port: int) -> List[Dict]:
        """Simulate DHT lookup (to be replaced with real Kademlia implementation)"""
        # In a real implementation, this would use the Kademlia protocol
        # to find peers closest to our node ID in the DHT
        return []

    async def _request_peer_lists(self):
        """Request peer lists from connected peers"""
        peer_list_request = NetworkMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PEER_LIST,
            payload={'request': True}
        )
        
        await self.broadcast_message(peer_list_request)

    async def _add_peer_from_discovery(self, peer_info: Dict):
        """Add peer discovered through various methods"""
        try:
            address = peer_info.get('address')
            port = peer_info.get('port', self.config.listen_port)
            protocol = ProtocolType[peer_info.get('protocol', 'TCP')]
            node_id = peer_info.get('node_id', '')
            
            if not address or await self._is_peer_banned(address):
                return
            
            # Create or update peer info
            peer_key = f"{address}:{port}"
            if peer_key not in self.peers:
                self.peers[peer_key] = PeerInfo(
                    node_id=node_id,
                    address=address,
                    port=port,
                    protocol=protocol,
                    version=peer_info.get('version', ''),
                    capabilities=peer_info.get('capabilities', []),
                    reputation=50  # Initial reputation for discovered peers
                )
            else:
                # Update existing peer
                self.peers[peer_key].last_seen = time.time()
                if node_id:
                    self.peers[peer_key].node_id = node_id
            
        except Exception as e:
            logger.error(f"Error adding discovered peer: {e}")

    async def _cleanup_peers(self):
        """Clean up old and low-reputation peers"""
        current_time = time.time()
        peers_to_remove = []
        
        for peer_key, peer in self.peers.items():
            # Remove very old peers
            if current_time - peer.last_seen > 86400:  # 24 hours
                peers_to_remove.append(peer_key)
            # Remove very low reputation peers
            elif peer.reputation < -50:
                peers_to_remove.append(peer_key)
        
        for peer_key in peers_to_remove:
            del self.peers[peer_key]

    async def _bootstrap_network(self):
        """Bootstrap to the network"""
        logger.info("Bootstrapping to network...")
        
        # Connect to bootstrap nodes
        for bootstrap_node in self.config.bootstrap_nodes:
            try:
                if '://' in bootstrap_node:
                    parsed = urlparse(bootstrap_node)
                    address = parsed.hostname
                    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
                    protocol = ProtocolType.HTTPS if parsed.scheme == 'https' else ProtocolType.HTTP
                else:
                    if ':' in bootstrap_node:
                        address, port_str = bootstrap_node.split(':', 1)
                        port = int(port_str)
                    else:
                        address = bootstrap_node
                        port = self.config.listen_port
                    protocol = ProtocolType.TCP
                
                await self.connect_to_peer(address, port, protocol)
                
            except Exception as e:
                logger.error(f"Failed to bootstrap to {bootstrap_node}: {e}")
        
        # Wait for initial connections
        await asyncio.sleep(2)
        
        if not self.is_connected():
            logger.warning("No initial bootstrap connections succeeded")
        else:
            logger.info(f"Bootstrapped with {len(self.connections)} connections")

    async def _nat_traversal(self):
        """Perform NAT traversal if enabled"""
        if not self.config.enable_nat_traversal:
            return
            
        while self.running:
            try:
                # Implement NAT traversal techniques like UPnP, STUN, etc.
                # This is a complex topic that would require additional dependencies
                await asyncio.sleep(3600)  # Run hourly
                
            except Exception as e:
                logger.error(f"NAT traversal error: {e}")
                await asyncio.sleep(600)

    async def _rate_limiter(self):
        """Manage rate limiting"""
        while self.running:
            try:
                current_time = time.time()
                
                # Reset rate limits periodically
                for connection_id, limit_data in list(self.rate_limits.items()):
                    if current_time - limit_data['last_reset'] >= 60:  # 1 minute
                        limit_data['message_count'] = 0
                        limit_data['bytes_sent'] = 0
                        limit_data['bytes_received'] = 0
                        limit_data['last_reset'] = current_time
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Rate limiter error: {e}")
                await asyncio.sleep(5)

    async def _ban_manager(self):
        """Manage banned peers"""
        while self.running:
            try:
                current_time = time.time()
                
                # Remove expired bans
                expired_bans = [
                    peer for peer, ban_until in self.banned_peers.items()
                    if ban_until <= current_time
                ]
                
                for peer in expired_bans:
                    del self.banned_peers[peer]
                    logger.info(f"Ban expired for peer: {peer}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Ban manager error: {e}")
                await asyncio.sleep(300)

    async def _check_rate_limit(self, connection_id: str, message_size: int) -> bool:
        """Check if rate limit is exceeded"""
        if connection_id not in self.rate_limits:
            return True
            
        limit_data = self.rate_limits[connection_id]
        
        # Check message count limit
        if limit_data['message_count'] >= self.config.rate_limit_per_peer:
            return False
        
        # Check bandwidth limit (optional)
        bandwidth_limit = self.config.rate_limit_per_peer * 1024  # 1KB per message average
        if limit_data['bytes_received'] + message_size > bandwidth_limit:
            return False
        
        # Update counters
        limit_data['message_count'] += 1
        limit_data['bytes_received'] += message_size
        
        return True

    async def _is_peer_banned(self, address: str) -> bool:
        """Check if peer is banned"""
        if address in self.banned_peers:
            if self.banned_peers[address] > time.time():
                return True
            else:
                # Ban expired
                del self.banned_peers[address]
        return False

    async def _update_peer_reputation(self, connection_id: str, delta: int):
        """Update peer reputation"""
        if connection_id in self.connections:
            address = self.connections[connection_id]['address'][0]
            await self._update_peer_reputation_by_address(address, delta)

    async def _update_peer_reputation_by_address(self, address: str, delta: int):
        """Update peer reputation by address"""
        for peer_key, peer in self.peers.items():
            if peer.address == address:
                peer.reputation += delta
                
                # Auto-ban if reputation drops too low
                if peer.reputation <= self.config.ban_threshold:
                    ban_until = time.time() + self.config.ban_duration
                    self.banned_peers[address] = ban_until
                    logger.warning(f"Auto-banned peer {address} for {self.config.ban_duration} seconds")
                
                break

    async def _penalize_peer(self, connection_id: str, penalty: int):
        """Penalize peer for bad behavior"""
        await self._update_peer_reputation(connection_id, penalty)

    async def _metrics_collector(self):
        """Collect and log network metrics"""
        while self.running:
            try:
                total_connections = len(self.connections)
                total_peers = len(self.peers)
                banned_peers = len([p for p in self.banned_peers.values() if p > time.time()])
                
                # Calculate overall metrics
                total_bytes_sent = sum(c['metrics'].bytes_sent for c in self.connections.values())
                total_bytes_received = sum(c['metrics'].bytes_received for c in self.connections.values())
                total_messages_sent = sum(c['metrics'].messages_sent for c in self.connections.values())
                total_messages_received = sum(c['metrics'].messages_received for c in self.connections.values())
                
                logger.info(
                    f"Network Metrics: Connections={total_connections}, "
                    f"Peers={total_peers}, Banned={banned_peers}, "
                    f"Sent={total_bytes_sent} bytes, Received={total_bytes_received} bytes, "
                    f"Messages Sent={total_messages_sent}, Received={total_messages_received}"
                )
                
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(300)

    async def _gossip_broadcaster(self):
        """Periodically broadcast gossip messages"""
        if not self.config.enable_gossip:
            return
            
        while self.running:
            try:
                # Create gossip message (could be about new blocks, transactions, etc.)
                gossip_message = NetworkMessage(
                    message_id=str(uuid.uuid4()),
                    message_type=MessageType.GOSSIP,
                    payload={
                        'node_id': self.node_id,
                        'timestamp': time.time(),
                        'content': 'network_heartbeat'
                    },
                    ttl=5  # Limit propagation
                )
                
                await self.broadcast_message(gossip_message, priority=0)
                
                await asyncio.sleep(30)  # Broadcast every 30 seconds
                
            except Exception as e:
                logger.error(f"Gossip broadcasting error: {e}")
                await asyncio.sleep(60)

    def _create_message_header(self, payload: bytes) -> bytes:
        """Create message header with magic, command, length, and checksum"""
        # Calculate checksum (first 4 bytes of double SHA256)
        checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
        
        # Create header structure
        header = struct.pack(
            '4s12sI4s',
            self.magic,  # 4 bytes magic
            b'RAYX_MSG',  # 12 bytes command (padded)
            len(payload),  # 4 bytes length
            checksum  # 4 bytes checksum
        )
        
        return header

    def _parse_message_header(self, data: bytes) -> Tuple[MessageHeader, bytes]:
        """Parse message header from data"""
        if len(data) < 24:  # Header size
            raise MessageError("Message too short for header")
        
        # Unpack header
        magic, command, length, checksum = struct.unpack('4s12sI4s', data[:24])
        
        # Extract payload
        payload = data[24:24+length]
        
        if len(payload) != length:
            raise MessageError("Payload length mismatch")
        
        header = MessageHeader()
        header.magic = magic
        header.command = command
        header.length = length
        header.checksum = checksum
        
        return header, payload

    def _serialize_message(self, message: NetworkMessage) -> bytes:
        """Serialize message to bytes"""
        try:
            # Convert to dict for serialization
            message_dict = {
                'message_id': message.message_id,
                'message_type': message.message_type.name,
                'payload': message.payload,
                'timestamp': message.timestamp,
                'ttl': message.ttl,
                'signature': message.signature,
                'source_node': message.source_node,
                'destination_node': message.destination_node,
                'priority': message.priority
            }
            
            # Use msgpack for efficient binary serialization
            return msgpack.packb(message_dict, use_bin_type=True)
            
        except Exception as e:
            logger.error(f"Message serialization error: {e}")
            raise MessageError(f"Serialization failed: {e}")

    def _deserialize_message(self, data: bytes) -> NetworkMessage:
        """Deserialize message from bytes"""
        try:
            message_dict = msgpack.unpackb(data, raw=False)
            
            # Convert back to NetworkMessage
            message = NetworkMessage(
                message_id=message_dict['message_id'],
                message_type=MessageType[message_dict['message_type']],
                payload=message_dict['payload'],
                timestamp=message_dict['timestamp'],
                ttl=message_dict['ttl'],
                signature=message_dict.get('signature'),
                source_node=message_dict.get('source_node'),
                destination_node=message_dict.get('destination_node'),
                priority=message_dict.get('priority', 1)
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Message deserialization error: {e}")
            raise MessageError(f"Deserialization failed: {e}")

    def _encrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Encrypt data using session key"""
        if connection_id not in self.session_keys:
            raise ConnectionError("No session key for encryption")
        
        try:
            # Use AES in GCM mode for authenticated encryption
            iv = get_random_bytes(12)
            cipher = AES.new(self.session_keys[connection_id], AES.MODE_GCM, nonce=iv)
            ciphertext, tag = cipher.encrypt_and_digest(data)
            
            # Return IV + ciphertext + tag
            return iv + ciphertext + tag
            
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise

    def _decrypt_data(self, data: bytes, connection_id: str) -> bytes:
        """Decrypt data using session key"""
        if connection_id not in self.session_keys:
            raise ConnectionError("No session key for decryption")
        
        try:
            # Extract IV, ciphertext, and tag
            iv = data[:12]
            ciphertext = data[12:-16]
            tag = data[-16:]
            
            cipher = AES.new(self.session_keys[connection_id], AES.MODE_GCM, nonce=iv)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            
            return plaintext
            
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib"""
        if not self.config.enable_compression:
            return data
            
        try:
            return zlib.compress(data)
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return data

    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data using zlib"""
        if not self.config.enable_compression:
            return data
            
        try:
            return zlib.decompress(data)
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return data

    def _get_public_key(self) -> bytes:
        """Get public key in compressed format"""
        public_key = self.private_key.public_key()
        return public_key.public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.CompressedPoint
        )

    def _sign_data(self, data: bytes) -> bytes:
        """Sign data with private key"""
        return self.private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )

    def _create_http_response(self, status_code: int, status_message: str, 
                            content: str = "", content_type: str = "text/plain") -> bytes:
        """Create HTTP response"""
        response = f"HTTP/1.1 {status_code} {status_message}\r\n"
        response += "Content-Type: {content_type}\r\n"
        response += f"Content-Length: {len(content)}\r\n"
        response += "Connection: close\r\n"
        response += "\r\n"
        response += content
        
        return response.encode('utf-8')

    async def _send_data(self, writer: asyncio.StreamWriter, data: bytes):
        """Send data with proper error handling"""
        try:
            writer.write(data)
            await writer.drain()
        except Exception as e:
            raise ConnectionError(f"Failed to send data: {e}")

    async def _receive_data(self, reader: asyncio.StreamReader) -> bytes:
        """Receive data with proper error handling"""
        try:
            # First read the header to know how much data to expect
            header_data = await reader.readexactly(24)
            header, _ = self._parse_message_header(header_data + b'\x00' * 24)  # Pad for parsing
            
            # Read the payload
            payload = await reader.readexactly(header.length)
            
            return header_data + payload
            
        except asyncio.IncompleteReadError:
            raise ConnectionError("Connection closed during data reception")
        except Exception as e:
            raise ConnectionError(f"Failed to receive data: {e}")

    async def _send_message_internal(self, writer: asyncio.StreamWriter, message: NetworkMessage, protocol: ProtocolType):
        """Internal method to send message"""
        serialized = self._serialize_message(message)
        
        if self.config.enable_compression:
            serialized = self._compress_data(serialized)
        
        # For handshake, we don't encrypt yet as session key isn't established
        if protocol == ProtocolType.TCP:
            await self._send_data(writer, serialized)

    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type].append(handler)

    def unregister_message_handler(self, message_type: MessageType, handler: Callable):
        """Unregister message handler"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type] = [
                h for h in self.message_handlers[message_type] if h != handler
            ]

    async def rpc_call(self, connection_id: str, method: str, params: Any, timeout: int = 30) -> Any:
        """Make RPC call to peer"""
        if connection_id not in self.connections:
            raise ConnectionError("Connection not found")
        
        request_id = str(uuid.uuid4())
        future = self.loop.create_future()
        self.pending_requests[request_id] = future
        
        rpc_message = NetworkMessage(
            message_id=request_id,
            message_type=MessageType.RPC_REQUEST,
            payload={'method': method, 'params': params}
        )
        
        try:
            await self.send_message(connection_id, rpc_message)
            
            # Wait for response with timeout
            return await asyncio.wait_for(future, timeout)
            
        except asyncio.TimeoutError:
            del self.pending_requests[request_id]
            raise ConnectionError("RPC call timeout")
        except Exception as e:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
            raise ConnectionError(f"RPC call failed: {e}")

class UDPProtocol:
    """UDP protocol handler"""
    def __init__(self, network: 'AdvancedP2PNetwork'):
        self.network = network
        self.transport = None
    
    def connection_made(self, transport: DatagramProtocol):
        self.transport = transport
    
    def datagram_received(self, data: bytes, addr: tuple):
        asyncio.create_task(self.network._handle_udp_connection(data, addr))
    
    def error_received(self, exc: Exception):
        logger.error(f"UDP error: {exc}")
    
    def connection_lost(self, exc: Optional[Exception]):
        logger.info("UDP connection closed")

# Example usage and testing
async def main():
    """Main function to demonstrate the network"""
    # Create configuration
    config = NodeConfig(
        network_type=NetworkType.TESTNET,
        listen_ip="0.0.0.0",
        listen_port=30303,
        max_connections=10,
        bootstrap_nodes=[
            "seed1.rayonix.site:30303",
            "seed2.rayonix.site:30303"
        ],
        dns_seeds=[
            "seed.rayonix.site",
            "backup.rayonix.site"
        ]
    )
    
    # Create network instance
    network = AdvancedP2PNetwork(config)
    
    # Register message handlers
    def handle_block(connection_id: str, message: NetworkMessage):
        logger.info(f"Received block from {connection_id}")
    
    network.register_message_handler(MessageType.BLOCK, handle_block)
    
    try:
        # Start network
        await network.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await network.stop()

if __name__ == "__main__":
    asyncio.run(main())
              