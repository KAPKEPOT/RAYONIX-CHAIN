from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
from collections import deque
import time
import uuid

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
    DHT = auto()
    DHT_PING = auto()
    DHT_PONG = auto()
    DHT_FIND_NODE = auto()
    DHT_FIND_NODE_RESPONSE = auto()
    DHT_FIND_VALUE = auto()
    DHT_FIND_VALUE_RESPONSE = auto()
    DHT_STORE = auto()
    DHT_STORE_RESPONSE = auto()

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
    enable_peer_exchange: bool = True
    peer_exchange_interval: int = 300  # 5 minutes
    max_peer_age: int = 86400  # 24 hours
    min_peer_reputation: int = -50
    dht_bootstrap_nodes: List[Tuple[str, int]] = field(default_factory=list)
    dht_k: int = 20  # Kademlia k parameter
    dht_alpha: int = 3  # Kademlia parallelism parameter
    dht_storage_size: int = 10000  # Max storage records
    enable_dht: bool = True

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
    source: str = "unknown"  # How we discovered this peer

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
    nonce: Optional[str] = None  # For replay protection

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
    last_ping_time: float = 0.0
    last_pong_time: float = 0.0

@dataclass
class MessageHeader:
    """Network message header structure"""
    magic: bytes = b'RAYX'  # Network magic number
    command: bytes = b'\x00' * 12  # 12-byte command name
    length: int = 0  # Payload length
    checksum: bytes = b'\x00' * 4  # First 4 bytes of sha256(sha256(payload))
    version: int = 1  # Protocol version

@dataclass
class RateLimitData:
    """Rate limiting data structure"""
    message_count: int = 0
    last_reset: float = field(default_factory=time.time)
    bytes_sent: int = 0
    bytes_received: int = 0
    last_penalty: float = 0.0
    penalty_level: int = 0

@dataclass
class DHTNode:
    """DHT node information"""
    node_id: str
    address: str
    port: int
    last_seen: float = field(default_factory=time.time)
    distance: int = 0  # XOR distance from our node ID

@dataclass
class RoutingTable:
    """Kademlia routing table structure"""
    buckets: List[List[DHTNode]] = field(default_factory=list)
    size: int = 20  # K-bucket size
    last_refresh: float = field(default_factory=time.time)