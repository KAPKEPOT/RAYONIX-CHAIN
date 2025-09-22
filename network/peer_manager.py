import asyncio
import time
import logging
from typing import Dict, List, Optional, Set, Tuple
import aiodns
from urllib.parse import urlparse
import random

from .models import PeerInfo, ProtocolType, NetworkType
from .exceptions import PeerBannedError
from .utils import NetworkUtils, ValidationUtils

logger = logging.getLogger("AdvancedNetwork")

class PeerManager:
    """Manages peer discovery, tracking, and scoring"""
    
    def __init__(self, config):
        self.config = config
        self.peers: Dict[str, PeerInfo] = {}
        self.banned_peers: Dict[str, float] = {}  # peer_id -> ban_until_timestamp
        self.whitelist: Set[str] = set()
        self.blacklist: Set[str] = set()
        self.dns_resolver = aiodns.DNSResolver()
        self.seen_gossip_messages: Dict[str, float] = {}  # message_hash -> expiration time
        self.peer_exchange_history: Dict[str, float] = {}  # peer_id -> last exchange time
    
    async def is_peer_banned(self, address: str) -> bool:
        """Check if peer is banned"""
        if address in self.banned_peers:
            if self.banned_peers[address] > time.time():
                return True
            else:
                # Ban expired
                del self.banned_peers[address]
        return False
    
    async def ban_peer(self, address: str, duration: int = None):
        """Ban a peer for specified duration"""
        ban_duration = duration or self.config.ban_duration
        self.banned_peers[address] = time.time() + ban_duration
        logger.warning(f"Banned peer {address} for {ban_duration} seconds")
    
    async def update_peer_reputation(self, connection_id: str, delta: int):
        """Update peer reputation"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        if connection_id in network.connection_manager.connections:
            address = network.connection_manager.connections[connection_id]['address'][0]
            await self.update_peer_reputation_by_address(address, delta)
    
    async def update_peer_reputation_by_address(self, address: str, delta: int):
        """Update peer reputation by address"""
        for peer_key, peer in self.peers.items():
            if peer.address == address:
                peer.reputation += delta
                
                # Auto-ban if reputation drops too low
                if peer.reputation <= self.config.ban_threshold:
                    await self.ban_peer(address)
                    logger.warning(f"Auto-banned peer {address} due to low reputation: {peer.reputation}")
                
                break
    
    async def add_peer_from_discovery(self, peer_info: Dict):
        """Add peer discovered through various methods"""
        try:
            address = peer_info.get('address')
            port = peer_info.get('port', self.config.listen_port)
            protocol_name = peer_info.get('protocol', 'TCP')
            node_id = peer_info.get('node_id', '')
            source = peer_info.get('source', 'unknown')
            
            if not address or await self.is_peer_banned(address):
                return
            
            # Validate protocol
            try:
                protocol = ProtocolType[protocol_name]
            except KeyError:
                protocol = ProtocolType.TCP
            
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
                    reputation=peer_info.get('reputation', 50),  # Initial reputation
                    source=source
                )
                logger.info(f"Discovered new peer: {address}:{port} via {source}")
            else:
                # Update existing peer
                self.peers[peer_key].last_seen = time.time()
                if node_id:
                    self.peers[peer_key].node_id = node_id
                if source != 'unknown':
                    self.peers[peer_key].source = source
                if network.dht:
                	await network.dht
                	await network.dht.add_peer_from_discovery(peer_info)
            
        except Exception as e:
            logger.error(f"Error adding discovered peer: {e}")
    
    def get_peer_by_address(self, address: str) -> Optional[PeerInfo]:
        """Get peer info by address"""
        for peer_key, peer in self.peers.items():
            if peer.address == address:
                return peer
        return None
    
    def get_best_peers(self, count: int = 10) -> List[PeerInfo]:
        """Get the best peers by reputation"""
        sorted_peers = sorted(
            self.peers.values(),
            key=lambda p: p.reputation,
            reverse=True
        )
        return sorted_peers[:count]
    
    async def discover_peers(self):
        """Discover new peers through various methods"""
        try:
            # DNS-based discovery
            if self.config.dns_seeds:
                await self.dns_discovery()
            
            # DHT-based discovery
            if self.config.enable_dht:
                await self.dht_discovery()
            
            # Request peer lists from connected peers
            if self.config.enable_peer_exchange:
                await self.request_peer_lists()
            
            # Clean up old peers
            await self.cleanup_peers()
            
        except Exception as e:
            logger.error(f"Peer discovery error: {e}")
    
    async def dns_discovery(self):
        """Discover peers through DNS seeds"""
        for dns_seed in self.config.dns_seeds:
            try:
                answers = await self.dns_resolver.query(dns_seed, 'A')
                for answer in answers:
                    await self.add_peer_from_discovery({
                        'address': answer.host,
                        'port': self.config.listen_port,
                        'protocol': ProtocolType.TCP.name,
                        'source': 'dns'
                    })
            except Exception as e:
                logger.error(f"DNS discovery failed for {dns_seed}: {e}")
    
    async def dht_discovery(self):
        """Discover peers through Distributed Hash Table"""
        try:
            # Query bootstrap nodes
            for bootstrap_node in self.config.dht_bootstrap_nodes:
                address, port = bootstrap_node
                discovered_peers = await self._dht_lookup(address, port)
                for peer_info in discovered_peers:
                    await self.add_peer_from_discovery(peer_info)
                    
        except Exception as e:
            logger.error(f"DHT discovery error: {e}")
    
    async def _dht_lookup(self, address: str, port: int) -> List[Dict]:
        """Perform DHT lookup for peers"""
        # This would be implemented with a real Kademlia DHT
        # For now, return empty list
        return []
    
    async def request_peer_lists(self):
        """Request peer lists from connected peers"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        # Only request from peers we haven't recently exchanged with
        current_time = time.time()
        eligible_peers = [
            conn_id for conn_id in network.connection_manager.connections
            if self.peer_exchange_history.get(conn_id, 0) < current_time - self.config.peer_exchange_interval
        ]
        
        if eligible_peers:
            peer_list_request = {
                'message_id': NetworkUtils.generate_node_id(),
                'message_type': 'PEER_LIST',
                'payload': {'request': True}
            }
            
            # Send to a random eligible peer
            target_peer = random.choice(eligible_peers)
            self.peer_exchange_history[target_peer] = current_time
            
            await network.send_message(target_peer, peer_list_request)
    
    async def cleanup_peers(self):
        """Clean up old and low-reputation peers"""
        current_time = time.time()
        peers_to_remove = []
        
        for peer_key, peer in self.peers.items():
            # Remove very old peers
            if current_time - peer.last_seen > self.config.max_peer_age:
                peers_to_remove.append(peer_key)
            # Remove very low reputation peers
            elif peer.reputation < self.config.min_peer_reputation:
                peers_to_remove.append(peer_key)
        
        for peer_key in peers_to_remove:
            del self.peers[peer_key]
        
        # Clean up old gossip messages
        self._cleanup_seen_gossip_messages()
    
    def _cleanup_seen_gossip_messages(self):
        """Clean up old seen gossip messages"""
        current_time = time.time()
        messages_to_remove = [
            msg_hash for msg_hash, expiry in self.seen_gossip_messages.items()
            if expiry < current_time
        ]
        
        for msg_hash in messages_to_remove:
            del self.seen_gossip_messages[msg_hash]
    
    async def has_seen_gossip_message(self, message_hash: str) -> bool:
        """Check if we've seen a gossip message recently"""
        return message_hash in self.seen_gossip_messages
    
    async def mark_gossip_message_seen(self, message_hash: str, ttl: int):
        """Mark a gossip message as seen"""
        # Set expiration based on TTL (add some buffer)
        self.seen_gossip_messages[message_hash] = time.time() + (ttl * 2) + 60
    
    async def bootstrap_network(self):
        """Bootstrap to the network"""
        logger.info("Bootstrapping to network...")
        
        # Connect to bootstrap nodes
        successful_bootstraps = 0
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
                
                from .core import AdvancedP2PNetwork
                network = AdvancedP2PNetwork.instance()
                result = await network.connection_manager.connect_to_peer(address, port, protocol)
                
                if result:
                    successful_bootstraps += 1
                    # Add to peer list
                    await self.add_peer_from_discovery({
                        'address': address,
                        'port': port,
                        'protocol': protocol.name,
                        'source': 'bootstrap'
                    })
                
            except Exception as e:
                logger.error(f"Failed to bootstrap to {bootstrap_node}: {e}")
        
        logger.info(f"Bootstrap completed: {successful_bootstraps} successful connections")
    
    def get_peer_stats(self) -> Dict:
        """Get peer statistics"""
        stats = {
            'total': len(self.peers),
            'banned': len(self.banned_peers),
            'by_protocol': defaultdict(int),
            'by_source': defaultdict(int),
            'reputation_distribution': defaultdict(int)
        }
        
        for peer in self.peers.values():
            stats['by_protocol'][peer.protocol.name] += 1
            stats['by_source'][peer.source] += 1
            
            # Group reputation scores
            if peer.reputation >= 80:
                stats['reputation_distribution']['excellent'] += 1
            elif peer.reputation >= 60:
                stats['reputation_distribution']['good'] += 1
            elif peer.reputation >= 40:
                stats['reputation_distribution']['fair'] += 1
            elif peer.reputation >= 20:
                stats['reputation_distribution']['poor'] += 1
            else:
                stats['reputation_distribution']['bad'] += 1
        
        return stats