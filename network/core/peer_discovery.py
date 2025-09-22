import asyncio
import logging
import random
import time
from typing import List, Dict, Any
from network.interfaces.discovery_interface import IPeerDiscovery
from network.exceptions import NetworkError
from network.models.peer_info import PeerInfo
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType, ProtocolType

logger = logging.getLogger("PeerDiscovery")

class PeerDiscovery(IPeerDiscovery):
    """Peer discovery implementation"""
    
    def __init__(self, network):
        self.network = network
        self.discovered_peers: Dict[str, PeerInfo] = {}
        self.last_discovery_time = 0
    
    async def discover_peers(self) -> List[Dict]:
        """Discover new peers"""
        discovered = []
        
        try:
            # DNS-based discovery
            dns_peers = await self._discover_via_dns()
            discovered.extend(dns_peers)
            
            # DHT-based discovery (if enabled)
            if self.network.config.enable_dht:
                dht_peers = await self._discover_via_dht()
                discovered.extend(dht_peers)
            
            # Gossip-based discovery
            gossip_peers = await self._discover_via_gossip()
            discovered.extend(gossip_peers)
            
            # Bootstrap nodes
            bootstrap_peers = await self._discover_via_bootstrap()
            discovered.extend(bootstrap_peers)
            
            self.last_discovery_time = time.time()
            logger.info(f"Discovered {len(discovered)} new peers")
            
        except Exception as e:
            logger.error(f"Peer discovery error: {e}")
        
        return discovered
    
    async def _discover_via_dns(self) -> List[Dict]:
        """Discover peers via DNS seeds"""
        discovered = []
        
        for dns_seed in self.network.config.dns_seeds:
            try:
                # In a real implementation, this would do DNS lookups
                # For now, we'll simulate some peers
                if "mainnet" in dns_seed:
                    discovered.append({"address": "1.2.3.4", "port": 30303, "protocol": "tcp"})
                    discovered.append({"address": "5.6.7.8", "port": 30303, "protocol": "tcp"})
                elif "testnet" in dns_seed:
                    discovered.append({"address": "9.10.11.12", "port": 30304, "protocol": "tcp"})
                
            except Exception as e:
                logger.error(f"DNS discovery error for {dns_seed}: {e}")
        
        return discovered
    
    async def _discover_via_dht(self) -> List[Dict]:
        """Discover peers via DHT"""
        discovered = []
        
        try:
            # In a real implementation, this would use a DHT library
            # For now, we'll simulate some DHT peers
            for i in range(3):
                discovered.append({
                    "address": f"192.168.{i}.{i+1}",
                    "port": 30303 + i,
                    "protocol": random.choice(["tcp", "udp"])
                })
                
        except Exception as e:
            logger.error(f"DHT discovery error: {e}")
        
        return discovered
    
    async def _discover_via_gossip(self) -> List[Dict]:
        """Discover peers via gossip protocol"""
        discovered = []
        
        try:
            # Request peer lists from connected peers
            peer_list_message = NetworkMessage(
                message_id=f"peer_req_{time.time()}",
                message_type=MessageType.PEER_LIST,
                payload={"request": True},
                source_node=self.network.node_id
            )
            
            # Broadcast request
            await self.network.broadcast_message(peer_list_message)
            
            # Responses will be handled by the message processor
            
        except Exception as e:
            logger.error(f"Gossip discovery error: {e}")
        
        return discovered
    
    async def _discover_via_bootstrap(self) -> List[Dict]:
        """Discover peers via bootstrap nodes"""
        discovered = []
        
        for bootstrap_node in self.network.config.bootstrap_nodes:
            try:
                # Parse bootstrap node address
                if '://' in bootstrap_node:
                    protocol, address = bootstrap_node.split('://')
                    if ':' in address:
                        addr, port = address.split(':')
                        discovered.append({
                            "address": addr,
                            "port": int(port),
                            "protocol": protocol
                        })
                else:
                    if ':' in bootstrap_node:
                        addr, port = bootstrap_node.split(':')
                        discovered.append({
                            "address": addr,
                            "port": int(port),
                            "protocol": "tcp"
                        })
                
            except Exception as e:
                logger.error(f"Bootstrap node error for {bootstrap_node}: {e}")
        
        return discovered
    
    async def bootstrap_network(self):
        """Bootstrap to the network"""
        logger.info("Bootstrapping network...")
        
        try:
            # Discover initial peers
            discovered_peers = await self.discover_peers()
            
            # Connect to discovered peers
            connected_count = 0
            for peer in discovered_peers:
                if connected_count >= self.network.config.max_connections:
                    break
                
                try:
                    connection_id = await self.network.connection_manager.connect_to_peer(
                        peer["address"], peer["port"], peer["protocol"]
                    )
                    
                    if connection_id:
                        connected_count += 1
                        logger.debug(f"Connected to bootstrap peer: {peer['address']}:{peer['port']}")
                        
                except Exception as e:
                    logger.debug(f"Failed to connect to bootstrap peer {peer['address']}:{e}")
            
            logger.info(f"Bootstrapped with {connected_count} peers")
            
        except Exception as e:
            logger.error(f"Network bootstrap failed: {e}")
            raise NetworkError(f"Bootstrap failed: {e}")
    
    async def maintain_peer_list(self):
        """Maintain and update peer list"""
        try:
            # Remove stale peers
            current_time = time.time()
            stale_peers = []
            
            for peer_id, peer_info in self.discovered_peers.items():
                if current_time - peer_info.last_seen > 3600:  # 1 hour
                    stale_peers.append(peer_id)
            
            for peer_id in stale_peers:
                del self.discovered_peers[peer_id]
            
            # Discover new peers if needed
            if (len(self.discovered_peers) < self.network.config.max_peers and
                current_time - self.last_discovery_time > 300):  # 5 minutes
                await self.discover_peers()
                
        except Exception as e:
            logger.error(f"Peer list maintenance error: {e}")
    
    async def request_peer_lists(self):
        """Request peer lists from connected peers"""
        try:
            if not self.network.connection_manager.is_connected():
                return
            
            # Create peer list request message
            peer_request = NetworkMessage(
                message_id=f"peer_req_{time.time()}",
                message_type=MessageType.PEER_LIST,
                payload={"request": True},
                source_node=self.network.node_id
            )
            
            # Send to a few random peers (not all to avoid flooding)
            connected_peers = list(self.network.connections.keys())
            if connected_peers:
                random_peers = random.sample(connected_peers, min(3, len(connected_peers)))
                
                for peer_id in random_peers:
                    try:
                        await self.network.message_processor.send_message(peer_id, peer_request)
                    except Exception as e:
                        logger.debug(f"Failed to send peer request to {peer_id}: {e}")
                        
        except Exception as e:
            logger.error(f"Peer list request error: {e}")
    
    def add_peer(self, peer_info: PeerInfo):
        """Add a peer to the discovered list"""
        peer_id = f"{peer_info.address}_{peer_info.port}"
        
        if peer_id not in self.discovered_peers:
            self.discovered_peers[peer_id] = peer_info
            logger.debug(f"Added new peer: {peer_info.address}:{peer_info.port}")
        else:
            # Update existing peer
            existing = self.discovered_peers[peer_id]
            existing.last_seen = peer_info.last_seen
            existing.reputation = peer_info.reputation
            existing.capabilities = peer_info.capabilities
    
    def get_best_peers(self, count: int = 10) -> List[PeerInfo]:
        """Get best peers by reputation"""
        peers = list(self.discovered_peers.values())
        peers.sort(key=lambda x: x.reputation, reverse=True)
        return peers[:count]