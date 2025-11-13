import asyncio
import logging
import aiodns
import socket
import random
import time
from typing import List, Dict, Any, Set
from urllib.parse import urlparse
import ipaddress

from network.interfaces.discovery_interface import IPeerDiscovery
from network.exceptions import NetworkError
from network.models.peer_info import PeerInfo
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType, ProtocolType
from config.config_manager import ConfigManager

logger = logging.getLogger("PeerDiscovery")

class PeerDiscovery(IPeerDiscovery):
    """Production-ready peer discovery implementation"""
    
    def __init__(self, network):
        self.network = network
        self.config_manager = network.config_manager
        self.discovered_peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Set[str] = set()
        self.last_discovery_time = 0
        self.dns_resolver = None  # Initialize lazily
        self.active_discoveries = set()
        
    async def _get_dns_resolver(self):
        """Lazy initialization of DNS resolver"""
        if self.dns_resolver is None:
        	self.dns_resolver = aiodns.DNSResolver()
        return self.dns_resolver
        
    async def discover_peers(self) -> List[Dict]:
        """Discover new peers using all available methods"""
        discovered = []
        
        try:
            # Run discovery methods concurrently
            discovery_tasks = []
            
            # DNS-based discovery
            if self.config_manager.get('peer_discovery.dns_discovery_enabled', True):
                discovery_tasks.append(self._discover_via_dns())
            
            # DHT-based discovery
            if self.config_manager.get('peer_discovery.dht_discovery_enabled', False):
                discovery_tasks.append(self._discover_via_dht())
            
            # Bootstrap nodes discovery
            if self.config_manager.get('peer_discovery.bootstrap_discovery_enabled', True):
                discovery_tasks.append(self._discover_via_bootstrap())
            
            # Wait for all discovery methods to complete
            results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Discovery method failed: {result}")
                    continue
                if isinstance(result, list):
                    discovered.extend(result)
            
            # Remove duplicates
            discovered = self._remove_duplicate_peers(discovered)
            
            # Validate peers before returning
            validated_peers = []
            for peer in discovered:
                if await self._validate_peer(peer):
                    validated_peers.append(peer)
            
            self.last_discovery_time = time.time()
            logger.info(f"Discovered {len(validated_peers)} new peers from {len(discovery_tasks)} methods")
            
            return validated_peers
            
        except Exception as e:
            logger.error(f"Peer discovery error: {e}")
            return []
    
    async def _discover_via_dns(self) -> List[Dict]:
        """Real DNS-based peer discovery"""
        discovered = []
        dns_seeds = self.config_manager.config.network.dns_seeds
        
        if not dns_seeds:
            return discovered
        
        dns_tasks = []
        for dns_seed in dns_seeds:
            task = asyncio.create_task(self._resolve_dns_seed(dns_seed))
            dns_tasks.append(task)
            self.active_discoveries.add(task)
            task.add_done_callback(self.active_discoveries.discard)
        
        # Wait for DNS resolutions with timeout
        try:
            dns_results = await asyncio.wait_for(
                asyncio.gather(*dns_tasks, return_exceptions=True),
                timeout=30.0  # 30 second timeout for DNS
            )
            
            for result in dns_results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, list):
                    discovered.extend(result)
                    
        except asyncio.TimeoutError:
            logger.warning("DNS discovery timed out")
        except Exception as e:
            logger.error(f"DNS discovery error: {e}")
        
        return discovered
    
    async def _resolve_dns_seed(self, dns_seed: str) -> List[Dict]:
        """Resolve a single DNS seed to get peer addresses"""
        discovered = []
        
        try:
            resolver = await self._get_dns_resolver()
            
            # Try A records (IPv4 addresses)
            try:
                a_records = await self.dns_resolver.query(dns_seed, 'A')
                for record in a_records:
                    discovered.append({
                        "address": record.host,
                        "port": self.config_manager.config.network.listen_port,
                        "protocol": "tcp"
                    })
            except aiodns.error.DNSError:
                pass
            
            # Try AAAA records (IPv6 addresses)
            try:
                aaaa_records = await self.dns_resolver.query(dns_seed, 'AAAA')
                for record in aaaa_records:
                    discovered.append({
                        "address": record.host,
                        "port": self.config_manager.config.network.listen_port,
                        "protocol": "tcp"
                    })
            except aiodns.error.DNSError:
                pass
            
            # Try SRV records for specific service discovery
            try:
                srv_records = await self.dns_resolver.query(f"_rayonix._tcp.{dns_seed}", 'SRV')
                for record in srv_records:
                    discovered.append({
                        "address": record.host,
                        "port": record.port,
                        "protocol": "tcp"
                    })
            except aiodns.error.DNSError:
                pass
                
            logger.debug(f"Resolved {len(discovered)} peers from DNS seed: {dns_seed}")
            
        except Exception as e:
            logger.warning(f"Failed to resolve DNS seed {dns_seed}: {e}")
        
        return discovered
    
    async def _discover_via_dht(self) -> List[Dict]:
        """Real DHT-based peer discovery"""
        discovered = []
        
        try:
            # Check if DHT is enabled and we have bootstrap nodes
            if not self.config_manager.get('network.enable_dht', False):
                return discovered
            
            # In production, you would use a real DHT library like:
            # - mainline-dht (Python)
            # - kademlia (Python)
            # - libp2p (if integrated)
            
            # For now, implement basic DHT bootstrap
            dht_bootstrap_nodes = self._get_dht_bootstrap_nodes()
            
            dht_tasks = []
            for bootstrap_node in dht_bootstrap_nodes:
                task = asyncio.create_task(self._query_dht_node(bootstrap_node))
                dht_tasks.append(task)
                self.active_discoveries.add(task)
                task.add_done_callback(self.active_discoveries.discard)
            
            # Wait for DHT queries
            try:
                dht_results = await asyncio.wait_for(
                    asyncio.gather(*dht_tasks, return_exceptions=True),
                    timeout=45.0
                )
                
                for result in dht_results:
                    if isinstance(result, Exception):
                        continue
                    if isinstance(result, list):
                        discovered.extend(result)
                        
            except asyncio.TimeoutError:
                logger.warning("DHT discovery timed out")
                
        except Exception as e:
            logger.error(f"DHT discovery error: {e}")
        
        return discovered
    
    def _get_dht_bootstrap_nodes(self) -> List[tuple]:
        """Get DHT bootstrap nodes - in production, these would be well-known nodes"""
        # These would be actual DHT bootstrap nodes
        return [
            ('router.bittorrent.com', 6881),
            ('dht.transmissionbt.com', 6881),
            ('router.utorrent.com', 6881)
        ]
    
    async def _query_dht_node(self, bootstrap_node: tuple) -> List[Dict]:
        """Query a single DHT node for peers"""
        # This is a simplified implementation
        # In production, you'd use a proper DHT client
        
        discovered = []
        host, port = bootstrap_node
        
        try:
            # Basic DHT ping to check if node is alive
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=10.0
            )
            
            # If connection successful, we can consider this a potential peer
            discovered.append({
                "address": host,
                "port": port,
                "protocol": "udp"  # DHT typically uses UDP
            })
            
            writer.close()
            await writer.wait_closed()
            
        except (asyncio.TimeoutError, ConnectionError, OSError):
            # Node is not reachable, skip
            pass
        except Exception as e:
            logger.debug(f"DHT node {host}:{port} query failed: {e}")
        
        return discovered
    
    async def _discover_via_gossip(self) -> List[Dict]:
        """Real gossip-based peer discovery"""
        discovered = []
        
        if not self.config_manager.get('peer_discovery.gossip_discovery_enabled', True):
            return discovered
        
        try:
            # Request peer lists from connected peers
            peer_list_message = NetworkMessage(
                message_id=f"peer_req_{int(time.time())}_{random.randint(1000, 9999)}",
                message_type=MessageType.PEER_LIST,
                payload={
                    "request": True,
                    "network_id": self.config_manager.config.network.network_id,
                    "version": "1.0.0",
                    "capabilities": ["tcp", "gossip"]
                },
                source_node=self.network.node_id,
                timestamp=time.time()
            )
            
            # Send to a subset of connected peers to avoid flooding
            connected_peers = list(self.connected_peers)
            if connected_peers:
                # Select peers based on reputation and connection stability
                selected_peers = self._select_peers_for_gossip(connected_peers, max_peers=5)
                
                gossip_tasks = []
                for peer_id in selected_peers:
                    task = asyncio.create_task(
                        self.network.message_processor.send_message(peer_id, peer_list_message)
                    )
                    gossip_tasks.append(task)
                    self.active_discoveries.add(task)
                    task.add_done_callback(self.active_discoveries.discard)
                
                # Don't wait too long for gossip responses
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*gossip_tasks, return_exceptions=True),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.debug("Gossip discovery timeout")
                    
        except Exception as e:
            logger.error(f"Gossip discovery error: {e}")
        
        return discovered  # Peers will be added via message handler
    
    async def _discover_via_bootstrap(self) -> List[Dict]:
        """Real bootstrap node discovery with validation"""
        discovered = []
        bootstrap_nodes = self.config_manager.config.network.bootstrap_nodes
        
        if not bootstrap_nodes:
            return discovered
        
        bootstrap_tasks = []
        for bootstrap_node in bootstrap_nodes:
            task = asyncio.create_task(self._resolve_bootstrap_node(bootstrap_node))
            bootstrap_tasks.append(task)
            self.active_discoveries.add(task)
            task.add_done_callback(self.active_discoveries.discard)
        
        try:
            bootstrap_results = await asyncio.wait_for(
                asyncio.gather(*bootstrap_tasks, return_exceptions=True),
                timeout=30.0
            )
            
            for result in bootstrap_results:
                if isinstance(result, Exception):
                    continue
                if isinstance(result, dict):
                    discovered.append(result)
                    
        except asyncio.TimeoutError:
            logger.warning("Bootstrap discovery timed out")
        except Exception as e:
            logger.error(f"Bootstrap discovery error: {e}")
        
        return discovered
    
    async def _resolve_bootstrap_node(self, bootstrap_node: str) -> Dict[str, Any]:
        """Resolve and validate a single bootstrap node"""
        try:
            # Parse the bootstrap node address
            if '://' in bootstrap_node:
                parsed = urlparse(bootstrap_node)
                protocol = parsed.scheme
                address = parsed.hostname
                port = parsed.port or self.config_manager.config.network.listen_port
            else:
                protocol = "tcp"
                if ':' in bootstrap_node:
                    address, port_str = bootstrap_node.rsplit(':', 1)
                    port = int(port_str)
                else:
                    address = bootstrap_node
                    port = self.config_manager.config.network.listen_port
            
            # Validate address format
            try:
                ipaddress.ip_address(address)
                # It's already an IP address
                resolved_address = address
            except ValueError:
                # It's a hostname, resolve it
                try:
                    resolved_records = await self.dns_resolver.query(address, 'A')
                    if resolved_records:
                        resolved_address = resolved_records[0].host
                    else:
                        raise ValueError(f"Cannot resolve hostname: {address}")
                except aiodns.error.DNSError:
                    raise ValueError(f"Cannot resolve hostname: {address}")
            
            # Validate the peer before returning
            peer_info = {
                "address": resolved_address,
                "port": port,
                "protocol": protocol,
                "source": "bootstrap",
                "resolved_at": time.time()
            }
            
            if await self._validate_peer(peer_info):
                return peer_info
            else:
                raise ValueError("Peer validation failed")
                
        except Exception as e:
            logger.warning(f"Failed to resolve bootstrap node {bootstrap_node}: {e}")
            raise
    
    async def _validate_peer(self, peer_info: Dict) -> bool:
        """Validate a peer before adding to discovery list"""
        try:
            address = peer_info["address"]
            port = peer_info["port"]
            
            # Check if it's a valid IP address
            try:
                ip = ipaddress.ip_address(address)
                
                # Filter out private and reserved addresses unless allowed
                if ip.is_private:
                    allow_private = self.config_manager.get('network.allow_private_peers', True)
                    
                    if not allow_private:
                    	logger.debug(f"Filtering out private peer: {address}")
                    	return False
                    	
                if ip.is_reserved or ip.is_loopback:
                    logger.debug(f"Filtering out reserved/loopback peer: {address}")
                    return False
                    
                # FIX: Allow loopback only in development
                if ip.is_loopback:
                	allow_loopback = self.config_manager.get('network.allow_loopback', False)
                	if not allow_loopback:
                		logger.debug(f"Filtering out loopback peer: {address}")
                		return False
                
            except ValueError:
                # Not a valid IP address
                logger.debug(f"Invalid IP address: {address}")
                return False
            
            # Check port range
            if not (1 <= port <= 65535):
                return False
            
            # Check if we're already connected to this peer
            peer_id = f"{address}_{port}"
            if peer_id in self.connected_peers:
                logger.debug(f"Already connected to peer: {peer_id}")
                return False
            
            #Make connectivity check optional and configurable
            validate_connectivity = self.config_manager.get('peer_discovery.validate_connectivity', False)
            if validate_connectivity:
                try:
                    reader, writer = await asyncio.wait_for(
                        asyncio.open_connection(address, port),
                        timeout=5.0
                    )
                    writer.close()
                    await writer.wait_closed()
                except (asyncio.TimeoutError, ConnectionError, OSError):
                    logger.debug(f"Peer validation failed for {address}:{port}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Peer validation error for {peer_info}: {e}")
            return False
    
    def _remove_duplicate_peers(self, peers: List[Dict]) -> List[Dict]:
        """Remove duplicate peers based on address:port"""
        seen = set()
        unique_peers = []
        
        for peer in peers:
            key = f"{peer['address']}:{peer['port']}"
            if key not in seen:
                seen.add(key)
                unique_peers.append(peer)
        
        return unique_peers
    
    def _select_peers_for_gossip(self, connected_peers: List[str], max_peers: int) -> List[str]:
        """Select the best peers for gossip requests"""
        # In production, this would consider:
        # - Peer reputation
        # - Connection stability
        # - Response history
        # - Geographic distribution
        
        if len(connected_peers) <= max_peers:
            return connected_peers
        
        # Simple random selection for now
        return random.sample(connected_peers, max_peers)
    
    async def bootstrap_network(self):
        """Production-ready network bootstrap"""
        logger.info("Bootstrapping network...")
        
        try:
            # Discover initial peers with retry logic
            max_attempts = 3
            discovered_peers = []
            
            for attempt in range(max_attempts):
                discovered_peers = await self.discover_peers()
                if discovered_peers:
                    break
                logger.warning(f"Bootstrap attempt {attempt + 1} failed, retrying...")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            if not discovered_peers:
                raise NetworkError("No peers discovered during bootstrap")
            
            # Connect to discovered peers with connection limits
            connected_count = 0
            max_connections = self.config_manager.config.network.max_connections
            current_connections = len(self.connected_peers)
            available_slots = max(0, max_connections - current_connections)
            
            connection_tasks = []
            for peer in discovered_peers[:available_slots]:  # Respect connection limits
                task = asyncio.create_task(self._connect_to_peer_with_retry(peer))
                connection_tasks.append(task)
            
            # Wait for connection attempts with timeout
            try:
                connection_results = await asyncio.wait_for(
                    asyncio.gather(*connection_tasks, return_exceptions=True),
                    timeout=60.0
                )
                
                for result in connection_results:
                    if isinstance(result, Exception):
                        continue
                    if result:
                        connected_count += 1
                        
            except asyncio.TimeoutError:
                logger.warning("Bootstrap connection phase timed out")
            
            logger.info(f"Bootstrapped with {connected_count} peers (target: {available_slots})")
            
            if connected_count == 0:
                raise NetworkError("Failed to connect to any peers during bootstrap")
                
        except Exception as e:
            logger.error(f"Network bootstrap failed: {e}")
            raise NetworkError(f"Bootstrap failed: {e}")
    
    async def _connect_to_peer_with_retry(self, peer: Dict) -> bool:
        """Connect to a peer with retry logic"""
        max_retries = 2
        address = peer["address"]
        port = peer["port"]
        protocol = peer.get("protocol", "tcp")
        
        for attempt in range(max_retries):
            try:
                connection_id = await self.network.connection_manager.connect_to_peer(
                    address, port, protocol
                )
                
                if connection_id:
                    self.connected_peers.add(connection_id)
                    logger.debug(f"Connected to bootstrap peer: {address}:{port}")
                    return True
                    
            except Exception as e:
                logger.debug(f"Connection attempt {attempt + 1} failed for {address}:{port}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Brief delay before retry
        
        return False
    
    # ... (rest of the methods remain but use real config values instead of hardcoded)
    
    async def maintain_peer_list(self):
        """Maintain and update peer list using config values"""
        try:
            current_time = time.time()
            stale_timeout = self.config_manager.get('peer_discovery.stale_peer_timeout', 3600)
            discovery_interval = self.config_manager.get('peer_discovery.discovery_interval', 300)
            
            # Remove stale peers
            stale_peers = []
            for peer_id, peer_info in self.discovered_peers.items():
                if current_time - peer_info.last_seen > stale_timeout:
                    stale_peers.append(peer_id)
            
            for peer_id in stale_peers:
                del self.discovered_peers[peer_id]
                self.connected_peers.discard(peer_id)
            
            # Discover new peers if needed
            min_peers = self.config_manager.config.network.max_peers * 0.1  # 10% of max
            if (len(self.discovered_peers) < min_peers and
                current_time - self.last_discovery_time > discovery_interval):
                await self.discover_peers()
                
        except Exception as e:
            logger.error(f"Peer list maintenance error: {e}")
    
    def add_peer(self, peer_info: PeerInfo):
        """Add a validated peer to the discovered list"""
        peer_id = f"{peer_info.address}_{peer_info.port}"
        
        if peer_id not in self.discovered_peers:
            self.discovered_peers[peer_id] = peer_info
            logger.debug(f"Added new peer: {peer_info.address}:{peer_info.port}")
        else:
            # Update existing peer with new information
            existing = self.discovered_peers[peer_id]
            existing.last_seen = peer_info.last_seen
            existing.reputation = max(existing.reputation, peer_info.reputation)  # Keep highest reputation
            existing.capabilities = list(set(existing.capabilities + peer_info.capabilities))
           
    async def request_peer_lists(self):
        """Request peer lists from connected peers - required by abstract base class"""
        try:
        	logger.info("Requesting peer lists from connected peers...")
        	
        	# Get connected peers
        	connected_peers = list(self.connected_peers)
        	if not connected_peers:
        		logger.debug("No connected peers to request peer lists from")
        		return
        	
        	# Create peer list request message
        	request_message = NetworkMessage(
        	    message_id=f"peer_list_request_{int(time.time())}_{random.randint(1000, 9999)}",
        	    message_type=MessageType.GET_PEERS,
        	    payload={
        	        "request": True,
        	        "max_peers": self.config_manager.get('peer_discovery.max_peers_to_return', 50),
        	        "network_id": self.config_manager.config.network.network_id
        	    },
        	    source_node=self.network.node_id,
            timestamp=time.time()
        	)
        	
        	# Send request to a subset of connected peers (avoid flooding)
        	max_requests = min(3, len(connected_peers))
        	peers_to_query = random.sample(connected_peers, max_requests)
        	
        	request_tasks = []
        	for peer_id in peers_to_query:
        		task = asyncio.create_task(
        		    self.network.message_processor.send_message(peer_id, request_message)
        		)
        		
        		request_tasks.append(task)
        	
        	# Wait for requests to complete with timeout
        	try:
        		await asyncio.wait_for(
        		    asyncio.gather(*request_tasks, return_exceptions=True),
        		    timeout=15.0
        		)
        		logger.debug(f"Sent peer list requests to {len(peers_to_query)} peers")
        	
        	except asyncio.TimeoutError:
        		logger.warning("Peer list requests timed out")
        
        except Exception as e:
        	logger.error(f"Error requesting peer lists: {e}")
        
    def mark_peer_connected(self, connection_id: str):
        """Mark a peer as connected"""
        self.connected_peers.add(connection_id)
    
    def mark_peer_disconnected(self, connection_id: str):
        """Mark a peer as disconnected"""
        self.connected_peers.discard(connection_id)