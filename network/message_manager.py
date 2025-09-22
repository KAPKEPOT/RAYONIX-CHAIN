import asyncio
import time
import logging
from typing import Dict, List, Callable, Any, Deque, Optional
from collections import defaultdict, deque

from .models import NetworkMessage, MessageType, ConnectionMetrics
from .exceptions import MessageError, SerializationError
from .utils import SerializationUtils, CompressionUtils, NetworkUtils

logger = logging.getLogger("AdvancedNetwork")

class MessageManager:
    """Manages message processing, serialization, and routing"""
    
    def __init__(self, config):
        self.config = config
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.pending_requests: Dict[str, asyncio.Future] = {}
        
        # Priority queues for message processing
        self.priority_queues: Dict[int, asyncio.Queue] = {
            0: asyncio.Queue(maxsize=1000),  # Low priority
            1: asyncio.Queue(maxsize=2000),  # Normal priority
            2: asyncio.Queue(maxsize=500),   # High priority
            3: asyncio.Queue(maxsize=100)    # Critical priority
        }
        
        # Message processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'last_reset': time.time(),
            'errors': 0,
            'by_type': defaultdict(int)
        }
    
    def serialize_message(self, message: Any) -> bytes:
        """Serialize message to bytes"""
        try:
            return SerializationUtils.serialize_message(message)
        except Exception as e:
            logger.error(f"Message serialization error: {e}")
            raise SerializationError(f"Serialization failed: {e}")
    
    def deserialize_message(self, data: bytes) -> Any:
        """Deserialize message from bytes"""
        try:
            return SerializationUtils.deserialize_message(data)
        except Exception as e:
            logger.error(f"Message deserialization error: {e}")
            raise SerializationError(f"Deserialization failed: {e}")
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress data if enabled"""
        if not self.config.enable_compression:
            return data
        return CompressionUtils.compress_data(data)
    
    def decompress_data(self, data: bytes) -> bytes:
        """Decompress data if enabled"""
        if not self.config.enable_compression:
            return data
        return CompressionUtils.decompress_data(data)
    
    async def process_message(self, connection_id: str, raw_data: bytes, 
                            security_manager, metrics: ConnectionMetrics) -> bool:
        """Process incoming message"""
        try:
            # Parse message header
            header, payload = NetworkUtils.parse_message_header(raw_data)
            
            # Verify magic number
            if header['magic'] != security_manager.magic:
                logger.warning(f"Invalid magic number from {connection_id}")
                return False
            
            # Verify checksum
            expected_checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
            if header['checksum'] != expected_checksum:
                logger.warning(f"Invalid checksum from {connection_id}")
                return False
            
            # Check message size
            if len(payload) > self.config.max_message_size:
                logger.warning(f"Oversized message from {connection_id}")
                return False
            
            # Decrypt if encryption enabled
            if self.config.enable_encryption:
                try:
                    payload = security_manager.decrypt_data(payload, connection_id)
                except Exception as e:
                    logger.error(f"Decryption failed for {connection_id}: {e}")
                    return False
            
            # Decompress if compression enabled
            if self.config.enable_compression:
                try:
                    payload = self.decompress_data(payload)
                except Exception as e:
                    logger.error(f"Decompression failed for {connection_id}: {e}")
                    return False
            
            # Deserialize message
            try:
                message_data = self.deserialize_message(payload)
            except Exception as e:
                logger.error(f"Deserialization failed for {connection_id}: {e}")
                return False
            
            # Create NetworkMessage object
            message = NetworkMessage(
                message_id=message_data.get('message_id', NetworkUtils.generate_node_id()),
                message_type=MessageType[message_data.get('message_type', 'PING')],
                payload=message_data.get('payload', {}),
                timestamp=message_data.get('timestamp', time.time()),
                ttl=message_data.get('ttl', 10),
                signature=message_data.get('signature'),
                source_node=message_data.get('source_node'),
                destination_node=message_data.get('destination_node'),
                priority=message_data.get('priority', 1),
                nonce=message_data.get('nonce')
            )
            
            # Update metrics
            metrics.messages_received += 1
            metrics.bytes_received += len(raw_data)
            metrics.last_activity = time.time()
            
            # Add to appropriate priority queue
            try:
                await self.priority_queues[message.priority].put((connection_id, message))
                self.processing_stats['total_processed'] += 1
                self.processing_stats['by_type'][message.message_type] += 1
                return True
            except asyncio.QueueFull:
                logger.warning(f"Priority queue {message.priority} is full, dropping message")
                return False
            
        except Exception as e:
            logger.error(f"Message processing error for {connection_id}: {e}")
            self.processing_stats['errors'] += 1
            return False
    
    async def prepare_message_for_sending(self, message: NetworkMessage, 
                                         connection_id: str, security_manager) -> bytes:
        """Prepare message for sending (serialize, compress, encrypt)"""
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
                'priority': message.priority,
                'nonce': message.nonce or NetworkUtils.generate_nonce()
            }
            
            # Serialize message
            serialized = self.serialize_message(message_dict)
            
            # Compress if enabled
            if self.config.enable_compression:
                serialized = self.compress_data(serialized)
            
            # Encrypt if enabled
            if self.config.enable_encryption:
                serialized = security_manager.encrypt_data(serialized, connection_id)
            
            # Create message header
            header = NetworkUtils.create_message_header(serialized, security_manager.magic)
            
            return header + serialized
            
        except Exception as e:
            logger.error(f"Message preparation error: {e}")
            raise MessageError(f"Failed to prepare message: {e}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type].append(handler)
    
    def unregister_message_handler(self, message_type: MessageType, handler: Callable):
        """Unregister message handler"""
        if message_type in self.message_handlers:
            self.message_handlers[message_type] = [
                h for h in self.message_handlers[message_type] if h != handler
            ]
    
    async def handle_message(self, connection_id: str, message: NetworkMessage):
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
            handler_method = getattr(self, f"_handle_{message.message_type.name.lower()}", None)
            if handler_method:
                await handler_method(connection_id, message)
            
        except Exception as e:
            logger.error(f"Message handling error for {connection_id}: {e}")
            self.processing_stats['errors'] += 1
    
    async def _handle_ping(self, connection_id: str, message: NetworkMessage):
        """Handle ping message"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        pong_message = NetworkMessage(
            message_id=NetworkUtils.generate_node_id(),
            message_type=MessageType.PONG,
            payload={
                'timestamp': time.time(), 
                'original_ping_id': message.message_id,
                'node_id': network.config.node_id
            }
        )
        
        await network.send_message(connection_id, pong_message)
    
    async def _handle_pong(self, connection_id: str, message: NetworkMessage):
        """Handle pong message"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        if connection_id in network.connection_manager.connections:
            # Update latency
            latency = time.time() - message.payload['timestamp']
            network.connection_manager.connections[connection_id]['metrics'].latency_history.append(latency)
            network.connection_manager.connections[connection_id]['metrics'].last_pong_time = time.time()
            
            # Update peer reputation
            await network.peer_manager.update_peer_reputation(connection_id, 1)
    
    async def _handle_peer_list(self, connection_id: str, message: NetworkMessage):
        """Handle peer list message"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        try:
            peers = message.payload.get('peers', [])
            valid_peers = 0
            
            for peer_info in peers:
                if network.peer_manager.validate_peer_info(peer_info):
                    peer_id = peer_info.get('node_id')
                    if peer_id and peer_id != network.config.node_id:
                        # Add to peer discovery
                        await network.peer_manager.add_peer_from_discovery(peer_info)
                        valid_peers += 1
            
            # Update reputation for sharing peers
            if valid_peers > 0:
                reputation_bonus = min(valid_peers * 2, 10)  # Max 10 points
                await network.peer_manager.update_peer_reputation(connection_id, reputation_bonus)
                logger.info(f"Added {valid_peers} new peers from {connection_id}")
            
        except Exception as e:
            logger.error(f"Peer list handling error: {e}")
    
    async def _handle_handshake(self, connection_id: str, message: NetworkMessage):
        """Handle handshake message"""
        # Already handled during connection establishment
        pass
    
    async def _handle_sync_request(self, connection_id: str, message: NetworkMessage):
        """Handle sync request"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        try:
            # Process sync request and prepare response
            sync_response = NetworkMessage(
                message_id=NetworkUtils.generate_node_id(),
                message_type=MessageType.SYNC_RESPONSE,
                payload={
                    'status': 'success', 
                    'data': [],
                    'request_id': message.message_id
                }
            )
            
            await network.send_message(connection_id, sync_response)
            
        except Exception as e:
            logger.error(f"Sync request handling error: {e}")
    
    async def _handle_sync_response(self, connection_id: str, message: NetworkMessage):
        """Handle sync response"""
        # Process sync data - would typically involve updating local state
        logger.debug(f"Received sync response from {connection_id}")
    
    async def _handle_gossip(self, connection_id: str, message: NetworkMessage):
        """Handle gossip message"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        try:
            # Check TTL
            if message.ttl <= 0:
                return
            
            # Check if we've seen this message recently (prevent loops)
            message_hash = hashlib.sha256(message.message_id.encode()).hexdigest()
            if await network.peer_manager.has_seen_gossip_message(message_hash):
                return
                
            await network.peer_manager.mark_gossip_message_seen(message_hash, message.ttl)
            
            # Decrement TTL and rebroadcast
            message.ttl -= 1
            await network.broadcast_message(message, exclude=[connection_id])
            
        except Exception as e:
            logger.error(f"Gossip handling error: {e}")
    
    async def _handle_rpc_request(self, connection_id: str, message: NetworkMessage):
        """Handle RPC request"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        try:
            # Process RPC and send response
            method = message.payload.get('method', '')
            params = message.payload.get('params', {})
            
            # Here you would implement actual RPC method handling
            result = await self._execute_rpc_method(method, params)
            
            response = NetworkMessage(
                message_id=NetworkUtils.generate_node_id(),
                message_type=MessageType.RPC_RESPONSE,
                payload={
                    'request_id': message.message_id, 
                    'result': result,
                    'success': True
                }
            )
            
            await network.send_message(connection_id, response)
            
        except Exception as e:
            logger.error(f"RPC request handling error: {e}")
            
            # Send error response
            error_response = NetworkMessage(
                message_id=NetworkUtils.generate_node_id(),
                message_type=MessageType.RPC_RESPONSE,
                payload={
                    'request_id': message.message_id,
                    'error': str(e),
                    'success': False
                }
            )
            
            await network.send_message(connection_id, error_response)
    
    async def _handle_rpc_response(self, connection_id: str, message: NetworkMessage):
        """Handle RPC response"""
        # Complete pending request
        request_id = message.payload.get('request_id')
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            if not future.done():
                if message.payload.get('success', False):
                    future.set_result(message.payload.get('result'))
                else:
                    future.set_exception(Exception(message.payload.get('error', 'Unknown error')))
                del self.pending_requests[request_id]
    
    async def _execute_rpc_method(self, method: str, params: Dict) -> Any:
        """Execute RPC method (placeholder implementation)"""
        # In a real implementation, this would dispatch to registered RPC handlers
        if method == 'ping':
            return {'response': 'pong', 'timestamp': time.time()}
        elif method == 'get_info':
            from .core import AdvancedP2PNetwork
            network = AdvancedP2PNetwork.instance()
            return {
                'node_id': network.config.node_id,
                'version': '1.0',
                'network': network.config.network_type.name,
                'peers': len(network.peer_manager.peers),
                'connections': len(network.connection_manager.connections)
            }
        else:
            raise Exception(f"Unknown RPC method: {method}")
    
    async def priority_message_processor(self, priority: int):
        """Process messages from a specific priority queue"""
        from .core import AdvancedP2PNetwork
        network = AdvancedP2PNetwork.instance()
        
        while network.running:
            try:
                connection_id, message = await self.priority_queues[priority].get()
                await self.handle_message(connection_id, message)
                self.priority_queues[priority].task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Priority {priority} message processor error: {e}")
    
    def get_processing_stats(self) -> Dict:
        """Get message processing statistics"""
        return self.processing_stats.copy()
    
    def reset_processing_stats(self):
        """Reset processing statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'last_reset': time.time(),
            'errors': 0,
            'by_type': defaultdict(int)
        }