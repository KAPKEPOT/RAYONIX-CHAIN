import asyncio
import logging
import time
from typing import Callable, Any, Dict, List
from network.interfaces.processor_interface import IMessageProcessor
from network.exceptions import MessageError
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType, ProtocolType, ConnectionState
from network.models.peer_info import PeerInfo


logger = logging.getLogger("MessageProcessor")

@dataclass
class SendResult:
    """Result of a message send operation"""
    success: bool
    duration: float
    bytes_sent: int
    retry_count: int = 0
    error: Optional[str] = None
    
class MessageProcessor(IMessageProcessor):
    """Message processing implementation"""
    
    def __init__(self, network):
        self.network = network
        self.handlers: Dict[MessageType, List[Callable]] = {}
        self._send_attempts: Dict[str, int] = {}  # Track send attempts per connection
        self._circuit_breakers: Dict[str, bool] = {}  # Circuit breaker pattern
    
    async def process_message(self, connection_id: str, message: NetworkMessage):
        """Process incoming message"""
        try:
            # Update connection activity
            self.network.connection_manager.update_connection_activity(connection_id)
            
            # Log message receipt
            logger.debug(f"Processing {message.message_type.name} from {connection_id}")
            
            # Call registered handlers
            if message.message_type in self.handlers:
                for handler in self.handlers[message.message_type]:
                    try:
                        await handler(connection_id, message)
                    except Exception as e:
                        logger.error(f"Handler error for {message.message_type}: {e}")
            
            # Handle specific message types
            if message.message_type == MessageType.PING:
                await self._handle_ping(connection_id, message)
            elif message.message_type == MessageType.PONG:
                await self._handle_pong(connection_id, message)
            elif message.message_type == MessageType.PEER_LIST:
                await self._handle_peer_list(connection_id, message)
            elif message.message_type == MessageType.HANDSHAKE:
                await self._handle_handshake(connection_id, message)
            
        except Exception as e:
            logger.error(f"Message processing error: {e}")
            await self.network.penalize_peer(connection_id, -5)
    
    async def _handle_ping(self, connection_id: str, message: NetworkMessage):
        """Handle ping message"""
        try:
            # Send pong response
            pong_message = NetworkMessage(
                message_id=f"pong_{time.time()}",
                message_type=MessageType.PONG,
                payload={"original_timestamp": message.payload.get("timestamp")},
                source_node=self.network.node_id
            )
            
            await self.send_message(connection_id, pong_message)
            
        except Exception as e:
            logger.error(f"Ping handling error: {e}")
    
    async def _handle_pong(self, connection_id: str, message: NetworkMessage):
        """Handle pong message"""
        try:
            # Calculate latency
            original_timestamp = message.payload.get("original_timestamp")
            if original_timestamp:
                latency = time.time() - original_timestamp
                
                # Update connection metrics
                self.network.metrics_collector.update_connection_metrics(
                    connection_id, latency=latency
                )
                
                # Update peer info if available
                if connection_id in self.network.connections:
                    peer_info = self.network.connections[connection_id].get('peer_info')
                    if peer_info:
                        peer_info.latency = latency
                        peer_info.last_seen = time.time()
            
        except Exception as e:
            logger.error(f"Pong handling error: {e}")
    
    async def _handle_peer_list(self, connection_id: str, message: NetworkMessage):
        """Handle peer list message"""
        try:
            if message.payload.get("request"):
                # Send our peer list
                peers_to_share = self.network.peer_discovery.get_best_peers(10)
                peer_data = []
                
                for peer in peers_to_share:
                    peer_data.append({
                        "address": peer.address,
                        "port": peer.port,
                        "protocol": peer.protocol.name.lower(),
                        "capabilities": peer.capabilities,
                        "reputation": peer.reputation
                    })
                
                response = NetworkMessage(
                    message_id=f"peer_resp_{time.time()}",
                    message_type=MessageType.PEER_LIST,
                    payload={"peers": peer_data},
                    source_node=self.network.node_id
                )
                
                await self.send_message(connection_id, response)
                
            else:
                # Process received peer list
                peers = message.payload.get("peers", [])
                
                for peer_data in peers:
                    try:
                        peer_info = PeerInfo(
                            node_id="",  # Will be set during connection
                            address=peer_data["address"],
                            port=peer_data["port"],
                            protocol=ProtocolType[peer_data["protocol"].upper()],
                            version="1.0.0",  # Default
                            capabilities=peer_data.get("capabilities", []),
                            reputation=peer_data.get("reputation", 50),
                            last_seen=time.time()
                        )
                        
                        self.network.peer_discovery.add_peer(peer_info)
                        
                    except Exception as e:
                        logger.debug(f"Invalid peer data: {e}")
            
        except Exception as e:
            logger.error(f"Peer list handling error: {e}")
    
    async def _handle_handshake(self, connection_id: str, message: NetworkMessage):
        """Handle handshake message"""
        try:
            # Verify handshake and establish secure connection
            success = await self.network.security_manager.process_handshake(
                connection_id, message
            )
            
            if success:
                # Update connection state
                if connection_id in self.network.connections:
                    peer_info = self.network.connections[connection_id].get('peer_info')
                    if peer_info:
                        peer_info.state = ConnectionState.READY
                        peer_info.node_id = message.source_node
                        peer_info.capabilities = message.payload.get("capabilities", [])
                        peer_info.version = message.payload.get("version", "1.0.0")
            
        except Exception as e:
            logger.error(f"Handshake handling error: {e}")
            await self.network.penalize_peer(connection_id, -10)
    
    def register_handler(self, message_type: Any, handler: Callable):
        """Register message handler"""
        if message_type not in self.handlers:
            self.handlers[message_type] = []
        
        self.handlers[message_type].append(handler)
        logger.debug(f"Registered handler for {message_type.name}")
    
    def unregister_handler(self, message_type: Any, handler: Callable):
        """Unregister message handler"""
        if message_type in self.handlers and handler in self.handlers[message_type]:
            self.handlers[message_type].remove(handler)
            logger.debug(f"Unregistered handler for {message_type.name}")
    
    async def send_message(self, connection_id: str, message: NetworkMessage) -> SendResult:
        """
        Send message to specific connection - TCP ONLY for P2P messaging
        Returns detailed SendResult with metrics and error information
        """
        start_time = time.time()
        message_size = self._calculate_message_size(message)
        retry_count = 0
        
        # Input validation
        validation_error = self._validate_inputs(connection_id, message)
        if validation_error:
            return SendResult(
                success=False, 
                duration=time.time() - start_time,
                bytes_sent=0,
                error=validation_error
            )
        
        try:
            # Check circuit breaker
            if self._is_circuit_open(connection_id):
                return SendResult(
                    success=False,
                    duration=time.time() - start_time,
                    bytes_sent=0,
                    error="Circuit breaker open"
                )
            
            # Get connection and validate
            connection = self.network.connections.get(connection_id)
            if not connection:
                return SendResult(
                    success=False,
                    duration=time.time() - start_time,
                    bytes_sent=0,
                    error="Connection not found"
                )
            
            # Protocol validation - TCP only for P2P
            protocol = connection.get('protocol')
            if protocol != 'tcp':
                return SendResult(
                    success=False,
                    duration=time.time() - start_time,
                    bytes_sent=0,
                    error=f"Protocol {protocol} not supported for P2P"
                )
            
            # Pre-send checks
            health_check = await self._perform_pre_send_checks(connection_id, connection, message)
            if not health_check.success:
                return health_check
            
            # Send with retry logic
            result = await self._send_with_retry(connection_id, message, message_size)
            result.duration = time.time() - start_time
            result.retry_count = retry_count
            
            # Update circuit breaker based on result
            self._update_circuit_breaker(connection_id, result.success)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Unexpected error sending to {connection_id}: {e}", exc_info=True)
            return SendResult(
                success=False,
                duration=duration,
                bytes_sent=0,
                retry_count=retry_count,
                error=f"Unexpected error: {str(e)}"
            )
    
    async def _send_with_retry(self, connection_id: str, message: NetworkMessage, message_size: int, max_retries: int = 3) -> SendResult:
        """Send message with exponential backoff retry logic"""
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                # Check rate limiting before each attempt
                if not await self._check_outgoing_rate_limit(connection_id, message_size):
                    last_error = "Outgoing rate limit exceeded"
                    if attempt == max_retries:
                        break
                    await self._wait_for_retry(attempt, connection_id)
                    continue
                
                # Attempt to send
                success = await self.network.tcp_handler.send_message(connection_id, message)
                
                if success:
                    # Update metrics on success
                    self._update_success_metrics(connection_id, message_size)
                    
                    if attempt > 0:
                        logger.info(f"Message send succeeded on retry {attempt} for {connection_id}")
                    
                    return SendResult(
                        success=True,
                        duration=0,  # Will be set by caller
                        bytes_sent=message_size,
                        retry_count=attempt
                    )
                else:
                    last_error = "TCP handler returned failure"
                    if attempt < max_retries:
                        await self._wait_for_retry(attempt, connection_id)
                    
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries:
                    await self._wait_for_retry(attempt, connection_id)
                else:
                    logger.error(f"Send failed after {max_retries + 1} attempts for {connection_id}: {e}")
        
        # All retries failed
        self._update_failure_metrics(connection_id, message_size)
        return SendResult(
            success=False,
            duration=0,  # Will be set by caller
            bytes_sent=0,
            retry_count=max_retries,
            error=last_error
        )
    
    async def _wait_for_retry(self, attempt: int, connection_id: str):
        """Wait before retry with exponential backoff and jitter"""
        delay = self._calculate_retry_delay(attempt)
        logger.warning(
            f"Send failed for {connection_id}, "
            f"retrying in {delay:.2f}s (attempt {attempt + 1})"
        )
        await asyncio.sleep(delay)
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = 0.5  # 500ms
        max_delay = 10.0  # 10 seconds max
        delay = min(base_delay * (2 ** attempt), max_delay)
        
        # Add jitter to avoid thundering herd
        jitter = random.uniform(0.8, 1.2)
        return delay * jitter
    
    def _validate_inputs(self, connection_id: str, message: NetworkMessage) -> Optional[str]:
        """Validate input parameters"""
        if not connection_id or not isinstance(connection_id, str):
            return "Invalid connection_id"
        
        if not message or not isinstance(message, NetworkMessage):
            return "Invalid message object"
        
        if not message.message_id or not isinstance(message.message_id, str):
            return "Message missing valid message_id"
        
        if not message.message_type:
            return "Message missing message_type"
        
        # Check message size limits
        message_size = self._calculate_message_size(message)
        if hasattr(self.network.config, 'max_message_size'):
            if message_size > self.network.config.max_message_size:
                return f"Message too large: {message_size} bytes"
        
        # Check TTL for gossip messages
        if hasattr(message, 'ttl') and message.ttl <= 0:
            return "Message TTL expired"
        
        return None
    
    def _calculate_message_size(self, message: NetworkMessage) -> int:
        """Calculate approximate message size"""
        try:
            size = len(message.message_id) if message.message_id else 0
            size += len(str(message.payload)) if message.payload else 0
            size += len(message.source_node) if message.source_node else 0
            return size
        except:
            return 0
    
    async def _perform_pre_send_checks(self, connection_id: str, connection: dict, message: NetworkMessage) -> SendResult:
        """Perform all pre-send health and validation checks"""
        
        # Check connection health
        if not await self._is_connection_healthy(connection_id, connection):
            return SendResult(
                success=False,
                duration=0,
                bytes_sent=0,
                error="Connection not healthy"
            )
        
        # Check peer reputation
        peer_info = connection.get('peer_info')
        if peer_info and peer_info.reputation < -50:
            return SendResult(
                success=False,
                duration=0,
                bytes_sent=0,
                error="Peer reputation too low"
            )
        
        return SendResult(success=True, duration=0, bytes_sent=0)
    
    async def _is_connection_healthy(self, connection_id: str, connection: dict) -> bool:
        """Check if connection is healthy and ready for messaging"""
        try:
            # Check last activity timeout
            last_activity = connection.get('last_activity', 0)
            if time.time() - last_activity > self.network.config.connection_timeout:
                logger.warning(f"Connection {connection_id} is stale")
                return False
            
            # Check connection state
            peer_info = connection.get('peer_info')
            if peer_info and peer_info.state != ConnectionState.READY:
                logger.warning(f"Connection {connection_id} not in READY state: {peer_info.state}")
                return False
            
            # TCP-specific health check
            writer = connection.get('writer')
            if not writer or writer.is_closing():
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Health check error for {connection_id}: {e}")
            return False
    
    async def _check_outgoing_rate_limit(self, connection_id: str, message_size: int) -> bool:
        """Check outgoing rate limiting"""
        # Use the network's rate limiter if available
        if hasattr(self.network, 'rate_limiter'):
            return await self.network.rate_limiter.check_outgoing_rate_limit(connection_id, message_size)
        return True
    
    def _update_success_metrics(self, connection_id: str, message_size: int):
        """Update metrics on successful send"""
        try:
            self.network.metrics_collector.update_connection_metrics(
                connection_id, 
                bytes_sent=message_size, 
                messages_sent=1
            )
            # Reset circuit breaker on success
            self._circuit_breakers[connection_id] = False
        except Exception as e:
            logger.debug(f"Metrics update error: {e}")
    
    def _update_failure_metrics(self, connection_id: str, message_size: int):
        """Update metrics on failed send"""
        try:
            self.network.metrics_collector.update_connection_metrics(
                connection_id,
                error_count=1
            )
        except Exception as e:
            logger.debug(f"Failure metrics error: {e}")
    
    def _is_circuit_open(self, connection_id: str) -> bool:
        """Check if circuit breaker is open for this connection"""
        return self._circuit_breakers.get(connection_id, False)
    
    def _update_circuit_breaker(self, connection_id: str, success: bool):
        """Update circuit breaker state"""
        if success:
            self._circuit_breakers[connection_id] = False
        else:
            # After multiple failures, open the circuit
            fail_count = self._send_attempts.get(connection_id, 0) + 1
            self._send_attempts[connection_id] = fail_count
            
            if fail_count >= 5:  # Open circuit after 5 consecutive failures
                self._circuit_breakers[connection_id] = True
                logger.error(f"Circuit breaker opened for {connection_id}")

    async def broadcast_message(self, message: NetworkMessage, exclude: list = None) -> Dict[str, SendResult]:
        """
        Broadcast message to all connections with individual result tracking
        Returns dictionary of connection_id -> SendResult
        """
        exclude = exclude or []
        connection_ids = [cid for cid in self.network.connections.keys() if cid not in exclude]
        results = {}
        
        # Batch processing for performance
        batch_size = 10
        for i in range(0, len(connection_ids), batch_size):
            batch = connection_ids[i:i + batch_size]
            
            # Process batch concurrently with semaphore to limit resource usage
            semaphore = asyncio.Semaphore(5)
            
            async def send_to_connection(conn_id):
                async with semaphore:
                    result = await self.send_message(conn_id, message)
                    results[conn_id] = result
                    return result
            
            # Execute batch concurrently
            await asyncio.gather(*[
                send_to_connection(conn_id) for conn_id in batch
            ], return_exceptions=True)
            
            # Small delay between batches to prevent overwhelming the system
            if i + batch_size < len(connection_ids):
                await asyncio.sleep(0.05)
        
        return results