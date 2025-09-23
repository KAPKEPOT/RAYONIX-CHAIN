# consensus/network/protocol.py
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Callable, Any, Set
import logging
from dataclasses import asdict
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('NetworkProtocol')

class MessageType:
    """Network message types for consensus protocol"""
    PROPOSAL = "proposal"
    VOTE = "vote"
    STATE_SYNC_REQUEST = "state_sync_request"
    STATE_SYNC_RESPONSE = "state_sync_response"
    VALIDATOR_UPDATE = "validator_update"
    EVIDENCE = "evidence"
    PING = "ping"
    PONG = "pong"

class NetworkProtocol:
    """Production-ready network communication protocol for consensus"""
    
    def __init__(self, consensus_engine: Any):
        self.consensus_engine = consensus_engine
        self.config = consensus_engine.config
        
        # Network state
        self.connected_peers: Set[str] = set()
        self.validator_peers: Dict[str, Dict] = {}  # validator_address -> peer info
        self.message_handlers: Dict[str, Callable] = {}
        self.pending_messages: Dict[str, List] = {}  # peer_id -> messages
        
        # Async components
        self.loop = asyncio.new_event_loop()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.websocket_server = None
        self.websocket_clients: Dict[str, Any] = {}
        
        # Message tracking for deduplication
        self.seen_messages: Set[str] = set()
        self.message_sequence: int = 0
        
        # Initialize message handlers
        self._register_message_handlers()
        
        # Start network services
        self._start_network_services()
    
    def _register_message_handlers(self):
        """Register handlers for different message types"""
        self.message_handlers = {
            MessageType.PROPOSAL: self._handle_proposal_message,
            MessageType.VOTE: self._handle_vote_message,
            MessageType.STATE_SYNC_REQUEST: self._handle_state_sync_request,
            MessageType.STATE_SYNC_RESPONSE: self._handle_state_sync_response,
            MessageType.VALIDATOR_UPDATE: self._handle_validator_update,
            MessageType.EVIDENCE: self._handle_evidence_message,
            MessageType.PING: self._handle_ping_message,
            MessageType.PONG: self._handle_pong_message
        }
    
    def _start_network_services(self):
        """Start network services in background threads"""
        def run_websocket_server():
            asyncio.set_event_loop(self.loop)
            try:
                start_server = websockets.serve(
                    self._handle_websocket_connection,
                    self.config.network_host,
                    self.config.network_port
                )
                self.loop.run_until_complete(start_server)
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"WebSocket server error: {e}")
        
        def run_peer_discovery():
            while getattr(self.consensus_engine, '_running', True):
                try:
                    self._discover_peers()
                    time.sleep(30)  # Discover peers every 30 seconds
                except Exception as e:
                    logger.error(f"Peer discovery error: {e}")
                    time.sleep(60)
        
        # Start WebSocket server
        server_thread = threading.Thread(target=run_websocket_server, daemon=True)
        server_thread.start()
        
        # Start peer discovery
        discovery_thread = threading.Thread(target=run_peer_discovery, daemon=True)
        discovery_thread.start()
        
        logger.info(f"Started network services on {self.config.network_host}:{self.config.network_port}")
    
    async def _handle_websocket_connection(self, websocket, path):
        """Handle incoming WebSocket connections"""
        peer_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        try:
            self.websocket_clients[peer_id] = websocket
            self.connected_peers.add(peer_id)
            
            logger.info(f"New connection from {peer_id}")
            
            # Send welcome message with our validator info
            await self._send_welcome_message(websocket)
            
            async for message in websocket:
                await self._process_incoming_message(message, peer_id)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection closed with {peer_id}")
        except Exception as e:
            logger.error(f"Error handling connection from {peer_id}: {e}")
        finally:
            self.connected_peers.discard(peer_id)
            self.websocket_clients.pop(peer_id, None)
    
    async def _send_welcome_message(self, websocket):
        """Send welcome message to new connection"""
        welcome_msg = {
            'type': 'welcome',
            'node_id': self.config.node_id,
            'height': self.consensus_engine.height,
            'validator_address': getattr(self.consensus_engine, 'validator_address', ''),
            'timestamp': time.time()
        }
        
        await websocket.send(json.dumps(welcome_msg))
    
    async def _process_incoming_message(self, message_data: str, peer_id: str):
        """Process incoming message from peer"""
        try:
            message = json.loads(message_data)
            message_type = message.get('type')
            
            if not message_type:
                logger.warning(f"Received message without type from {peer_id}")
                return
            
            # Check for duplicate messages
            message_hash = self._calculate_message_hash(message)
            if message_hash in self.seen_messages:
                return  # Ignore duplicate
            
            self.seen_messages.add(message_hash)
            
            # Route to appropriate handler
            handler = self.message_handlers.get(message_type)
            if handler:
                await handler(message, peer_id)
            else:
                logger.warning(f"Unknown message type: {message_type} from {peer_id}")
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON message from {peer_id}")
        except Exception as e:
            logger.error(f"Error processing message from {peer_id}: {e}")
    
    async def _handle_proposal_message(self, message: Dict, peer_id: str):
        """Handle block proposal message"""
        try:
            from consensus.models.blocks import BlockProposal
            
            proposal_data = message.get('proposal')
            if not proposal_data:
                logger.warning("Proposal message missing proposal data")
                return
            
            # Validate message signature
            if not self._validate_message_signature(message):
                logger.warning("Invalid signature on proposal message")
                return
            
            # Create BlockProposal object
            proposal = BlockProposal.from_dict(proposal_data)
            
            # Pass to consensus engine
            self.consensus_engine.receive_proposal(proposal)
            
            logger.debug(f"Processed proposal from {peer_id} for height {proposal.height}")
            
        except Exception as e:
            logger.error(f"Error handling proposal message: {e}")
    
    async def _handle_vote_message(self, message: Dict, peer_id: str):
        """Handle vote message"""
        try:
            from consensus.models.votes import Vote
            
            vote_data = message.get('vote')
            if not vote_data:
                logger.warning("Vote message missing vote data")
                return
            
            # Validate message signature
            if not self._validate_message_signature(message):
                logger.warning("Invalid signature on vote message")
                return
            
            # Create Vote object
            vote = Vote.from_dict(vote_data)
            
            # Pass to consensus engine
            self.consensus_engine.receive_vote(vote)
            
            logger.debug(f"Processed {vote.vote_type.name} from {peer_id} for height {vote.height}")
            
        except Exception as e:
            logger.error(f"Error handling vote message: {e}")
    
    async def _handle_state_sync_request(self, message: Dict, peer_id: str):
        """Handle state synchronization request"""
        try:
            height = message.get('height')
            if height is None:
                logger.warning("State sync request missing height")
                return
            
            # Prepare state sync response
            sync_data = self._prepare_state_sync_data(height)
            
            response = {
                'type': MessageType.STATE_SYNC_RESPONSE,
                'height': height,
                'sync_data': sync_data,
                'timestamp': time.time(),
                'sequence': self._get_next_sequence()
            }
            
            # Sign the response
            self._sign_message(response)
            
            # Send response
            await self._send_to_peer(peer_id, response)
            
            logger.info(f"Sent state sync response to {peer_id} for height {height}")
            
        except Exception as e:
            logger.error(f"Error handling state sync request: {e}")
    
    async def _handle_state_sync_response(self, message: Dict, peer_id: str):
        """Handle state synchronization response"""
        try:
            sync_data = message.get('sync_data')
            if not sync_data:
                logger.warning("State sync response missing sync data")
                return
            
            # Validate message signature
            if not self._validate_message_signature(message):
                logger.warning("Invalid signature on state sync response")
                return
            
            # Apply state sync data
            self._apply_state_sync_data(sync_data)
            
            logger.info(f"Applied state sync from {peer_id}")
            
        except Exception as e:
            logger.error(f"Error handling state sync response: {e}")
    
    async def _handle_validator_update(self, message: Dict, peer_id: str):
        """Handle validator update message"""
        try:
            update_data = message.get('update')
            if not update_data:
                logger.warning("Validator update missing update data")
                return
            
            # Validate message signature
            if not self._validate_message_signature(message):
                logger.warning("Invalid signature on validator update")
                return
            
            # Process validator update
            self._process_validator_update(update_data)
            
            logger.info(f"Processed validator update from {peer_id}")
            
        except Exception as e:
            logger.error(f"Error handling validator update: {e}")
    
    async def _handle_evidence_message(self, message: Dict, peer_id: str):
        """Handle evidence of misbehavior"""
        try:
            evidence = message.get('evidence')
            if not evidence:
                logger.warning("Evidence message missing evidence data")
                return
            
            # Validate message signature
            if not self._validate_message_signature(message):
                logger.warning("Invalid signature on evidence message")
                return
            
            # Pass evidence to slashing manager
            validator_address = evidence.get('validator_address')
            if validator_address:
                self.consensus_engine.slashing_manager.slash_validator(
                    validator_address, evidence, peer_id
                )
            
            logger.info(f"Processed evidence from {peer_id} against {validator_address}")
            
        except Exception as e:
            logger.error(f"Error handling evidence message: {e}")
    
    async def _handle_ping_message(self, message: Dict, peer_id: str):
        """Handle ping message"""
        try:
            # Respond with pong
            pong_msg = {
                'type': MessageType.PONG,
                'timestamp': time.time(),
                'sequence': message.get('sequence', 0)
            }
            
            self._sign_message(pong_msg)
            await self._send_to_peer(peer_id, pong_msg)
            
        except Exception as e:
            logger.error(f"Error handling ping message: {e}")
    
    async def _handle_pong_message(self, message: Dict, peer_id: str):
        """Handle pong message (update latency)"""
        try:
            sent_timestamp = message.get('original_timestamp', 0)
            if sent_timestamp:
                latency = time.time() - sent_timestamp
                # Update peer latency information
                if peer_id in self.validator_peers:
                    self.validator_peers[peer_id]['latency'] = latency
                    self.validator_peers[peer_id]['last_seen'] = time.time()
            
        except Exception as e:
            logger.error(f"Error handling pong message: {e}")
    
    def broadcast_proposal(self, proposal: Any):
        """Broadcast block proposal to all peers"""
        try:
            message = {
                'type': MessageType.PROPOSAL,
                'proposal': proposal.to_dict(),
                'timestamp': time.time(),
                'sequence': self._get_next_sequence()
            }
            
            self._sign_message(message)
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message(message), 
                self.loop
            )
            
            logger.info(f"Broadcast proposal for height {proposal.height}")
            
        except Exception as e:
            logger.error(f"Error broadcasting proposal: {e}")
    
    def broadcast_vote(self, vote: Any):
        """Broadcast vote to all peers"""
        try:
            message = {
                'type': MessageType.VOTE,
                'vote': vote.to_dict(),
                'timestamp': time.time(),
                'sequence': self._get_next_sequence()
            }
            
            self._sign_message(message)
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message(message), 
                self.loop
            )
            
            logger.debug(f"Broadcast {vote.vote_type.name} for height {vote.height}")
            
        except Exception as e:
            logger.error(f"Error broadcasting vote: {e}")
    
    def broadcast_evidence(self, evidence: Dict):
        """Broadcast evidence of misbehavior"""
        try:
            message = {
                'type': MessageType.EVIDENCE,
                'evidence': evidence,
                'timestamp': time.time(),
                'sequence': self._get_next_sequence()
            }
            
            self._sign_message(message)
            asyncio.run_coroutine_threadsafe(
                self._broadcast_message(message), 
                self.loop
            )
            
            logger.info("Broadcast evidence of misbehavior")
            
        except Exception as e:
            logger.error(f"Error broadcasting evidence: {e}")
    
    async def _broadcast_message(self, message: Dict):
        """Broadcast message to all connected peers"""
        tasks = []
        for peer_id in list(self.connected_peers):
            task = self._send_to_peer(peer_id, message)
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _send_to_peer(self, peer_id: str, message: Dict):
        """Send message to specific peer"""
        try:
            websocket = self.websocket_clients.get(peer_id)
            if websocket and not websocket.closed:
                await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to {peer_id}: {e}")
            # Remove disconnected peer
            self.connected_peers.discard(peer_id)
            self.websocket_clients.pop(peer_id, None)
    
    def _discover_peers(self):
        """Discover and connect to new peers"""
        # Implementation would depend on specific peer discovery mechanism
        # This could include DNS seeds, hardcoded bootstrap nodes, etc.
        pass
    
    def _validate_message_signature(self, message: Dict) -> bool:
        """Validate message signature"""
        try:
            from consensus.crypto.signing import CryptoManager
            crypto_manager = CryptoManager()
            
            signature = message.get('signature')
            public_key = message.get('public_key')
            signing_data = self._get_message_signing_data(message)
            
            if not signature or not public_key:
                return False
            
            return crypto_manager.verify_signature(public_key, signing_data, signature)
            
        except Exception as e:
            logger.error(f"Error validating message signature: {e}")
            return False
    
    def _sign_message(self, message: Dict):
        """Sign outgoing message"""
        try:
            from consensus.crypto.signing import CryptoManager
            crypto_manager = CryptoManager()
            
            signing_data = self._get_message_signing_data(message)
            signature = crypto_manager.sign_data(signing_data)
            
            message['signature'] = signature
            message['public_key'] = getattr(self.config, 'node_public_key', '')
            
        except Exception as e:
            logger.error(f"Error signing message: {e}")
    
    def _get_message_signing_data(self, message: Dict) -> bytes:
        """Get data that should be signed for a message"""
        # Exclude signature and public_key from signing data
        signing_data = message.copy()
        signing_data.pop('signature', None)
        signing_data.pop('public_key', None)
        
        return json.dumps(signing_data, sort_keys=True).encode()
    
    def _calculate_message_hash(self, message: Dict) -> str:
        """Calculate unique hash for message deduplication"""
        import hashlib
        message_string = json.dumps(message, sort_keys=True)
        return hashlib.sha256(message_string.encode()).hexdigest()
    
    def _get_next_sequence(self) -> int:
        """Get next message sequence number"""
        self.message_sequence += 1
        return self.message_sequence
    
    def _prepare_state_sync_data(self, height: int) -> Dict:
        """Prepare state synchronization data for given height"""
        # This would include blocks, validator sets, etc.
        return {
            'height': height,
            'blocks': [],  # Would include actual block data
            'validators': {k: v.to_dict() for k, v in self.consensus_engine.validators.items()},
            'app_hash': getattr(self.consensus_engine.abci, 'app_hash', '')
        }
    
    def _apply_state_sync_data(self, sync_data: Dict):
        """Apply state synchronization data"""
        # This would update the node's state to match the sync data
        pass
    
    def _process_validator_update(self, update_data: Dict):
        """Process validator update"""
        # This would update local validator information
        pass
    
    def get_network_status(self) -> Dict:
        """Get current network status"""
        return {
            'connected_peers': len(self.connected_peers),
            'total_peers': len(self.validator_peers),
            'message_sequence': self.message_sequence,
            'seen_messages': len(self.seen_messages),
            'pending_messages': sum(len(messages) for messages in self.pending_messages.values())
        }
    
    def shutdown(self):
        """Shutdown network protocol"""
        try:
            # Close all connections
            for websocket in self.websocket_clients.values():
                asyncio.run_coroutine_threadsafe(websocket.close(), self.loop)
            
            # Stop event loop
            self.loop.call_soon_threadsafe(self.loop.stop)
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            logger.info("Network protocol shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during network shutdown: {e}")