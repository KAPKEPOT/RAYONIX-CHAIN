# consensus/network/protocol.py
import logging
import time
from typing import Dict, Any, List, Optional
from network.models.network_message import NetworkMessage
from network.config.network_types import MessageType

logger = logging.getLogger('ConsensusProtocol')

class ConsensusMessageType:
    """Consensus-specific message types that extend base network types"""
    PROPOSAL = "consensus_proposal"
    VOTE = "consensus_vote" 
    STATE_SYNC_REQUEST = "consensus_state_sync_request"
    STATE_SYNC_RESPONSE = "consensus_state_sync_response"
    VALIDATOR_UPDATE = "consensus_validator_update"
    EVIDENCE = "consensus_evidence"

class ConsensusNetworkProtocol:
    """
    Thin wrapper around P2P network for consensus messaging.
    Handles ONLY consensus-specific protocol logic.
    """
    
    def __init__(self, consensus_engine: Any, p2p_network: Any):
        if not consensus_engine or not p2p_network:
            raise ValueError("consensus_engine and p2p_network are required")
            
        self.consensus_engine = consensus_engine
        self.p2p_network = p2p_network
        self.config = consensus_engine.config
        
        # Register consensus message handlers with the network layer
        self._register_handlers()
        
        logger.info("Consensus network protocol initialized")
    
    def _register_handlers(self):
        """Register consensus message handlers with the network layer"""
        # Convert consensus message types to network message types
        message_mapping = {
            ConsensusMessageType.PROPOSAL: self._handle_proposal,
            ConsensusMessageType.VOTE: self._handle_vote,
            ConsensusMessageType.STATE_SYNC_REQUEST: self._handle_state_sync_request,
            ConsensusMessageType.STATE_SYNC_RESPONSE: self._handle_state_sync_response,
            ConsensusMessageType.VALIDATOR_UPDATE: self._handle_validator_update,
            ConsensusMessageType.EVIDENCE: self._handle_evidence,
        }
        
        for message_type, handler in message_mapping.items():
            # Use base network message type with custom payload
            self.p2p_network.register_message_handler(
                MessageType.CUSTOM,  # Use CUSTOM type for consensus messages
                self._create_consensus_handler(handler, message_type)
            )
    
    def _create_consensus_handler(self, handler, expected_type: str):
        """Create a wrapper handler that filters by consensus message type"""
        async def consensus_handler(connection_id: str, message: NetworkMessage):
            # Check if this is a consensus message of the expected type
            if (message.payload and 
                isinstance(message.payload, dict) and 
                message.payload.get('consensus_type') == expected_type):
                await handler(connection_id, message)
        
        return consensus_handler
    
    async def _handle_proposal(self, connection_id: str, message: NetworkMessage):
        """Handle block proposal message - just pass to consensus engine"""
        try:
            proposal_data = message.payload.get('data', {})
            
            # Validate message signature if present
            if not self._validate_consensus_message(message):
                logger.warning(f"Invalid signature on proposal from {connection_id}")
                return
            
            # Pass to consensus engine
            self.consensus_engine.receive_proposal(proposal_data)
            
            logger.debug(f"Processed proposal from {connection_id} for height {proposal_data.get('height')}")
            
        except Exception as e:
            logger.error(f"Error handling proposal: {e}")
    
    async def _handle_vote(self, connection_id: str, message: NetworkMessage):
        """Handle vote message - just pass to consensus engine"""
        try:
            vote_data = message.payload.get('data', {})
            
            # Validate message signature if present
            if not self._validate_consensus_message(message):
                logger.warning(f"Invalid signature on vote from {connection_id}")
                return
            
            # Pass to consensus engine
            self.consensus_engine.receive_vote(vote_data)
            
            logger.debug(f"Processed vote from {connection_id} for height {vote_data.get('height')}")
            
        except Exception as e:
            logger.error(f"Error handling vote: {e}")
    
    async def _handle_state_sync_request(self, connection_id: str, message: NetworkMessage):
        """Handle state synchronization request"""
        try:
            height = message.payload.get('data', {}).get('height')
            if height is None:
                return
            
            # Prepare state sync data using consensus engine
            sync_data = self.consensus_engine.prepare_state_sync_data(height)
            
            # Send response back
            response_payload = {
                'consensus_type': ConsensusMessageType.STATE_SYNC_RESPONSE,
                'data': {
                    'height': height,
                    'sync_data': sync_data
                }
            }
            
            await self._send_to_peer(connection_id, response_payload)
            
        except Exception as e:
            logger.error(f"Error handling state sync request: {e}")
    
    async def _handle_state_sync_response(self, connection_id: str, message: NetworkMessage):
        """Handle state synchronization response"""
        try:
            sync_data = message.payload.get('data', {}).get('sync_data')
            if not sync_data:
                return
            
            # Validate message signature if present
            if not self._validate_consensus_message(message):
                logger.warning(f"Invalid signature on state sync response from {connection_id}")
                return
            
            # Apply state sync data using consensus engine
            self.consensus_engine.apply_state_sync_data(sync_data)
            
        except Exception as e:
            logger.error(f"Error handling state sync response: {e}")
    
    async def _handle_validator_update(self, connection_id: str, message: NetworkMessage):
        """Handle validator update message"""
        try:
            update_data = message.payload.get('data', {})
            
            # Validate message signature if present
            if not self._validate_consensus_message(message):
                logger.warning(f"Invalid signature on validator update from {connection_id}")
                return
            
            # Process validator update using consensus engine
            self.consensus_engine.process_validator_update(update_data)
            
        except Exception as e:
            logger.error(f"Error handling validator update: {e}")
    
    async def _handle_evidence(self, connection_id: str, message: NetworkMessage):
        """Handle evidence of misbehavior"""
        try:
            evidence_data = message.payload.get('data', {})
            
            # Validate message signature if present
            if not self._validate_consensus_message(message):
                logger.warning(f"Invalid signature on evidence from {connection_id}")
                return
            
            # Pass evidence to slashing manager
            validator_address = evidence_data.get('validator_address')
            if validator_address and hasattr(self.consensus_engine, 'slashing_manager'):
                self.consensus_engine.slashing_manager.process_evidence(
                    validator_address, evidence_data, connection_id
                )
            
        except Exception as e:
            logger.error(f"Error handling evidence: {e}")
    
    def _validate_consensus_message(self, message: NetworkMessage) -> bool:
        """Validate consensus message signature"""
        try:
            # Use consensus engine's crypto for validation
            if hasattr(self.consensus_engine, 'crypto_manager'):
                signing_data = self._get_signing_data(message.payload)
                signature = message.payload.get('signature')
                public_key = message.payload.get('public_key')
                
                if signature and public_key:
                    return self.consensus_engine.crypto_manager.verify_signature(
                        public_key, signing_data, signature
                    )
            
            # If no crypto manager or no signature, accept message (for testing/development)
            return True
            
        except Exception as e:
            logger.error(f"Message validation error: {e}")
            return False
    
    def _get_signing_data(self, payload: Dict) -> bytes:
        """Get data that should be signed for consensus messages"""
        import json
        signing_data = payload.copy()
        signing_data.pop('signature', None)
        signing_data.pop('public_key', None)
        return json.dumps(signing_data, sort_keys=True).encode()
    
    def _sign_payload(self, payload: Dict):
        """Sign outgoing consensus message payload"""
        try:
            if hasattr(self.consensus_engine, 'crypto_manager'):
                signing_data = self._get_signing_data(payload)
                signature = self.consensus_engine.crypto_manager.sign_data(signing_data)
                
                payload['signature'] = signature
                payload['public_key'] = getattr(self.config, 'node_public_key', '')
                
        except Exception as e:
            logger.error(f"Error signing payload: {e}")
    
    async def _send_to_peer(self, peer_id: str, payload: Dict):
        """Send consensus message to specific peer"""
        try:
            message = NetworkMessage(
                message_id=f"consensus_{time.time()}",
                message_type=MessageType.CUSTOM,  # Use CUSTOM type for consensus
                payload=payload,
                source_node=self.p2p_network.node_id
            )
            
            return await self.p2p_network.send_message(peer_id, message)
            
        except Exception as e:
            logger.error(f"Error sending to peer {peer_id}: {e}")
            return False
    
    # === PUBLIC API ===
    
    def broadcast_proposal(self, proposal: Any):
        """Broadcast block proposal to all peers"""
        try:
            payload = {
                'consensus_type': ConsensusMessageType.PROPOSAL,
                'data': proposal.to_dict() if hasattr(proposal, 'to_dict') else proposal,
                'timestamp': time.time()
            }
            
            self._sign_payload(payload)
            
            message = NetworkMessage(
                message_id=f"proposal_{time.time()}",
                message_type=MessageType.CUSTOM,
                payload=payload,
                source_node=self.p2p_network.node_id
            )
            
            # Use existing broadcast infrastructure
            asyncio.create_task(self.p2p_network.broadcast_message(message))
            
            logger.info(f"Broadcast proposal for height {getattr(proposal, 'height', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error broadcasting proposal: {e}")
    
    def broadcast_vote(self, vote: Any):
        """Broadcast vote to all peers"""
        try:
            payload = {
                'consensus_type': ConsensusMessageType.VOTE,
                'data': vote.to_dict() if hasattr(vote, 'to_dict') else vote,
                'timestamp': time.time()
            }
            
            self._sign_payload(payload)
            
            message = NetworkMessage(
                message_id=f"vote_{time.time()}",
                message_type=MessageType.CUSTOM,
                payload=payload,
                source_node=self.p2p_network.node_id
            )
            
            # Use existing broadcast infrastructure
            asyncio.create_task(self.p2p_network.broadcast_message(message))
            
            logger.debug(f"Broadcast vote for height {getattr(vote, 'height', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error broadcasting vote: {e}")
    
    def broadcast_evidence(self, evidence: Dict):
        """Broadcast evidence of misbehavior"""
        try:
            payload = {
                'consensus_type': ConsensusMessageType.EVIDENCE,
                'data': evidence,
                'timestamp': time.time()
            }
            
            self._sign_payload(payload)
            
            message = NetworkMessage(
                message_id=f"evidence_{time.time()}",
                message_type=MessageType.CUSTOM,
                payload=payload,
                source_node=self.p2p_network.node_id
            )
            
            # Use existing broadcast infrastructure
            asyncio.create_task(self.p2p_network.broadcast_message(message))
            
            logger.info("Broadcast evidence of misbehavior")
            
        except Exception as e:
            logger.error(f"Error broadcasting evidence: {e}")
    
    def broadcast_validator_update(self, update_data: Dict):
        """Broadcast validator update"""
        try:
            payload = {
                'consensus_type': ConsensusMessageType.VALIDATOR_UPDATE,
                'data': update_data,
                'timestamp': time.time()
            }
            
            self._sign_payload(payload)
            
            message = NetworkMessage(
                message_id=f"validator_update_{time.time()}",
                message_type=MessageType.CUSTOM,
                payload=payload,
                source_node=self.p2p_network.node_id
            )
            
            # Use existing broadcast infrastructure
            asyncio.create_task(self.p2p_network.broadcast_message(message))
            
        except Exception as e:
            logger.error(f"Error broadcasting validator update: {e}")
    
    async def request_state_sync(self, peer_id: str, height: int):
        """Request state synchronization from specific peer"""
        try:
            payload = {
                'consensus_type': ConsensusMessageType.STATE_SYNC_REQUEST,
                'data': {'height': height},
                'timestamp': time.time()
            }
            
            self._sign_payload(payload)
            
            return await self._send_to_peer(peer_id, payload)
            
        except Exception as e:
            logger.error(f"Error requesting state sync: {e}")
            return False
    
    def get_network_status(self) -> Dict:
        """Get network status from underlying P2P network"""
        try:
            return self.p2p_network.get_network_stats()
        except Exception as e:
            logger.error(f"Error getting network status: {e}")
            return {}
    
    def get_connected_validators(self) -> List[Dict]:
        """Get list of connected validator peers"""
        try:
            # This would filter peers to only those that are validators
            all_peers = self.p2p_network.get_peers()
            return [peer for peer in all_peers if peer.get('is_validator', False)]
        except Exception as e:
            logger.error(f"Error getting connected validators: {e}")
            return []
    
    def shutdown(self):
        """Shutdown consensus protocol - network cleanup handled by P2P network"""
        logger.info("Consensus protocol shutdown")
        # No cleanup needed - P2P network manages its own lifecycle