"""
gRPC API implementation for consensus system
"""

import grpc
from typing import Dict, List, Optional, Any
import logging
from concurrent import futures
import time

from ..exceptions import ConsensusError
from ..utils import RateLimiter

logger = logging.getLogger('consensus.api')

# Import generated gRPC protobuf files
# These would be generated from .proto files
try:
    import consensus_pb2
    import consensus_pb2_grpc
    from google.protobuf import empty_pb2
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("gRPC dependencies not available")

class ConsensusServicer:
    """gRPC servicer for consensus API"""
    
    def __init__(self, max_workers: int = 10, max_message_size: int = 100 * 1024 * 1024):
        if not GRPC_AVAILABLE:
            raise ConsensusError("gRPC dependencies not available")
        
        self.max_workers = max_workers
        self.max_message_size = max_message_size
        self.rate_limiter = RateLimiter(1000, 1.0)  # 1000 requests per second
        self.server = None
    
    def start(self, host: str = "127.0.0.1", port: int = 26658) -> None:
        """Start gRPC server"""
        try:
            self.server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers),
                options=[
                    ('grpc.max_send_message_length', self.max_message_size),
                    ('grpc.max_receive_message_length', self.max_message_size),
                ]
            )
            
            # Add servicer to server
            consensus_pb2_grpc.add_ConsensusServicer_to_server(self, self.server)
            
            # Start server
            self.server.add_insecure_port(f"{host}:{port}")
            self.server.start()
            
            logger.info(f"gRPC server started on {host}:{port}")
            
        except Exception as e:
            raise ConsensusError(f"Failed to start gRPC server: {e}")
    
    def stop(self) -> None:
        """Stop gRPC server"""
        if self.server:
            self.server.stop(0)
            logger.info("gRPC server stopped")
    
    def GetHealth(self, request, context):
        """gRPC health check"""
        if not self.rate_limiter.acquire():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded")
            return consensus_pb2.HealthResponse()
        
        try:
            return consensus_pb2.HealthResponse(
                status=consensus_pb2.HealthResponse.HEALTHY,
                timestamp=int(time.time())
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Health check failed: {e}")
            return consensus_pb2.HealthResponse()
    
    def GetConsensusStatus(self, request, context):
        """Get consensus status"""
        if not self.rate_limiter.acquire():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded")
            return consensus_pb2.ConsensusStatus()
        
        try:
            # This would get actual consensus state
            return consensus_pb2.ConsensusStatus(
                height=0,
                round=0,
                step="unknown",
                validators_count=0,
                peers_count=0
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get consensus status: {e}")
            return consensus_pb2.ConsensusStatus()
    
    def GetValidators(self, request, context):
        """Get validators"""
        if not self.rate_limiter.acquire():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded")
            return consensus_pb2.ValidatorsResponse()
        
        try:
            # This would get actual validator data
            validator = consensus_pb2.Validator(
                address='0x' + '0' * 40,
                stake=1000,
                status=consensus_pb2.Validator.ACTIVE,
                voting_power=100
            )
            return consensus_pb2.ValidatorsResponse(validators=[validator])
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get validators: {e}")
            return consensus_pb2.ValidatorsResponse()
    
    def GetBlock(self, request, context):
        """Get block by height"""
        if not self.rate_limiter.acquire():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded")
            return consensus_pb2.BlockResponse()
        
        try:
            height = request.height
            # This would get actual block data
            return consensus_pb2.BlockResponse(
                height=height,
                hash='0' * 64,
                timestamp=int(time.time()),
                validator='0x' + '0' * 40
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to get block: {e}")
            return consensus_pb2.BlockResponse()
    
    def BroadcastTransaction(self, request, context):
        """Broadcast transaction"""
        if not self.rate_limiter.acquire():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded")
            return consensus_pb2.TransactionResponse()
        
        try:
            transaction_data = request.transaction
            # This would actually broadcast the transaction
            return consensus_pb2.TransactionResponse(
                hash='0x' + '0' * 64,
                status=consensus_pb2.TransactionResponse.PENDING
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Failed to broadcast transaction: {e}")
            return consensus_pb2.TransactionResponse()

class GRPCClient:
    """gRPC client for consensus API"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 26658):
        if not GRPC_AVAILABLE:
            raise ConsensusError("gRPC dependencies not available")
        
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = consensus_pb2_grpc.ConsensusStub(self.channel)
    
    def close(self) -> None:
        """Close gRPC channel"""
        self.channel.close()
    
    def get_health(self) -> Optional[Dict]:
        """Get health status"""
        try:
            response = self.stub.GetHealth(empty_pb2.Empty())
            return {
                'status': consensus_pb2.HealthResponse.Status.Name(response.status),
                'timestamp': response.timestamp
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC health check failed: {e}")
            return None
    
    def get_consensus_status(self) -> Optional[Dict]:
        """Get consensus status"""
        try:
            response = self.stub.GetConsensusStatus(empty_pb2.Empty())
            return {
                'height': response.height,
                'round': response.round,
                'step': response.step,
                'validators_count': response.validators_count,
                'peers_count': response.peers_count
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC consensus status failed: {e}")
            return None
    
    def get_validators(self) -> Optional[List[Dict]]:
        """Get validators"""
        try:
            response = self.stub.GetValidators(empty_pb2.Empty())
            return [
                {
                    'address': validator.address,
                    'stake': validator.stake,
                    'status': consensus_pb2.Validator.Status.Name(validator.status),
                    'voting_power': validator.voting_power
                }
                for validator in response.validators
            ]
        except grpc.RpcError as e:
            logger.error(f"gRPC get validators failed: {e}")
            return None
    
    def get_block(self, height: int) -> Optional[Dict]:
        """Get block by height"""
        try:
            request = consensus_pb2.BlockRequest(height=height)
            response = self.stub.GetBlock(request)
            return {
                'height': response.height,
                'hash': response.hash,
                'timestamp': response.timestamp,
                'validator': response.validator
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC get block failed: {e}")
            return None
    
    def broadcast_transaction(self, transaction: bytes) -> Optional[Dict]:
        """Broadcast transaction"""
        try:
            request = consensus_pb2.TransactionRequest(transaction=transaction)
            response = self.stub.BroadcastTransaction(request)
            return {
                'hash': response.hash,
                'status': consensus_pb2.TransactionResponse.Status.Name(response.status)
            }
        except grpc.RpcError as e:
            logger.error(f"gRPC broadcast transaction failed: {e}")
            return None