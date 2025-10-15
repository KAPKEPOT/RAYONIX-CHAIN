# blockchain/consensus/grpc_client.py
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
import grpc
from google.protobuf import json_format

from blockchain.consensus.proto import consensus_pb2, consensus_pb2_grpc

logger = logging.getLogger(__name__)

class RayonixGRPCClient:
    """
    gRPC client for RAYONIX Consensus Engine
    """
    
    def __init__(self, host: str = "localhost", port: int = 50051, 
                 max_retries: int = 3, timeout: int = 30):
        self.host = host
        self.port = port
        self.max_retries = max_retries
        self.timeout = timeout
        self.channel = None
        self.stub = None
        self._connect_task = None
        
    async def connect(self):
        """Establish connection to gRPC server"""
        if self.channel is None:
            try:
                self.channel = grpc.aio.insecure_channel(
                    f"{self.host}:{self.port}",
                    options=[
                        ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
                        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                        ('grpc.keepalive_time_ms', 10000),
                        ('grpc.keepalive_timeout_ms', 5000),
                    ]
                )
                self.stub = consensus_pb2_grpc.ConsensusServiceStub(self.channel)
                logger.info(f"Connected to consensus engine at {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Failed to connect to consensus engine: {e}")
                raise
                
    async def disconnect(self):
        """Close connection"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
            
    async def ensure_connected(self):
        """Ensure connection is established"""
        if self.channel is None:
            await self.connect()
            
        # Test connection with health check
        try:
            await self.health_check()
        except Exception:
            await self.connect()
            
    async def process_slot(self, slot: int, parent_block_hash: str, 
                          validators: List[Dict], network_state: Dict) -> Dict[str, Any]:
        """
        Process a slot through consensus engine
        """
        await self.ensure_connected()
        
        # Convert Python types to protobuf
        validators_pb = [self._validator_to_proto(v) for v in validators]
        network_state_pb = self._network_state_to_proto(network_state)
        
        request = consensus_pb2.SlotRequest(
            slot=slot,
            parent_block_hash=parent_block_hash,
            validators=validators_pb,
            network_state=network_state_pb
        )
        
        for attempt in range(self.max_retries):
            try:
                response = await self.stub.ProcessSlot(
                    request, 
                    timeout=self.timeout
                )
                return self._slot_response_to_dict(response)
                
            except grpc.RpcError as e:
                logger.warning(f"Slot processing attempt {attempt + 1} failed: {e.code()} - {e.details()}")
                if attempt == self.max_retries - 1:
                    logger.error(f"All slot processing attempts failed for slot {slot}")
                    raise
                await asyncio.sleep(1 << attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Unexpected error processing slot {slot}: {e}")
                raise
                
    async def process_epoch(self, epoch: int, network_state: Dict) -> Dict[str, Any]:
        """
        Process epoch transition
        """
        await self.ensure_connected()
        
        request = consensus_pb2.EpochRequest(
            epoch=epoch,
            network_state=self._network_state_to_proto(network_state)
        )
        
        try:
            response = await self.stub.ProcessEpoch(request, timeout=self.timeout)
            return self._epoch_response_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"Epoch processing failed: {e.code()} - {e.details()}")
            raise
            
    async def get_validator_info(self, validator_id: str) -> Optional[Dict[str, Any]]:
        """
        Get validator information from consensus
        """
        await self.ensure_connected()
        
        request = consensus_pb2.ValidatorRequest(validator_id=validator_id)
        
        try:
            response = await self.stub.GetValidatorInfo(request, timeout=10)
            if response.info:
                return self._validator_info_to_dict(response.info)
            return None
        except grpc.RpcError as e:
            logger.warning(f"Failed to get validator info for {validator_id}: {e}")
            return None
            
    async def health_check(self) -> bool:
        """
        Check if consensus engine is healthy
        """
        if self.stub is None:
            return False
            
        try:
            request = consensus_pb2.HealthCheckRequest()
            response = await self.stub.HealthCheck(request, timeout=5)
            return response.status == "SERVING"
        except grpc.RpcError:
            return False
            
    async def optimize_parameters(self, historical_data: List[Dict]) -> Dict[str, Any]:
        """
        Optimize consensus parameters
        """
        await self.ensure_connected()
        
        historical_data_pb = [self._historical_data_to_proto(d) for d in historical_data]
        request = consensus_pb2.OptimizeRequest(historical_data=historical_data_pb)
        
        try:
            response = await self.stub.OptimizeParameters(request, timeout=60)
            return self._optimize_response_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"Parameter optimization failed: {e}")
            raise
            
    async def handle_fork(self, fork_data: Dict) -> Dict[str, Any]:
        """
        Handle fork situation
        """
        await self.ensure_connected()
        
        fork_data_pb = self._fork_data_to_proto(fork_data)
        request = consensus_pb2.ForkRequest(fork_data=fork_data_pb)
        
        try:
            response = await self.stub.HandleFork(request, timeout=30)
            return self._fork_response_to_dict(response)
        except grpc.RpcError as e:
            logger.error(f"Fork handling failed: {e}")
            raise
            
    async def stream_events(self, event_types: List[str]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time consensus events
        """
        await self.ensure_connected()
        
        request = consensus_pb2.StreamRequest(event_types=event_types)
        
        try:
            async for event in self.stub.StreamEvents(request):
                yield self._event_to_dict(event)
        except grpc.RpcError as e:
            logger.error(f"Event streaming failed: {e}")
            
    # Conversion methods
    def _validator_to_proto(self, validator: Dict) -> consensus_pb2.Validator:
        return consensus_pb2.Validator(
            validator_id=validator['validator_id'],
            stake=validator.get('stake', 0),
            reliability_score=validator.get('reliability_score', 0.0),
            total_score=validator.get('total_score', 0.0),
            performance=self._performance_to_proto(validator.get('performance', {})),
            time_lived=self._time_lived_to_proto(validator.get('time_lived_components', {}))
        )
        
    def _network_state_to_proto(self, network_state: Dict) -> consensus_pb2.NetworkState:
        return consensus_pb2.NetworkState(
            block_height=network_state.get('block_height', 0),
            total_stake=network_state.get('total_stake', 0),
            validator_count=network_state.get('validator_count', 0),
            network_load=network_state.get('network_load', 0.0),
            security_parameter=network_state.get('security_parameter', 1.0),
            decentralization_index=network_state.get('decentralization_index', 0.0),
            average_latency=network_state.get('average_latency', 0.0),
            fork_probability=network_state.get('fork_probability', 0.0),
            economic_indicators=self._economic_indicators_to_proto(
                network_state.get('economic_indicators', {})
            )
        )
        
    def _performance_to_proto(self, performance: Dict) -> consensus_pb2.PerformanceMetrics:
        return consensus_pb2.PerformanceMetrics(
            uptime_percentage=performance.get('uptime_percentage', 0.0),
            blocks_proposed=performance.get('blocks_proposed', 0),
            blocks_missed=performance.get('blocks_missed', 0),
            average_latency_ms=performance.get('average_latency_ms', 0.0),
            consecutive_successes=performance.get('consecutive_successes', 0)
        )
        
    def _time_lived_to_proto(self, time_lived: Dict) -> consensus_pb2.TimeLivedComponents:
        return consensus_pb2.TimeLivedComponents(
            exponential_moving_average=time_lived.get('exponential_moving_average', 0.0),
            cumulative_reliability=time_lived.get('cumulative_reliability', 0.0),
            activation_epoch=time_lived.get('activation_epoch', 0),
            last_reliability_update=time_lived.get('last_reliability_update', 0)
        )
        
    def _economic_indicators_to_proto(self, indicators: Dict) -> consensus_pb2.EconomicIndicators:
        return consensus_pb2.EconomicIndicators(
            inflation_rate=indicators.get('inflation_rate', 0.0),
            stake_ratio=indicators.get('stake_ratio', 0.0),
            reward_rate=indicators.get('reward_rate', 0.0),
            penalty_rate=indicators.get('penalty_rate', 0.0)
        )
        
    def _slot_response_to_dict(self, response: consensus_pb2.SlotResponse) -> Dict[str, Any]:
        return {
            'is_leader': response.is_leader,
            'selected_validator': response.selected_validator,
            'validator_scores': dict(response.validator_scores),
            'vrf_output': self._vrf_to_dict(response.vrf_output) if response.vrf_output else None,
            'metrics': self._metrics_to_dict(response.metrics) if response.metrics else None,
            'error_message': response.error_message
        }
        
    def _epoch_response_to_dict(self, response: consensus_pb2.EpochResponse) -> Dict[str, Any]:
        result = {}
        if response.result:
            result = {
                'stake_components': dict(response.result.stake_components),
                'stake_power': dict(response.result.stake_power),
                'time_lived_components': dict(response.result.time_lived_components),
                'reliability_components': dict(response.result.reliability_components),
                'comprehensive_scores': dict(response.result.comprehensive_scores),
                'slot_results': [self._slot_result_to_dict(sr) for sr in response.result.slot_results],
                'reward_distribution': self._reward_distribution_to_dict(response.result.reward_distribution)
            }
            
        return {
            'epoch': response.epoch,
            'result': result,
            'metrics': self._metrics_to_dict(response.metrics) if response.metrics else None
        }
        
    def _slot_result_to_dict(self, slot_result: consensus_pb2.SlotResult) -> Dict[str, Any]:
        return {
            'slot': slot_result.slot,
            'leader_selection': self._leader_selection_to_dict(slot_result.leader_selection),
            'vrf_output': self._vrf_to_dict(slot_result.vrf_output),
            'temperature': slot_result.temperature,
            'anomaly_detection': self._anomaly_detection_to_dict(slot_result.anomaly_detection)
        }
        
    def _leader_selection_to_dict(self, selection: consensus_pb2.LeaderSelection) -> Dict[str, Any]:
        return {
            'selected_validator': selection.selected_validator,
            'selection_proof': selection.selection_proof.hex() if selection.selection_proof else None,
            'selection_score': selection.selection_score
        }
        
    def _vrf_to_dict(self, vrf: consensus_pb2.VRFOutput) -> Dict[str, Any]:
        return {
            'proof': vrf.proof.hex() if vrf.proof else None,
            'output': vrf.output.hex() if vrf.output else None,
            'validator_id': vrf.validator_id
        }
        
    def _anomaly_detection_to_dict(self, anomaly: consensus_pb2.AnomalyDetection) -> Dict[str, Any]:
        return {
            'anomaly_detected': anomaly.anomaly_detected,
            'anomaly_type': anomaly.anomaly_type,
            'severity': anomaly.severity,
            'affected_validators': list(anomaly.affected_validators)
        }
        
    def _reward_distribution_to_dict(self, distribution: consensus_pb2.RewardDistribution) -> Dict[str, Any]:
        return {
            'rewards': dict(distribution.rewards),
            'total_distributed': distribution.total_distributed,
            'foundation_reward': distribution.foundation_reward
        }
        
    def _metrics_to_dict(self, metrics: consensus_pb2.ConsensusMetrics) -> Dict[str, Any]:
        return {
            'epoch': metrics.epoch,
            'slot': metrics.slot,
            'validator_count': metrics.validator_count,
            'total_stake': metrics.total_stake,
            'average_score': metrics.average_score,
            'gini_coefficient': metrics.gini_coefficient,
            'entropy': metrics.entropy,
            'security_score': metrics.security_score,
            'health_score': metrics.health_score
        }
        
    def _validator_info_to_dict(self, info: consensus_pb2.ValidatorInfo) -> Dict[str, Any]:
        return {
            'validator_id': info.validator_id,
            'stake': info.stake,
            'reliability_score': info.reliability_score,
            'total_score': info.total_score,
            'selection_probability': info.selection_probability,
            'status': info.status,
            'last_active_slot': info.last_active_slot
        }
        
    def _optimize_response_to_dict(self, response: consensus_pb2.OptimizeResponse) -> Dict[str, Any]:
        return {
            'parameter_adjustments': dict(response.parameter_adjustments),
            'expected_improvement': response.expected_improvement,
            'risk_level': response.risk_level
        }
        
    def _fork_response_to_dict(self, response: consensus_pb2.ForkResponse) -> Dict[str, Any]:
        plan = {}
        if response.plan:
            plan = {
                'target_height': response.plan.target_height,
                'blocks_to_rollback': list(response.plan.blocks_to_rollback),
                'blocks_to_add': list(response.plan.blocks_to_add),
                'common_ancestor': response.plan.common_ancestor
            }
            
        return {
            'requires_reorganization': response.requires_reorganization,
            'plan': plan,
            'emergency_level': response.emergency_level
        }
        
    def _historical_data_to_proto(self, data: Dict) -> consensus_pb2.HistoricalData:
        return consensus_pb2.HistoricalData(
            epoch=data.get('epoch', 0),
            metrics=self._metrics_to_proto(data.get('metrics', {})),
            validator_performance=data.get('validator_performance', {}),
            network_state=self._network_state_to_proto(data.get('network_state', {}))
        )
        
    def _fork_data_to_proto(self, data: Dict) -> consensus_pb2.ForkData:
        return consensus_pb2.ForkData(
            fork_hash=data.get('fork_hash', ''),
            fork_height=data.get('fork_height', 0),
            conflicting_validators=data.get('conflicting_validators', []),
            fork_severity=data.get('fork_severity', 0.0)
        )
        
    def _event_to_dict(self, event: consensus_pb2.ConsensusEvent) -> Dict[str, Any]:
        return {
            'type': event.type,
            'timestamp': event.timestamp,
            'data': event.data,
            'severity': event.severity
        }
        
    def _metrics_to_proto(self, metrics: Dict) -> consensus_pb2.ConsensusMetrics:
        return consensus_pb2.ConsensusMetrics(
            epoch=metrics.get('epoch', 0),
            slot=metrics.get('slot', 0),
            validator_count=metrics.get('validator_count', 0),
            total_stake=metrics.get('total_stake', 0),
            average_score=metrics.get('average_score', 0.0),
            gini_coefficient=metrics.get('gini_coefficient', 0.0),
            entropy=metrics.get('entropy', 0.0),
            security_score=metrics.get('security_score', 0.0),
            health_score=metrics.get('health_score', 0.0)
        )