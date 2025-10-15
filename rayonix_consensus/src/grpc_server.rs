// consensus/src/grpc_server.rs
use tonic::{transport::Server, Request, Response, Status};
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json;

use crate::hybrid_orchestrator::HybridOrchestrator;
use crate::types::*;

#[derive(Default)]
pub struct ConsensusServiceImpl {
    orchestrator: Arc<HybridOrchestrator>,
    is_healthy: Arc<RwLock<bool>>,
}

impl ConsensusServiceImpl {
    pub fn new(orchestrator: Arc<HybridOrchestrator>) -> Self {
        Self {
            orchestrator,
            is_healthy: Arc::new(RwLock::new(true)),
        }
    }
}

#[tonic::async_trait]
impl ConsensusService for ConsensusServiceImpl {
    async fn process_slot(
        &self,
        request: Request<SlotRequest>,
    ) -> Result<Response<SlotResponse>, Status> {
        let slot_data = request.into_inner();
        
        // Convert protobuf to internal types
        let validators = convert_validators_from_proto(slot_data.validators);
        let network_state = convert_network_state_from_proto(slot_data.network_state);
        
        let result = self.orchestrator
            .process_slot(
                slot_data.slot,
                &slot_data.parent_block_hash,
                &validators,
                &network_state,
            )
            .await
            .map_err(|e| Status::internal(format!("Slot processing failed: {}", e)))?;

        // Convert internal result to protobuf
        let response = SlotResponse {
            is_leader: is_current_validator_leader(&result.leader_selection),
            selected_validator: result.leader_selection.selected_validator,
            validator_scores: result.comprehensive_scores.scores,
            vrf_output: Some(convert_vrf_to_proto(&result.vrf_output)),
            metrics: Some(convert_metrics_to_proto(&result.slot_metrics)),
            error_message: String::new(),
        };

        Ok(Response::new(response))
    }

    async fn process_epoch(
        &self,
        request: Request<EpochRequest>,
    ) -> Result<Response<EpochResponse>, Status> {
        let epoch_data = request.into_inner();
        let network_state = convert_network_state_from_proto(epoch_data.network_state);
        
        let result = self.orchestrator
            .execute_consensus_epoch(epoch_data.epoch, &network_state)
            .await
            .map_err(|e| Status::internal(format!("Epoch processing failed: {}", e)))?;

        let response = EpochResponse {
            epoch: result.epoch,
            result: Some(convert_epoch_result_to_proto(&result)),
            metrics: Some(convert_metrics_to_proto(&result.epoch_metrics)),
        };

        Ok(Response::new(response))
    }

    async fn get_validator_info(
        &self,
        request: Request<ValidatorRequest>,
    ) -> Result<Response<ValidatorResponse>, Status> {
        let validator_id = request.into_inner().validator_id;
        
        // In a real implementation, you'd fetch this from the orchestrator
        let validator_info = ValidatorInfo {
            validator_id: validator_id.clone(),
            stake: 1000, // Example
            reliability_score: 0.95,
            total_score: 0.88,
            selection_probability: 0.02,
            status: "active".to_string(),
            last_active_slot: 1000,
        };

        let response = ValidatorResponse {
            info: Some(validator_info),
        };

        Ok(Response::new(response))
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        let is_healthy = *self.is_healthy.read().await;
        
        let response = HealthCheckResponse {
            status: if is_healthy { "SERVING".to_string() } else { "NOT_SERVING".to_string() },
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: 0, // You'd track this
        };

        Ok(Response::new(response))
    }

    async fn optimize_parameters(
        &self,
        request: Request<OptimizeRequest>,
    ) -> Result<Response<OptimizeResponse>, Status> {
        let optimization_data = request.into_inner();
        let historical_data = convert_historical_data_from_proto(optimization_data.historical_data);
        
        let result = self.orchestrator
            .optimize_hybrid_parameters(&historical_data, &NetworkState::default())
            .await
            .map_err(|e| Status::internal(format!("Parameter optimization failed: {}", e)))?;

        let response = OptimizeResponse {
            parameter_adjustments: result.parameter_adjustments,
            expected_improvement: result.expected_improvement,
            risk_level: result.implementation_risk.to_string(),
        };

        Ok(Response::new(response))
    }

    async fn handle_fork(
        &self,
        request: Request<ForkRequest>,
    ) -> Result<Response<ForkResponse>, Status> {
        let fork_data = request.into_inner().fork_data
            .ok_or_else(|| Status::invalid_argument("Missing fork data"))?;

        let crisis_type = CrisisType::Fork;
        let severity = CrisisSeverity::from(fork_data.fork_severity);
        let triggering_events = vec![CrisisEvent::ForkDetected];

        let emergency_response = self.orchestrator
            .handle_emergency_situation(
                crisis_type,
                severity,
                triggering_events,
                &NetworkState::default(),
            )
            .await
            .map_err(|e| Status::internal(format!("Fork handling failed: {}", e)))?;

        let response = ForkResponse {
            requires_reorganization: true, // You'd determine this based on response
            plan: Some(ReorganizationPlan::default()), // You'd create this
            emergency_level: "high".to_string(),
        };

        Ok(Response::new(response))
    }

    type StreamEventsStream = ReceiverStream<Result<ConsensusEvent, Status>>;

    async fn stream_events(
        &self,
        request: Request<StreamRequest>,
    ) -> Result<Response<Self::StreamEventsStream>, Status> {
        let event_types = request.into_inner().event_types;
        let (tx, rx) = tokio::sync::mpsc::channel(100);

        // In real implementation, you'd hook this into the orchestrator's event system
        tokio::spawn(async move {
            // Example event streaming
            let event = ConsensusEvent {
                type: "slot_processed".to_string(),
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                data: serde_json::to_vec(&serde_json::json!({"slot": 1000})).unwrap(),
                severity: "info".to_string(),
            };

            let _ = tx.send(Ok(event)).await;
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

// Conversion functions
fn convert_validators_from_proto(validators: Vec<Validator>) -> Vec<ActiveValidator> {
    validators.into_iter().map(|v| ActiveValidator {
        identity: ValidatorIdentity { id: v.validator_id },
        stake_state: StakeState {
            total_stake: v.stake,
            ..Default::default()
        },
        performance: convert_performance_from_proto(v.performance.unwrap_or_default()),
        time_lived_state: convert_time_lived_from_proto(v.time_lived.unwrap_or_default()),
        activation_epoch: 0, // You'd set this
        current_score: v.total_score,
    }).collect()
}

fn convert_network_state_from_proto(network_state: NetworkState) -> NetworkStateInternal {
    NetworkStateInternal {
        block_height: network_state.block_height,
        total_stake: network_state.total_stake,
        validator_count: network_state.validator_count,
        network_load: network_state.network_load,
        security_parameter: network_state.security_parameter,
        decentralization_index: network_state.decentralization_index,
        average_latency: network_state.average_latency,
        fork_probability: network_state.fork_probability,
        economic_indicators: convert_economic_indicators_from_proto(
            network_state.economic_indicators.unwrap_or_default()
        ),
        ..Default::default()
    }
}

fn convert_performance_from_proto(metrics: PerformanceMetrics) -> ValidatorPerformance {
    ValidatorPerformance {
        uptime_percentage: metrics.uptime_percentage,
        blocks_proposed: metrics.blocks_proposed,
        blocks_missed: metrics.blocks_missed,
        average_latency_ms: metrics.average_latency_ms,
        consecutive_successes: metrics.consecutive_successes,
        ..Default::default()
    }
}

// Add more conversion functions as needed...

// Main server function
pub async fn run_grpc_server(
    orchestrator: Arc<HybridOrchestrator>,
    address: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = address.parse()?;
    let service = ConsensusServiceImpl::new(orchestrator);

    println!("Starting gRPC server on {}", addr);

    Server::builder()
        .add_service(ConsensusServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}