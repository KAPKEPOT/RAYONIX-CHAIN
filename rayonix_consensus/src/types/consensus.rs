// consensus/types/consensus.rs
use std::collections::BTreeMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorId(pub [u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveValidator {
    pub identity: ValidatorIdentity,
    pub stake_state: StakeState,
    pub time_lived_state: TimeLivedState,
    pub performance: ValidatorPerformance,
    pub current_score: f64,
    pub activation_epoch: Epoch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorIdentity {
    pub id: ValidatorId,
    pub consensus_key: PublicKey,
    pub withdrawal_key: PublicKey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeState {
    pub effective_stake: u128,
    pub pending_stake: u128,
    pub withdrawal_epoch: Option<Epoch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeLivedState {
    pub exponential_moving_average: f64,
    pub cumulative_reliability: f64,
    pub last_reliability_update: Epoch,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformance {
    pub blocks_proposed: u64,
    pub blocks_missed: u64,
    pub attestations_made: u64,
    pub attestations_missed: u64,
    pub sync_committee_participation: u64,
    pub sync_committee_misses: u64,
    pub average_latency_ms: f64,
    pub uptime_percentage: f64,
    pub historical_performance: BTreeMap<Epoch, PerformanceSnapshot>,
}

// Core consensus primitives
pub type Epoch = u64;
pub type Slot = u64;
pub type BlockHash = [u8; 32];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    pub current_epoch: Epoch,
    pub current_slot: Slot,
    pub total_network_stake: u128,
    pub active_validators: usize,
    pub total_validators: usize,
    pub security_level: SecurityLevel,
    pub average_propagation_time_ms: f64,
    pub max_propagation_time_ms: f64,
    pub transaction_volume: f64,
    pub decentralization_index: f64,
    pub network_load: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusState {
    pub finalized_checkpoint: Checkpoint,
    pub justified_checkpoint: Checkpoint,
    pub current_slot: Slot,
    pub previous_justified_epoch: Epoch,
    pub previous_finalized_epoch: Epoch,
}