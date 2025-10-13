// consensus/types/validator.rs
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Serialize, Deserialize};
use bincode::{Encode, Decode};
use rayon::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize, Encode, Decode)]
pub struct ValidatorId([u8; 32]);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorIdentity {
    pub id: ValidatorId,
    pub consensus_key: [u8; 48],
    pub operational_key: [u8; 32],
    pub withdrawal_credentials: [u8; 32],
    pub fee_recipient: [u8; 20],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorStakeState {
    pub effective_stake: u128,
    pub pending_stake_changes: BTreeMap<Epoch, StakeChange>,
    pub slashable_balance: u128,
    pub withdrawal_queue: Vec<WithdrawalRequest>,
    pub last_stake_update: Epoch,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorPerformanceMetrics {
    pub blocks_proposed: u64,
    pub blocks_missed: u64,
    pub attestations_made: u64,
    pub attestations_missed: u64,
    pub sync_committee_participation: u64,
    pub sync_committee_misses: u64,
    pub average_latency_ms: f64,
    pub uptime_percentage: f64,
    pub consecutive_successes: u32,
    pub consecutive_failures: u32,
    pub historical_performance: BTreeMap<Epoch, PerformanceSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeLivedState {
    pub exponential_moving_average: f64,
    pub cumulative_reliability: f64,
    pub reliability_decay_factor: f64,
    pub last_reliability_update: Epoch,
    pub reliability_history: BTreeMap<Epoch, ReliabilityEntry>,
    pub grace_period_remaining: u64,
    pub forced_exit_epoch: Option<Epoch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorScoreComponents {
    pub stake_component: f64,
    pub time_lived_component: f64,
    pub performance_component: f64,
    pub stochastic_component: f64,
    pub correction_factor: f64,
    pub volatility_estimate: f64,
    pub score_confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveValidator {
    pub identity: ValidatorIdentity,
    pub stake_state: ValidatorStakeState,
    pub performance: ValidatorPerformanceMetrics,
    pub time_lived_state: TimeLivedState,
    pub score_components: ValidatorScoreComponents,
    pub current_score: f64,
    pub status: ValidatorStatus,
    pub activation_epoch: Epoch,
    pub exit_epoch: Option<Epoch>,
    pub slashing_history: Vec<SlashingEvent>,
    pub reward_history: BTreeMap<Epoch, RewardDistribution>,
    pub metadata: ValidatorMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JailedValidator {
    pub validator: ActiveValidator,
    pub jail_epoch: Epoch,
    pub release_epoch: Epoch,
    pub jail_reason: JailReason,
    pub slash_amount: u128,
    pub appeal_status: AppealStatus,
    pub rehabilitation_requirements: RehabilitationCriteria,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingValidator {
    pub identity: ValidatorIdentity,
    pub deposited_stake: u128,
    pub submission_epoch: Epoch,
    pub activation_eligibility_epoch: Epoch,
    pub queue_position: u64,
    pub validation_status: ValidationStatus,
    pub technical_requirements: TechnicalCompliance,
}