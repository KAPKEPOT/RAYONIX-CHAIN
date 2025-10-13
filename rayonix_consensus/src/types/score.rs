// consensus/types/score.rs
use std::collections::BTreeMap;
use serde::{Serialize, Deserialize};
use statrs::distribution::{Normal, Continuous};
use statrs::statistics::Statistics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreCalculationContext {
    pub epoch: Epoch,
    pub total_network_stake: u128,
    pub active_validator_count: u64,
    pub network_load_factor: f64,
    pub security_parameter: f64,
    pub decentralization_index: f64,
    pub previous_epoch_scores: BTreeMap<ValidatorId, f64>,
    pub score_distribution_stats: ScoreStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreStatistics {
    pub mean: f64,
    pub standard_deviation: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub gini_coefficient: f64,
    pub entropy: f64,
    pub percentile_25: f64,
    pub percentile_50: f64,
    pub percentile_75: f64,
    pub percentile_95: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticParameters {
    pub base_volatility: f64,
    pub mean_reversion_speed: f64,
    pub volatility_clustering_factor: f64,
    pub jump_diffusion_intensity: f64,
    pub jump_size_distribution: Normal,
    pub correlation_matrix: Option<Vec<Vec<f64>>>,
    pub random_seed: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightParameters {
    pub alpha_stake: f64,
    pub beta_time_lived: f64,
    pub gamma_performance: f64,
    pub delta_penalty: f64,
    pub epsilon_correction: f64,
    pub adaptive_learning_rate: f64,
    pub momentum_factor: f64,
    pub regularization_parameter: f64,
}