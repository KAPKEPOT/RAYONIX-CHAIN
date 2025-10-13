// consensus/types/election.rs
use std::collections::{BTreeMap, HashMap};
use serde::{Deserialize, Serialize};
use bls::{Signature, PublicKey, AggregateSignature};
use merlin::Transcript;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderSelectionResult {
    pub selected_validator: ValidatorId,
    pub slot: Slot,
    pub vrf_output: VRFOutput,
    pub selection_proof: SelectionProof,
    pub probabilities: BTreeMap<ValidatorId, f64>,
    pub temperature: f64,
    pub selection_entropy: f64,
    pub confidence_level: f64,
    pub anti_correlation_metrics: AntiCorrelationMetrics,
    pub geographic_distribution: GeographicDistribution,
    pub temporal_patterns: TemporalPatternAnalysis,
    pub security_verification: SecurityVerification,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRFOutput {
    pub output: [u8; 32],
    pub proof: VRFProof,
    pub randomness: [u8; 32],
    pub slot: Slot,
    pub input: VRFInput,
    pub beacon_contribution: [u8; 32],
    pub quality_metrics: VRFQualityMetrics,
    pub cryptographic_verification: CryptographicVerification,
    pub bias_resistance_analysis: BiasResistanceAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionProof {
    pub challenge: [u8; 32],
    pub signature: [u8; 96],
    pub transcript_commitment: [u8; 32],
    pub proof_timestamp: u64,
    pub proof_version: u8,
    pub merkle_proof: MerkleProof,
    pub aggregate_verification: AggregateVerification,
    pub zero_knowledge_component: ZeroKnowledgeComponent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedProbabilityDistribution {
    pub probabilities: BTreeMap<ValidatorId, f64>,
    pub temperature: f64,
    pub softmax_ensemble: SoftmaxEnsemble,
    pub quality_metrics: ProbabilityQualityMetrics,
    pub uncertainty_quantification: UncertaintyQuantification,
    pub temporal_stability: TemporalStabilityAnalysis,
    pub distribution_entropy: f64,
    pub fairness_indicators: FairnessIndicators,
    pub security_implications: SecurityImplications,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityQualityMetrics {
    pub entropy: f64,
    pub gini_coefficient: f64,
    pub herfindahl_hirschman_index: f64,
    pub effective_number: f64,
    pub concentration_ratio: f64,
    pub fairness_index: f64,
    pub stability_metric: f64,
    pub confidence_level: f64,
    pub numerical_stability: NumericalStabilityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftmaxEnsemble {
    pub variants: Vec<SoftmaxVariant>,
    pub weights: BTreeMap<SoftmaxVariant, f64>,
    pub performance_metrics: BTreeMap<SoftmaxVariant, VariantPerformance>,
    pub adaptive_learning: AdaptiveLearningState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionContext {
    pub current_epoch: Epoch,
    pub current_slot: Slot,
    pub parent_block_hash: BlockHash,
    pub network_conditions: NetworkConditions,
    pub security_parameters: SecurityParameters,
    pub economic_indicators: EconomicIndicators,
    pub historical_context: HistoricalContext,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorScore {
    pub validator_id: ValidatorId,
    pub comprehensive_score: f64,
    pub stake_component: f64,
    pub time_lived_component: f64,
    pub performance_component: f64,
    pub stochastic_component: f64,
    pub correction_factor: f64,
    pub volatility_estimate: f64,
    pub score_confidence_interval: (f64, f64),
    pub historical_consistency: HistoricalConsistency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessedScore {
    pub validator_id: ValidatorId,
    pub raw_score: f64,
    pub outlier_corrected: f64,
    pub variance_stabilized: f64,
    pub network_normalized: f64,
    pub statistical_z_score: f64,
    pub confidence_weight: f64,
    pub robustness_indicator: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureOptimization {
    pub optimal_temperature: f64,
    pub distribution_analysis: DistributionAnalysis,
    pub security_analysis: SecurityAnalysis,
    pub historical_analysis: HistoricalAnalysis,
    pub economic_analysis: EconomicAnalysis,
    pub confidence_level: f64,
    pub stability_assessment: StabilityAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRFQualityMetrics {
    pub bias_resistance: f64,
    pub unpredictability: f64,
    pub verifiability: f64,
    pub efficiency: f64,
    pub overall_quality: f64,
    pub cryptographic_strength: f64,
    pub randomness_quality: f64,
    pub resistance_analysis: ResistanceAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateVRFProof {
    pub aggregate_proof: AggregateSignature,
    pub aggregate_output: [u8; 32],
    pub verification_transcript: [u8; 32],
    pub participant_count: u32,
    pub quality_metrics: AggregateVRFQuality,
    pub cryptographic_verification: AggregateCryptographicVerification,
    pub efficiency_metrics: AggregationEfficiency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedValidator {
    pub validator_id: ValidatorId,
    pub selection_probability: f64,
    pub cumulative_probability: f64,
    pub random_index: usize,
    pub selection_quality: f64,
    pub anti_correlation_status: AntiCorrelationStatus,
    pub geographic_compliance: GeographicCompliance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionVerification {
    pub valid: bool,
    pub claimed_leader: ValidatorId,
    pub slot: Slot,
    pub probability: f64,
    pub confidence_score: f64,
    pub verification_timestamp: u64,
    pub cryptographic_verification: CryptographicVerificationResult,
    pub statistical_verification: StatisticalVerification,
    pub security_audit: SecurityAuditResult,
}

// Advanced election-specific enums
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SoftmaxVariant {
    Standard,
    Sparse,
    Robust,
    TemperatureScaled,
    EntropyMaximizing,
    MultiplicativeWeights,
    BoltzmannExploration,
    AdaptiveRegularization,
    FairnessConstrained,
    SecurityOptimized,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProbabilityError {
    InvalidSelectionSignature,
    VRFConsistencyError(String),
    ProbabilityError(String),
    NumericalUnderflow,
    ZeroSumProbabilities,
    EmptyInput,
    InvalidProbabilitySum(f64),
    InvalidProbabilityValue(ValidatorId, f64),
    ProbabilityOutOfRange(ValidatorId, f64),
    NumericalStabilizationFailed,
    NumericalInstability(ValidatorId, f64),
    DistributionValidationFailed(String),
    TemperatureOptimizationFailed(String),
    EnsembleCombinationError(String),
    ConstraintApplicationError(String),
    StochasticRefinementError(String),
    QualityAssessmentError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VRFError {
    ComputationError(String),
    VerificationError(String),
    AggregationError(String),
    RandomnessError(String),
    QualityThresholdExceeded,
    CryptographicError(String),
    BiasDetectionError(String),
    EfficiencyConstraintViolated(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElectionSecurityLevel {
    Normal,
    Elevated,
    High,
    Critical,
}

// Advanced VRF Types
pub type VRFProof = [u8; 96];
pub type VRFInput = [u8; 32];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicVerification {
    pub signature_valid: bool,
    pub public_key_valid: bool,
    pub transcript_consistent: bool,
    pub timestamp_valid: bool,
    pub version_compatible: bool,
    pub overall_cryptographic_integrity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasResistanceAnalysis {
    pub statistical_bias: f64,
    pub cryptographic_bias: f64,
    pub temporal_bias: f64,
    pub overall_bias_resistance: f64,
    pub bias_confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateVRFQuality {
    pub aggregation_efficiency: f64,
    pub collective_unpredictability: f64,
    pub verification_speed: f64,
    pub robustness: f64,
    pub scalability: f64,
    pub fault_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiCorrelationMetrics {
    pub temporal_correlation: f64,
    pub spatial_correlation: f64,
    pub stake_correlation: f64,
    pub geographic_correlation: f64,
    pub overall_anti_correlation_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicDistribution {
    pub region_balance: f64,
    pub latency_optimization: f64,
    pub regulatory_compliance: f64,
    pub network_topology_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatternAnalysis {
    pub recent_selection_frequency: f64,
    pub historical_pattern_consistency: f64,
    pub temporal_fairness: f64,
    pub pattern_anomaly_detection: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityVerification {
    pub sybil_resistance: f64,
    pub nothing_at_stake_resistance: f64,
    pub long_range_attack_resistance: f64,
    pub grinding_attack_resistance: f64,
    pub overall_security_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyQuantification {
    pub probability_variance: f64,
    pub confidence_intervals: BTreeMap<ValidatorId, (f64, f64)>,
    pub prediction_uncertainty: f64,
    pub model_risk: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalStabilityAnalysis {
    pub short_term_stability: f64,
    pub medium_term_stability: f64,
    pub long_term_stability: f64,
    pub regime_change_detection: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessIndicators {
    pub individual_fairness: f64,
    pub group_fairness: f64,
    pub procedural_fairness: f64,
    pub distributive_fairness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityImplications {
    pub attack_surface_analysis: f64,
    pub incentive_compatibility: f64,
    pub game_theoretic_stability: f64,
    pub cryptographic_security: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalStabilityMetrics {
    pub condition_number: f64,
    pub floating_point_error: f64,
    pub numerical_precision: f64,
    pub stability_margin: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantPerformance {
    pub historical_accuracy: f64,
    pub stability_contribution: f64,
    pub fairness_contribution: f64,
    pub security_contribution: f64,
    pub computational_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningState {
    pub learning_rate: f64,
    pub exploration_factor: f64,
    pub performance_history: Vec<f64>,
    pub adaptation_speed: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConditions {
    pub latency_distribution: LatencyDistribution,
    pub bandwidth_utilization: f64,
    pub node_connectivity: f64,
    pub network_congestion: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityParameters {
    pub minimum_security_level: ElectionSecurityLevel,
    pub attack_resistance_requirements: AttackResistanceRequirements,
    pub cryptographic_standards: CryptographicStandards,
    pub audit_requirements: AuditRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicIndicators {
    pub inflation_rate: f64,
    pub stake_concentration: f64,
    pub validator_economics: ValidatorEconomics,
    pub incentive_structure: IncentiveStructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalContext {
    pub recent_elections: Vec<EpochPerformance>,
    pub pattern_analysis: PatternAnalysis,
    pub anomaly_detection: AnomalyDetection,
    pub trend_analysis: TrendAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalConsistency {
    pub score_stability: f64,
    pub performance_consistency: f64,
    pub behavioral_patterns: f64,
    pub reliability_trend: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    pub skewness: f64,
    pub kurtosis: f64,
    pub multimodality: f64,
    tail_behavior: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAnalysis {
    pub threat_level: f64,
    pub vulnerability_assessment: f64,
    pub risk_mitigation: f64,
    pub security_optimization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalAnalysis {
    pub pattern_consistency: f64,
    pub regime_stability: f64,
    pub adaptation_effectiveness: f64,
    pub learning_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EconomicAnalysis {
    pub incentive_alignment: f64,
    pub economic_efficiency: f64,
    pub stake_optimization: f64,
    pub reward_distribution: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAssessment {
    pub sensitivity_analysis: f64,
    pub robustness_evaluation: f64,
    pub convergence_properties: f64,
    pub stability_margins: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResistanceAnalysis {
    pub cryptographic_resistance: f64,
    pub statistical_resistance: f64,
    pub implementation_resistance: f64,
    pub overall_resistance_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateCryptographicVerification {
    pub signature_aggregation_valid: bool,
    pub participant_verification: bool,
    pub consistency_check: bool,
    pub security_guarantees: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationEfficiency {
    pub computational_complexity: f64,
    pub communication_overhead: f64,
    pub scalability_metrics: f64,
    pub resource_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntiCorrelationStatus {
    pub temporal_compliance: bool,
    pub spatial_compliance: bool,
    pub stake_compliance: bool,
    pub overall_compliance: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicCompliance {
    pub region_balance: bool,
    pub latency_requirements: bool,
    pub regulatory_requirements: bool,
    pub topology_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CryptographicVerificationResult {
    pub signature_verification: bool,
    pub key_validation: bool,
    pub proof_verification: bool,
    pub timestamp_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalVerification {
    pub probability_validation: bool,
    pub distribution_validation: bool,
    pub randomness_validation: bool,
    pub fairness_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAuditResult {
    pub sybil_attack_audit: bool,
    pub grinding_attack_audit: bool,
    pub nothing_at_stake_audit: bool,
    pub long_range_attack_audit: bool,
}

// Election Configuration with comprehensive parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionConfig {
    pub base_temperature: f64,
    pub min_temperature: f64,
    pub max_temperature: f64,
    pub default_softmax_variant: SoftmaxVariant,
    pub vrf_quality_threshold: f64,
    pub selection_confidence_threshold: f64,
    pub anti_correlation_window: u64,
    pub geographic_distribution_weight: f64,
    pub temporal_anti_correlation_strength: f64,
    pub numerical_precision_requirements: NumericalPrecisionRequirements,
    pub security_parameters: ElectionSecurityParameters,
    pub fairness_constraints: FairnessConstraints,
    pub performance_requirements: PerformanceRequirements,
    pub adaptive_learning_config: AdaptiveLearningConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalPrecisionRequirements {
    pub floating_point_precision: f64,
    pub error_tolerance: f64,
    pub stability_threshold: f64,
    pub convergence_criteria: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectionSecurityParameters {
    pub minimum_cryptographic_strength: u32,
    pub bias_resistance_requirement: f64,
    pub attack_resistance_level: f64,
    pub audit_frequency: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConstraints {
    pub individual_fairness_bound: f64,
    pub group_fairness_bound: f64,
    pub temporal_fairness_bound: f64,
    pub geographic_fairness_bound: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRequirements {
    pub computation_time_limit: u64,
    pub memory_usage_limit: u64,
    pub scalability_requirements: ScalabilityRequirements,
    pub efficiency_metrics: EfficiencyMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningConfig {
    pub learning_rate: f64,
    pub exploration_rate: f64,
    pub adaptation_speed: f64,
    pub performance_weight: f64,
}

impl Default for ElectionConfig {
    fn default() -> Self {
        Self {
            base_temperature: 1.0,
            min_temperature: 0.1,
            max_temperature: 10.0,
            default_softmax_variant: SoftmaxVariant::Standard,
            vrf_quality_threshold: 0.8,
            selection_confidence_threshold: 0.9,
            anti_correlation_window: 64,
            geographic_distribution_weight: 0.3,
            temporal_anti_correlation_strength: 0.5,
            numerical_precision_requirements: NumericalPrecisionRequirements {
                floating_point_precision: 1e-12,
                error_tolerance: 1e-8,
                stability_threshold: 1e-6,
                convergence_criteria: 1e-10,
            },
            security_parameters: ElectionSecurityParameters {
                minimum_cryptographic_strength: 128,
                bias_resistance_requirement: 0.95,
                attack_resistance_level: 0.9,
                audit_frequency: 1000,
            },
            fairness_constraints: FairnessConstraints {
                individual_fairness_bound: 0.1,
                group_fairness_bound: 0.05,
                temporal_fairness_bound: 0.15,
                geographic_fairness_bound: 0.2,
            },
            performance_requirements: PerformanceRequirements {
                computation_time_limit: 1000,
                memory_usage_limit: 1024 * 1024 * 100, // 100 MB
                scalability_requirements: ScalabilityRequirements::default(),
                efficiency_metrics: EfficiencyMetrics::default(),
            },
            adaptive_learning_config: AdaptiveLearningConfig {
                learning_rate: 0.01,
                exploration_rate: 0.1,
                adaptation_speed: 0.05,
                performance_weight: 0.7,
            },
        }
    }
}