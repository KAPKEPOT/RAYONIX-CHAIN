// consensus/core/election/leader_selector.rs
use crate::types::*;
use std::collections::{BTreeMap, BinaryHeap, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Categorical, Continuous, Normal},
    statistics::Statistics,
};
use rand::prelude::*;
use rand_distr::{WeightedAliasIndex, Beta};
use bls::AggregateSignature;
use merlin::Transcript;

pub struct LeaderSelector {
    vrf_provider: Arc<dyn VRFProvider>,
    probability_calculator: ProbabilityCalculator,
    anti_correlation_engine: AntiCorrelationEngine,
    geographic_distributor: GeographicDistributor,
    temporal_scheduler: TemporalScheduler,
    selection_verifier: SelectionVerifier,
    entropy_manager: EntropyManager,
}

impl LeaderSelector {
    pub async fn select_leader_for_slot(
        &self,
        slot: Slot,
        validators: &[ActiveValidator],
        parent_block_hash: BlockHash,
        network_state: &NetworkState,
    ) -> Result<LeaderSelectionResult, ElectionError> {
        // Phase 1: Pre-selection validation and setup
        self.validate_selection_conditions(slot, validators, network_state).await?;
        
        // Phase 2: Calculate selection probabilities with advanced softmax
        let selection_probabilities = self.calculate_selection_probabilities(validators, slot, network_state).await?;
        
        // Phase 3: Generate verifiable random function output
        let vrf_output = self.generate_vrf_output(slot, parent_block_hash, validators).await?;
        
        // Phase 4: Apply anti-correlation measures
        let anti_correlated_probabilities = self.apply_anti_correlation_measures(
            &selection_probabilities, 
            slot, 
            validators
        ).await?;
        
        // Phase 5: Apply geographic distribution constraints
        let geographically_constrained = self.apply_geographic_constraints(
            &anti_correlated_probabilities, 
            validators, 
            network_state
        ).await?;
        
        // Phase 6: Perform weighted random selection with VRF
        let selected_validator = self.perform_weighted_selection(
            &geographically_constrained, 
            vrf_output.randomness
        ).await?;
        
        // Phase 7: Generate cryptographic proof of selection
        let selection_proof = self.generate_selection_proof(
            &selected_validator, 
            &vrf_output, 
            slot
        ).await?;
        
        // Phase 8: Verify selection integrity
        self.verify_selection_integrity(
            &selected_validator, 
            &selection_proof, 
            &vrf_output, 
            slot
        ).await?;
        
        let result = LeaderSelectionResult {
            selected_validator: selected_validator.validator_id,
            slot,
            vrf_output: vrf_output.clone(),
            selection_proof,
            probabilities: geographically_constrained,
            temperature: self.probability_calculator.get_temperature().await,
            selection_entropy: self.calculate_selection_entropy(&geographically_constrained).await?,
            confidence_level: self.calculate_selection_confidence(&selected_validator, &geographically_constrained).await?,
        };
        
        // Phase 9: Update selection history and statistics
        self.update_selection_history(result.clone()).await?;
        
        Ok(result)
    }
    
    async fn calculate_selection_probabilities(
        &self,
        validators: &[ActiveValidator],
        slot: Slot,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, f64>, ElectionError> {
        let scores: Vec<f64> = validators.iter().map(|v| v.current_score).collect();
        
        // Calculate adaptive temperature based on network conditions
        let temperature = self.calculate_adaptive_temperature(scores, network_state).await?;
        
        // Apply numerical stability: subtract max score before exponentiation
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f64> = scores.iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();
        
        let sum_exp: f64 = exp_scores.iter().sum();
        
        let probabilities: BTreeMap<ValidatorId, f64> = validators.iter()
            .zip(exp_scores.iter())
            .map(|(validator, &exp_score)| {
                (validator.identity.id, exp_score / sum_exp)
            })
            .collect();
        
        // Validate probability distribution properties
        self.validate_probability_distribution(&probabilities).await?;
        
        Ok(probabilities)
    }
    
    async fn calculate_adaptive_temperature(
        &self,
        scores: Vec<f64>,
        network_state: &NetworkState,
    ) -> Result<f64, ElectionError> {
        let base_temperature = 1.0;
        
        // Calculate score distribution statistics
        let mean_score = scores.mean();
        let score_variance = scores.variance();
        let gini_coefficient = self.calculate_gini_coefficient(&scores).await?;
        
        // Adjust temperature based on inequality measures
        let inequality_factor = 1.0 + gini_coefficient.powi(2);
        
        // Adjust based on network security requirements
        let security_factor = match network_state.security_level {
            SecurityLevel::High => 0.7,
            SecurityLevel::Medium => 1.0,
            SecurityLevel::Low => 1.3,
        };
        
        // Adjust based on network size and decentralization
        let size_factor = (network_state.validator_count as f64).ln().max(1.0);
        
        let adaptive_temperature = base_temperature * inequality_factor * security_factor / size_factor;
        
        // Apply bounds
        Ok(adaptive_temperature.max(0.1).min(10.0))
    }
    
    async fn apply_anti_correlation_measures(
        &self,
        probabilities: &BTreeMap<ValidatorId, f64>,
        slot: Slot,
        validators: &[ActiveValidator],
    ) -> Result<BTreeMap<ValidatorId, f64>, ElectionError> {
        let mut adjusted_probabilities = probabilities.clone();
        
        // Get recent selection history
        let recent_selections = self.get_recent_selections(slot).await?;
        
        // Apply temporal anti-correlation
        for (validator_id, probability) in &mut adjusted_probabilities {
            let recent_frequency = self.calculate_recent_selection_frequency(*validator_id, &recent_selections).await;
            let temporal_penalty = self.calculate_temporal_penalty(recent_frequency).await?;
            *probability *= temporal_penalty;
        }
        
        // Apply spatial anti-correlation (geographic distribution)
        let geographic_adjustments = self.calculate_geographic_adjustments(validators, &recent_selections).await?;
        for (validator_id, probability) in &mut adjusted_probabilities {
            if let Some(adjustment) = geographic_adjustments.get(validator_id) {
                *probability *= adjustment;
            }
        }
        
        // Apply stake-based anti-correlation to prevent dominance
        let stake_adjustments = self.calculate_stake_anti_correlation(validators).await?;
        for (validator_id, probability) in &mut adjusted_probabilities {
            if let Some(adjustment) = stake_adjustments.get(validator_id) {
                *probability *= adjustment;
            }
        }
        
        // Renormalize probabilities after adjustments
        let total: f64 = adjusted_probabilities.values().sum();
        for probability in adjusted_probabilities.values_mut() {
            *probability /= total;
        }
        
        Ok(adjusted_probabilities)
    }
    
    async fn perform_weighted_selection(
        &self,
        probabilities: &BTreeMap<ValidatorId, f64>,
        vrf_randomness: [u8; 32],
    ) -> Result<SelectedValidator, ElectionError> {
        let validator_ids: Vec<ValidatorId> = probabilities.keys().cloned().collect();
        let probability_values: Vec<f64> = probabilities.values().cloned().collect();
        
        // Use VRF output as seed for deterministic randomness
        let mut rng = ChaChaRng::from_seed(vrf_randomness);
        
        // Use alias method for O(1) weighted random selection
        let dist = WeightedAliasIndex::new(probability_values)
            .map_err(|e| ElectionError::ProbabilityError(e.to_string()))?;
        
        let selected_index = dist.sample(&mut rng);
        let selected_validator_id = validator_ids[selected_index];
        
        // Find the complete validator information
        let selected_validator = self.find_validator_by_id(selected_validator_id, probabilities.keys()).await?;
        
        Ok(SelectedValidator {
            validator_id: selected_validator_id,
            selection_probability: probabilities[&selected_validator_id],
            cumulative_probability: self.calculate_cumulative_probability(selected_index, &probability_values).await?,
            random_index: selected_index,
            selection_quality: self.calculate_selection_quality(selected_index, &probability_values).await?,
        })
    }
    
    async fn generate_selection_proof(
        &self,
        selected_validator: &SelectedValidator,
        vrf_output: &VRFOutput,
        slot: Slot,
    ) -> Result<SelectionProof, ElectionError> {
        let mut transcript = Transcript::new(b"RAYONIX_LEADER_SELECTION");
        
        // Commit to selection parameters
        transcript.append_message(b"slot", &slot.to_be_bytes());
        transcript.append_message(b"validator_id", &selected_validator.validator_id.0);
        transcript.append_message(b"vrf_output", &vrf_output.output);
        transcript.append_message(b"probability", &selected_validator.selection_probability.to_be_bytes());
        
        // Generate proof challenge
        let mut challenge = [0u8; 32];
        transcript.challenge_bytes(b"selection_challenge", &mut challenge);
        
        // Sign the proof with the selector's key
        let signature = self.sign_selection_proof(&challenge, slot).await?;
        
        Ok(SelectionProof {
            challenge,
            signature,
            transcript_commitment: transcript.clone().into(),
            proof_timestamp: self.get_current_timestamp().await,
            proof_version: SELECTION_PROOF_VERSION,
        })
    }
    
    pub async fn verify_leader_selection(
        &self,
        claimed_leader: ValidatorId,
        selection_proof: &SelectionProof,
        vrf_output: &VRFOutput,
        slot: Slot,
        validators: &[ActiveValidator],
    ) -> Result<SelectionVerification, ElectionError> {
        // Phase 1: Verify proof signature
        let signature_valid = self.verify_selection_signature(selection_proof, slot).await?;
        if !signature_valid {
            return Err(ElectionError::InvalidSelectionSignature);
        }
        
        // Phase 2: Recompute selection probabilities
        let probabilities = self.calculate_selection_probabilities(validators, slot, &NetworkState::default()).await?;
        
        // Phase 3: Verify VRF output consistency
        let vrf_consistent = self.verify_vrf_consistency(vrf_output, slot, validators).await?;
        if !vrf_consistent {
            return Err(ElectionError::VRFConsistencyError);
        }
        
        // Phase 4: Verify selection matches probabilities and randomness
        let selection_valid = self.verify_selection_match(
            claimed_leader, 
            &probabilities, 
            vrf_output.randomness
        ).await?;
        
        // Phase 5: Check anti-correlation constraints
        let constraints_satisfied = self.verify_anti_correlation_constraints(claimed_leader, slot).await?;
        
        Ok(SelectionVerification {
            valid: signature_valid && vrf_consistent && selection_valid && constraints_satisfied,
            claimed_leader,
            slot,
            probability: probabilities.get(&claimed_leader).copied().unwrap_or(0.0),
            confidence_score: self.calculate_verification_confidence(
                signature_valid, 
                vrf_consistent, 
                selection_valid, 
                constraints_satisfied
            ).await?,
            verification_timestamp: self.get_current_timestamp().await,
        })
    }
}

pub struct AntiCorrelationEngine {
    selection_history: Arc<RwLock<SelectionHistory>>,
    correlation_analyzer: CorrelationAnalyzer,
    penalty_calculator: PenaltyCalculator,
    constraint_solver: ConstraintSolver,
}

impl AntiCorrelationEngine {
    pub async fn calculate_temporal_penalty(
        &self,
        recent_selection_frequency: f64,
    ) -> Result<f64, ElectionError> {
        // Exponential penalty for frequent recent selections
        let base_penalty = 0.8; // 20% reduction for recent selections
        let penalty_exponent = 4.0; // Strong penalty for high frequency
        
        let penalty = base_penalty.powf(recent_selection_frequency * penalty_exponent);
        
        Ok(penalty.max(0.1).min(1.0)) // Bound between 10% and 100%
    }
    
    pub async fn calculate_geographic_adjustments(
        &self,
        validators: &[ActiveValidator],
        recent_selections: &[SelectionRecord],
    ) -> Result<BTreeMap<ValidatorId, f64>, ElectionError> {
        let mut adjustments = BTreeMap::new();
        
        // Group validators by geographic region
        let regions = self.group_validators_by_region(validators).await?;
        
        // Calculate recent selection frequency per region
        let region_frequencies = self.calculate_region_selection_frequencies(&regions, recent_selections).await?;
        
        for (region, validators_in_region) in regions {
            let region_frequency = region_frequencies.get(&region).copied().unwrap_or(0.0);
            
            // Boost probability for underrepresented regions
            let region_adjustment = if region_frequency < 0.1 {
                1.5 // 50% boost for very underrepresented regions
            } else if region_frequency < 0.2 {
                1.2 // 20% boost for underrepresented regions
            } else {
                1.0 // No adjustment for well-represented regions
            };
            
            for validator in validators_in_region {
                adjustments.insert(validator.identity.id, region_adjustment);
            }
        }
        
        Ok(adjustments)
    }
    
    pub async fn calculate_stake_anti_correlation(
        &self,
        validators: &[ActiveValidator],
    ) -> Result<BTreeMap<ValidatorId, f64>, ElectionError> {
        let mut adjustments = BTreeMap::new();
        
        let total_stake: u128 = validators.iter()
            .map(|v| v.stake_state.effective_stake)
            .sum();
        
        for validator in validators {
            let stake_ratio = validator.stake_state.effective_stake as f64 / total_stake as f64;
            
            // Apply stake-based anti-correlation to prevent dominance
            let stake_adjustment = if stake_ratio > 0.1 {
                0.7 // 30% penalty for large stakeholders
            } else if stake_ratio > 0.05 {
                0.9 // 10% penalty for medium stakeholders
            } else {
                1.0 // No penalty for small stakeholders
            };
            
            adjustments.insert(validator.identity.id, stake_adjustment);
        }
        
        Ok(adjustments)
    }
}

pub struct ProbabilityCalculator {
    temperature_controller: TemperatureController,
    softmax_variants: SoftmaxVariantRegistry,
    numerical_stabilizer: NumericalStabilizer,
    distribution_validator: DistributionValidator,
}

impl ProbabilityCalculator {
    pub async fn calculate_softmax_probabilities(
        &self,
        scores: &[f64],
        temperature: f64,
        variant: SoftmaxVariant,
    ) -> Result<Vec<f64>, ElectionError> {
        match variant {
            SoftmaxVariant::Standard => self.standard_softmax(scores, temperature).await,
            SoftmaxVariant::Sparse => self.sparse_softmax(scores, temperature).await,
            SoftmaxVariant::TemperatureScaled => self.temperature_scaled_softmax(scores, temperature).await,
            SoftmaxVariant::Robust => self.robust_softmax(scores, temperature).await,
        }
    }
    
    async fn standard_softmax(
        &self,
        scores: &[f64],
        temperature: f64,
    ) -> Result<Vec<f64>, ElectionError> {
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f64> = scores.iter()
            .map(|&s| ((s - max_score) / temperature).exp())
            .collect();
        
        let sum_exp: f64 = exp_scores.iter().sum();
        
        Ok(exp_scores.iter().map(|&s| s / sum_exp).collect())
    }
    
    async fn sparse_softmax(
        &self,
        scores: &[f64],
        temperature: f64,
    ) -> Result<Vec<f64>, ElectionError> {
        // Sparse softmax encourages more equal distribution
        let standard_probs = self.standard_softmax(scores, temperature).await?;
        
        // Apply sparsification transformation
        let sparsified: Vec<f64> = standard_probs.iter()
            .map(|&p| p.powf(0.8)) // Sparsification exponent
            .collect();
        
        let sum: f64 = sparsified.iter().sum();
        
        Ok(sparsified.iter().map(|&s| s / sum).collect())
    }
    
    async fn robust_softmax(
        &self,
        scores: &[f64],
        temperature: f64,
    ) -> Result<Vec<f64>, ElectionError> {
        // Robust softmax reduces sensitivity to outlier scores
        let median_score = self.calculate_median(scores).await?;
        let mad = self.calculate_median_absolute_deviation(scores, median_score).await?;
        
        // Winsorize scores to reduce outlier influence
        let winsorized_scores: Vec<f64> = scores.iter()
            .map(|&s| {
                if (s - median_score).abs() > 3.0 * mad {
                    median_score + 3.0 * mad * (s - median_score).signum()
                } else {
                    s
                }
            })
            .collect();
        
        self.standard_softmax(&winsorized_scores, temperature).await
    }
    
    async fn calculate_median(&self, scores: &[f64]) -> Result<f64, ElectionError> {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_scores.len();
        if n % 2 == 1 {
            Ok(sorted_scores[n / 2])
        } else {
            Ok((sorted_scores[n / 2 - 1] + sorted_scores[n / 2]) / 2.0)
        }
    }
}

pub struct GeographicDistributor {
    region_detector: RegionDetector,
    latency_analyzer: LatencyAnalyzer,
    topology_mapper: TopologyMapper,
    distribution_optimizer: DistributionOptimizer,
}

impl GeographicDistributor {
    pub async fn apply_geographic_constraints(
        &self,
        probabilities: &BTreeMap<ValidatorId, f64>,
        validators: &[ActiveValidator],
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, f64>, ElectionError> {
        let mut constrained_probabilities = probabilities.clone();
        
        // Detect validator regions based on network topology
        let regions = self.detect_validator_regions(validators).await?;
        
        // Calculate current geographic distribution
        let current_distribution = self.calculate_current_distribution(&regions, probabilities).await?;
        
        // Calculate optimal geographic distribution
        let optimal_distribution = self.calculate_optimal_distribution(&regions, network_state).await?;
        
        // Apply constraints to move toward optimal distribution
        for (region, validators_in_region) in regions {
            let current_prob = current_distribution.get(&region).copied().unwrap_or(0.0);
            let optimal_prob = optimal_distribution.get(&region).copied().unwrap_or(0.0);
            
            let adjustment_factor = if optimal_prob > 0.0 {
                optimal_prob / current_prob.max(1e-10)
            } else {
                0.0
            };
            
            for validator in validators_in_region {
                if let Some(prob) = constrained_probabilities.get_mut(&validator.identity.id) {
                    *prob *= adjustment_factor;
                }
            }
        }
        
        // Renormalize probabilities
        let total: f64 = constrained_probabilities.values().sum();
        for prob in constrained_probabilities.values_mut() {
            *prob /= total;
        }
        
        Ok(constrained_probabilities)
    }
    
    async fn calculate_optimal_distribution(
        &self,
        regions: &BTreeMap<Region, Vec<&ActiveValidator>>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<Region, f64>, ElectionError> {
        let mut optimal_distribution = BTreeMap::new();
        
        // Strategy 1: Proportional to population density
        let population_based = self.calculate_population_based_distribution(regions).await?;
        
        // Strategy 2: Proportional to network latency optimization
        let latency_based = self.calculate_latency_optimized_distribution(regions, network_state).await?;
        
        // Strategy 3: Considering political and regulatory diversity
        let diversity_based = self.calculate_diversity_based_distribution(regions).await?;
        
        // Combine strategies with weights
        for region in regions.keys() {
            let population_weight = 0.4;
            let latency_weight = 0.4;
            let diversity_weight = 0.2;
            
            let combined_prob = population_weight * population_based.get(region).copied().unwrap_or(0.0) +
                              latency_weight * latency_based.get(region).copied().unwrap_or(0.0) +
                              diversity_weight * diversity_based.get(region).copied().unwrap_or(0.0);
            
            optimal_distribution.insert(region.clone(), combined_prob);
        }
        
        // Normalize to sum to 1
        let total: f64 = optimal_distribution.values().sum();
        for prob in optimal_distribution.values_mut() {
            *prob /= total;
        }
        
        Ok(optimal_distribution)
    }
}