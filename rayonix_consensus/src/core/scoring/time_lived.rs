// consensus/core/scoring/time_lived.rs
use crate::types::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Exponential, Gamma, Normal, Continuous},
    statistics::Statistics,
};

pub struct TimeLivedEngine {
    reliability_tracker: ReliabilityTracker,
    decay_engine: TimeDecayEngine,
    memory_integrator: MemoryIntegrator,
    trend_analyzer: TrendAnalyzer,
    time_lived_store: Arc<RwLock<BTreeMap<ValidatorId, TimeLivedState>>>,
}

impl TimeLivedEngine {
    pub async fn calculate_time_lived_components(
        &self,
        validators: &[ActiveValidator],
        current_epoch: Epoch,
        network_state: &NetworkState,
    ) -> Result<TimeLivedComponents, TimeLivedError> {
        // Phase 1: Calculate base reliability metrics
        let reliability_components = self.calculate_reliability_components(validators, current_epoch).await?;
        
        // Phase 2: Apply temporal decay to historical performance
        let decayed_components = self.apply_temporal_decay(reliability_components, current_epoch).await?;
        
        // Phase 3: Integrate memory effects with adaptive weighting
        let memory_integrated = self.integrate_memory_effects(decayed_components, validators).await?;
        
        // Phase 4: Analyze temporal trends and patterns
        let trend_analyzed = self.analyze_temporal_trends(memory_integrated, validators).await?;
        
        // Phase 5: Calculate final time-lived components
        let final_components = self.calculate_final_components(trend_analyzed, network_state).await?;

        Ok(TimeLivedComponents {
            components: final_components,
            reliability_metrics: self.calculate_reliability_metrics(&final_components).await?,
            temporal_stability: self.assess_temporal_stability(&final_components).await?,
            memory_effectiveness: self.assess_memory_effectiveness(&final_components).await?,
        })
    }

    async fn calculate_reliability_components(
        &self,
        validators: &[ActiveValidator],
        current_epoch: Epoch,
    ) -> Result<BTreeMap<ValidatorId, ReliabilityComponent>, TimeLivedError> {
        let reliability_components: BTreeMap<ValidatorId, ReliabilityComponent> = validators
            .par_iter()
            .map(|validator| {
                let performance = &validator.performance;
                let time_lived_state = &validator.time_lived_state;
                
                // Calculate current epoch reliability
                let epoch_reliability = self.calculate_epoch_reliability(performance, current_epoch).await?;
                
                // Calculate exponential moving average
                let ema_reliability = self.update_exponential_moving_average(
                    time_lived_state.exponential_moving_average,
                    epoch_reliability,
                    current_epoch,
                    validator.identity.id
                ).await?;
                
                // Calculate cumulative reliability with saturation
                let cumulative_reliability = self.update_cumulative_reliability(
                    time_lived_state.cumulative_reliability,
                    epoch_reliability,
                    current_epoch
                ).await?;
                
                // Calculate reliability trend
                let reliability_trend = self.calculate_reliability_trend(validator.identity.id, current_epoch).await?;
                
                let component = ReliabilityComponent {
                    validator_id: validator.identity.id,
                    epoch_reliability,
                    exponential_moving_average: ema_reliability,
                    cumulative_reliability,
                    reliability_trend,
                    consecutive_successes: performance.consecutive_successes,
                    consecutive_failures: performance.consecutive_failures,
                    stability_score: self.calculate_stability_score(ema_reliability, reliability_trend).await?,
                };
                
                Ok((validator.identity.id, component))
            })
            .collect::<Result<BTreeMap<_, _>, TimeLivedError>>()?;

        Ok(reliability_components)
    }

    async fn calculate_epoch_reliability(
        &self,
        performance: &ValidatorPerformance,
        current_epoch: Epoch,
    ) -> Result<f64, TimeLivedError> {
        let mut component_scores = Vec::new();
        let mut total_weight = 0.0;
        
        // Block production reliability (40% weight)
        if performance.blocks_proposed + performance.blocks_missed > 0 {
            let block_reliability = performance.blocks_proposed as f64 / 
                (performance.blocks_proposed + performance.blocks_missed) as f64;
            component_scores.push(block_reliability * 0.4);
            total_weight += 0.4;
        }
        
        // Attestation reliability (35% weight)
        if performance.attestations_made + performance.attestations_missed > 0 {
            let attestation_reliability = performance.attestations_made as f64 / 
                (performance.attestations_made + performance.attestations_missed) as f64;
            component_scores.push(attestation_reliability * 0.35);
            total_weight += 0.35;
        }
        
        // Sync committee reliability (15% weight)
        if performance.sync_committee_participation + performance.sync_committee_misses > 0 {
            let sync_reliability = performance.sync_committee_participation as f64 / 
                (performance.sync_committee_participation + performance.sync_committee_misses) as f64;
            component_scores.push(sync_reliability * 0.15);
            total_weight += 0.15;
        }
        
        // Latency performance (10% weight)
        let latency_score = self.calculate_latency_score(performance.average_latency_ms).await?;
        component_scores.push(latency_score * 0.1);
        total_weight += 0.1;
        
        // Calculate weighted average
        let weighted_score: f64 = component_scores.iter().sum();
        let normalized_score = if total_weight > 0.0 {
            weighted_score / total_weight
        } else {
            0.0
        };
        
        // Apply consecutive performance adjustments
        let consecutive_adjustment = self.calculate_consecutive_adjustment(performance).await?;
        let adjusted_score = (normalized_score * consecutive_adjustment)
            .max(0.0)
            .min(1.0);
        
        Ok(adjusted_score)
    }

    async fn update_exponential_moving_average(
        &self,
        previous_ema: f64,
        current_value: f64,
        current_epoch: Epoch,
        validator_id: ValidatorId,
    ) -> Result<f64, TimeLivedError> {
        // Calculate adaptive decay factor based on validator history and network conditions
        let decay_factor = self.calculate_adaptive_decay_factor(validator_id, current_epoch).await?;
        
        // Update EMA: EMA_t = α * current + (1-α) * EMA_{t-1}
        let updated_ema = (decay_factor * current_value) + ((1.0 - decay_factor) * previous_ema);
        
        Ok(updated_ema.max(0.0).min(1.0))
    }

    async fn calculate_adaptive_decay_factor(
        &self,
        validator_id: ValidatorId,
        current_epoch: Epoch,
    ) -> Result<f64, TimeLivedError> {
        let store = self.time_lived_store.read().await;
        let validator_state = store.get(&validator_id);
        
        let base_decay = 0.1; // Base decay factor
        
        guard let Some(state) = validator_state else {
            return Ok(base_decay); // Use base decay for new validators
        };
        
        // Calculate stability-based adjustment
        let stability = self.calculate_reliability_stability(state).await?;
        
        // Calculate tenure-based adjustment
        let tenure = current_epoch - state.activation_epoch;
        let tenure_factor = self.calculate_tenure_factor(tenure).await?;
        
        // Calculate performance-based adjustment
        let performance_factor = self.calculate_performance_factor(state).await?;
        
        // Combine factors for adaptive decay
        let adaptive_decay = base_decay * (1.0 - stability * 0.3) * tenure_factor * performance_factor;
        
        Ok(adaptive_decay.max(0.05).min(0.3))
    }

    async fn apply_temporal_decay(
        &self,
        reliability_components: BTreeMap<ValidatorId, ReliabilityComponent>,
        current_epoch: Epoch,
    ) -> Result<BTreeMap<ValidatorId, DecayedComponent>, TimeLivedError> {
        let decayed_components: BTreeMap<ValidatorId, DecayedComponent> = reliability_components
            .par_iter()
            .map(|(validator_id, component)| {
                let time_since_update = (current_epoch - component.last_update_epoch) as f64;
                
                // Apply exponential decay to EMA
                let ema_decay_factor = (-0.01 * time_since_update).exp();
                let decayed_ema = component.exponential_moving_average * ema_decay_factor;
                
                // Apply slower decay to cumulative reliability
                let cumulative_decay_factor = (-0.001 * time_since_update).exp();
                let decayed_cumulative = component.cumulative_reliability * cumulative_decay_factor;
                
                // Calculate decay impact metrics
                let decay_impact = self.calculate_decay_impact(ema_decay_factor, cumulative_decay_factor).await?;
                
                let decayed = DecayedComponent {
                    validator_id: *validator_id,
                    original_ema: component.exponential_moving_average,
                    decayed_ema,
                    original_cumulative: component.cumulative_reliability,
                    decayed_cumulative,
                    ema_decay_factor,
                    cumulative_decay_factor,
                    decay_impact,
                    temporal_consistency: self.assess_temporal_consistency(decayed_ema, decayed_cumulative).await?,
                };
                
                Ok((*validator_id, decayed))
            })
            .collect::<Result<BTreeMap<_, _>, TimeLivedError>>()?;

        Ok(decayed_components)
    }

    async fn integrate_memory_effects(
        &self,
        decayed_components: BTreeMap<ValidatorId, DecayedComponent>,
        validators: &[ActiveValidator],
    ) -> Result<BTreeMap<ValidatorId, MemoryIntegratedComponent>, TimeLivedError> {
        let memory_integrated: BTreeMap<ValidatorId, MemoryIntegratedComponent> = decayed_components
            .par_iter()
            .map(|(validator_id, decayed)| {
                let validator = validators.iter()
                    .find(|v| v.identity.id == *validator_id)
                    .ok_or(TimeLivedError::ValidatorNotFound(*validator_id))?;
                
                // Calculate memory kernel weights
                let kernel_weights = self.calculate_memory_kernel_weights(validator).await?;
                
                // Integrate historical performance with memory kernel
                let memory_integrated_value = self.integrate_with_memory_kernel(
                    decayed.decayed_ema,
                    &validator.performance.historical_performance,
                    &kernel_weights
                ).await?;
                
                // Calculate memory effectiveness
                let memory_effectiveness = self.calculate_memory_effectiveness(
                    memory_integrated_value,
                    decayed.decayed_ema
                ).await?;
                
                let integrated = MemoryIntegratedComponent {
                    validator_id: *validator_id,
                    decayed_ema: decayed.decayed_ema,
                    memory_integrated_value,
                    kernel_weights,
                    memory_effectiveness,
                    historical_depth: validator.performance.historical_performance.len(),
                    memory_coherence: self.assess_memory_coherence(&kernel_weights).await?,
                };
                
                Ok((*validator_id, integrated))
            })
            .collect::<Result<BTreeMap<_, _>, TimeLivedError>>()?;

        Ok(memory_integrated)
    }

    async fn integrate_with_memory_kernel(
        &self,
        current_ema: f64,
        historical_performance: &[HistoricalPerformance],
        kernel_weights: &[f64],
    ) -> Result<f64, TimeLivedError> {
        if historical_performance.is_empty() {
            return Ok(current_ema);
        }
        
        let mut integrated_value = current_ema * kernel_weights[0];
        
        for (i, (performance, weight)) in historical_performance.iter().zip(kernel_weights.iter().skip(1)).enumerate() {
            integrated_value += performance.reliability_score * weight;
        }
        
        // Normalize by total weight
        let total_weight: f64 = kernel_weights.iter().sum();
        let normalized_value = if total_weight > 0.0 {
            integrated_value / total_weight
        } else {
            current_ema
        };
        
        Ok(normalized_value.max(0.0).min(1.0))
    }

    async fn calculate_memory_kernel_weights(
        &self,
        validator: &ActiveValidator,
    ) -> Result<Vec<f64>, TimeLivedError> {
        let historical_depth = validator.performance.historical_performance.len();
        let mut weights = Vec::with_capacity(historical_depth + 1);
        
        // Current performance weight (highest)
        weights.push(0.5); // 50% weight for current performance
        
        // Historical performance weights with exponential decay
        let decay_rate = 0.8; // 20% decay per epoch
        let mut historical_weight = 0.5; // Remaining 50% distributed among history
        
        for i in 0..historical_depth {
            let weight = historical_weight * decay_rate.powf(i as f64);
            weights.push(weight);
        }
        
        // Normalize weights to sum to 1.0
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            for weight in &mut weights {
                *weight /= total;
            }
        }
        
        Ok(weights)
    }

    pub async fn handle_validator_misbehavior(
        &self,
        validator_id: ValidatorId,
        offense: &MisbehaviorOffense,
        current_epoch: Epoch,
    ) -> Result<TimeLivedPenalty, TimeLivedError> {
        let mut store = self.time_lived_store.write().await;
        
        guard let Some(state) = store.get_mut(&validator_id) else {
            return Err(TimeLivedError::ValidatorNotFound(validator_id));
        };

        // Calculate reliability penalty based on offense severity
        let penalty_severity = self.calculate_penalty_severity(offense).await?;
        
        // Apply penalty to reliability metrics
        state.exponential_moving_average *= 1.0 - penalty_severity;
        state.cumulative_reliability *= 1.0 - penalty_severity;
        
        // Reset consecutive successes
        state.consecutive_successes = 0;
        state.consecutive_failures += 1;
        
        // Update penalty history
        state.penalty_history.push(PenaltyRecord {
            epoch: current_epoch,
            severity: penalty_severity,
            offense_type: offense.offense_type.clone(),
        });

        Ok(TimeLivedPenalty {
            validator_id,
            penalty_severity,
            new_ema: state.exponential_moving_average,
            new_cumulative: state.cumulative_reliability,
            recovery_time: self.calculate_recovery_time(penalty_severity, state).await?,
        })
    }

    async fn calculate_penalty_severity(
        &self,
        offense: &MisbehaviorOffense,
    ) -> Result<f64, TimeLivedError> {
        let base_severity = match offense.offense_type {
            MisbehaviorType::DoubleSigning => 0.5,
            MisbehaviorType::LivenessViolation => 0.1,
            MisbehaviorType::NetworkViolation => 0.05,
            MisbehaviorType::PerformanceDegradation => 0.02,
        };

        // Adjust for repeat offenses
        let repeat_multiplier = if offense.repeat_count > 0 {
            1.0 + (offense.repeat_count as f64 * 0.3)
        } else {
            1.0
        };

        let total_severity = base_severity * repeat_multiplier;
        
        Ok(total_severity.min(0.9)) // Cap at 90% penalty
    }
}