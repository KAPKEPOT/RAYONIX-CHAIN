// consensus/core/scoring/weights_manager.rs
use crate::types::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Beta, Continuous},
    statistics::Statistics,
};
use nalgebra::{DVector, DMatrix};

pub struct WeightsManager {
    adaptive_engine: AdaptiveWeightEngine,
    equilibrium_analyzer: EquilibriumAnalyzer,
    security_balancer: SecurityBalancer,
    performance_optimizer: PerformanceOptimizer,
    weight_history: Arc<RwLock<BTreeMap<Epoch, WeightDistribution>>>,
}

impl WeightsManager {
    pub async fn calculate_optimal_weights(
        &self,
        validators: &[ActiveValidator],
        stake_components: &StakeComponents,
        time_lived_components: &TimeLivedComponents,
        network_state: &NetworkState,
        current_epoch: Epoch,
    ) -> Result<OptimalWeights, WeightsError> {
        // Phase 1: Calculate base weight components
        let base_weights = self.calculate_base_weights(stake_components, time_lived_components).await?;
        
        // Phase 2: Apply adaptive adjustments based on network conditions
        let adaptive_weights = self.apply_adaptive_adjustments(base_weights, network_state).await?;
        
        // Phase 3: Apply equilibrium optimization
        let equilibrium_weights = self.apply_equilibrium_optimization(adaptive_weights, validators).await?;
        
        // Phase 4: Apply security balancing
        let security_balanced = self.apply_security_balancing(equilibrium_weights, network_state).await?;
        
        // Phase 5: Apply performance optimization
        let performance_optimized = self.apply_performance_optimization(security_balanced, validators).await?;
        
        // Phase 6: Normalize final weights
        let normalized_weights = self.normalize_weights(performance_optimized).await?;

        let optimal_weights = OptimalWeights {
            epoch: current_epoch,
            base_weights,
            adaptive_weights,
            equilibrium_weights,
            security_balanced,
            performance_optimized,
            normalized_weights,
            weight_metrics: self.calculate_weight_metrics(&normalized_weights).await?,
            stability_analysis: self.analyze_weight_stability(&normalized_weights).await?,
        };

        // Store weight history
        self.update_weight_history(current_epoch, optimal_weights.clone()).await;

        Ok(optimal_weights)
    }

    async fn calculate_base_weights(
        &self,
        stake_components: &StakeComponents,
        time_lived_components: &TimeLivedComponents,
    ) -> Result<BaseWeights, WeightsError> {
        let mut base_weights = BTreeMap::new();
        
        // Calculate stake-based weights
        let stake_weights = self.calculate_stake_based_weights(stake_components).await?;
        
        // Calculate time-lived-based weights
        let time_lived_weights = self.calculate_time_lived_weights(time_lived_components).await?;
        
        // Combine with initial fixed ratio
        let initial_stake_ratio = 0.6; // 60% stake, 40% time-lived initially
        let initial_time_lived_ratio = 0.4;
        
        for validator_id in stake_weights.keys() {
            let stake_weight = stake_weights.get(validator_id).unwrap_or(&0.0);
            let time_lived_weight = time_lived_weights.get(validator_id).unwrap_or(&0.0);
            
            let combined_weight = (stake_weight * initial_stake_ratio) + 
                                (time_lived_weight * initial_time_lived_ratio);
            
            base_weights.insert(*validator_id, BaseWeight {
                validator_id: *validator_id,
                stake_component: *stake_weight,
                time_lived_component: *time_lived_weight,
                combined_weight,
                stake_ratio: initial_stake_ratio,
                time_lived_ratio: initial_time_lived_ratio,
            });
        }
        
        Ok(BaseWeights {
            weights: base_weights,
            stake_weights,
            time_lived_weights,
            stake_ratio: initial_stake_ratio,
            time_lived_ratio: initial_time_lived_ratio,
        })
    }

    async fn calculate_stake_based_weights(
        &self,
        stake_components: &StakeComponents,
    ) -> Result<BTreeMap<ValidatorId, f64>, WeightsError> {
        let total_component: f64 = stake_components.components.values()
            .map(|c| c.final_component)
            .sum();
        
        if total_component == 0.0 {
            return Err(WeightsError::ZeroTotalComponent);
        }
        
        let stake_weights: BTreeMap<ValidatorId, f64> = stake_components.components
            .par_iter()
            .map(|(validator_id, component)| {
                let normalized_weight = component.final_component / total_component;
                Ok((*validator_id, normalized_weight))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(stake_weights)
    }

    async fn calculate_time_lived_weights(
        &self,
        time_lived_components: &TimeLivedComponents,
    ) -> Result<BTreeMap<ValidatorId, f64>, WeightsError> {
        let total_component: f64 = time_lived_components.components.values()
            .map(|c| c.final_component)
            .sum();
        
        if total_component == 0.0 {
            return Err(WeightsError::ZeroTotalComponent);
        }
        
        let time_lived_weights: BTreeMap<ValidatorId, f64> = time_lived_components.components
            .par_iter()
            .map(|(validator_id, component)| {
                let normalized_weight = component.final_component / total_component;
                Ok((*validator_id, normalized_weight))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(time_lived_weights)
    }

    async fn apply_adaptive_adjustments(
        &self,
        base_weights: BaseWeights,
        network_state: &NetworkState,
    ) -> Result<AdaptiveWeights, WeightsError> {
        // Calculate adaptive ratios based on network conditions
        let adaptive_ratios = self.calculate_adaptive_ratios(network_state).await?;
        
        let adaptive_weights: BTreeMap<ValidatorId, AdaptiveWeight> = base_weights.weights
            .par_iter()
            .map(|(validator_id, base)| {
                let stake_component = base.stake_component;
                let time_lived_component = base.time_lived_component;
                
                // Apply adaptive ratios
                let adaptive_weight = (stake_component * adaptive_ratios.stake_ratio) + 
                                    (time_lived_component * adaptive_ratios.time_lived_ratio);
                
                // Apply network condition adjustments
                let network_adjusted = self.apply_network_adjustments(
                    adaptive_weight, 
                    validator_id, 
                    network_state
                ).await?;
                
                let adaptive = AdaptiveWeight {
                    validator_id: *validator_id,
                    base_weight: base.combined_weight,
                    adaptive_weight: network_adjusted,
                    stake_ratio: adaptive_ratios.stake_ratio,
                    time_lived_ratio: adaptive_ratios.time_lived_ratio,
                    network_adjustment: self.calculate_network_adjustment_factor(validator_id, network_state).await?,
                    adaptation_magnitude: (network_adjusted - base.combined_weight).abs(),
                };
                
                Ok((*validator_id, adaptive))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(AdaptiveWeights {
            weights: adaptive_weights,
            adaptive_ratios,
            network_state: network_state.clone(),
        })
    }

    async fn calculate_adaptive_ratios(
        &self,
        network_state: &NetworkState,
    ) -> Result<AdaptiveRatios, WeightsError> {
        let base_stake_ratio = 0.6;
        let base_time_lived_ratio = 0.4;
        
        // Adjust based on network security level
        let security_adjustment = match network_state.security_level {
            SecurityLevel::High => 0.1,   // Increase stake weight for security
            SecurityLevel::Medium => 0.0, // No adjustment
            SecurityLevel::Low => -0.1,   // Decrease stake weight for decentralization
        };
        
        // Adjust based on network age
        let age_adjustment = if network_state.network_age_epochs < 1000 {
            -0.1 // Favor time-lived for young networks
        } else if network_state.network_age_epochs < 10000 {
            0.0 // Balanced approach
        } else {
            0.1 // Favor stake for mature networks
        };
        
        // Adjust based on decentralization
        let decentralization_adjustment = if network_state.decentralization_index < 0.3 {
            -0.2 // Strongly favor time-lived for centralized networks
        } else if network_state.decentralization_index < 0.7 {
            0.0 // Balanced approach
        } else {
            0.1 // Slightly favor stake for decentralized networks
        };
        
        let total_adjustment = security_adjustment + age_adjustment + decentralization_adjustment;
        
        let adjusted_stake_ratio = (base_stake_ratio + total_adjustment)
            .max(0.2)
            .min(0.8);
        
        let adjusted_time_lived_ratio = 1.0 - adjusted_stake_ratio;
        
        Ok(AdaptiveRatios {
            stake_ratio: adjusted_stake_ratio,
            time_lived_ratio: adjusted_time_lived_ratio,
            security_adjustment,
            age_adjustment,
            decentralization_adjustment,
            total_adjustment,
        })
    }

    async fn apply_equilibrium_optimization(
        &self,
        adaptive_weights: AdaptiveWeights,
        validators: &[ActiveValidator],
    ) -> Result<EquilibriumWeights, WeightsError> {
        let equilibrium_analysis = self.analyze_weight_equilibrium(&adaptive_weights.weights, validators).await?;
        
        let equilibrium_weights: BTreeMap<ValidatorId, EquilibriumWeight> = adaptive_weights.weights
            .par_iter()
            .map(|(validator_id, adaptive)| {
                let current_weight = adaptive.adaptive_weight;
                
                // Calculate equilibrium correction
                let equilibrium_correction = self.calculate_equilibrium_correction(
                    *validator_id, 
                    current_weight, 
                    &equilibrium_analysis
                ).await?;
                
                let corrected_weight = current_weight * equilibrium_correction;
                
                let equilibrium = EquilibriumWeight {
                    validator_id: *validator_id,
                    pre_correction_weight: current_weight,
                    corrected_weight,
                    equilibrium_correction,
                    equilibrium_deviation: self.calculate_equilibrium_deviation(current_weight, &equilibrium_analysis).await?,
                    stability_contribution: self.assess_stability_contribution(equilibrium_correction).await?,
                };
                
                Ok((*validator_id, equilibrium))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(EquilibriumWeights {
            weights: equilibrium_weights,
            equilibrium_analysis,
        })
    }

    async fn apply_security_balancing(
        &self,
        equilibrium_weights: EquilibriumWeights,
        network_state: &NetworkState,
    ) -> Result<SecurityBalancedWeights, WeightsError> {
        let security_analysis = self.analyze_security_implications(&equilibrium_weights.weights, network_state).await?;
        
        let security_balanced: BTreeMap<ValidatorId, SecurityBalancedWeight> = equilibrium_weights.weights
            .par_iter()
            .map(|(validator_id, equilibrium)| {
                let current_weight = equilibrium.corrected_weight;
                
                // Apply security balancing
                let security_balance = self.calculate_security_balance(*validator_id, current_weight, &security_analysis).await?;
                
                let balanced_weight = current_weight * security_balance;
                
                let balanced = SecurityBalancedWeight {
                    validator_id: *validator_id,
                    pre_balance_weight: current_weight,
                    balanced_weight,
                    security_balance,
                    security_risk: self.assess_security_risk(*validator_id, current_weight).await?,
                    attack_resistance: self.calculate_attack_resistance(balanced_weight).await?,
                };
                
                Ok((*validator_id, balanced))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(SecurityBalancedWeights {
            weights: security_balanced,
            security_analysis,
        })
    }

    async fn apply_performance_optimization(
        &self,
        security_balanced: SecurityBalancedWeights,
        validators: &[ActiveValidator],
    ) -> Result<PerformanceOptimizedWeights, WeightsError> {
        let performance_analysis = self.analyze_performance_impact(&security_balanced.weights, validators).await?;
        
        let performance_optimized: BTreeMap<ValidatorId, PerformanceOptimizedWeight> = security_balanced.weights
            .par_iter()
            .map(|(validator_id, balanced)| {
                let current_weight = balanced.balanced_weight;
                
                // Apply performance optimization
                let performance_optimization = self.calculate_performance_optimization(
                    *validator_id, 
                    current_weight, 
                    validators
                ).await?;
                
                let optimized_weight = current_weight * performance_optimization;
                
                let optimized = PerformanceOptimizedWeight {
                    validator_id: *validator_id,
                    pre_optimization_weight: current_weight,
                    optimized_weight,
                    performance_optimization,
                    performance_impact: self.assess_performance_impact(optimized_weight).await?,
                    efficiency_gain: self.calculate_efficiency_gain(performance_optimization).await?,
                };
                
                Ok((*validator_id, optimized))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(PerformanceOptimizedWeights {
            weights: performance_optimized,
            performance_analysis,
        })
    }

    async fn normalize_weights(
        &self,
        performance_optimized: PerformanceOptimizedWeights,
    ) -> Result<BTreeMap<ValidatorId, NormalizedWeight>, WeightsError> {
        let total_weight: f64 = performance_optimized.weights.values()
            .map(|w| w.optimized_weight)
            .sum();
        
        if total_weight == 0.0 {
            return Err(WeightsError::ZeroTotalWeight);
        }
        
        let normalized_weights: BTreeMap<ValidatorId, NormalizedWeight> = performance_optimized.weights
            .par_iter()
            .map(|(validator_id, optimized)| {
                let normalized_value = optimized.optimized_weight / total_weight;
                
                let normalized = NormalizedWeight {
                    validator_id: *validator_id,
                    normalized_value,
                    raw_value: optimized.optimized_weight,
                    normalization_factor: 1.0 / total_weight,
                    final_weight: normalized_value,
                };
                
                Ok((*validator_id, normalized))
            })
            .collect::<Result<BTreeMap<_, _>, WeightsError>>()?;

        Ok(normalized_weights)
    }

    pub async fn calculate_weight_metrics(
        &self,
        normalized_weights: &BTreeMap<ValidatorId, NormalizedWeight>,
    ) -> Result<WeightMetrics, WeightsError> {
        let weight_values: Vec<f64> = normalized_weights.values()
            .map(|w| w.final_weight)
            .collect();
        
        Ok(WeightMetrics {
            gini_coefficient: self.calculate_gini_coefficient(&weight_values).await?,
            entropy: self.calculate_weight_entropy(&weight_values).await?,
            herfindahl_index: self.calculate_herfindahl_index(&weight_values).await?,
            coefficient_of_variation: weight_values.std_dev() / weight_values.mean().max(1e-10),
            weight_concentration: self.assess_weight_concentration(&weight_values).await?,
            balance_score: self.calculate_balance_score(&weight_values).await?,
        })
    }

    async fn calculate_gini_coefficient(&self, values: &[f64]) -> Result<f64, WeightsError> {
        if values.is_empty() {
            return Ok(0.0);
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len() as f64;
        let mut numerator = 0.0;
        let sum_values: f64 = sorted_values.iter().sum();

        if sum_values == 0.0 {
            return Ok(0.0);
        }

        for (i, value) in sorted_values.iter().enumerate() {
            numerator += *value * (2.0 * (i as f64) - n + 1.0);
        }

        Ok((numerator / (n * sum_values)).max(0.0).min(1.0))
    }

    async fn calculate_weight_entropy(&self, values: &[f64]) -> Result<f64, WeightsError> {
        let total: f64 = values.iter().sum();
        if total == 0.0 {
            return Ok(0.0);
        }

        let entropy: f64 = values.iter()
            .map(|&value| {
                let probability = value / total;
                if probability > 0.0 {
                    -probability * probability.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = (values.len() as f64).ln();
        let normalized_entropy = if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        };

        Ok(normalized_entropy.max(0.0).min(1.0))
    }

    async fn update_weight_history(&self, epoch: Epoch, weights: OptimalWeights) {
        let mut history = self.weight_history.write().await;
        
        if history.len() >= 1000 {
            history.pop_first();
        }
        
        history.insert(epoch, weights);
    }
}