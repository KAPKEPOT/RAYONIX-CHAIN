// consensus/mechanisms/pos/power_calculator.rs
use crate::types::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Gamma, Continuous},
    statistics::Statistics,
};
use nalgebra::{DVector, DMatrix, SVD};

pub struct PowerCalculator {
    nonlinear_transforms: NonlinearTransformRegistry,
    security_models: SecurityModelSuite,
    economic_models: EconomicModelEngine,
    power_history: Arc<RwLock<BTreeMap<Epoch, PowerDistribution>>>,
}

impl PowerCalculator {
    pub async fn calculate_validator_power(
        &self,
        validators: &[ActiveValidator],
        stake_components: &StakeComponents,
        network_state: &NetworkState,
        current_epoch: Epoch,
    ) -> Result<PowerDistribution, PowerError> {
        // Phase 1: Calculate base power from stake components
        let base_power = self.calculate_base_power(validators, stake_components).await?;
        
        // Phase 2: Apply nonlinear power transformation
        let transformed_power = self.apply_nonlinear_transformation(base_power, network_state).await?;
        
        // Phase 3: Apply security-based adjustments
        let security_adjusted = self.apply_security_adjustments(transformed_power, network_state).await?;
        
        // Phase 4: Apply economic equilibrium corrections
        let equilibrium_corrected = self.apply_equilibrium_corrections(security_adjusted, network_state).await?;
        
        // Phase 5: Normalize power distribution
        let normalized_power = self.normalize_power_distribution(equilibrium_corrected).await?;
        
        let distribution = PowerDistribution {
            epoch: current_epoch,
            base_power,
            transformed_power,
            security_adjusted,
            equilibrium_corrected,
            normalized_power,
            power_metrics: self.calculate_power_metrics(&normalized_power).await?,
            concentration_analysis: self.analyze_power_concentration(&normalized_power).await?,
        };
        
        // Store power history
        self.update_power_history(current_epoch, distribution.clone()).await;
        
        Ok(distribution)
    }

    async fn calculate_base_power(
        &self,
        validators: &[ActiveValidator],
        stake_components: &StakeComponents,
    ) -> Result<BTreeMap<ValidatorId, BasePower>, PowerError> {
        let base_power: BTreeMap<ValidatorId, BasePower> = validators
            .par_iter()
            .map(|validator| {
                let stake_component = stake_components.components.get(&validator.identity.id)
                    .map(|c| c.final_component)
                    .unwrap_or(0.0);
                
                // Calculate performance-based power adjustment
                let performance_adjustment = self.calculate_performance_adjustment(validator).await?;
                
                // Calculate network contribution factor
                let network_contribution = self.calculate_network_contribution(validator).await?;
                
                let base_power_value = stake_component * performance_adjustment * network_contribution;
                
                let base_power = BasePower {
                    validator_id: validator.identity.id,
                    stake_component,
                    performance_adjustment,
                    network_contribution,
                    base_power_value,
                    raw_power: stake_component,
                };
                
                Ok((validator.identity.id, base_power))
            })
            .collect::<Result<BTreeMap<_, _>, PowerError>>()?;

        Ok(base_power)
    }

    async fn apply_nonlinear_transformation(
        &self,
        base_power: BTreeMap<ValidatorId, BasePower>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, TransformedPower>, PowerError> {
        let power_exponent = self.calculate_power_exponent(network_state).await?;
        
        let transformed_power: BTreeMap<ValidatorId, TransformedPower> = base_power
            .par_iter()
            .map(|(validator_id, base)| {
                let base_value = base.base_power_value;
                
                // Apply power law transformation
                let power_law_transformed = base_value.powf(power_exponent);
                
                // Apply sigmoid activation for bounded output
                let sigmoid_activated = 1.0 / (1.0 + (-8.0 * (power_law_transformed - 0.5)).exp());
                
                // Calculate transformation metrics
                let transformation_ratio = sigmoid_activated / base_value.max(1e-10);
                
                let transformed = TransformedPower {
                    validator_id: *validator_id,
                    base_power_value: base_value,
                    power_law_transformed,
                    sigmoid_activated,
                    power_exponent,
                    transformation_ratio,
                    transformation_effect: self.assess_transformation_effect(transformation_ratio).await?,
                };
                
                Ok((*validator_id, transformed))
            })
            .collect::<Result<BTreeMap<_, _>, PowerError>>()?;

        Ok(transformed_power)
    }

    async fn apply_security_adjustments(
        &self,
        transformed_power: BTreeMap<ValidatorId, TransformedPower>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, SecurityAdjustedPower>, PowerError> {
        let security_multiplier = self.calculate_security_multiplier(network_state).await?;
        let decentralization_factor = self.calculate_decentralization_factor(network_state).await?;
        
        let security_adjusted: BTreeMap<ValidatorId, SecurityAdjustedPower> = transformed_power
            .par_iter()
            .map(|(validator_id, transformed)| {
                let base_power = transformed.sigmoid_activated;
                
                // Apply security multiplier
                let security_adjusted = base_power * security_multiplier;
                
                // Apply decentralization factor
                let decentralization_adjusted = security_adjusted * decentralization_factor;
                
                // Apply anti-collusion adjustment
                let anti_collusion_adjusted = self.apply_anti_collusion_adjustment(
                    *validator_id, 
                    decentralization_adjusted, 
                    network_state
                ).await?;
                
                let adjusted = SecurityAdjustedPower {
                    validator_id: *validator_id,
                    base_power,
                    security_adjusted,
                    decentralization_adjusted,
                    anti_collusion_adjusted,
                    security_multiplier,
                    decentralization_factor,
                    final_adjusted_power: anti_collusion_adjusted,
                };
                
                Ok((*validator_id, adjusted))
            })
            .collect::<Result<BTreeMap<_, _>, PowerError>>()?;

        Ok(security_adjusted)
    }

    async fn apply_equilibrium_corrections(
        &self,
        security_adjusted: BTreeMap<ValidatorId, SecurityAdjustedPower>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, EquilibriumCorrectedPower>, PowerError> {
        let equilibrium_analysis = self.analyze_power_equilibrium(&security_adjusted, network_state).await?;
        
        let equilibrium_corrected: BTreeMap<ValidatorId, EquilibriumCorrectedPower> = security_adjusted
            .par_iter()
            .map(|(validator_id, adjusted)| {
                let current_power = adjusted.final_adjusted_power;
                
                // Calculate equilibrium correction
                let equilibrium_correction = self.calculate_equilibrium_correction(
                    *validator_id, 
                    current_power, 
                    &equilibrium_analysis
                ).await?;
                
                let corrected_power = current_power * equilibrium_correction;
                
                let corrected = EquilibriumCorrectedPower {
                    validator_id: *validator_id,
                    pre_correction_power: current_power,
                    corrected_power,
                    equilibrium_correction,
                    equilibrium_deviation: self.calculate_equilibrium_deviation(
                        current_power, 
                        &equilibrium_analysis
                    ).await?,
                    stability_impact: self.assess_stability_impact(equilibrium_correction).await?,
                };
                
                Ok((*validator_id, corrected))
            })
            .collect::<Result<BTreeMap<_, _>, PowerError>>()?;

        Ok(equilibrium_corrected)
    }

    async fn normalize_power_distribution(
        &self,
        equilibrium_corrected: BTreeMap<ValidatorId, EquilibriumCorrectedPower>,
    ) -> Result<BTreeMap<ValidatorId, NormalizedPower>, PowerError> {
        let total_power: f64 = equilibrium_corrected.values()
            .map(|p| p.corrected_power)
            .sum();
        
        if total_power == 0.0 {
            return Err(PowerError::ZeroTotalPower);
        }
        
        let normalized_power: BTreeMap<ValidatorId, NormalizedPower> = equilibrium_corrected
            .par_iter()
            .map(|(validator_id, corrected)| {
                let normalized_value = corrected.corrected_power / total_power;
                
                let normalized = NormalizedPower {
                    validator_id: *validator_id,
                    normalized_value,
                    raw_value: corrected.corrected_power,
                    normalization_factor: 1.0 / total_power,
                    final_power: normalized_value,
                };
                
                Ok((*validator_id, normalized))
            })
            .collect::<Result<BTreeMap<_, _>, PowerError>>()?;

        Ok(normalized_power)
    }

    pub async fn calculate_power_metrics(
        &self,
        power_distribution: &BTreeMap<ValidatorId, NormalizedPower>,
    ) -> Result<PowerMetrics, PowerError> {
        let power_values: Vec<f64> = power_distribution.values()
            .map(|p| p.final_power)
            .collect();
        
        Ok(PowerMetrics {
            gini_coefficient: self.calculate_gini_coefficient(&power_values).await?,
            entropy: self.calculate_power_entropy(&power_values).await?,
            herfindahl_index: self.calculate_herfindahl_index(&power_values).await?,
            coefficient_of_variation: power_values.std_dev() / power_values.mean().max(1e-10),
            power_concentration: self.assess_power_concentration(&power_values).await?,
            decentralization_score: self.calculate_decentralization_score(&power_values).await?,
        })
    }

    async fn calculate_gini_coefficient(&self, values: &[f64]) -> Result<f64, PowerError> {
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

    async fn calculate_power_entropy(&self, values: &[f64]) -> Result<f64, PowerError> {
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

    async fn update_power_history(&self, epoch: Epoch, distribution: PowerDistribution) {
        let mut history = self.power_history.write().await;
        
        if history.len() >= 1000 {
            history.pop_first();
        }
        
        history.insert(epoch, distribution);
    }
}