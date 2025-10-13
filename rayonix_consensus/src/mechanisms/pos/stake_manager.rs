// consensus/mechanisms/pos/stake_manager.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Gamma, Beta, Continuous},
    statistics::Statistics,
};
use nalgebra::{DVector, DMatrix, SVD};

pub struct StakeManager {
    stake_registry: Arc<RwLock<BTreeMap<ValidatorId, StakeState>>>,
    power_calculator: PowerCalculator,
    delegation_engine: DelegationEngine,
    slash_protector: SlashProtector,
    economic_analyzer: EconomicAnalyzer,
    historical_stake: Arc<RwLock<BTreeMap<Epoch, StakeDistribution>>>,
}

impl StakeManager {
    pub async fn process_epoch_stake_updates(
        &self,
        current_epoch: Epoch,
        validators: &[ActiveValidator],
        economic_params: &EconomicParameters,
        network_state: &NetworkState,
    ) -> Result<StakeUpdateResult, StakeError> {
        // Phase 1: Calculate effective stake with progressive scaling
        let effective_stakes = self.calculate_effective_stakes(validators, economic_params).await?;
        
        // Phase 2: Apply stake concentration penalties
        let penalized_stakes = self.apply_concentration_penalties(effective_stakes, network_state).await?;
        
        // Phase 3: Calculate stake power with nonlinear transformation
        let stake_power = self.calculate_stake_power(penalized_stakes, network_state).await?;
        
        // Phase 4: Update delegation relationships
        let delegation_updates = self.update_delegation_relationships(stake_power, current_epoch).await?;
        
        // Phase 5: Apply economic adjustments
        let economic_adjustments = self.apply_economic_adjustments(stake_power, economic_params).await?;
        
        // Phase 6: Calculate final stake components
        let stake_components = self.calculate_stake_components(economic_adjustments, validators).await?;

        Ok(StakeUpdateResult {
            epoch: current_epoch,
            effective_stakes,
            penalized_stakes,
            stake_power,
            delegation_updates,
            economic_adjustments,
            stake_components,
            distribution_metrics: self.calculate_distribution_metrics(&stake_components).await?,
        })
    }

    async fn calculate_effective_stakes(
        &self,
        validators: &[ActiveValidator],
        economic_params: &EconomicParameters,
    ) -> Result<BTreeMap<ValidatorId, EffectiveStake>, StakeError> {
        let total_network_stake: u128 = validators.iter()
            .map(|v| v.stake_state.total_stake)
            .sum();

        let effective_stakes: BTreeMap<ValidatorId, EffectiveStake> = validators
            .par_iter()
            .map(|validator| {
                let raw_stake = validator.stake_state.total_stake;
                let stake_ratio = raw_stake as f64 / total_network_stake as f64;
                
                // Apply progressive scaling to prevent dominance
                let progressive_factor = self.calculate_progressive_scaling(stake_ratio, economic_params).await?;
                
                // Calculate stake age bonus
                let age_bonus = self.calculate_stake_age_bonus(validator).await?;
                
                // Calculate delegation efficiency
                let delegation_efficiency = self.calculate_delegation_efficiency(validator).await?;
                
                let effective_stake = (raw_stake as f64 * progressive_factor * (1.0 + age_bonus) * delegation_efficiency) as u128;
                
                let effective = EffectiveStake {
                    validator_id: validator.identity.id,
                    raw_stake,
                    effective_stake,
                    progressive_factor,
                    age_bonus,
                    delegation_efficiency,
                    stake_ratio,
                    effective_ratio: effective_stake as f64 / total_network_stake as f64,
                };
                
                Ok((validator.identity.id, effective))
            })
            .collect::<Result<BTreeMap<_, _>, StakeError>>()?;

        Ok(effective_stakes)
    }

    async fn calculate_progressive_scaling(
        &self,
        stake_ratio: f64,
        economic_params: &EconomicParameters,
    ) -> Result<f64, StakeError> {
        // Progressive scaling reduces influence of very large stakeholders
        // Uses logistic function for smooth transition
        let scaling_center = economic_params.progressive_scaling_center;
        let scaling_sharpness = economic_params.progressive_scaling_sharpness;
        
        let progressive_factor = 1.0 / (1.0 + (-scaling_sharpness * (stake_ratio - scaling_center)).exp());
        
        Ok(progressive_factor.max(economic_params.min_scaling_factor)
            .min(economic_params.max_scaling_factor))
    }

    async fn apply_concentration_penalties(
        &self,
        effective_stakes: BTreeMap<ValidatorId, EffectiveStake>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, PenalizedStake>, StakeError> {
        let total_effective: u128 = effective_stakes.values().map(|s| s.effective_stake).sum();
        
        let penalized_stakes: BTreeMap<ValidatorId, PenalizedStake> = effective_stakes
            .par_iter()
            .map(|(validator_id, effective)| {
                let effective_ratio = effective.effective_ratio;
                
                // Calculate concentration penalty using power law
                let concentration_penalty = if effective_ratio > network_state.concentration_threshold {
                    let excess_ratio = effective_ratio - network_state.concentration_threshold;
                    (excess_ratio / network_state.concentration_threshold).powf(network_state.penalty_exponent)
                } else {
                    0.0
                };
                
                let penalized_stake = (effective.effective_stake as f64 * (1.0 - concentration_penalty)) as u128;
                
                let penalized = PenalizedStake {
                    validator_id: *validator_id,
                    pre_penalty_stake: effective.effective_stake,
                    penalized_stake,
                    concentration_penalty,
                    penalty_severity: self.assess_penalty_severity(concentration_penalty).await?,
                };
                
                Ok((*validator_id, penalized))
            })
            .collect::<Result<BTreeMap<_, _>, StakeError>>()?;

        Ok(penalized_stakes)
    }

    async fn calculate_stake_power(
        &self,
        penalized_stakes: BTreeMap<ValidatorId, PenalizedStake>,
        network_state: &NetworkState,
    ) -> Result<BTreeMap<ValidatorId, StakePower>, StakeError> {
        let total_penalized: u128 = penalized_stakes.values().map(|s| s.penalized_stake).sum();
        
        let stake_power: BTreeMap<ValidatorId, StakePower> = penalized_stakes
            .par_iter()
            .map(|(validator_id, penalized)| {
                let base_power_ratio = penalized.penalized_stake as f64 / total_penalized as f64;
                
                // Apply nonlinear power transformation
                let power_exponent = self.calculate_power_exponent(network_state).await?;
                let transformed_power = base_power_ratio.powf(power_exponent);
                
                // Apply network security multiplier
                let security_multiplier = self.calculate_security_multiplier(network_state).await?;
                
                let final_power = transformed_power * security_multiplier;
                
                let power = StakePower {
                    validator_id: *validator_id,
                    base_power_ratio,
                    transformed_power,
                    final_power,
                    power_exponent,
                    security_multiplier,
                    influence_weight: self.calculate_influence_weight(final_power).await?,
                };
                
                Ok((*validator_id, power))
            })
            .collect::<Result<BTreeMap<_, _>, StakeError>>()?;

        Ok(stake_power)
    }

    pub async fn handle_slashing_event(
        &self,
        validator_id: ValidatorId,
        offense: &SlashingOffense,
        current_epoch: Epoch,
    ) -> Result<SlashingResult, StakeError> {
        let mut registry = self.stake_registry.write().await;
        
        guard let Some(stake_state) = registry.get_mut(&validator_id) else {
            return Err(StakeError::ValidatorNotFound(validator_id));
        };

        // Calculate slashing penalty based on offense severity
        let penalty_fraction = self.calculate_slashing_penalty(offense).await?;
        
        // Apply penalty to stake
        let penalty_amount = (stake_state.total_stake as f64 * penalty_fraction) as u128;
        let new_stake = stake_state.total_stake.saturating_sub(penalty_amount);
        
        // Update stake state
        stake_state.total_stake = new_stake;
        stake_state.slashed_amount += penalty_amount;
        stake_state.last_slashing_epoch = current_epoch;
        
        // Apply additional reputation penalties
        stake_state.reputation_score *= 1.0 - penalty_fraction;
        
        Ok(SlashingResult {
            validator_id,
            penalty_fraction,
            penalty_amount,
            new_stake,
            slashing_epoch: current_epoch,
            recovery_parameters: self.calculate_recovery_parameters(offense, penalty_fraction).await?,
        })
    }

    async fn calculate_slashing_penalty(
        &self,
        offense: &SlashingOffense,
    ) -> Result<f64, StakeError> {
        let base_penalty = match offense.severity {
            OffenseSeverity::Low => 0.01,    // 1% penalty
            OffenseSeverity::Medium => 0.05, // 5% penalty
            OffenseSeverity::High => 0.10,   // 10% penalty
            OffenseSeverity::Critical => 0.33, // 33% penalty
        };

        // Adjust for repeat offenses
        let repeat_multiplier = if offense.repeat_count > 0 {
            1.0 + (offense.repeat_count as f64 * 0.5)
        } else {
            1.0
        };

        let total_penalty = base_penalty * repeat_multiplier;
        
        Ok(total_penalty.min(0.9)) // Cap at 90% penalty
    }
}

pub struct PowerCalculator {
    nonlinear_transforms: NonlinearTransformRegistry,
    security_models: SecurityModelSuite,
    economic_models: EconomicModelEngine,
}

impl PowerCalculator {
    pub async fn calculate_power_exponent(
        &self,
        network_state: &NetworkState,
    ) -> Result<f64, StakeError> {
        let base_exponent = 0.5; // Square root transformation by default
        
        // Adjust based on decentralization level
        let decentralization_adjustment = if network_state.decentralization_index < 0.3 {
            0.7 // More aggressive transformation for centralized networks
        } else if network_state.decentralization_index < 0.7 {
            0.5 // Moderate transformation
        } else {
            0.3 // Less aggressive transformation for decentralized networks
        };
        
        // Adjust based on network security
        let security_adjustment = match network_state.security_level {
            SecurityLevel::High => 0.8,
            SecurityLevel::Medium => 1.0,
            SecurityLevel::Low => 1.2,
        };
        
        let final_exponent = base_exponent * decentralization_adjustment * security_adjustment;
        
        Ok(final_exponent.max(0.1).min(1.0))
    }

    pub async fn calculate_security_multiplier(
        &self,
        network_state: &NetworkState,
    ) -> Result<f64, StakeError> {
        let base_multiplier = 1.0;
        
        // Adjust based on active validator count
        let validator_adjustment = if network_state.validator_count < 100 {
            0.8 // Lower multiplier for small networks
        } else if network_state.validator_count < 1000 {
            1.0 // Standard multiplier
        } else {
            1.2 // Higher multiplier for large networks
        };
        
        // Adjust based on network age and stability
        let stability_adjustment = network_state.network_stability.min(2.0).max(0.5);
        
        let final_multiplier = base_multiplier * validator_adjustment * stability_adjustment;
        
        Ok(final_multiplier.max(0.5).min(2.0))
    }
}