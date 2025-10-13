// consensus/economics/rewards/calculator.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Gamma, Beta, Poisson, Continuous},
    statistics::{Statistics, Distribution},
};
use nalgebra::{DVector, DMatrix, SVD, SymmetricEigen};
use rand::prelude::*;
use rand_distr::{Gamma as RandGamma, Beta as RandBeta};

pub struct RewardCalculator {
    base_reward_models: BaseRewardModelSuite,
    performance_adjuster: PerformanceAdjuster,
    inflation_scheduler: InflationScheduler,
    treasury_manager: TreasuryManager,
    reward_optimizer: RewardOptimizer,
    historical_analyzer: HistoricalAnalyzer,
}

impl RewardCalculator {
    pub async fn calculate_epoch_rewards(
        &self,
        validators: &[ActiveValidator],
        epoch: Epoch,
        network_metrics: &NetworkMetrics,
        economic_params: &EconomicParameters,
    ) -> Result<EpochRewardCalculation, RewardError> {
        // Phase 1: Calculate total reward pool
        let total_reward_pool = self.calculate_total_reward_pool(epoch, network_metrics, economic_params).await?;
        
        // Phase 2: Calculate base reward distribution
        let base_reward_distribution = self.calculate_base_reward_distribution(
            validators, 
            total_reward_pool, 
            economic_params
        ).await?;
        
        // Phase 3: Apply performance-based adjustments
        let performance_adjusted_rewards = self.apply_performance_adjustments(
            &base_reward_distribution, 
            validators, 
            epoch, 
            network_metrics
        ).await?;
        
        // Phase 4: Apply time-lived bonuses
        let time_lived_adjusted_rewards = self.apply_time_lived_bonuses(
            &performance_adjusted_rewards, 
            validators, 
            epoch
        ).await?;
        
        // Phase 5: Apply network condition modifiers
        let network_adjusted_rewards = self.apply_network_condition_modifiers(
            &time_lived_adjusted_rewards, 
            network_metrics, 
            economic_params
        ).await?;
        
        // Phase 6: Calculate treasury allocation
        let treasury_allocation = self.calculate_treasury_allocation(total_reward_pool, economic_params).await?;
        
        // Phase 7: Final reward distribution with constraints
        let final_distribution = self.apply_distribution_constraints(
            &network_adjusted_rewards, 
            total_reward_pool, 
            treasury_allocation,
            validators
        ).await?;

        Ok(EpochRewardCalculation {
            epoch,
            total_reward_pool,
            base_reward_distribution,
            performance_adjusted_rewards,
            time_lived_adjusted_rewards,
            network_adjusted_rewards,
            final_distribution,
            treasury_allocation,
            reward_metrics: self.calculate_reward_metrics(&final_distribution, validators).await?,
            economic_efficiency: self.calculate_economic_efficiency(&final_distribution, network_metrics).await?,
        })
    }

    async fn calculate_total_reward_pool(
        &self,
        epoch: Epoch,
        network_metrics: &NetworkMetrics,
        economic_params: &EconomicParameters,
    ) -> Result<u128, RewardError> {
        // Component 1: Base block reward
        let base_block_reward = self.calculate_base_block_reward(epoch, economic_params).await?;
        
        // Component 2: Transaction fees
        let transaction_fees = network_metrics.total_transaction_fees;
        
        // Component 3: MEV rewards (if applicable)
        let mev_rewards = self.calculate_mev_rewards(network_metrics).await?;
        
        // Component 4: Network activity bonus
        let activity_bonus = self.calculate_network_activity_bonus(network_metrics, economic_params).await?;
        
        // Component 5: Inflation adjustment
        let inflation_adjustment = self.calculate_inflation_adjustment(epoch, economic_params).await?;
        
        let total_reward = base_block_reward + transaction_fees + mev_rewards + activity_bonus;
        let inflation_adjusted_reward = (total_reward as f64 * inflation_adjustment) as u128;
        
        // Apply maximum emission constraints
        let max_emission = self.calculate_max_emission(epoch, economic_params).await?;
        let constrained_reward = inflation_adjusted_reward.min(max_emission);
        
        Ok(constrained_reward)
    }

    async fn calculate_base_block_reward(
        &self,
        epoch: Epoch,
        economic_params: &EconomicParameters,
    ) -> Result<u128, RewardError> {
        let base_reward = economic_params.base_reward_per_block;
        
        // Apply emission schedule
        let emission_factor = self.calculate_emission_factor(epoch, economic_params).await?;
        
        // Apply network security multiplier
        let security_multiplier = self.calculate_security_multiplier(economic_params).await?;
        
        let adjusted_reward = (base_reward as f64 * emission_factor * security_multiplier) as u128;
        
        Ok(adjusted_reward)
    }

    async fn calculate_base_reward_distribution(
        &self,
        validators: &[ActiveValidator],
        total_reward_pool: u128,
        economic_params: &EconomicParameters,
    ) -> Result<BaseRewardDistribution, RewardError> {
        let validator_count = validators.len();
        let total_stake: u128 = validators.iter()
            .map(|v| v.stake_state.effective_stake)
            .sum();
        
        let base_distribution: BTreeMap<ValidatorId, BaseRewardComponent> = validators
            .par_iter()
            .map(|validator| {
                // Component 1: Stake-proportional reward
                let stake_proportional = self.calculate_stake_proportional_reward(
                    validator, 
                    total_stake, 
                    total_reward_pool, 
                    economic_params
                ).await?;
                
                // Component 2: Equal distribution component
                let equal_distribution = self.calculate_equal_distribution_component(
                    validator_count, 
                    total_reward_pool, 
                    economic_params
                ).await?;
                
                // Component 3: Participation base reward
                let participation_base = self.calculate_participation_base_reward(validator, economic_params).await?;
                
                let total_base_reward = stake_proportional.amount + equal_distribution.amount + participation_base.amount;
                
                let base_component = BaseRewardComponent {
                    stake_proportional,
                    equal_distribution,
                    participation_base,
                    total_base_reward,
                    base_reward_share: total_base_reward as f64 / total_reward_pool as f64,
                };
                
                Ok((validator.identity.id, base_component))
            })
            .collect::<Result<BTreeMap<_, _>, RewardError>>()?;
        
        Ok(BaseRewardDistribution {
            distribution: base_distribution,
            total_stake,
            validator_count: validator_count as u32,
            average_base_reward: total_reward_pool / validator_count as u128,
            gini_coefficient: self.calculate_base_reward_gini(&base_distribution).await?,
        })
    }

    async fn calculate_stake_proportional_reward(
        &self,
        validator: &ActiveValidator,
        total_stake: u128,
        total_reward_pool: u128,
        economic_params: &EconomicParameters,
    ) -> Result<StakeProportionalReward, RewardError> {
        let validator_stake = validator.stake_state.effective_stake;
        
        if total_stake == 0 {
            return Ok(StakeProportionalReward::zero());
        }
        
        let stake_ratio = validator_stake as f64 / total_stake as f64;
        
        // Apply progressive taxation to stake rewards
        let progressivity_factor = self.calculate_progressivity_factor(stake_ratio, economic_params).await?;
        
        // Apply stake concentration penalties
        let concentration_penalty = self.calculate_stake_concentration_penalty(stake_ratio, economic_params).await?;
        
        let adjusted_stake_ratio = stake_ratio * progressivity_factor * (1.0 - concentration_penalty);
        
        let stake_reward_amount = (total_reward_pool as f64 * adjusted_stake_ratio * economic_params.stake_reward_weight) as u128;
        
        Ok(StakeProportionalReward {
            amount: stake_reward_amount,
            stake_ratio,
            progressivity_factor,
            concentration_penalty,
            adjusted_stake_ratio,
            raw_stake_reward: (total_reward_pool as f64 * stake_ratio * economic_params.stake_reward_weight) as u128,
        })
    }

    async fn apply_performance_adjustments(
        &self,
        base_distribution: &BaseRewardDistribution,
        validators: &[ActiveValidator],
        epoch: Epoch,
        network_metrics: &NetworkMetrics,
    ) -> Result<PerformanceAdjustedRewards, RewardError> {
        let performance_adjusted: BTreeMap<ValidatorId, PerformanceAdjustedReward> = validators
            .par_iter()
            .map(|validator| {
                let base_reward = base_distribution.distribution.get(&validator.identity.id)
                    .map(|c| c.total_base_reward)
                    .unwrap_or(0);
                
                // Calculate performance multiplier
                let performance_multiplier = self.calculate_performance_multiplier(validator, epoch, network_metrics).await?;
                
                // Calculate reliability bonus
                let reliability_bonus = self.calculate_reliability_bonus(validator, epoch).await?;
                
                // Calculate participation penalty
                let participation_penalty = self.calculate_participation_penalty(validator, epoch).await?;
                
                // Calculate latency adjustment
                let latency_adjustment = self.calculate_latency_adjustment(validator, network_metrics).await?;
                
                let adjusted_reward = ((base_reward as f64 * performance_multiplier * reliability_bonus * latency_adjustment) - participation_penalty).max(0.0) as u128;
                
                let performance_reward = PerformanceAdjustedReward {
                    base_reward,
                    adjusted_reward,
                    performance_multiplier,
                    reliability_bonus,
                    participation_penalty: participation_penalty as u128,
                    latency_adjustment,
                    performance_score: self.calculate_overall_performance_score(validator).await?,
                };
                
                Ok((validator.identity.id, performance_reward))
            })
            .collect::<Result<BTreeMap<_, _>, RewardError>>()?;
        
        Ok(PerformanceAdjustedRewards {
            distribution: performance_adjusted,
            total_adjusted_reward: performance_adjusted.values().map(|r| r.adjusted_reward).sum(),
            average_performance_multiplier: performance_adjusted.values().map(|r| r.performance_multiplier).sum::<f64>() / performance_adjusted.len() as f64,
            performance_entropy: self.calculate_performance_entropy(&performance_adjusted).await?,
        })
    }

    async fn calculate_performance_multiplier(
        &self,
        validator: &ActiveValidator,
        epoch: Epoch,
        network_metrics: &NetworkMetrics,
    ) -> Result<f64, RewardError> {
        let performance = &validator.performance;
        
        // Component 1: Block production performance
        let block_production_score = if performance.blocks_proposed + performance.blocks_missed > 0 {
            performance.blocks_proposed as f64 / (performance.blocks_proposed + performance.blocks_missed) as f64
        } else {
            1.0
        };
        
        // Component 2: Attestation performance
        let attestation_score = if performance.attestations_made + performance.attestations_missed > 0 {
            performance.attestations_made as f64 / (performance.attestations_made + performance.attestations_missed) as f64
        } else {
            1.0
        };
        
        // Component 3: Sync committee performance
        let sync_committee_score = if performance.sync_committee_participation + performance.sync_committee_misses > 0 {
            performance.sync_committee_participation as f64 / 
            (performance.sync_committee_participation + performance.sync_committee_misses) as f64
        } else {
            1.0
        };
        
        // Component 4: Latency performance (inverse relationship)
        let average_network_latency = network_metrics.average_block_propagation_latency_ms;
        let latency_score = if performance.average_latency_ms > 0.0 && average_network_latency > 0.0 {
            (average_network_latency / performance.average_latency_ms).min(2.0).max(0.5)
        } else {
            1.0
        };
        
        // Component 5: Uptime performance
        let uptime_score = performance.uptime_percentage / 100.0;
        
        // Weighted combination of performance components
        let weighted_score = (block_production_score * 0.35) +
                           (attestation_score * 0.30) +
                           (sync_committee_score * 0.15) +
                           (latency_score * 0.10) +
                           (uptime_score * 0.10);
        
        // Convert to multiplier with bounds
        let multiplier = 0.5 + weighted_score; // Range: 0.5 to 1.5
        
        Ok(multiplier.max(0.5).min(1.5))
    }

    async fn apply_time_lived_bonuses(
        &self,
        performance_rewards: &PerformanceAdjustedRewards,
        validators: &[ActiveValidator],
        epoch: Epoch,
    ) -> Result<TimeLivedAdjustedRewards, RewardError> {
        let time_lived_adjusted: BTreeMap<ValidatorId, TimeLivedAdjustedReward> = validators
            .par_iter()
            .map(|validator| {
                let performance_reward = performance_rewards.distribution.get(&validator.identity.id)
                    .map(|r| r.adjusted_reward)
                    .unwrap_or(0);
                
                // Calculate time-lived bonus factor
                let time_lived_bonus = self.calculate_time_lived_bonus_factor(validator, epoch).await?;
                
                // Calculate reliability persistence bonus
                let persistence_bonus = self.calculate_persistence_bonus(validator).await?;
                
                // Calculate tenure multiplier
                let tenure_multiplier = self.calculate_tenure_multiplier(validator, epoch).await?;
                
                let total_time_bonus = time_lived_bonus * persistence_bonus * tenure_multiplier;
                let time_adjusted_reward = (performance_reward as f64 * total_time_bonus) as u128;
                
                let time_reward = TimeLivedAdjustedReward {
                    performance_reward,
                    time_adjusted_reward,
                    time_lived_bonus,
                    persistence_bonus,
                    tenure_multiplier,
                    total_time_bonus,
                    effective_tenure: self.calculate_effective_tenure(validator, epoch).await?,
                };
                
                Ok((validator.identity.id, time_reward))
            })
            .collect::<Result<BTreeMap<_, _>, RewardError>>()?;
        
        Ok(TimeLivedAdjustedRewards {
            distribution: time_lived_adjusted,
            total_time_adjusted_reward: time_lived_adjusted.values().map(|r| r.time_adjusted_reward).sum(),
            average_time_bonus: time_lived_adjusted.values().map(|r| r.total_time_bonus).sum::<f64>() / time_lived_adjusted.len() as f64,
            tenure_distribution: self.analyze_tenure_distribution(&time_lived_adjusted).await?,
        })
    }

    async fn calculate_time_lived_bonus_factor(
        &self,
        validator: &ActiveValidator,
        current_epoch: Epoch,
    ) -> Result<f64, RewardError> {
        let activation_epoch = validator.activation_epoch;
        let tenure_epochs = current_epoch.saturating_sub(activation_epoch);
        
        // Convert tenure to years (assuming 6-second slots)
        let tenure_years = tenure_epochs as f64 * 6.0 / (365.25 * 24.0 * 3600.0);
        
        // Logistic growth function for time-lived bonus
        let growth_rate = 2.0; // Controls how quickly bonus grows
        let carrying_capacity = 1.5; // Maximum bonus factor
        let midpoint = 1.0; // Year where bonus is half maximum
        
        let logistic_bonus = carrying_capacity / (1.0 + (-growth_rate * (tenure_years - midpoint)).exp());
        
        // Apply minimum and maximum bounds
        let bounded_bonus = logistic_bonus.max(1.0).min(carrying_capacity);
        
        Ok(bounded_bonus)
    }

    async fn apply_network_condition_modifiers(
        &self,
        time_lived_rewards: &TimeLivedAdjustedRewards,
        network_metrics: &NetworkMetrics,
        economic_params: &EconomicParameters,
    ) -> Result<NetworkAdjustedRewards, RewardError> {
        let network_adjusted: BTreeMap<ValidatorId, NetworkAdjustedReward> = time_lived_rewards.dribution
            .par_iter()
            .map(|(validator_id, time_reward)| {
                let time_adjusted_reward = time_reward.time_adjusted_reward;
                
                // Network load modifier
                let load_modifier = self.calculate_network_load_modifier(network_metrics, economic_params).await?;
                
                // Security level modifier
                let security_modifier = self.calculate_security_modifier(network_metrics, economic_params).await?;
                
                // Decentralization bonus
                let decentralization_bonus = self.calculate_decentralization_bonus(network_metrics, economic_params).await?;
                
                // Economic stability modifier
                let economic_stability_modifier = self.calculate_economic_stability_modifier(network_metrics, economic_params).await?;
                
                let total_network_modifier = load_modifier * security_modifier * decentralization_bonus * economic_stability_modifier;
                
                let network_adjusted_reward = (time_adjusted_reward as f64 * total_network_modifier) as u128;
                
                let network_reward = NetworkAdjustedReward {
                    time_adjusted_reward,
                    network_adjusted_reward,
                    load_modifier,
                    security_modifier,
                    decentralization_bonus,
                    economic_stability_modifier,
                    total_network_modifier,
                    network_contribution: self.calculate_network_contribution(validator_id, network_metrics).await?,
                };
                
                Ok((*validator_id, network_reward))
            })
            .collect::<Result<BTreeMap<_, _>, RewardError>>()?;
        
        Ok(NetworkAdjustedRewards {
            distribution: network_adjusted,
            total_network_adjusted_reward: network_adjusted.values().map(|r| r.network_adjusted_reward).sum(),
            average_network_modifier: network_adjusted.values().map(|r| r.total_network_modifier).sum::<f64>() / network_adjusted.len() as f64,
            network_efficiency: self.calculate_network_efficiency(&network_adjusted, network_metrics).await?,
        })
    }

    async fn apply_distribution_constraints(
        &self,
        network_rewards: &NetworkAdjustedRewards,
        total_reward_pool: u128,
        treasury_allocation: u128,
        validators: &[ActiveValidator],
    ) -> Result<FinalRewardDistribution, RewardError> {
        let available_rewards = total_reward_pool.saturating_sub(treasury_allocation);
        let current_total: u128 = network_rewards.distribution.values().map(|r| r.network_adjusted_reward).sum();
        
        // Calculate scaling factor to fit within available rewards
        let scaling_factor = if current_total > 0 {
            available_rewards as f64 / current_total as f64
        } else {
            1.0
        };
        
        let final_distribution: BTreeMap<ValidatorId, FinalValidatorReward> = network_rewards.dribution
            .par_iter()
            .map(|(validator_id, network_reward)| {
                let scaled_reward = (network_reward.network_adjusted_reward as f64 * scaling_factor) as u128;
                
                // Apply minimum reward guarantee
                let min_reward = self.calculate_minimum_reward(validators, available_rewards).await?;
                let reward_with_min = scaled_reward.max(min_reward);
                
                // Apply maximum reward cap
                let max_reward = self.calculate_maximum_reward(validators, available_rewards).await?;
                let capped_reward = reward_with_min.min(max_reward);
                
                // Calculate tax and fees
                let (tax_amount, net_reward) = self.calculate_tax_and_fees(capped_reward, validators, validator_id).await?;
                
                let final_reward = FinalValidatorReward {
                    gross_reward: capped_reward,
                    tax_amount,
                    net_reward,
                    scaling_factor_applied: scaling_factor,
                    min_reward_guarantee: min_reward,
                    max_reward_cap: max_reward,
                    tax_rate: self.calculate_effective_tax_rate(capped_reward, tax_amount).await?,
                };
                
                Ok((*validator_id, final_reward))
            })
            .collect::<Result<BTreeMap<_, _>, RewardError>>()?;
        
        // Verify distribution constraints
        self.verify_distribution_constraints(&final_distribution, available_rewards).await?;
        
        Ok(FinalRewardDistribution {
            distribution: final_distribution,
            total_distributed: final_distribution.values().map(|r| r.net_reward).sum(),
            treasury_allocation,
            scaling_factor,
            distribution_efficiency: self.calculate_distribution_efficiency(&final_distribution, validators).await?,
            fairness_metrics: self.calculate_reward_fairness_metrics(&final_distribution, validators).await?,
        })
    }

    pub async fn calculate_treasury_allocation(
        &self,
        total_reward_pool: u128,
        economic_params: &EconomicParameters,
    ) -> Result<TreasuryAllocation, RewardError> {
        let base_treasury_percentage = economic_params.treasury_percentage;
        
        // Dynamic treasury allocation based on network needs
        let dynamic_adjustment = self.calculate_dynamic_treasury_adjustment(economic_params).await?;
        
        let total_treasury_percentage = base_treasury_percentage * dynamic_adjustment;
        let treasury_amount = (total_reward_pool as f64 * total_treasury_percentage) as u128;
        
        // Allocate treasury to different funds
        let development_fund = (treasury_amount as f64 * economic_params.development_fund_allocation) as u128;
        let security_fund = (treasury_amount as f64 * economic_params.security_fund_allocation) as u128;
        let community_fund = (treasury_amount as f64 * economic_params.community_fund_allocation) as u128;
        let reserve_fund = treasury_amount - development_fund - security_fund - community_fund;
        
        Ok(TreasuryAllocation {
            total_amount: treasury_amount,
            development_fund,
            security_fund,
            community_fund,
            reserve_fund,
            treasury_percentage: total_treasury_percentage,
            dynamic_adjustment,
            allocation_efficiency: self.calculate_treasury_allocation_efficiency(
                development_fund, 
                security_fund, 
                community_fund, 
                reserve_fund
            ).await?,
        })
    }
}

pub struct PerformanceAdjuster {
    performance_metrics: PerformanceMetricsCalculator,
    reliability_analyzer: ReliabilityAnalyzer,
    latency_optimizer: LatencyOptimizer,
    participation_tracker: ParticipationTracker,
}

impl PerformanceAdjuster {
    pub async fn calculate_reliability_bonus(
        &self,
        validator: &ActiveValidator,
        current_epoch: Epoch,
    ) -> Result<f64, RewardError> {
        let reliability_ema = validator.time_lived_state.exponential_moving_average;
        
        // Convert EMA to bonus using sigmoid function
        let sigmoid_center = 0.8; // 80% reliability
        let sigmoid_steepness = 10.0; // Controls transition sharpness
        
        let reliability_bonus = 1.0 / (1.0 + (-sigmoid_steepness * (reliability_ema - sigmoid_center)).exp());
        
        // Apply bounds
        let bounded_bonus = reliability_bonus.max(0.8).min(1.2);
        
        Ok(bounded_bonus)
    }

    pub async fn calculate_participation_penalty(
        &self,
        validator: &ActiveValidator,
        epoch: Epoch,
    ) -> Result<f64, RewardError> {
        let performance = &validator.performance;
        
        // Calculate missed participation rate
        let total_expected = performance.blocks_proposed + performance.blocks_missed + 
                           performance.attestations_made + performance.attestations_missed +
                           performance.sync_committee_participation + performance.sync_committee_misses;
        
        let total_missed = performance.blocks_missed + performance.attestations_missed + performance.sync_committee_misses;
        
        let missed_rate = if total_expected > 0 {
            total_missed as f64 / total_expected as f64
        } else {
            0.0
        };
        
        // Calculate penalty using power function for progressive penalties
        let penalty_exponent = 2.0; // Quadratic penalty
        let penalty_amount = missed_rate.powf(penalty_exponent) * 0.5; // Maximum 50% penalty
        
        Ok(penalty_amount)
    }
}

pub struct InflationScheduler {
    emission_curve: EmissionCurveCalculator,
    monetary_policy: MonetaryPolicyEngine,
    supply_tracker: SupplyTracker,
    inflation_targeter: InflationTargeter,
}

impl InflationScheduler {
    pub async fn calculate_emission_factor(
        &self,
        epoch: Epoch,
        economic_params: &EconomicParameters,
    ) -> Result<f64, RewardError> {
        let total_supply = self.supply_tracker.get_total_supply(epoch).await?;
        let max_supply = economic_params.max_supply;
        
        // Calculate current inflation rate
        let current_inflation = self.calculate_current_inflation_rate(epoch, economic_params).await?;
        let target_inflation = economic_params.target_inflation_rate;
        
        // Calculate supply utilization
        let supply_utilization = if max_supply > 0 {
            total_supply as f64 / max_supply as f64
        } else {
            0.0
        };
        
        // Apply emission schedule (e.g., halving events)
        let halving_factor = self.calculate_halving_factor(epoch, economic_params).await?;
        
        // Calculate inflation adjustment
        let inflation_adjustment = if current_inflation > target_inflation {
            // Reduce emissions if inflation is too high
            1.0 - (current_inflation - target_inflation) * 2.0
        } else {
            // Increase emissions if inflation is too low
            1.0 + (target_inflation - current_inflation) * 1.5
        };
        
        let emission_factor = halving_factor * inflation_adjustment * (1.0 - supply_utilization * 0.5);
        
        Ok(emission_factor.max(0.1).min(2.0))
    }

    async fn calculate_halving_factor(
        &self,
        epoch: Epoch,
        economic_params: &EconomicParameters,
    ) -> Result<f64, RewardError> {
        let halving_interval = economic_params.halving_interval_epochs;
        let halving_count = epoch / halving_interval;
        
        // Each halving reduces emission by 50%
        let halving_factor = 0.5_f64.powi(halving_count as i32);
        
        Ok(halving_factor)
    }
}