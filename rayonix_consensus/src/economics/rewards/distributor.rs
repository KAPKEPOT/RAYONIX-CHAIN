// consensus/economics/rewards/distributor.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use rayon::prelude::*;
use statrs::statistics::Statistics;

pub struct RewardDistributor {
    fairness_engine: FairnessEngine,
    inflation_curve: InflationCurve,
    reward_pool_calculator: RewardPoolCalculator,
    distribution_verifier: DistributionVerifier,
    historical_distributions: Arc<RwLock<BTreeMap<Epoch, RewardDistribution>>>,
    treasury_manager: TreasuryManager,
}

impl RewardDistributor {
    pub async fn calculate_epoch_rewards(
        &self,
        validators: &[ActiveValidator],
        epoch: Epoch,
        total_transaction_fees: u128,
        network_metrics: &NetworkMetrics,
    ) -> Result<RewardDistribution, RewardError> {
        // Phase 1: Calculate total reward pool
        let total_reward_pool = self.calculate_total_reward_pool(epoch, total_transaction_fees, network_metrics).await?;
        
        // Phase 2: Calculate individual validator rewards
        let validator_rewards = self.calculate_validator_rewards(validators, total_reward_pool, epoch).await?;
        
        // Phase 3: Apply fairness corrections
        let fair_rewards = self.apply_fairness_corrections(validator_rewards, validators, epoch).await?;
        
        // Phase 4: Verify distribution integrity
        self.verify_distribution_integrity(&fair_rewards, total_reward_pool).await?;
        
        // Phase 5: Allocate treasury portion
        let (final_rewards, treasury_allocation) = self.allocate_treasury_portion(fair_rewards, total_reward_pool).await?;
        
        let distribution = RewardDistribution {
            epoch,
            total_reward_pool,
            validator_rewards: final_rewards,
            treasury_allocation,
            distribution_hash: self.calculate_distribution_hash(&final_rewards).await,
            merkle_root: self.calculate_merkle_root(&final_rewards).await?,
            timestamp: self.get_current_timestamp().await,
        };
        
        // Phase 6: Store distribution record
        self.store_distribution_record(distribution.clone()).await?;
        
        Ok(distribution)
    }
    
    async fn calculate_validator_rewards(
        &self,
        validators: &[ActiveValidator],
        total_reward_pool: u128,
        epoch: Epoch,
    ) -> Result<BTreeMap<ValidatorId, ValidatorReward>, RewardError> {
        let scores: Vec<f64> = validators.iter().map(|v| v.current_score).collect();
        
        // Calculate power weights with fairness exponent
        let power_weights: Vec<f64> = scores.par_iter()
            .map(|&score| score.powf(self.fairness_engine.fairness_exponent))
            .collect();
        
        let total_power: f64 = power_weights.par_iter().sum();
        
        if total_power == 0.0 {
            return Err(RewardError::ZeroTotalPower);
        }
        
        // Calculate base rewards
        let base_rewards: BTreeMap<ValidatorId, u128> = validators.par_iter()
            .zip(power_weights.par_iter())
            .map(|(validator, &weight)| {
                let reward_share = weight / total_power;
                let base_reward = (total_reward_pool as f64 * reward_share) as u128;
                (validator.identity.id, base_reward)
            })
            .collect();
        
        // Apply performance-based adjustments
        let performance_adjusted_rewards = self.apply_performance_adjustments(base_rewards, validators, epoch).await?;
        
        // Apply progressive taxation
        let taxed_rewards = self.apply_progressive_taxation(performance_adjusted_rewards, validators).await?;
        
        Ok(taxed_rewards)
    }
    
    async fn apply_fairness_corrections(
        &self,
        rewards: BTreeMap<ValidatorId, ValidatorReward>,
        validators: &[ActiveValidator],
        epoch: Epoch,
    ) -> Result<BTreeMap<ValidatorId, ValidatorReward>, RewardError> {
        let mut corrected_rewards = rewards;
        
        // Calculate Gini coefficient of current distribution
        let reward_values: Vec<f64> = corrected_rewards.values()
            .map(|r| r.amount as f64)
            .collect();
        let current_gini = self.calculate_gini_coefficient(&reward_values);
        
        // Apply corrections if inequality is too high
        if current_gini > self.fairness_engine.max_allowed_gini {
            corrected_rewards = self.redistribute_for_fairness(corrected_rewards, validators, current_gini).await?;
        }
        
        // Ensure minimum reward for active participants
        corrected_rewards = self.ensure_minimum_rewards(corrected_rewards, validators).await?;
        
        // Cap maximum reward to prevent dominance
        corrected_rewards = self.apply_maximum_reward_cap(corrected_rewards, validators).await?;
        
        Ok(corrected_rewards)
    }
    
    async fn apply_performance_adjustments(
        &self,
        base_rewards: BTreeMap<ValidatorId, u128>,
        validators: &[ActiveValidator],
        epoch: Epoch,
    ) -> Result<BTreeMap<ValidatorId, ValidatorReward>, RewardError> {
        let mut adjusted_rewards = BTreeMap::new();
        
        for (validator_id, base_reward) in base_rewards {
            let validator = validators.iter()
                .find(|v| v.identity.id == validator_id)
                .ok_or(RewardError::ValidatorNotFound(validator_id))?;
            
            // Calculate performance multiplier
            let performance_multiplier = self.calculate_performance_multiplier(validator, epoch).await?;
            
            // Calculate reliability bonus
            let reliability_bonus = self.calculate_reliability_bonus(validator).await?;
            
            // Calculate participation penalty
            let participation_penalty = self.calculate_participation_penalty(validator, epoch).await?;
            
            let adjusted_amount = ((base_reward as f64 * performance_multiplier * reliability_bonus) - participation_penalty).max(0.0) as u128;
            
            let reward = ValidatorReward {
                validator_id,
                amount: adjusted_amount,
                base_amount: base_reward,
                performance_multiplier,
                reliability_bonus,
                participation_penalty: participation_penalty as u128,
                tax_amount: 0, // Will be calculated in taxation phase
                net_amount: adjusted_amount, // Preliminary
                reward_components: RewardComponents {
                    stake_component: (base_reward as f64 * 0.4) as u128,
                    performance_component: (base_reward as f64 * 0.3) as u128,
                    time_lived_component: (base_reward as f64 * 0.3) as u128,
                    bonus_components: BTreeMap::new(),
                },
            };
            
            adjusted_rewards.insert(validator_id, reward);
        }
        
        Ok(adjusted_rewards)
    }
    
    async fn calculate_performance_multiplier(
        &self,
        validator: &ActiveValidator,
        epoch: Epoch,
    ) -> Result<f64, RewardError> {
        let performance = &validator.performance;
        
        // Block proposal performance
        let block_success_rate = if performance.blocks_proposed + performance.blocks_missed > 0 {
            performance.blocks_proposed as f64 / (performance.blocks_proposed + performance.blocks_missed) as f64
        } else {
            1.0
        };
        
        // Attestation performance
        let attestation_success_rate = if performance.attestations_made + performance.attestations_missed > 0 {
            performance.attestations_made as f64 / (performance.attestations_made + performance.attestations_missed) as f64
        } else {
            1.0
        };
        
        // Sync committee performance
        let sync_success_rate = if performance.sync_committee_participation + performance.sync_committee_misses > 0 {
            performance.sync_committee_participation as f64 / 
            (performance.sync_committee_participation + performance.sync_committee_misses) as f64
        } else {
            1.0
        };
        
        // Latency performance (inverse relationship)
        let latency_factor = (1000.0 / (performance.average_latency_ms.max(1.0))).min(2.0).max(0.5);
        
        // Combined performance score
        let combined_performance = (block_success_rate * 0.4 + 
                                  attestation_success_rate * 0.3 + 
                                  sync_success_rate * 0.2 + 
                                  latency_factor * 0.1) * validator.performance.uptime_percentage;
        
        // Convert to multiplier (0.5 to 1.5 range)
        let multiplier = 0.5 + combined_performance;
        
        Ok(multiplier.max(0.5).min(1.5))
    }
}