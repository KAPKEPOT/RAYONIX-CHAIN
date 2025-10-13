// consensus/security/slashing/penalty_calculator.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Exponential, Gamma, Poisson, Continuous},
    statistics::{Statistics, Distribution},
};
use nalgebra::{DVector, DMatrix, SVD};
use rand::prelude::*;
use rand_distr::{Gamma as RandGamma, Beta};

pub struct PenaltyCalculator {
    base_penalty_models: BasePenaltyModelSuite,
    escalation_engine: EscalationEngine,
    economic_impact_analyzer: EconomicImpactAnalyzer,
    rehabilitation_calculator: RehabilitationCalculator,
    cross_validator_penalty: CrossValidatorPenaltyEngine,
    historical_penalty_analyzer: HistoricalPenaltyAnalyzer,
}

impl PenaltyCalculator {
    pub async fn calculate_comprehensive_penalty(
        &self,
        offense: &SlashingOffense,
        validator: &ActiveValidator,
        network_context: &NetworkContext,
        historical_offenses: &[HistoricalOffense],
    ) -> Result<ComprehensivePenalty, PenaltyError> {
        // Phase 1: Calculate base penalty using multiple models
        let base_penalty_components = self.calculate_base_penalty_components(offense, validator, network_context).await?;
        
        // Phase 2: Apply escalation based on historical behavior
        let escalation_components = self.apply_escalation_penalties(
            &base_penalty_components, 
            validator, 
            historical_offenses,
            network_context
        ).await?;
        
        // Phase 3: Calculate economic impact adjustments
        let economic_impact = self.calculate_economic_impact(
            &escalation_components, 
            validator, 
            network_context
        ).await?;
        
        // Phase 4: Apply cross-validator correlation penalties
        let correlation_penalties = self.apply_cross_validator_penalties(
            validator, 
            offense, 
            &escalation_components, 
            network_context
        ).await?;
        
        // Phase 5: Calculate rehabilitation requirements
        let rehabilitation_plan = self.calculate_rehabilitation_requirements(
            validator, 
            offense, 
            &escalation_components,
            historical_offenses
        ).await?;
        
        // Phase 6: Combine all penalty components
        let final_penalty = self.combine_penalty_components(
            &base_penalty_components,
            &escalation_components,
            &economic_impact,
            &correlation_penalties,
            network_context
        ).await?;

        Ok(ComprehensivePenalty {
            base_components: base_penalty_components,
            escalation_components,
            economic_impact,
            correlation_penalties,
            rehabilitation_plan,
            final_penalty,
            penalty_entropy: self.calculate_penalty_entropy(&final_penalty).await?,
            deterrence_factor: self.calculate_deterrence_factor(validator, offense, network_context).await?,
        })
    }

    async fn calculate_base_penalty_components(
        &self,
        offense: &SlashingOffense,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<BasePenaltyComponents, PenaltyError> {
        let effective_stake = validator.stake_state.effective_stake;
        
        // Component 1: Severity-based penalty
        let severity_penalty = self.calculate_severity_based_penalty(offense, effective_stake, context).await?;
        
        // Component 2: Stake-proportional penalty
        let stake_proportional_penalty = self.calculate_stake_proportional_penalty(offense, validator, context).await?;
        
        // Component 3: Network impact penalty
        let network_impact_penalty = self.calculate_network_impact_penalty(offense, validator, context).await?;
        
        // Component 4: Time-decay adjusted penalty
        let time_adjusted_penalty = self.calculate_time_adjusted_penalty(offense, validator, context).await?;
        
        // Component 5: Reputation damage penalty
        let reputation_penalty = self.calculate_reputation_penalty(offense, validator, context).await?;

        Ok(BasePenaltyComponents {
            severity_penalty,
            stake_proportional_penalty,
            network_impact_penalty,
            time_adjusted_penalty,
            reputation_penalty,
            combined_base_penalty: self.combine_base_penalties(
                severity_penalty,
                stake_proportional_penalty,
                network_impact_penalty,
                time_adjusted_penalty,
                reputation_penalty,
                context
            ).await?,
        })
    }

    async fn calculate_severity_based_penalty(
        &self,
        offense: &SlashingOffense,
        effective_stake: u128,
        context: &NetworkContext,
    ) -> Result<SeverityPenalty, PenaltyError> {
        let base_stake = effective_stake as f64;
        
        match offense.severity {
            OffenseSeverity::Critical => {
                // Critical offenses: 5-15% of effective stake
                let base_percentage = 0.08; // 8% base
                let confidence_adjustment = offense.confidence_score * 0.07; // Up to 7% additional
                let total_percentage = (base_percentage + confidence_adjustment).min(0.15).max(0.05);
                
                let penalty_amount = (base_stake * total_percentage) as u128;
                
                Ok(SeverityPenalty {
                    penalty_amount,
                    base_percentage,
                    confidence_adjustment,
                    total_percentage,
                    severity_multiplier: 3.0,
                    minimum_penalty: (base_stake * 0.05) as u128,
                    maximum_penalty: (base_stake * 0.15) as u128,
                })
            }
            OffenseSeverity::High => {
                // High offenses: 2-8% of effective stake
                let base_percentage = 0.04;
                let confidence_adjustment = offense.confidence_score * 0.04;
                let total_percentage = (base_percentage + confidence_adjustment).min(0.08).max(0.02);
                
                let penalty_amount = (base_stake * total_percentage) as u128;
                
                Ok(SeverityPenalty {
                    penalty_amount,
                    base_percentage,
                    confidence_adjustment,
                    total_percentage,
                    severity_multiplier: 2.0,
                    minimum_penalty: (base_stake * 0.02) as u128,
                    maximum_penalty: (base_stake * 0.08) as u128,
                })
            }
            OffenseSeverity::Medium => {
                // Medium offenses: 0.5-3% of effective stake
                let base_percentage = 0.015;
                let confidence_adjustment = offense.confidence_score * 0.015;
                let total_percentage = (base_percentage + confidence_adjustment).min(0.03).max(0.005);
                
                let penalty_amount = (base_stake * total_percentage) as u128;
                
                Ok(SeverityPenalty {
                    penalty_amount,
                    base_percentage,
                    confidence_adjustment,
                    total_percentage,
                    severity_multiplier: 1.0,
                    minimum_penalty: (base_stake * 0.005) as u128,
                    maximum_penalty: (base_stake * 0.03) as u128,
                })
            }
            OffenseSeverity::Low => {
                // Low offenses: 0.1-1% of effective stake
                let base_percentage = 0.003;
                let confidence_adjustment = offense.confidence_score * 0.007;
                let total_percentage = (base_percentage + confidence_adjustment).min(0.01).max(0.001);
                
                let penalty_amount = (base_stake * total_percentage) as u128;
                
                Ok(SeverityPenalty {
                    penalty_amount,
                    base_percentage,
                    confidence_adjustment,
                    total_percentage,
                    severity_multiplier: 0.5,
                    minimum_penalty: (base_stake * 0.001) as u128,
                    maximum_penalty: (base_stake * 0.01) as u128,
                })
            }
        }
    }

    async fn calculate_stake_proportional_penalty(
        &self,
        offense: &SlashingOffense,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<StakeProportionalPenalty, PenaltyError> {
        let effective_stake = validator.stake_state.effective_stake;
        let total_network_stake = context.total_network_stake;
        
        // Calculate stake concentration factor
        let stake_concentration = effective_stake as f64 / total_network_stake as f64;
        
        // Higher concentration leads to higher penalties (progressive taxation principle)
        let concentration_multiplier = if stake_concentration > 0.1 {
            2.0 // 100% increase for large stakeholders
        } else if stake_concentration > 0.05 {
            1.5 // 50% increase for medium stakeholders
        } else if stake_concentration > 0.01 {
            1.2 // 20% increase for small-medium stakeholders
        } else {
            1.0 // No increase for small stakeholders
        };
        
        // Base penalty scaled by stake concentration
        let base_penalty = match offense.severity {
            OffenseSeverity::Critical => 0.05,
            OffenseSeverity::High => 0.025,
            OffenseSeverity::Medium => 0.01,
            OffenseSeverity::Low => 0.002,
        };
        
        let adjusted_penalty_percentage = base_penalty * concentration_multiplier;
        let penalty_amount = (effective_stake as f64 * adjusted_penalty_percentage) as u128;
        
        Ok(StakeProportionalPenalty {
            penalty_amount,
            stake_concentration,
            concentration_multiplier,
            base_penalty_percentage: base_penalty,
            adjusted_penalty_percentage,
            progressivity_factor: self.calculate_progressivity_factor(stake_concentration).await?,
        })
    }

    async fn apply_escalation_penalties(
        &self,
        base_components: &BasePenaltyComponents,
        validator: &ActiveValidator,
        historical_offenses: &[HistoricalOffense],
        context: &NetworkContext,
    ) -> Result<EscalationComponents, PenaltyError> {
        let historical_count = historical_offenses.len() as u32;
        
        if historical_count == 0 {
            return Ok(EscalationComponents::no_escalation(base_components.combined_base_penalty));
        }
        
        // Calculate escalation factors using multiple models
        let exponential_escalation = self.calculate_exponential_escalation(historical_count, base_components).await?;
        let geometric_escalation = self.calculate_geometric_escalation(historical_count, base_components).await?;
        let recency_weighted_escalation = self.calculate_recency_weighted_escalation(historical_offenses, base_components).await?;
        
        // Combine escalation models using Bayesian model averaging
        let combined_escalation = self.combine_escalation_models(
            exponential_escalation,
            geometric_escalation,
            recency_weighted_escalation,
            historical_offenses
        ).await?;
        
        // Calculate final escalated penalty
        let escalated_penalty = (base_components.combined_base_penalty as f64 * combined_escalation.combined_multiplier) as u128;
        
        // Apply maximum escalation cap
        let max_escalated_penalty = (validator.stake_state.effective_stake as f64 * context.max_penalty_percentage) as u128;
        let final_escalated_penalty = escalated_penalty.min(max_escalated_penalty);
        
        Ok(EscalationComponents {
            historical_offense_count: historical_count,
            exponential_escalation,
            geometric_escalation,
            recency_weighted_escalation,
            combined_escalation,
            escalated_penalty: final_escalated_penalty,
            escalation_cap_applied: escalated_penalty > max_escalated_penalty,
            rehabilitation_impact: self.calculate_escalation_rehabilitation_impact(historical_count).await?,
        })
    }

    async fn calculate_exponential_escalation(
        &self,
        offense_count: u32,
        base_components: &BasePenaltyComponents,
    ) -> Result<ExponentialEscalation, PenaltyError> {
        // Exponential escalation: penalty grows as e^(λ * n)
        let lambda = 0.5; // Escalation rate parameter
        let exponent = lambda * offense_count as f64;
        let multiplier = exponent.exp();
        
        // Cap exponential growth to prevent excessive penalties
        let capped_multiplier = multiplier.min(10.0); // Maximum 10x multiplier
        
        let base_penalty = base_components.combined_base_penalty;
        let escalated_amount = (base_penalty as f64 * capped_multiplier) as u128;
        
        Ok(ExponentialEscalation {
            offense_count,
            lambda_parameter: lambda,
            raw_multiplier: multiplier,
            capped_multiplier,
            escalated_amount,
            growth_rate: self.calculate_exponential_growth_rate(offense_count, lambda).await?,
        })
    }

    async fn calculate_economic_impact(
        &self,
        escalation_components: &EscalationComponents,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<EconomicImpact, PenaltyError> {
        let penalty_amount = escalation_components.escalated_penalty;
        let validator_stake = validator.stake_state.effective_stake;
        
        // Calculate stake reduction impact
        let new_stake = validator_stake.saturating_sub(penalty_amount);
        let stake_reduction_ratio = if validator_stake > 0 {
            penalty_amount as f64 / validator_stake as f64
        } else {
            0.0
        };
        
        // Calculate voting power reduction
        let original_voting_power = self.calculate_voting_power(validator_stake, context).await?;
        let new_voting_power = self.calculate_voting_power(new_stake, context).await?;
        let voting_power_reduction = original_voting_power - new_voting_power;
        
        // Calculate economic opportunity cost
        let opportunity_cost = self.calculate_opportunity_cost(penalty_amount, validator, context).await?;
        
        // Calculate network economic impact
        let network_impact = self.calculate_network_economic_impact(penalty_amount, validator, context).await?;
        
        Ok(EconomicImpact {
            penalty_amount,
            stake_reduction_ratio,
            voting_power_reduction,
            opportunity_cost,
            network_impact,
            economic_entropy: self.calculate_economic_entropy(penalty_amount, validator_stake).await?,
            recovery_time: self.calculate_economic_recovery_time(penalty_amount, validator, context).await?,
        })
    }

    async fn calculate_opportunity_cost(
        &self,
        penalty_amount: u128,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<OpportunityCost, PenaltyError> {
        let expected_rewards = self.calculate_expected_rewards(validator, context).await?;
        let penalty_duration = self.calculate_penalty_duration(validator, context).await?;
        
        // Lost rewards during penalty period
        let lost_rewards = (expected_rewards * penalty_duration as f64) as u128;
        
        // Compound interest effect on lost stake
        let annual_yield = context.network_annual_yield;
        let compounding_effect = if penalty_duration > 0 {
            let periods = penalty_duration as f64 / 365.0; // Convert to years
            penalty_amount as f64 * ((1.0 + annual_yield).powf(periods) - 1.0)
        } else {
            0.0
        };
        
        let total_opportunity_cost = lost_rewards + compounding_effect as u128;
        
        Ok(OpportunityCost {
            lost_rewards,
            compounding_effect: compounding_effect as u128,
            total_opportunity_cost,
            penalty_duration,
            annual_yield,
            net_present_value: self.calculate_net_present_value(total_opportunity_cost, penalty_duration, annual_yield).await?,
        })
    }

    async fn apply_cross_validator_penalties(
        &self,
        validator: &ActiveValidator,
        offense: &SlashingOffense,
        escalation_components: &EscalationComponents,
        context: &NetworkContext,
    ) -> Result<CrossValidatorPenalties, PenaltyError> {
        let mut cross_penalties = Vec::new();
        
        // Penalty for correlated offenses (validators acting together)
        let correlation_penalty = self.calculate_correlation_penalty(validator, offense, context).await?;
        if correlation_penalty.penalty_amount > 0 {
            cross_penalties.push(correlation_penalty);
        }
        
        // Penalty for systemic risk contribution
        let systemic_risk_penalty = self.calculate_systemic_risk_penalty(validator, offense, context).await?;
        if systemic_risk_penalty.penalty_amount > 0 {
            cross_penalties.push(systemic_risk_penalty);
        }
        
        // Penalty for network topology impact
        let topology_penalty = self.calculate_topology_penalty(validator, context).await?;
        if topology_penalty.penalty_amount > 0 {
            cross_penalties.push(topology_penalty);
        }
        
        let total_cross_penalty: u128 = cross_penalties.iter().map(|p| p.penalty_amount).sum();
        
        Ok(CrossValidatorPenalties {
            penalties: cross_penalties,
            total_cross_penalty,
            correlation_strength: self.calculate_correlation_strength(validator, context).await?,
            systemic_risk_factor: self.calculate_systemic_risk_factor(validator, context).await?,
        })
    }

    async fn calculate_rehabilitation_requirements(
        &self,
        validator: &ActiveValidator,
        offense: &SlashingOffense,
        escalation_components: &EscalationComponents,
        historical_offenses: &[HistoricalOffense],
    ) -> Result<RehabilitationPlan, PenaltyError> {
        let offense_count = historical_offenses.len() as u32 + 1; // Include current offense
        
        // Calculate jail duration based on offense severity and history
        let base_jail_duration = self.calculate_base_jail_duration(offense).await?;
        let escalated_jail_duration = self.escalate_jail_duration(base_jail_duration, offense_count).await?;
        
        // Calculate probation period after jail
        let probation_period = self.calculate_probation_period(escalated_jail_duration, offense_count).await?;
        
        // Calculate retraining requirements
        let retraining_requirements = self.calculate_retraining_requirements(validator, offense).await?;
        
        // Calculate performance monitoring period
        let monitoring_period = self.calculate_monitoring_period(offense, offense_count).await?;
        
        // Calculate stake lock requirements
        let stake_lock_requirements = self.calculate_stake_lock_requirements(validator, escalation_components).await?;
        
        Ok(RehabilitationPlan {
            jail_duration: escalated_jail_duration,
            probation_period,
            retraining_requirements,
            monitoring_period,
            stake_lock_requirements,
            rehabilitation_score: self.calculate_rehabilitation_score(validator, historical_offenses).await?,
            success_probability: self.calculate_rehabilitation_success_probability(validator, offense).await?,
        })
    }

    async fn calculate_base_jail_duration(&self, offense: &SlashingOffense) -> Result<Epoch, PenaltyError> {
        match offense.severity {
            OffenseSeverity::Critical => Ok(8192),  // ~36 days at 6-second slots
            OffenseSeverity::High => Ok(4096),      // ~18 days
            OffenseSeverity::Medium => Ok(1024),    // ~4.5 days
            OffenseSeverity::Low => Ok(256),        // ~1.1 days
        }
    }

    async fn escalate_jail_duration(
        &self,
        base_duration: Epoch,
        offense_count: u32,
    ) -> Result<Epoch, PenaltyError> {
        if offense_count <= 1 {
            return Ok(base_duration);
        }
        
        // Exponential escalation for repeat offenses
        let escalation_factor = 2.0_f64.powi(offense_count as i32 - 1);
        let escalated_duration = (base_duration as f64 * escalation_factor) as Epoch;
        
        // Cap maximum jail duration
        let max_jail_duration = 65536; // ~4.5 months
        Ok(escalated_duration.min(max_jail_duration))
    }

    pub async fn calculate_whistleblower_reward(
        &self,
        penalty_amount: u128,
        offense: &SlashingOffense,
        reporter_credibility: f64,
        network_context: &NetworkContext,
    ) -> Result<WhistleblowerReward, PenaltyError> {
        let base_reward_percentage = 0.05; // 5% base reward
        
        // Adjust reward based on offense severity
        let severity_multiplier = match offense.severity {
            OffenseSeverity::Critical => 1.5,
            OffenseSeverity::High => 1.2,
            OffenseSeverity::Medium => 1.0,
            OffenseSeverity::Low => 0.8,
        };
        
        // Adjust based on reporter credibility
        let credibility_multiplier = reporter_credibility.max(0.1).min(2.0);
        
        // Adjust based on evidence quality
        let evidence_quality_multiplier = offense.confidence_score.max(0.5).min(1.5);
        
        let total_multiplier = severity_multiplier * credibility_multiplier * evidence_quality_multiplier;
        let reward_percentage = base_reward_percentage * total_multiplier;
        
        let base_reward = (penalty_amount as f64 * reward_percentage) as u128;
        
        // Apply minimum and maximum reward bounds
        let min_reward = 1_000_000; // 1 token minimum
        let max_reward = penalty_amount / 4; // Maximum 25% of penalty
        
        let final_reward = base_reward.max(min_reward).min(max_reward);
        
        Ok(WhistleblowerReward {
            base_reward,
            final_reward,
            reward_percentage,
            severity_multiplier,
            credibility_multiplier,
            evidence_quality_multiplier,
            total_multiplier,
            reward_breakdown: self.calculate_reward_breakdown(penalty_amount, final_reward).await?,
        })
    }
}

pub struct EconomicImpactAnalyzer {
    market_impact_model: MarketImpactModel,
    network_effects_calculator: NetworkEffectsCalculator,
    liquidity_impact_analyzer: LiquidityImpactAnalyzer,
    systemic_risk_assessor: SystemicRiskAssessor,
}

impl EconomicImpactAnalyzer {
    pub async fn calculate_network_economic_impact(
        &self,
        penalty_amount: u128,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<NetworkEconomicImpact, PenaltyError> {
        // Calculate market impact of penalty
        let market_impact = self.calculate_market_impact(penalty_amount, validator, context).await?;
        
        // Calculate network effects
        let network_effects = self.calculate_network_effects(validator, context).await?;
        
        // Calculate liquidity impact
        let liquidity_impact = self.calculate_liquidity_impact(penalty_amount, context).await?;
        
        // Calculate systemic risk contribution
        let systemic_risk = self.calculate_systemic_risk(validator, context).await?;
        
        Ok(NetworkEconomicImpact {
            market_impact,
            network_effects,
            liquidity_impact,
            systemic_risk,
            total_impact: self.combine_economic_impacts(&market_impact, &network_effects, &liquidity_impact, &systemic_risk).await?,
            impact_confidence: self.calculate_impact_confidence(validator, context).await?,
        })
    }

    async fn calculate_market_impact(
        &self,
        penalty_amount: u128,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<MarketImpact, PenaltyError> {
        let validator_stake = validator.stake_state.effective_stake;
        let total_stake = context.total_network_stake;
        
        // Price impact from stake reduction
        let stake_reduction_impact = penalty_amount as f64 / total_stake as f64;
        
        // Confidence impact from slashing event
        let confidence_impact = self.calculate_confidence_impact(validator, context).await?;
        
        // Volatility impact
        let volatility_impact = self.calculate_volatility_impact(penalty_amount, context).await?;
        
        Ok(MarketImpact {
            stake_reduction_impact,
            confidence_impact,
            volatility_impact,
            total_market_impact: stake_reduction_impact + confidence_impact + volatility_impact,
            market_cap_effect: self.calculate_market_cap_effect(penalty_amount, context).await?,
        })
    }

    async fn calculate_network_effects(
        &self,
        validator: &ActiveValidator,
        context: &NetworkContext,
    ) -> Result<NetworkEffects, PenaltyError> {
        // Calculate validator's network centrality
        let centrality = self.calculate_network_centrality(validator, context).await?;
        
        // Calculate impact on network security
        let security_impact = self.calculate_security_impact(validator, context).await?;
        
        // Calculate impact on network performance
        let performance_impact = self.calculate_performance_impact(validator, context).await?;
        
        Ok(NetworkEffects {
            centrality,
            security_impact,
            performance_impact,
            total_network_effect: centrality + security_impact + performance_impact,
            network_resilience: self.calculate_network_resilience(validator, context).await?,
        })
    }
}