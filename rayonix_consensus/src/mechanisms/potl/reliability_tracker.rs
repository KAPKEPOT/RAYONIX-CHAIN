// consensus/mechanisms/potl/reliability_tracker.rs
use crate::types::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::statistics::Statistics;

pub struct ReliabilityTracker {
    reliability_store: ReliabilityStore,
    decay_calculator: DecayCalculator,
    grace_period_manager: GracePeriodManager,
    anomaly_detector: ReliabilityAnomalyDetector,
    recovery_tracker: RecoveryTracker,
    historical_reliability: Arc<RwLock<BTreeMap<ValidatorId, VecDeque<ReliabilityRecord>>>>,
}

impl ReliabilityTracker {
    pub async fn update_reliability_metrics(
        &self,
        validator_id: ValidatorId,
        performance: &ValidatorPerformance,
        current_epoch: Epoch,
    ) -> Result<ReliabilityUpdate, ReliabilityError> {
        // Phase 1: Calculate current epoch reliability score
        let epoch_reliability = self.calculate_epoch_reliability(performance, current_epoch).await?;
        
        // Phase 2: Apply exponential moving average
        let updated_ema = self.update_exponential_moving_average(
            validator_id, 
            epoch_reliability, 
            current_epoch
        ).await?;
        
        // Phase 3: Update cumulative reliability
        let updated_cumulative = self.update_cumulative_reliability(
            validator_id, 
            epoch_reliability, 
            current_epoch
        ).await?;
        
        // Phase 4: Detect reliability anomalies
        let anomaly_detection = self.detect_reliability_anomalies(
            validator_id, 
            epoch_reliability, 
            &updated_ema
        ).await?;
        
        // Phase 5: Update grace period status
        let grace_period_update = self.update_grace_period(
            validator_id, 
            epoch_reliability, 
            current_epoch
        ).await?;
        
        // Phase 6: Track recovery progress if applicable
        let recovery_update = self.track_recovery_progress(
            validator_id, 
            epoch_reliability, 
            &anomaly_detection
        ).await?;
        
        let update = ReliabilityUpdate {
            validator_id,
            epoch: current_epoch,
            epoch_reliability,
            exponential_moving_average: updated_ema,
            cumulative_reliability: updated_cumulative,
            anomaly_detection,
            grace_period_status: grace_period_update,
            recovery_progress: recovery_update,
            reliability_trend: self.calculate_reliability_trend(validator_id).await?,
            confidence_interval: self.calculate_reliability_confidence(validator_id).await?,
        };
        
        // Phase 7: Store reliability record
        self.store_reliability_record(update.clone()).await?;
        
        Ok(update)
    }
    
    async fn calculate_epoch_reliability(
        &self,
        performance: &ValidatorPerformance,
        current_epoch: Epoch,
    ) -> Result<f64, ReliabilityError> {
        let mut component_scores = Vec::new();
        
        // Block proposal reliability (40% weight)
        if performance.blocks_proposed + performance.blocks_missed > 0 {
            let block_reliability = performance.blocks_proposed as f64 / 
                (performance.blocks_proposed + performance.blocks_missed) as f64;
            component_scores.push(block_reliability * 0.4);
        }
        
        // Attestation reliability (35% weight)
        if performance.attestations_made + performance.attestations_missed > 0 {
            let attestation_reliability = performance.attestations_made as f64 / 
                (performance.attestations_made + performance.attestations_missed) as f64;
            component_scores.push(attestation_reliability * 0.35);
        }
        
        // Sync committee reliability (15% weight)
        if performance.sync_committee_participation + performance.sync_committee_misses > 0 {
            let sync_reliability = performance.sync_committee_participation as f64 / 
                (performance.sync_committee_participation + performance.sync_committee_misses) as f64;
            component_scores.push(sync_reliability * 0.15);
        }
        
        // Latency performance (10% weight)
        let latency_score = self.calculate_latency_score(performance.average_latency_ms).await?;
        component_scores.push(latency_score * 0.1);
        
        // Calculate weighted average
        let weighted_score: f64 = component_scores.iter().sum();
        
        // Apply consecutive performance bonuses/penalties
        let consecutive_adjustment = self.calculate_consecutive_adjustment(performance).await?;
        let adjusted_score = (weighted_score * consecutive_adjustment)
            .max(0.0)
            .min(1.0);
        
        Ok(adjusted_score)
    }
    
    async fn update_exponential_moving_average(
        &self,
        validator_id: ValidatorId,
        current_reliability: f64,
        current_epoch: Epoch,
    ) -> Result<f64, ReliabilityError> {
        let historical_data = self.historical_reliability.read().await;
        let validator_history = historical_data.get(&validator_id);
        
        let previous_ema = validator_history
            .and_then(|history| history.back())
            .map(|record| record.exponential_moving_average)
            .unwrap_or(1.0); // Start with perfect reliability for new validators
        
        // Calculate adaptive decay factor based on validator history
        let decay_factor = self.calculate_adaptive_decay_factor(validator_id, validator_history).await?;
        
        // Update EMA: EMA_t = α * current + (1-α) * EMA_{t-1}
        let updated_ema = (decay_factor * current_reliability) + ((1.0 - decay_factor) * previous_ema);
        
        Ok(updated_ema.max(0.0).min(1.0))
    }
    
    async fn calculate_adaptive_decay_factor(
        &self,
        validator_id: ValidatorId,
        history: Option<&VecDeque<ReliabilityRecord>>,
    ) -> Result<f64, ReliabilityError> {
        let base_decay = self.reliability_config.base_decay_factor;
        
        guard let Some(history) = history else {
            return Ok(base_decay); // Use base decay for new validators
        };
        
        // Calculate reliability stability
        let stability = self.calculate_reliability_stability(history).await?;
        
        // Calculate validator tenure factor
        let tenure_factor = self.calculate_tenure_factor(history.len() as u64).await?;
        
        // Adjust decay based on stability and tenure
        // More stable validators get slower decay (smaller alpha)
        // Newer validators get faster decay to quickly establish reliability
        let adaptive_decay = base_decay * (1.0 - stability * 0.5) * tenure_factor;
        
        Ok(adaptive_decay.max(self.reliability_config.min_decay_factor)
            .min(self.reliability_config.max_decay_factor))
    }
    
    async fn detect_reliability_anomalies(
        &self,
        validator_id: ValidatorId,
        current_reliability: f64,
        current_ema: &f64,
    ) -> Result<ReliabilityAnomalyDetection, ReliabilityError> {
        let historical_data = self.historical_reliability.read().await;
        let validator_history = historical_data.get(&validator_id);
        
        guard let Some(history) = validator_history else {
            return Ok(ReliabilityAnomalyDetection::no_anomaly()); // No history for new validators
        };
        
        let recent_records: Vec<&ReliabilityRecord> = history.iter()
            .rev()
            .take(self.reliability_config.anomaly_detection_window as usize)
            .collect();
        
        if recent_records.len() < 3 {
            return Ok(ReliabilityAnomalyDetection::no_anomaly());
        }
        
        // Calculate statistical properties of recent reliability
        let recent_reliabilities: Vec<f64> = recent_records.iter()
            .map(|r| r.epoch_reliability)
            .collect();
        
        let mean_reliability = recent_reliabilities.mean();
        let std_dev = recent_reliabilities.std_dev();
        
        // Detect significant deviation from historical pattern
        let z_score = if std_dev > 0.0 {
            (current_reliability - mean_reliability) / std_dev
        } else {
            0.0
        };
        
        let deviation_anomaly = z_score.abs() > self.reliability_config.anomaly_z_threshold;
        
        // Detect sudden drop patterns
        let sudden_drop_detected = self.detect_sudden_drop_pattern(&recent_reliabilities, current_reliability).await?;
        
        // Detect recovery patterns
        let recovery_detected = self.detect_recovery_pattern(&recent_reliabilities, current_reliability).await?;
        
        Ok(ReliabilityAnomalyDetection {
            deviation_anomaly,
            sudden_drop_detected,
            recovery_detected,
            z_score,
            confidence: self.calculate_anomaly_confidence(z_score, sudden_drop_detected).await?,
            recommended_action: self.determine_anomaly_response(deviation_anomaly, sudden_drop_detected).await?,
        })
    }
    
    pub async fn calculate_time_lived_component(
        &self,
        validator_id: ValidatorId,
        current_epoch: Epoch,
    ) -> Result<TimeLivedComponent, ReliabilityError> {
        let historical_data = self.historical_reliability.read().await;
        let validator_history = historical_data.get(&validator_id);
        
        guard let Some(history) = validator_history else {
            return Ok(TimeLivedComponent::new_validator());
        };
        
        // Get current EMA
        let current_ema = history.back()
            .map(|record| record.exponential_moving_average)
            .unwrap_or(1.0);
        
        // Calculate tenure-based scaling
        let tenure = history.len() as u64;
        let tenure_factor = self.calculate_tenure_scaling_factor(tenure).await?;
        
        // Calculate stability bonus
        let stability_bonus = self.calculate_stability_bonus(history).await?;
        
        // Calculate reliability trend impact
        let trend_impact = self.calculate_trend_impact(history).await?;
        
        // Combine components with nonlinear transformation
        let base_component = current_ema * tenure_factor;
        let enhanced_component = base_component * (1.0 + stability_bonus) * (1.0 + trend_impact);
        
        // Apply sigmoid activation for bounded output
        let activated_component = 1.0 / (1.0 + (-8.0 * (enhanced_component - 0.5)).exp());
        
        Ok(TimeLivedComponent {
            value: activated_component,
            exponential_moving_average: current_ema,
            tenure,
            tenure_factor,
            stability_bonus,
            trend_impact,
            confidence: self.calculate_component_confidence(history).await?,
        })
    }
}