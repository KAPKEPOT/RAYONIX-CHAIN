// consensus/security/crisis/detector.rs
use crate::types::*;
use std::collections::{BTreeMap, HashMap, VecDeque, HashSet};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex, watch};
use rayon::prelude::*;
use statrs::{
    distribution::{Normal, Exponential, Poisson, Gamma, Continuous},
    statistics::{Statistics, Distribution},
};
use nalgebra::{DVector, DMatrix, SVD, SymmetricEigen};
use rand::prelude::*;
use rand_distr::{Normal as RandNormal, Poisson as RandPoisson};

pub struct CrisisDetector {
    health_metrics_collector: HealthMetricsCollector,
    anomaly_detection_engine: AnomalyDetectionEngine,
    threat_assessment_engine: ThreatAssessmentEngine,
    emergency_trigger_manager: EmergencyTriggerManager,
    mitigation_coordinator: MitigationCoordinator,
    crisis_history: Arc<RwLock<VecDeque<CrisisEvent>>>,
    alert_system: CrisisAlertSystem,
}

impl CrisisDetector {
    pub async fn monitor_consensus_health(
        &self,
        current_state: &ConsensusState,
        validators: &[ActiveValidator],
        network_metrics: &NetworkMetrics,
        economic_indicators: &EconomicIndicators,
    ) -> Result<ComprehensiveHealthAssessment, CrisisError> {
        // Phase 1: Collect comprehensive health metrics
        let health_metrics = self.health_metrics_collector.collect_comprehensive_metrics(
            current_state,
            validators,
            network_metrics,
            economic_indicators,
        ).await?;
        
        // Phase 2: Detect anomalies using multiple statistical models
        let anomaly_detection = self.anomaly_detection_engine.detect_anomalies(&health_metrics).await?;
        
        // Phase 3: Assess threat level and potential impact
        let threat_assessment = self.threat_assessment_engine.assess_threat_level(
            &anomaly_detection,
            &health_metrics,
            current_state,
        ).await?;
        
        // Phase 4: Check emergency triggers
        let triggered_emergencies = self.emergency_trigger_manager.check_triggers(
            &threat_assessment,
            &health_metrics,
            current_state,
        ).await?;
        
        // Phase 5: Calculate overall health score with confidence intervals
        let health_score = self.calculate_comprehensive_health_score(
            &health_metrics,
            &threat_assessment,
            &anomaly_detection,
        ).await?;
        
        // Phase 6: Determine required mitigation actions
        let mitigation_actions = self.determine_mitigation_actions(
            &threat_assessment,
            &triggered_emergencies,
            &health_metrics,
        ).await?;

        Ok(ComprehensiveHealthAssessment {
            health_score,
            health_metrics,
            anomaly_detection,
            threat_assessment,
            triggered_emergencies,
            mitigation_actions,
            assessment_timestamp: self.get_current_timestamp().await,
            confidence_level: self.calculate_assessment_confidence(&health_metrics, &anomaly_detection).await?,
            risk_quantiles: self.calculate_risk_quantiles(&threat_assessment).await?,
        })
    }

    async fn calculate_comprehensive_health_score(
        &self,
        metrics: &HealthMetrics,
        threat_assessment: &ThreatAssessment,
        anomaly_detection: &AnomalyDetectionResult,
    ) -> Result<HealthScore, CrisisError> {
        let component_scores = vec![
            // Liveness component (30% weight)
            self.calculate_liveness_component(&metrics.liveness_metrics, threat_assessment).await? * 0.3,
            
            // Safety component (30% weight)
            self.calculate_safety_component(&metrics.safety_metrics, threat_assessment).await? * 0.3,
            
            // Decentralization component (15% weight)
            self.calculate_decentralization_component(&metrics.decentralization_metrics).await? * 0.15,
            
            // Performance component (15% weight)
            self.calculate_performance_component(&metrics.performance_metrics).await? * 0.15,
            
            // Economic health component (10% weight)
            self.calculate_economic_health_component(&metrics.economic_metrics).await? * 0.1,
        ];

        let base_score: f64 = component_scores.iter().sum();
        
        // Apply threat-based discount using exponential decay
        let threat_discount = match threat_assessment.overall_threat_level {
            ThreatLevel::Critical => (-2.0).exp(), // ~13.5% of base score
            ThreatLevel::High => (-1.0).exp(),     // ~36.8% of base score
            ThreatLevel::Medium => (-0.5).exp(),   // ~60.7% of base score
            ThreatLevel::Low => (-0.2).exp(),      // ~81.9% of base score
            ThreatLevel::None => 1.0,              // 100% of base score
        };

        // Apply anomaly penalty using weighted sum of anomaly scores
        let anomaly_penalty = anomaly_detection.anomalies.iter()
            .map(|anomaly| anomaly.severity_score * anomaly.confidence)
            .sum::<f64>()
            .min(0.5); // Cap at 50% penalty

        let final_score = base_score * threat_discount * (1.0 - anomaly_penalty);
        
        // Calculate confidence interval using bootstrap resampling
        let confidence_interval = self.calculate_score_confidence_interval(
            &component_scores,
            base_score,
            final_score,
        ).await?;

        Ok(HealthScore {
            value: final_score.max(0.0).min(100.0),
            base_score,
            threat_discount,
            anomaly_penalty,
            component_scores: component_scores.into_iter().enumerate().collect(),
            confidence_interval,
            trend_analysis: self.analyze_health_trend(final_score).await?,
        })
    }

    pub async fn detect_liveness_violation(
        &self,
        current_slot: Slot,
        finalized_epoch: Epoch,
        block_production_metrics: &BlockProductionMetrics,
        network_conditions: &NetworkConditions,
    ) -> Result<LivenessViolationDetection, CrisisError> {
        // Phase 1: Calculate block production rate with statistical significance
        let production_analysis = self.analyze_block_production_rate(
            block_production_metrics,
            network_conditions,
        ).await?;
        
        // Phase 2: Analyze finality progression
        let finality_progression = self.analyze_finality_progression(finalized_epoch, current_slot).await?;
        
        // Phase 3: Detect chain growth stagnation
        let growth_stagnation = self.detect_growth_stagnation(current_slot, block_production_metrics).await?;
        
        // Phase 4: Assess validator participation patterns
        let participation_assessment = self.assess_validator_participation(block_production_metrics).await?;
        
        // Phase 5: Calculate composite liveness score
        let liveness_score = self.calculate_liveness_score(
            &production_analysis,
            &finality_progression,
            &growth_stagnation,
            &participation_assessment,
            network_conditions,
        ).await?;
        
        // Phase 6: Determine violation severity using statistical thresholds
        let violation_severity = if liveness_score < self.crisis_config.critical_liveness_threshold {
            ViolationSeverity::Critical
        } else if liveness_score < self.crisis_config.high_liveness_threshold {
            ViolationSeverity::High
        } else if liveness_score < self.crisis_config.medium_liveness_threshold {
            ViolationSeverity::Medium
        } else {
            ViolationSeverity::None
        };

        Ok(LivenessViolationDetection {
            liveness_score,
            violation_severity,
            production_analysis,
            finality_progression,
            growth_stagnation,
            participation_assessment,
            statistical_confidence: self.calculate_liveness_confidence(
                &production_analysis,
                &participation_assessment,
            ).await?,
            recommended_actions: self.generate_liveness_recovery_actions(violation_severity).await?,
            estimated_recovery_time: self.estimate_recovery_time(violation_severity, network_conditions).await?,
        })
    }

    async fn analyze_block_production_rate(
        &self,
        metrics: &BlockProductionMetrics,
        network_conditions: &NetworkConditions,
    ) -> Result<ProductionAnalysis, CrisisError> {
        let expected_blocks = metrics.expected_blocks;
        let produced_blocks = metrics.blocks_produced;
        
        if expected_blocks == 0 {
            return Ok(ProductionAnalysis::insufficient_data());
        }

        let production_rate = produced_blocks as f64 / expected_blocks as f64;
        
        // Calculate statistical significance using binomial test
        let binomial_confidence = self.calculate_binomial_confidence(
            produced_blocks,
            expected_blocks,
            network_conditions.nominal_production_rate,
        ).await?;
        
        // Calculate z-score for production deviation
        let expected_production = expected_blocks as f64 * network_conditions.nominal_production_rate;
        let production_std_dev = (expected_blocks as f64 * 
            network_conditions.nominal_production_rate * 
            (1.0 - network_conditions.nominal_production_rate)).sqrt();
        
        let z_score = if production_std_dev > 0.0 {
            (produced_blocks as f64 - expected_production) / production_std_dev
        } else {
            0.0
        };
        
        // Detect production anomalies using control chart rules
        let anomalies = self.detect_production_anomalies(metrics, network_conditions).await?;

        Ok(ProductionAnalysis {
            production_rate,
            expected_rate: network_conditions.nominal_production_rate,
            z_score,
            binomial_confidence,
            anomalies,
            trend: self.analyze_production_trend(metrics).await?,
            seasonality_effect: self.detect_production_seasonality(metrics).await?,
        })
    }

    async fn calculate_binomial_confidence(
        &self,
        successes: u64,
        trials: u64,
        expected_probability: f64,
    ) -> Result<f64, CrisisError> {
        use statrs::distribution::Binomial;
        
        if trials == 0 {
            return Ok(0.0);
        }

        let binomial = Binomial::new(expected_probability, trials)
            .map_err(|e| CrisisError::StatisticalError(e.to_string()))?;
        
        // Calculate two-tailed p-value
        let observed_probability = successes as f64 / trials as f64;
        let deviation = (observed_probability - expected_probability).abs();
        
        if deviation == 0.0 {
            return Ok(1.0);
        }
        
        // Calculate probability of observing this deviation or more extreme
        let lower_tail = if observed_probability < expected_probability {
            binomial.cdf(successes)
        } else {
            binomial.cdf((expected_probability * trials as f64) as u64)
        };
        
        let upper_tail = if observed_probability > expected_probability {
            1.0 - binomial.cdf(successes - 1)
        } else {
            1.0 - binomial.cdf((expected_probability * trials as f64) as u64 - 1)
        };
        
        let p_value = (lower_tail + upper_tail).min(1.0);
        let confidence = 1.0 - p_value;
        
        Ok(confidence.max(0.0).min(1.0))
    }

    pub async fn detect_safety_violation(
        &self,
        fork_events: &[ForkEvent],
        finality_proofs: &[FinalityProof],
        validator_behavior: &[ValidatorBehaviorMetrics],
        network_topology: &NetworkTopology,
    ) -> Result<SafetyViolationDetection, CrisisError> {
        // Phase 1: Check for finality reversions
        let finality_reversions = self.detect_finality_reversions(finality_proofs).await?;
        
        // Phase 2: Analyze fork depth and frequency
        let fork_analysis = self.analyze_fork_characteristics(fork_events, network_topology).await?;
        
        // Phase 3: Detect equivocation patterns
        let equivocation_patterns = self.detect_equivocation_patterns(validator_behavior).await?;
        
        // Phase 4: Assess consensus safety margins
        let safety_margins = self.assess_safety_margins(&finality_reversions, &fork_analysis).await?;
        
        // Phase 5: Calculate safety score using multiple indicators
        let safety_score = self.calculate_safety_score(
            &finality_reversions,
            &fork_analysis,
            &equivocation_patterns,
            &safety_margins,
        ).await?;
        
        // Phase 6: Determine safety violation level
        let violation_level = if safety_score < self.crisis_config.critical_safety_threshold {
            SafetyViolationLevel::Critical
        } else if safety_score < self.crisis_config.high_safety_threshold {
            SafetyViolationLevel::High
        } else if safety_score < self.crisis_config.medium_safety_threshold {
            SafetyViolationLevel::Medium
        } else {
            SafetyViolationLevel::None
        };

        Ok(SafetyViolationDetection {
            safety_score,
            violation_level,
            finality_reversions,
            fork_analysis,
            equivocation_patterns,
            safety_margins,
            emergency_protocols: self.activate_safety_protocols(violation_level).await?,
            chain_integrity_check: self.perform_chain_integrity_check().await?,
            byzantine_resilience: self.assess_byzantine_resilience(validator_behavior, network_topology).await?,
        })
    }

    pub async fn activate_emergency_protocol(
        &self,
        crisis_type: CrisisType,
        severity: CrisisSeverity,
        triggering_events: Vec<CrisisEvent>,
        network_state: &NetworkState,
    ) -> Result<EmergencyProtocolActivation, CrisisError> {
        // Phase 1: Validate emergency activation conditions
        self.validate_emergency_activation(crisis_type, severity, &triggering_events, network_state).await?;
        
        // Phase 2: Activate appropriate emergency protocol
        let protocol = match crisis_type {
            CrisisType::LivenessViolation => {
                self.activate_liveness_recovery_protocol(severity, &triggering_events, network_state).await?
            }
            CrisisType::SafetyViolation => {
                self.activate_safety_recovery_protocol(severity, &triggering_events, network_state).await?
            }
            CrisisType::EconomicAttack => {
                self.activate_economic_defense_protocol(severity, &triggering_events, network_state).await?
            }
            CrisisType::NetworkPartition => {
                self.activate_partition_recovery_protocol(severity, &triggering_events, network_state).await?
            }
            CrisisType::ValidatorCollusion => {
                self.activate_collusion_response_protocol(severity, &triggering_events, network_state).await?
            }
            CrisisType::GovernanceAttack => {
                self.activate_governance_defense_protocol(severity, &triggering_events, network_state).await?
            }
        };
        
        // Phase 3: Notify network participants
        self.alert_system.broadcast_emergency_alert(&protocol, network_state).await?;
        
        // Phase 4: Update crisis history
        self.record_crisis_event(crisis_type, severity, triggering_events, network_state).await?;
        
        // Phase 5: Initialize mitigation procedures
        self.mitigation_coordinator.initialize_mitigation(&protocol, network_state).await?;

        Ok(protocol)
    }

    async fn activate_liveness_recovery_protocol(
        &self,
        severity: CrisisSeverity,
        triggering_events: &[CrisisEvent],
        network_state: &NetworkState,
    ) -> Result<EmergencyProtocol, CrisisError> {
        let recovery_parameters = match severity {
            CrisisSeverity::Critical => LivenessRecoveryParameters {
                emergency_fork_required: true,
                validator_set_rotation: 0.5, // Rotate 50% of validators
                block_time_adjustment: 2.0, // Double block time temporarily
                gas_limit_adjustment: 0.5, // Reduce gas limit by 50%
                emergency_inflation: 0.1, // 10% emergency inflation for recovery
                recovery_timeout: 86400, // 24 hours
            },
            CrisisSeverity::High => LivenessRecoveryParameters {
                emergency_fork_required: false,
                validator_set_rotation: 0.3,
                block_time_adjustment: 1.5,
                gas_limit_adjustment: 0.7,
                emergency_inflation: 0.05,
                recovery_timeout: 43200, // 12 hours
            },
            CrisisSeverity::Medium => LivenessRecoveryParameters {
                emergency_fork_required: false,
                validator_set_rotation: 0.1,
                block_time_adjustment: 1.2,
                gas_limit_adjustment: 0.9,
                emergency_inflation: 0.02,
                recovery_timeout: 21600, // 6 hours
            },
            _ => LivenessRecoveryParameters::default(),
        };

        Ok(EmergencyProtocol {
            protocol_type: EmergencyProtocolType::LivenessRecovery,
            severity,
            activation_timestamp: self.get_current_timestamp().await,
            recovery_parameters: EmergencyRecoveryParameters::Liveness(recovery_parameters),
            mitigation_strategy: self.design_liveness_mitigation_strategy(severity, network_state).await?,
            success_probability: self.calculate_recovery_success_probability(severity, network_state).await?,
            economic_impact: self.estimate_economic_impact(severity, network_state).await?,
        })
    }
}

pub struct AnomalyDetectionEngine {
    statistical_models: StatisticalModelRegistry,
    machine_learning_models: MachineLearningModelSuite,
    temporal_analyzers: TemporalAnalysisEngine,
    multivariate_analyzers: MultivariateAnalysisEngine,
}

impl AnomalyDetectionEngine {
    pub async fn detect_anomalies(
        &self,
        metrics: &HealthMetrics,
    ) -> Result<AnomalyDetectionResult, CrisisError> {
        let mut all_anomalies = Vec::new();
        
        // Statistical anomaly detection
        let statistical_anomalies = self.detect_statistical_anomalies(metrics).await?;
        all_anomalies.extend(statistical_anomalies);
        
        // Temporal anomaly detection
        let temporal_anomalies = self.detect_temporal_anomalies(metrics).await?;
        all_anomalies.extend(temporal_anomalies);
        
        // Multivariate anomaly detection
        let multivariate_anomalies = self.detect_multivariate_anomalies(metrics).await?;
        all_anomalies.extend(multivariate_anomalies);
        
        // Machine learning based anomaly detection
        let ml_anomalies = self.detect_ml_anomalies(metrics).await?;
        all_anomalies.extend(ml_anomalies);
        
        // Cluster anomalies and remove duplicates
        let clustered_anomalies = self.cluster_anomalies(all_anomalies).await?;
        
        // Calculate overall anomaly score
        let anomaly_score = self.calculate_anomaly_score(&clustered_anomalies).await?;

        Ok(AnomalyDetectionResult {
            anomalies: clustered_anomalies,
            anomaly_score,
            detection_confidence: self.calculate_detection_confidence(&clustered_anomalies).await?,
            false_positive_estimate: self.estimate_false_positive_rate(&clustered_anomalies).await?,
        })
    }

    async fn detect_statistical_anomalies(
        &self,
        metrics: &HealthMetrics,
    ) -> Result<Vec<Anomaly>, CrisisError> {
        let mut anomalies = Vec::new();
        
        // Z-score based anomaly detection
        let z_score_anomalies = self.detect_z_score_anomalies(metrics).await?;
        anomalies.extend(z_score_anomalies);
        
        // IQR based anomaly detection
        let iqr_anomalies = self.detect_iqr_anomalies(metrics).await?;
        anomalies.extend(iqr_anomalies);
        
        // Grubbs' test for outliers
        let grubbs_anomalies = self.detect_grubbs_anomalies(metrics).await?;
        anomalies.extend(grubbs_anomalies);
        
        // Generalized ESD test for multiple outliers
        let gesd_anomalies = self.detect_gesd_anomalies(metrics).await?;
        anomalies.extend(gesd_anomalies);

        Ok(anomalies)
    }

    async fn detect_z_score_anomalies(
        &self,
        metrics: &HealthMetrics,
    ) -> Result<Vec<Anomaly>, CrisisError> {
        let mut anomalies = Vec::new();
        let z_threshold = 3.0; // 3 sigma threshold
        
        // Analyze block production rate
        let production_rate = metrics.liveness_metrics.block_production_rate;
        let production_mean = metrics.historical_baselines.average_production_rate;
        let production_std = metrics.historical_baselines.production_rate_std;
        
        if production_std > 0.0 {
            let z_score = (production_rate - production_mean).abs() / production_std;
            if z_score > z_threshold {
                anomalies.push(Anomaly {
                    metric: AnomalyMetric::BlockProductionRate,
                    value: production_rate,
                    z_score,
                    severity_score: self.calculate_anomaly_severity(z_score).await?,
                    confidence: self.calculate_z_score_confidence(z_score).await?,
                    description: format!("Unusual block production rate: z-score = {:.2}", z_score),
                });
            }
        }

        // Analyze finality delay
        let finality_delay = metrics.safety_metrics.average_finality_delay;
        let delay_mean = metrics.historical_baselines.average_finality_delay;
        let delay_std = metrics.historical_baselines.finality_delay_std;
        
        if delay_std > 0.0 {
            let z_score = (finality_delay - delay_mean).abs() / delay_std;
            if z_score > z_threshold {
                anomalies.push(Anomaly {
                    metric: AnomalyMetric::FinalityDelay,
                    value: finality_delay,
                    z_score,
                    severity_score: self.calculate_anomaly_severity(z_score).await?,
                    confidence: self.calculate_z_score_confidence(z_score).await?,
                    description: format!("Unusual finality delay: z-score = {:.2}", z_score),
                });
            }
        }

        Ok(anomalies)
    }

    async fn detect_multivariate_anomalies(
        &self,
        metrics: &HealthMetrics,
    ) -> Result<Vec<Anomaly>, CrisisError> {
        // Use Mahalanobis distance for multivariate outlier detection
        let feature_vector = self.extract_feature_vector(metrics).await?;
        let covariance_matrix = self.calculate_feature_covariance(metrics).await?;
        
        // Calculate Mahalanobis distance
        let mahalanobis_distance = self.calculate_mahalanobis_distance(&feature_vector, &covariance_matrix).await?;
        
        // Use chi-squared distribution for threshold
        let chi_squared_threshold = 13.8; // p < 0.001 for 4 degrees of freedom
        
        if mahalanobis_distance > chi_squared_threshold {
            return Ok(vec![Anomaly {
                metric: AnomalyMetric::MultivariateOutlier,
                value: mahalanobis_distance,
                z_score: mahalanobis_distance.sqrt(), // Approximate z-score
                severity_score: self.calculate_mahalanobis_severity(mahalanobis_distance).await?,
                confidence: self.calculate_mahalanobis_confidence(mahalanobis_distance).await?,
                description: format!("Multivariate anomaly detected: Mahalanobis distance = {:.2}", mahalanobis_distance),
            }]);
        }

        Ok(Vec::new())
    }

    async fn calculate_mahalanobis_distance(
        &self,
        feature_vector: &DVector<f64>,
        covariance_matrix: &DMatrix<f64>,
    ) -> Result<f64, CrisisError> {
        // Mahalanobis distance: D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        
        // Calculate inverse of covariance matrix
        let covariance_inv = covariance_matrix
            .try_inverse()
            .ok_or(CrisisError::MatrixInversionFailed)?;
        
        // Assume zero-mean features (centered during preprocessing)
        let distance_squared = feature_vector.transpose() * &covariance_inv * feature_vector;
        
        Ok(distance_squared[0])
    }
}

pub struct ThreatAssessmentEngine {
    risk_models: RiskModelRegistry,
    impact_analyzers: ImpactAnalysisEngine,
    propagation_models: PropagationModelEngine,
    game_theoretic_analyzers: GameTheoreticAnalyzer,
}

impl ThreatAssessmentEngine {
    pub async fn assess_threat_level(
        &self,
        anomaly_detection: &AnomalyDetectionResult,
        health_metrics: &HealthMetrics,
        current_state: &ConsensusState,
    ) -> Result<ThreatAssessment, CrisisError> {
        // Phase 1: Analyze individual threat vectors
        let threat_vectors = self.analyze_threat_vectors(anomaly_detection, health_metrics).await?;
        
        // Phase 2: Assess potential impact
        let impact_assessment = self.assess_potential_impact(&threat_vectors, current_state).await?;
        
        // Phase 3: Model threat propagation
        let propagation_analysis = self.model_threat_propagation(&threat_vectors, health_metrics).await?;
        
        // Phase 4: Calculate composite threat level
        let threat_level = self.calculate_composite_threat_level(
            &threat_vectors,
            &impact_assessment,
            &propagation_analysis,
        ).await?;
        
        // Phase 5: Assess attacker incentives and capabilities
        let attacker_analysis = self.assess_attacker_capabilities(health_metrics, current_state).await?;

        Ok(ThreatAssessment {
            threat_level,
            threat_vectors,
            impact_assessment,
            propagation_analysis,
            attacker_analysis,
            confidence: self.calculate_threat_assessment_confidence(
                &threat_vectors,
                &impact_assessment,
            ).await?,
            mitigation_priority: self.calculate_mitigation_priority(&threat_vectors).await?,
        })
    }

    async fn analyze_threat_vectors(
        &self,
        anomaly_detection: &AnomalyDetectionResult,
        health_metrics: &HealthMetrics,
    ) -> Result<Vec<ThreatVector>, CrisisError> {
        let mut threat_vectors = Vec::new();
        
        // Liveness threat vector
        let liveness_threat = self.assess_liveness_threat(anomaly_detection, health_metrics).await?;
        if liveness_threat.severity > ThreatSeverity::Low {
            threat_vectors.push(liveness_threat);
        }
        
        // Safety threat vector
        let safety_threat = self.assess_safety_threat(anomaly_detection, health_metrics).await?;
        if safety_threat.severity > ThreatSeverity::Low {
            threat_vectors.push(safety_threat);
        }
        
        // Economic threat vector
        let economic_threat = self.assess_economic_threat(anomaly_detection, health_metrics).await?;
        if economic_threat.severity > ThreatSeverity::Low {
            threat_vectors.push(economic_threat);
        }
        
        // Network threat vector
        let network_threat = self.assess_network_threat(anomaly_detection, health_metrics).await?;
        if network_threat.severity > ThreatSeverity::Low {
            threat_vectors.push(network_threat);
        }

        Ok(threat_vectors)
    }

    async fn calculate_composite_threat_level(
        &self,
        threat_vectors: &[ThreatVector],
        impact_assessment: &ImpactAssessment,
        propagation_analysis: &PropagationAnalysis,
    ) -> Result<ThreatLevel, CrisisError> {
        // Calculate weighted threat score
        let threat_score: f64 = threat_vectors.iter()
            .map(|vector| vector.severity as u8 as f64 * vector.probability)
            .sum();
        
        // Adjust for impact and propagation
        let impact_factor = impact_assessment.overall_impact as f64 / 10.0;
        let propagation_factor = propagation_analysis.propagation_risk;
        
        let composite_score = threat_score * impact_factor * propagation_factor;
        
        // Map to threat level
        if composite_score >= 8.0 {
            Ok(ThreatLevel::Critical)
        } else if composite_score >= 6.0 {
            Ok(ThreatLevel::High)
        } else if composite_score >= 4.0 {
            Ok(ThreatLevel::Medium)
        } else if composite_score >= 2.0 {
            Ok(ThreatLevel::Low)
        } else {
            Ok(ThreatLevel::None)
        }
    }
}