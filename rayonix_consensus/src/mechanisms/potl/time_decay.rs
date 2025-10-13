// consensus/mechanisms/potl/time_decay.rs
use crate::types::*;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use rayon::prelude::*;
use statrs::{
    distribution::{Exponential, Gamma, Normal, Continuous},
    statistics::{Statistics, Distribution},
};
use rand::prelude::*;
use rand_distr::{Gamma as RandGamma, Beta};

pub struct TimeDecayEngine {
    decay_functions: DecayFunctionRegistry,
    memory_kernels: MemoryKernelCalculator,
    half_life_calculator: HalfLifeCalculator,
    retention_optimizer: RetentionOptimizer,
    temporal_correlations: TemporalCorrelationEngine,
}

impl TimeDecayEngine {
    pub async fn apply_temporal_decay(
        &self,
        validator_states: &[ActiveValidator],
        current_epoch: Epoch,
        network_conditions: &NetworkConditions,
    ) -> Result<Vec<DecayedValidatorState>, TimeDecayError> {
        // Phase 1: Calculate adaptive decay parameters based on network state
        let decay_parameters = self.calculate_adaptive_decay_parameters(network_conditions).await?;
        
        // Phase 2: Apply multi-component decay to each validator
        let decayed_states: Vec<DecayedValidatorState> = validator_states
            .par_iter()
            .map(|validator| {
                self.apply_validator_decay(validator, current_epoch, &decay_parameters).await
            })
            .collect::<Result<Vec<_>, TimeDecayError>>()?;
        
        // Phase 3: Apply temporal correlation corrections
        let correlated_states = self.apply_temporal_corrections(decayed_states, current_epoch).await?;
        
        // Phase 4: Optimize memory retention across the network
        let optimized_states = self.optimize_retention(correlated_states, network_conditions).await?;
        
        Ok(optimized_states)
    }
    
    async fn apply_validator_decay(
        &self,
        validator: &ActiveValidator,
        current_epoch: Epoch,
        parameters: &DecayParameters,
    ) -> Result<DecayedValidatorState, TimeDecayError> {
        let time_since_update = (current_epoch - validator.time_lived_state.last_reliability_update) as f64;
        
        // Component 1: Exponential moving average decay
        let ema_decay = self.apply_ema_decay(
            validator.time_lived_state.exponential_moving_average,
            time_since_update,
            parameters
        ).await?;
        
        // Component 2: Cumulative reliability decay with saturation
        let cumulative_decay = self.apply_cumulative_decay(
            validator.time_lived_state.cumulative_reliability,
            time_since_update,
            validator.performance.uptime_percentage,
            parameters
        ).await?;
        
        // Component 3: Performance history decay with recency weighting
        let performance_decay = self.apply_performance_decay(
            &validator.performance.historical_performance,
            time_since_update,
            parameters
        ).await?;
        
        // Component 4: Reputation decay with nonlinear scaling
        let reputation_decay = self.apply_reputation_decay(
            validator.current_score,
            time_since_update,
            validator.activation_epoch,
            parameters
        ).await?;
        
        // Combine decay components with weighted fusion
        let combined_decay = self.fuse_decay_components(
            ema_decay,
            cumulative_decay,
            performance_decay,
            reputation_decay,
            validator
        ).await?;
        
        Ok(DecayedValidatorState {
            validator_id: validator.identity.id,
            original_ema: validator.time_lived_state.exponential_moving_average,
            decayed_ema: ema_decay.decayed_value,
            original_cumulative: validator.time_lived_state.cumulative_reliability,
            decayed_cumulative: cumulative_decay.decayed_value,
            performance_decay_factor: performance_decay.decay_factor,
            reputation_decay_factor: reputation_decay.decay_factor,
            combined_decay_factor: combined_decay.overall_decay,
            time_decay_entropy: self.calculate_decay_entropy(&combined_decay).await?,
            confidence_interval: self.calculate_decay_confidence(
                time_since_update,
                validator.performance.consecutive_successes
            ).await?,
        })
    }
    
    async fn apply_ema_decay(
        &self,
        current_ema: f64,
        time_delta: f64,
        parameters: &DecayParameters,
    ) -> Result<EMADecayResult, TimeDecayError> {
        // Adaptive EMA decay: EMA_t = α * current + (1-α) * EMA_{t-1}
        // With time decay: α becomes time-dependent
        
        let base_alpha = parameters.ema_decay_factor;
        let time_adjusted_alpha = base_alpha * (1.0 + parameters.time_sensitivity * time_delta);
        
        // Apply exponential decay to EMA value
        let decay_factor = (-time_adjusted_alpha * time_delta).exp();
        let decayed_ema = current_ema * decay_factor;
        
        // Calculate decay rate derivative for stability analysis
        let decay_rate = -time_adjusted_alpha * decayed_ema;
        
        Ok(EMADecayResult {
            original_value: current_ema,
            decayed_value: decayed_ema.max(0.0).min(1.0),
            decay_factor,
            decay_rate,
            time_constant: 1.0 / time_adjusted_alpha,
            half_life: self.calculate_half_life(time_adjusted_alpha).await?,
        })
    }
    
    async fn apply_cumulative_decay(
        &self,
        cumulative_reliability: f64,
        time_delta: f64,
        current_uptime: f64,
        parameters: &DecayParameters,
    ) -> Result<CumulativeDecayResult, TimeDecayError> {
        // Cumulative reliability follows power-law decay with saturation
        let saturation_threshold = parameters.cumulative_saturation;
        
        if cumulative_reliability >= saturation_threshold {
            // Saturated regime: slower decay
            let saturation_decay = parameters.saturated_decay_rate;
            let decay_factor = (-saturation_decay * time_delta).exp();
            let decayed_value = cumulative_reliability * decay_factor;
            
            Ok(CumulativeDecayResult {
                original_value: cumulative_reliability,
                decayed_value: decayed_value.max(0.0),
                decay_factor,
                decay_regime: DecayRegime::Saturated,
                saturation_effect: self.calculate_saturation_effect(cumulative_reliability, saturation_threshold).await?,
            })
        } else {
            // Linear growth regime: faster decay but with recovery potential
            let growth_decay = parameters.growth_decay_rate * (1.0 - current_uptime);
            let decay_factor = (-growth_decay * time_delta).exp();
            let decayed_value = cumulative_reliability * decay_factor;
            
            Ok(CumulativeDecayResult {
                original_value: cumulative_reliability,
                decayed_value: decayed_value.max(0.0),
                decay_factor,
                decay_regime: DecayRegime::Growing,
                recovery_potential: self.calculate_recovery_potential(decayed_value, current_uptime).await?,
            })
        }
    }
    
    async fn apply_performance_decay(
        &self,
        historical_performance: &BTreeMap<Epoch, PerformanceSnapshot>,
        time_delta: f64,
        parameters: &DecayParameters,
    ) -> Result<PerformanceDecayResult, TimeDecayError> {
        if historical_performance.is_empty() {
            return Ok(PerformanceDecayResult::default());
        }
        
        // Convert historical performance to time-weighted vector
        let performance_series: Vec<(f64, f64)> = historical_performance
            .iter()
            .map(|(&epoch, snapshot)| {
                let time_offset = time_delta - (epoch as f64);
                let performance_value = snapshot.overall_performance();
                (time_offset, performance_value)
            })
            .collect();
        
        // Apply kernel-based decay with temporal weighting
        let kernel_weights = self.calculate_kernel_weights(&performance_series, parameters).await?;
        
        // Calculate decayed performance using convolution with memory kernel
        let decayed_performance = performance_series
            .iter()
            .zip(kernel_weights.iter())
            .map(|((_, perf), weight)| perf * weight)
            .sum::<f64>();
        
        let total_weight: f64 = kernel_weights.iter().sum();
        let normalized_performance = if total_weight > 0.0 {
            decayed_performance / total_weight
        } else {
            0.0
        };
        
        Ok(PerformanceDecayResult {
            decay_factor: normalized_performance,
            kernel_weights,
            temporal_resolution: self.calculate_temporal_resolution(&performance_series).await?,
            memory_integral: self.calculate_memory_integral(&kernel_weights).await?,
        })
    }
    
    async fn calculate_kernel_weights(
        &self,
        performance_series: &[(f64, f64)],
        parameters: &DecayParameters,
    ) -> Result<Vec<f64>, TimeDecayError> {
        let weights: Vec<f64> = performance_series
            .par_iter()
            .map(|(time_offset, _)| {
                // Use Gamma distribution kernel for flexible memory decay
                let shape = parameters.memory_shape;
                let scale = parameters.memory_scale;
                
                let gamma_dist = Gamma::new(shape, scale)
                    .map_err(|e| TimeDecayError::DistributionError(e.to_string()))?;
                
                // Kernel value at given time offset
                let kernel_value = gamma_dist.pdf(*time_offset);
                Ok(kernel_value)
            })
            .collect::<Result<Vec<_>, TimeDecayError>>()?;
        
        // Normalize weights
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            Ok(weights.iter().map(|&w| w / total).collect())
        } else {
            Ok(weights)
        }
    }
    
    async fn apply_reputation_decay(
        &self,
        current_score: f64,
        time_delta: f64,
        activation_epoch: Epoch,
        parameters: &DecayParameters,
    ) -> Result<ReputationDecayResult, TimeDecayError> {
        let validator_age = time_delta + activation_epoch as f64;
        
        // Reputation decay follows a Weibull distribution for aging effects
        let shape_parameter = parameters.reputation_shape;
        let scale_parameter = parameters.reputation_scale * validator_age;
        
        let weibull_decay = (-(time_delta / scale_parameter).powf(shape_parameter)).exp();
        let decayed_reputation = current_score * weibull_decay;
        
        // Calculate reputation resilience based on historical stability
        let resilience_factor = self.calculate_reputation_resilience(current_score, validator_age).await?;
        
        Ok(ReputationDecayResult {
            original_reputation: current_score,
            decayed_reputation: decayed_reputation.max(0.0).min(1.0),
            decay_factor: weibull_decay,
            resilience_factor,
            aging_penalty: self.calculate_aging_penalty(validator_age, parameters).await?,
            recovery_time: self.calculate_recovery_time(decayed_reputation, resilience_factor).await?,
        })
    }
    
    pub async fn calculate_optimal_decay_parameters(
        &self,
        network_metrics: &NetworkMetrics,
        historical_decay_data: &[HistoricalDecayRecord],
    ) -> Result<OptimizedDecayParameters, TimeDecayError> {
        // Phase 1: Analyze historical decay patterns
        let pattern_analysis = self.analyze_decay_patterns(historical_decay_data).await?;
        
        // Phase 2: Fit decay models to historical data
        let model_fits = self.fit_decay_models(historical_decay_data).await?;
        
        // Phase 3: Calculate network-optimal parameters
        let network_optimal = self.calculate_network_optimal_parameters(network_metrics, &model_fits).await?;
        
        // Phase 4: Apply stability constraints
        let constrained_parameters = self.apply_stability_constraints(network_optimal, &pattern_analysis).await?;
        
        // Phase 5: Validate parameter feasibility
        self.validate_parameter_feasibility(&constrained_parameters).await?;
        
        Ok(constrained_parameters)
    }
    
    async fn analyze_decay_patterns(
        &self,
        historical_data: &[HistoricalDecayRecord],
    ) -> Result<DecayPatternAnalysis, TimeDecayError> {
        let decay_rates: Vec<f64> = historical_data
            .iter()
            .map(|record| record.decay_rate)
            .collect();
        
        let timescales: Vec<f64> = historical_data
            .iter()
            .map(|record| record.timescale)
            .collect();
        
        // Statistical analysis of decay patterns
        let mean_decay_rate = decay_rates.mean();
        let decay_variance = decay_rates.variance();
        let timescale_distribution = self.analyze_timescale_distribution(&timescales).await?;
        
        // Autocorrelation analysis for temporal patterns
        let autocorrelation = self.calculate_autocorrelation(&decay_rates).await?;
        
        // Cluster analysis for different decay regimes
        let decay_clusters = self.cluster_decay_patterns(historical_data).await?;
        
        Ok(DecayPatternAnalysis {
            mean_decay_rate,
            decay_variance,
            timescale_distribution,
            autocorrelation,
            decay_clusters,
            stationarity_test: self.test_stationarity(&decay_rates).await?,
            regime_switches: self.detect_regime_switches(historical_data).await?,
        })
    }
    
    async fn fit_decay_models(
        &self,
        historical_data: &[HistoricalDecayRecord],
    ) -> Result<DecayModelFits, TimeDecayError> {
        // Fit multiple decay models to historical data
        let exponential_fit = self.fit_exponential_model(historical_data).await?;
        let power_law_fit = self.fit_power_law_model(historical_data).await?;
        let weibull_fit = self.fit_weibull_model(historical_data).await?;
        let stretched_exponential_fit = self.fit_stretched_exponential_model(historical_data).await?;
        
        // Compare model fits using information criteria
        let model_comparison = self.compare_decay_models(
            &exponential_fit,
            &power_law_fit,
            &weibull_fit,
            &stretched_exponential_fit,
            historical_data
        ).await?;
        
        Ok(DecayModelFits {
            exponential: exponential_fit,
            power_law: power_law_fit,
            weibull: weibull_fit,
            stretched_exponential: stretched_exponential_fit,
            best_model: model_comparison.best_model,
            model_weights: model_comparison.model_weights,
            combined_predictions: self.combine_model_predictions(
                &exponential_fit,
                &power_law_fit,
                &weibull_fit,
                &stretched_exponential_fit,
                &model_comparison.model_weights
            ).await?,
        })
    }
}

pub struct MemoryKernelCalculator {
    kernel_functions: KernelFunctionRegistry,
    convolution_engine: ConvolutionEngine,
    spectral_analyzer: SpectralAnalyzer,
}

impl MemoryKernelCalculator {
    pub async fn calculate_memory_kernel(
        &self,
        time_series: &[f64],
        kernel_type: KernelType,
        bandwidth: f64,
    ) -> Result<MemoryKernel, TimeDecayError> {
        let kernel_values = match kernel_type {
            KernelType::Exponential => {
                self.calculate_exponential_kernel(time_series.len(), bandwidth).await?
            }
            KernelType::Gaussian => {
                self.calculate_gaussian_kernel(time_series.len(), bandwidth).await?
            }
            KernelType::Epanechnikov => {
                self.calculate_epanechnikov_kernel(time_series.len(), bandwidth).await?
            }
            KernelType::Gamma => {
                self.calculate_gamma_kernel(time_series.len(), bandwidth).await?
            }
        };
        
        // Apply convolution to smooth the time series
        let smoothed_series = self.convolution_engine.convolve(time_series, &kernel_values).await?;
        
        // Calculate spectral properties of the kernel
        let spectral_properties = self.spectral_analyzer.analyze_kernel(&kernel_values).await?;
        
        Ok(MemoryKernel {
            kernel_values,
            smoothed_series,
            spectral_properties,
            effective_bandwidth: self.calculate_effective_bandwidth(&kernel_values).await?,
            memory_length: self.calculate_memory_length(&kernel_values).await?,
        })
    }
    
    async fn calculate_exponential_kernel(
        &self,
        length: usize,
        bandwidth: f64,
    ) -> Result<Vec<f64>, TimeDecayError> {
        let kernel: Vec<f64> = (0..length)
            .map(|i| {
                let t = i as f64;
                (-t / bandwidth).exp()
            })
            .collect();
        
        // Normalize kernel
        let sum: f64 = kernel.iter().sum();
        Ok(kernel.iter().map(|&k| k / sum).collect())
    }
    
    async fn calculate_gamma_kernel(
        &self,
        length: usize,
        bandwidth: f64,
    ) -> Result<Vec<f64>, TimeDecayError> {
        let shape = 2.0; // Shape parameter for Gamma distribution
        let scale = bandwidth;
        
        let kernel: Vec<f64> = (0..length)
            .map(|i| {
                let t = i as f64;
                // Gamma PDF: t^(k-1) * e^(-t/θ) / (θ^k * Γ(k))
                t.powf(shape - 1.0) * (-t / scale).exp() / (scale.powf(shape) * gamma(shape))
            })
            .collect();
        
        // Normalize kernel
        let sum: f64 = kernel.iter().sum();
        Ok(kernel.iter().map(|&k| k / sum).collect())
    }
}

// Gamma function approximation
fn gamma(x: f64) -> f64 {
    statrs::function::gamma::gamma(x)
}