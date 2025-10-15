package utils

import (
    "fmt"
    "math"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
)

// RateLimiter implements a sophisticated, adaptive rate limiting system with physics-inspired dynamics
type RateLimiter struct {
    // Configuration
    config        *RateLimitConfig
    peerID        string
    
    // Token bucket state (atomic for concurrent access)
    tokens        atomic.Int64
    lastRefill    atomic.Int64 // Unix nanoseconds
    
    // Adaptive learning state
    learningRate  float64
    history       *RateLimitHistory
    adaptiveModel *AdaptiveRateModel
    
    // Physics-inspired properties
    pressure      float64 // System pressure (0-1)
    flowRate      float64 // Current flow rate
    resistance    float64 // System resistance
    
    // Statistics and monitoring
    stats         *RateLimitStats
    violationCount atomic.Uint64
    
    // Synchronization
    mu            sync.RWMutex
    lastUpdated   time.Time
}

// RateLimitConfig defines rate limiting configuration
type RateLimitConfig struct {
    // Basic token bucket parameters
    Rate         float64       `json:"rate"`          // Tokens per second
    Burst        int64         `json:"burst"`         // Maximum burst capacity
    Window       time.Duration `json:"window"`        // Time window for rate calculation
    
    // Adaptive learning parameters
    LearningRate float64       `json:"learning_rate"` // How quickly to adapt (0-1)
    Sensitivity  float64       `json:"sensitivity"`   // Sensitivity to violations (0-1)
    
    // Physics model parameters
    PressureThreshold float64 `json:"pressure_threshold"` // Pressure level for backpressure
    FlowDamping       float64 `json:"flow_damping"`       // Damping factor for flow oscillations
    ResistanceBase    float64 `json:"resistance_base"`    // Base resistance level
    
    // Enforcement parameters
    StrictMode    bool          `json:"strict_mode"`    // Whether to strictly enforce limits
    GracePeriod   time.Duration `json:"grace_period"`   // Grace period for new peers
    PenaltyFactor float64       `json:"penalty_factor"` // Penalty multiplier for violations
}

// RateLimitHistory tracks historical rate limiting data
type RateLimitHistory struct {
    windows      []*RateWindow
    currentWindow *RateWindow
    windowSize   time.Duration
    maxWindows   int
    mu           sync.RWMutex
}

// RateWindow represents rate data for a time window
type RateWindow struct {
    startTime    time.Time
    endTime      time.Time
    requestCount uint64
    tokenCount   uint64
    violationCount uint64
    pressure     float64
}

// AdaptiveRateModel implements machine learning for rate prediction
type AdaptiveRateModel struct {
    weights      []float64
    bias         float64
    features     *FeatureVector
    trainingData []*TrainingSample
    learningRate float64
    mu           sync.RWMutex
}

// FeatureVector contains features for rate prediction
type FeatureVector struct {
    RequestRate    float64   `json:"request_rate"`
    TokenUsage     float64   `json:"token_usage"`
    Pressure       float64   `json:"pressure"`
    TimeOfDay      float64   `json:"time_of_day"`
    DayOfWeek      float64   `json:"day_of_week"`
    NetworkLoad    float64   `json:"network_load"`
    PeerReputation float64   `json:"peer_reputation"`
}

// TrainingSample represents a training data point
type TrainingSample struct {
    Features *FeatureVector `json:"features"`
    Target   float64        `json:"target"` // Optimal rate
    Weight   float64        `json:"weight"`
    Timestamp time.Time     `json:"timestamp"`
}

// RateLimitStats provides comprehensive statistics
type RateLimitStats struct {
    TotalRequests   atomic.Uint64 `json:"total_requests"`
    AllowedRequests atomic.Uint64 `json:"allowed_requests"`
    DeniedRequests  atomic.Uint64 `json:"denied_requests"`
    TotalTokens     atomic.Uint64 `json:"total_tokens"`
    UsedTokens      atomic.Uint64 `json:"used_tokens"`
    
    CurrentRate     atomic.Float64 `json:"current_rate"`
    AverageRate     atomic.Float64 `json:"average_rate"`
    PeakRate        atomic.Float64 `json:"peak_rate"`
    
    PressureHistory *RollingStatistics `json:"-"`
    RateHistory     *RollingStatistics `json:"-"`
    
    LastViolation   atomic.Int64 `json:"last_violation"` // Unix nanoseconds
    ViolationStreak atomic.Uint32 `json:"violation_streak"`
    
    mu              sync.RWMutex
}

// NewRateLimiter creates a new adaptive rate limiter
func NewRateLimiter(peerID string, cfg *RateLimitConfig) *RateLimiter {
    if cfg == nil {
        cfg = &RateLimitConfig{
            Rate:            1000.0,
            Burst:           5000,
            Window:          time.Second,
            LearningRate:    0.1,
            Sensitivity:     0.5,
            PressureThreshold: 0.8,
            FlowDamping:     0.1,
            ResistanceBase:  1.0,
            StrictMode:      false,
            GracePeriod:     time.Minute * 5,
            PenaltyFactor:   1.5,
        }
    }
    
    now := time.Now()
    limiter := &RateLimiter{
        config:        cfg,
        peerID:        peerID,
        learningRate:  cfg.LearningRate,
        history:       NewRateLimitHistory(cfg.Window, 60), // 60 windows of history
        adaptiveModel: NewAdaptiveRateModel(cfg.LearningRate),
        pressure:      0.0,
        flowRate:      cfg.Rate,
        resistance:    cfg.ResistanceBase,
        stats:         NewRateLimitStats(),
        lastUpdated:   now,
    }
    
    // Initialize token bucket
    limiter.tokens.Store(cfg.Burst)
    limiter.lastRefill.Store(now.UnixNano())
    
    return limiter
}

// NewRateLimitHistory creates a new rate limit history tracker
func NewRateLimitHistory(windowSize time.Duration, maxWindows int) *RateLimitHistory {
    now := time.Now()
    return &RateLimitHistory{
        windows:      make([]*RateWindow, 0, maxWindows),
        currentWindow: &RateWindow{
            startTime: now,
            endTime:   now.Add(windowSize),
        },
        windowSize: windowSize,
        maxWindows: maxWindows,
    }
}

// NewAdaptiveRateModel creates a new adaptive rate model
func NewAdaptiveRateModel(learningRate float64) *AdaptiveRateModel {
    // 7 features: RequestRate, TokenUsage, Pressure, TimeOfDay, DayOfWeek, NetworkLoad, PeerReputation
    return &AdaptiveRateModel{
        weights:      make([]float64, 7),
        bias:         1.0,
        features:     &FeatureVector{},
        trainingData: make([]*TrainingSample, 0),
        learningRate: learningRate,
    }
}

// NewRateLimitStats creates new rate limit statistics
func NewRateLimitStats() *RateLimitStats {
    return &RateLimitStats{
        PressureHistory: NewRollingStatistics(1000),
        RateHistory:     NewRollingStatistics(1000),
    }
}

// Allow checks if a request should be allowed and updates internal state
func (rl *RateLimiter) Allow(tokens int64) (allowed bool, waitTime time.Duration, pressure float64) {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    now := time.Now()
    rl.stats.TotalRequests.Add(1)
    
    // Update token bucket
    rl.refillTokens(now)
    
    // Check if we're in grace period for new peers
    if rl.isInGracePeriod(now) {
        rl.stats.AllowedRequests.Add(1)
        rl.stats.UsedTokens.Add(uint64(tokens))
        return true, 0, rl.pressure
    }
    
    // Apply adaptive rate adjustment
    adjustedTokens := rl.applyAdaptiveAdjustment(tokens, now)
    
    // Check token availability
    currentTokens := rl.tokens.Load()
    if currentTokens >= adjustedTokens {
        // Allow request and consume tokens
        rl.tokens.Add(-adjustedTokens)
        rl.stats.AllowedRequests.Add(1)
        rl.stats.UsedTokens.Add(uint64(adjustedTokens))
        
        // Update flow rate and pressure
        rl.updateFlowDynamics(adjustedTokens, now)
        
        return true, 0, rl.pressure
    }
    
    // Request denied due to insufficient tokens
    rl.stats.DeniedRequests.Add(1)
    rl.recordViolation(now)
    
    // Calculate wait time
    deficit := adjustedTokens - currentTokens
    waitTime = time.Duration(float64(deficit)/rl.flowRate) * time.Second
    
    // Update pressure dynamics
    rl.updatePressureDynamics(deficit, now)
    
    return false, waitTime, rl.pressure
}

// AllowN checks if N requests should be allowed
func (rl *RateLimiter) AllowN(tokens int64, n int) (allowed bool, waitTime time.Duration, pressure float64) {
    return rl.Allow(tokens * int64(n))
}

// TryAllow attempts to allow a request without blocking
func (rl *RateLimiter) TryAllow(tokens int64) (allowed bool, pressure float64) {
    allowed, waitTime, pressure := rl.Allow(tokens)
    return allowed && waitTime == 0, pressure
}

// refillTokens refills tokens based on elapsed time
func (rl *RateLimiter) refillTokens(now time.Time) {
    lastRefill := time.Unix(0, rl.lastRefill.Load())
    elapsed := now.Sub(lastRefill).Seconds()
    
    if elapsed <= 0 {
        return
    }
    
    // Calculate tokens to add based on current flow rate
    tokensToAdd := int64(rl.flowRate * elapsed)
    if tokensToAdd > 0 {
        currentTokens := rl.tokens.Load()
        newTokens := currentTokens + tokensToAdd
        
        // Don't exceed burst capacity
        if newTokens > rl.config.Burst {
            newTokens = rl.config.Burst
        }
        
        rl.tokens.Store(newTokens)
        rl.lastRefill.Store(now.UnixNano())
        rl.stats.TotalTokens.Add(uint64(tokensToAdd))
    }
}

// applyAdaptiveAdjustment applies adaptive adjustments to token cost
func (rl *RateLimiter) applyAdaptiveAdjustment(tokens int64, now time.Time) int64 {
    // Base adjustment based on pressure
    pressureAdjustment := 1.0 + (rl.pressure * 0.5) // Increase cost under pressure
    
    // Reputation-based adjustment
    reputationAdjustment := rl.calculateReputationAdjustment()
    
    // Time-based adjustment (circadian rhythm)
    timeAdjustment := rl.calculateTimeAdjustment(now)
    
    // Network load adjustment
    loadAdjustment := rl.calculateLoadAdjustment()
    
    // Combine adjustments
    totalAdjustment := pressureAdjustment * reputationAdjustment * timeAdjustment * loadAdjustment
    
    adjustedTokens := float64(tokens) * totalAdjustment
    
    // Apply machine learning prediction if available
    if mlAdjustment := rl.predictOptimalAdjustment(now); mlAdjustment > 0 {
        adjustedTokens *= mlAdjustment
    }
    
    return int64(math.Ceil(adjustedTokens))
}

// calculateReputationAdjustment calculates adjustment based on peer reputation
func (rl *RateLimiter) calculateReputationAdjustment() float64 {
    // In a real implementation, this would query the peer's reputation
    // For now, use a simple model based on violation history
    violationStreak := rl.stats.ViolationStreak.Load()
    
    if violationStreak == 0 {
        return 1.0 // No penalty for good behavior
    }
    
    // Exponential penalty for repeated violations
    penalty := math.Pow(rl.config.PenaltyFactor, float64(violationStreak))
    return math.Max(1.0, penalty)
}

// calculateTimeAdjustment calculates time-based rate adjustments
func (rl *RateLimiter) calculateTimeAdjustment(now time.Time) float64 {
    hour := float64(now.Hour())
    day := float64(now.Weekday())
    
    // Simple circadian model: higher rates during active hours
    // This is a simplified model - in production, use more sophisticated time series analysis
    hourAdjustment := 1.0 + 0.3*math.Sin((hour-12.0)*math.Pi/12.0)
    dayAdjustment := 1.0 + 0.1*math.Sin((day-3.0)*math.Pi/3.5)
    
    return hourAdjustment * dayAdjustment
}

// calculateLoadAdjustment calculates adjustment based on network load
func (rl *RateLimiter) calculateLoadAdjustment() float64 {
    // In a real implementation, this would query network-wide load metrics
    // For now, use pressure as a proxy for load
    return 1.0 + (rl.pressure * 0.3)
}

// predictOptimalAdjustment uses machine learning to predict optimal rate adjustment
func (rl *RateLimiter) predictOptimalAdjustment(now time.Time) float64 {
    rl.adaptiveModel.mu.RLock()
    defer rl.adaptiveModel.mu.RUnlock()
    
    // Update features
    rl.updateFeatures(now)
    
    // Simple linear prediction
    prediction := rl.adaptiveModel.bias
    for i, weight := range rl.adaptiveModel.weights {
        featureValue := rl.getFeatureValue(i)
        prediction += weight * featureValue
    }
    
    return math.Max(0.1, prediction) // Ensure positive adjustment
}

// updateFeatures updates the feature vector for machine learning
func (rl *RateLimiter) updateFeatures(now time.Time) {
    features := rl.adaptiveModel.features
    
    // Request rate feature
    features.RequestRate = rl.stats.CurrentRate.Load()
    
    // Token usage feature
    totalTokens := rl.stats.TotalTokens.Load()
    usedTokens := rl.stats.UsedTokens.Load()
    if totalTokens > 0 {
        features.TokenUsage = float64(usedTokens) / float64(totalTokens)
    }
    
    // Pressure feature
    features.Pressure = rl.pressure
    
    // Time features
    features.TimeOfDay = float64(now.Hour()) / 24.0
    features.DayOfWeek = float64(now.Weekday()) / 7.0
    
    // Network load (simplified - use pressure as proxy)
    features.NetworkLoad = rl.pressure
    
    // Peer reputation (simplified - use violation streak)
    violationStreak := rl.stats.ViolationStreak.Load()
    features.PeerReputation = 1.0 / (1.0 + float64(violationStreak))
}

// getFeatureValue gets the value of a specific feature
func (rl *RateLimiter) getFeatureValue(index int) float64 {
    features := rl.adaptiveModel.features
    
    switch index {
    case 0:
        return features.RequestRate
    case 1:
        return features.TokenUsage
    case 2:
        return features.Pressure
    case 3:
        return features.TimeOfDay
    case 4:
        return features.DayOfWeek
    case 5:
        return features.NetworkLoad
    case 6:
        return features.PeerReputation
    default:
        return 0.0
    }
}

// updateFlowDynamics updates flow rate and pressure based on current usage
func (rl *RateLimiter) updateFlowDynamics(tokens int64, now time.Time) {
    // Update current rate
    currentRate := rl.calculateCurrentRate(now)
    rl.stats.CurrentRate.Store(currentRate)
    rl.stats.RateHistory.Add(currentRate)
    
    // Update flow rate with damping to prevent oscillations
    targetFlow := rl.config.Rate * (1.0 - rl.pressure)
    flowDelta := targetFlow - rl.flowRate
    rl.flowRate += flowDelta * rl.config.FlowDamping
    
    // Update pressure based on token usage
    usageRatio := float64(tokens) / float64(rl.config.Burst)
    rl.pressure = rl.pressure*0.9 + usageRatio*0.1 // Exponential moving average
    
    rl.stats.PressureHistory.Add(rl.pressure)
    
    // Update resistance based on pressure
    rl.resistance = rl.config.ResistanceBase * (1.0 + rl.pressure*2.0)
}

// updatePressureDynamics updates pressure when requests are denied
func (rl *RateLimiter) updatePressureDynamics(deficit int64, now time.Time) {
    // Increase pressure when there are deficits
    deficitPressure := float64(deficit) / float64(rl.config.Burst)
    rl.pressure = math.Min(1.0, rl.pressure+deficitPressure)
    
    rl.stats.PressureHistory.Add(rl.pressure)
}

// calculateCurrentRate calculates the current request rate
func (rl *RateLimiter) calculateCurrentRate(now time.Time) float64 {
    rl.history.mu.Lock()
    defer rl.history.mu.Unlock()
    
    // Ensure current window is up to date
    rl.history.ensureCurrentWindow(now)
    
    windowDuration := now.Sub(rl.history.currentWindow.startTime).Seconds()
    if windowDuration > 0 {
        rate := float64(rl.history.currentWindow.requestCount) / windowDuration
        return rate
    }
    
    return 0.0
}

// recordViolation records a rate limit violation
func (rl *RateLimiter) recordViolation(now time.Time) {
    rl.violationCount.Add(1)
    rl.stats.LastViolation.Store(now.UnixNano())
    
    currentStreak := rl.stats.ViolationStreak.Load()
    rl.stats.ViolationStreak.Store(currentStreak + 1)
    
    // Add training sample for machine learning
    rl.addTrainingSample(now)
}

// addTrainingSample adds a training sample for adaptive learning
func (rl *RateLimiter) addTrainingSample(now time.Time) {
    rl.adaptiveModel.mu.Lock()
    defer rl.adaptiveModel.mu.Unlock()
    
    sample := &TrainingSample{
        Features:  rl.adaptiveModel.features,
        Target:    rl.calculateOptimalRate(),
        Weight:    1.0,
        Timestamp: now,
    }
    
    rl.adaptiveModel.trainingData = append(rl.adaptiveModel.trainingData, sample)
    
    // Limit training data size
    if len(rl.adaptiveModel.trainingData) > 1000 {
        rl.adaptiveModel.trainingData = rl.adaptiveModel.trainingData[1:]
    }
    
    // Retrain model periodically
    if len(rl.adaptiveModel.trainingData)%100 == 0 {
        go rl.retrainModel()
    }
}

// calculateOptimalRate calculates the optimal rate based on current conditions
func (rl *RateLimiter) calculateOptimalRate() float64 {
    // Simple heuristic: reduce rate under high pressure, increase under low pressure
    baseRate := rl.config.Rate
    pressureFactor := 1.0 - rl.pressure
    
    // Consider violation history
    violationFactor := 1.0 / (1.0 + float64(rl.stats.ViolationStreak.Load())*0.1)
    
    return baseRate * pressureFactor * violationFactor
}

// retrainModel retrains the machine learning model
func (rl *RateLimiter) retrainModel() {
    rl.adaptiveModel.mu.Lock()
    defer rl.adaptiveModel.mu.Unlock()
    
    if len(rl.adaptiveModel.trainingData) < 10 {
        return // Not enough data for training
    }
    
    // Simple gradient descent (in production, use more sophisticated ML)
    for epoch := 0; epoch < 10; epoch++ {
        for _, sample := range rl.adaptiveModel.trainingData {
            prediction := rl.adaptiveModel.bias
            for i, weight := range rl.adaptiveModel.weights {
                featureValue := rl.getFeatureValueFromSample(sample, i)
                prediction += weight * featureValue
            }
            
            error := sample.Target - prediction
            
            // Update weights
            rl.adaptiveModel.bias += rl.adaptiveModel.learningRate * error
            for i := range rl.adaptiveModel.weights {
                featureValue := rl.getFeatureValueFromSample(sample, i)
                rl.adaptiveModel.weights[i] += rl.adaptiveModel.learningRate * error * featureValue
            }
        }
    }
}

// getFeatureValueFromSample gets feature value from a training sample
func (rl *RateLimiter) getFeatureValueFromSample(sample *TrainingSample, index int) float64 {
    switch index {
    case 0:
        return sample.Features.RequestRate
    case 1:
        return sample.Features.TokenUsage
    case 2:
        return sample.Features.Pressure
    case 3:
        return sample.Features.TimeOfDay
    case 4:
        return sample.Features.DayOfWeek
    case 5:
        return sample.Features.NetworkLoad
    case 6:
        return sample.Features.PeerReputation
    default:
        return 0.0
    }
}

// isInGracePeriod checks if the peer is in the grace period
func (rl *RateLimiter) isInGracePeriod(now time.Time) bool {
    if !rl.config.StrictMode {
        return true
    }
    
    establishedTime := time.Unix(0, rl.stats.LastViolation.Load())
    if establishedTime.IsZero() {
        establishedTime = rl.lastUpdated
    }
    
    return now.Sub(establishedTime) < rl.config.GracePeriod
}

// ensureCurrentWindow ensures the current window is up to date
func (rh *RateLimitHistory) ensureCurrentWindow(now time.Time) {
    if now.After(rh.currentWindow.endTime) {
        // Move current window to history
        rh.windows = append(rh.windows, rh.currentWindow)
        
        // Maintain window count limit
        if len(rh.windows) > rh.maxWindows {
            rh.windows = rh.windows[1:]
        }
        
        // Create new current window
        rh.currentWindow = &RateWindow{
            startTime: now,
            endTime:   now.Add(rh.windowSize),
        }
    }
}

// GetStats returns current rate limiter statistics
func (rl *RateLimiter) GetStats() *RateLimitStats {
    rl.mu.RLock()
    defer rl.mu.RUnlock()
    
    // Update average rate
    totalRequests := rl.stats.TotalRequests.Load()
    if totalRequests > 0 {
        allowedRequests := rl.stats.AllowedRequests.Load()
        averageRate := float64(allowedRequests) / time.Since(rl.lastUpdated).Seconds()
        rl.stats.AverageRate.Store(averageRate)
    }
    
    // Update peak rate
    currentRate := rl.stats.CurrentRate.Load()
    peakRate := rl.stats.PeakRate.Load()
    if currentRate > peakRate {
        rl.stats.PeakRate.Store(currentRate)
    }
    
    return rl.stats
}

// Reset resets the rate limiter state
func (rl *RateLimiter) Reset() {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    now := time.Now()
    rl.tokens.Store(rl.config.Burst)
    rl.lastRefill.Store(now.UnixNano())
    rl.pressure = 0.0
    rl.flowRate = rl.config.Rate
    rl.resistance = rl.config.ResistanceBase
    rl.lastUpdated = now
    
    // Reset statistics
    rl.stats = NewRateLimitStats()
    rl.violationCount.Store(0)
}

// UpdateConfig updates the rate limiter configuration
func (rl *RateLimiter) UpdateConfig(newConfig *RateLimitConfig) {
    rl.mu.Lock()
    defer rl.mu.Unlock()
    
    rl.config = newConfig
    rl.learningRate = newConfig.LearningRate
    rl.adaptiveModel.learningRate = newConfig.LearningRate
    
    // Reset tokens if burst capacity changed
    currentTokens := rl.tokens.Load()
    if currentTokens > newConfig.Burst {
        rl.tokens.Store(newConfig.Burst)
    }
}

// GetPressure returns the current system pressure
func (rl *RateLimiter) GetPressure() float64 {
    rl.mu.RLock()
    defer rl.mu.RUnlock()
    return rl.pressure
}

// GetFlowRate returns the current flow rate
func (rl *RateLimiter) GetFlowRate() float64 {
    rl.mu.RLock()
    defer rl.mu.RUnlock()
    return rl.flowRate
}

// GetResistance returns the current system resistance
func (rl *RateLimiter) GetResistance() float64 {
    rl.mu.RLock()
    defer rl.mu.RUnlock()
    return rl.resistance
}

// String returns a string representation of the rate limiter
func (rl *RateLimiter) String() string {
    stats := rl.GetStats()
    return fmt.Sprintf("RateLimiter[Peer:%s, Pressure:%.3f, Flow:%.1f/s, Allowed:%d/%d]", 
        rl.peerID[:8], rl.pressure, rl.flowRate, 
        stats.AllowedRequests.Load(), stats.TotalRequests.Load())
}