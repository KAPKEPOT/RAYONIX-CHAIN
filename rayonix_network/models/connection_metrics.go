package models

import (
    "encoding/json"
    "fmt"
    "math"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/utils"
)

// ConnectionMetrics provides comprehensive performance tracking for network connections
type ConnectionMetrics struct {
    // Connection identification
    ConnectionID  string    `json:"connection_id" msgpack:"connection_id"`
    PeerID        string    `json:"peer_id" msgpack:"peer_id"`
    Protocol      string    `json:"protocol" msgpack:"protocol"`
    
    // Timing and duration
    EstablishedAt time.Time `json:"established_at" msgpack:"established_at"`
    LastActivity  time.Time `json:"last_activity" msgpack:"last_activity"`
    Uptime        time.Duration `json:"uptime" msgpack:"uptime"`
    
    // Atomic counters for high-frequency updates
    bytesSent        uint64
    bytesReceived    uint64
    messagesSent     uint64
    messagesReceived uint64
    errorsEncountered uint64
    retriesAttempted uint64
    
    // Latency tracking with rolling statistics
    latencyStats    *utils.RollingStatistics
    rttStats        *utils.RollingStatistics
    
    // Message size statistics
    messageSizeStats *utils.RollingStatistics
    
    // Success/failure rates
    successCount    uint64
    failureCount    uint64
    timeoutCount    uint64
    
    // Bandwidth calculations
    bandwidthStats  *BandwidthCalculator
    
    // Connection quality metrics
    qualityScore    float64
    stabilityScore  float64
    reliabilityScore float64
    
    // Physics-inspired metrics
    potentialEnergy  float64
    kineticEnergy    float64
    entropyContribution float64
    forceMagnitude   float64
    
    // Resource utilization
    cpuUsage        float64
    memoryUsage     uint64
    bufferUsage     uint64
    
    // Protocol-specific metrics
    tcpMetrics      *TCPMetrics
    udpMetrics      *UDPMetrics
    wsMetrics       *WebSocketMetrics
    
    // Security metrics
    securityMetrics *SecurityMetrics
    
    // Synchronization
    mu              sync.RWMutex
    lastUpdated     time.Time
    version         uint64
}

// BandwidthCalculator tracks bandwidth usage over time windows
type BandwidthCalculator struct {
    windows        []*BandwidthWindow
    currentWindow  *BandwidthWindow
    windowSize     time.Duration
    mu             sync.RWMutex
}

// BandwidthWindow represents bandwidth usage in a time window
type BandwidthWindow struct {
    startTime     time.Time
    endTime       time.Time
    bytesSent     uint64
    bytesReceived uint64
    messageCount  uint64
}

// TCPMetrics tracks TCP-specific connection metrics
type TCPMetrics struct {
    segmentsSent     uint64
    segmentsReceived uint64
    retransmissions  uint64
    congestionWindow uint32
    rttVariance      float64
    ssthresh         uint32
    state            string
}

// UDPMetrics tracks UDP-specific connection metrics
type UDPMetrics struct {
    datagramsSent    uint64
    datagramsReceived uint64
    datagramsLost    uint64
    jitter           time.Duration
    reorderBuffer    uint32
}

// WebSocketMetrics tracks WebSocket-specific metrics
type WebSocketMetrics struct {
    framesSent       uint64
    framesReceived   uint64
    pingCount        uint64
    pongCount        uint64
    closeCount       uint64
    compressionRatio float64
}

// SecurityMetrics tracks security-related metrics
type SecurityMetrics struct {
    handshakesCompleted uint64
    handshakesFailed    uint64
    authFailures        uint64
    encryptionFailures  uint64
    decryptionFailures  uint64
    replayAttempts      uint64
    signatureFailures   uint64
}

// NewConnectionMetrics creates a new ConnectionMetrics instance
func NewConnectionMetrics(connectionID, peerID, protocol string) *ConnectionMetrics {
    now := time.Now()
    
    return &ConnectionMetrics{
        ConnectionID:   connectionID,
        PeerID:         peerID,
        Protocol:       protocol,
        EstablishedAt:  now,
        LastActivity:   now,
        Uptime:         0,
        latencyStats:   utils.NewRollingStatistics(1000),
        rttStats:       utils.NewRollingStatistics(1000),
        messageSizeStats: utils.NewRollingStatistics(5000),
        bandwidthStats: NewBandwidthCalculator(time.Minute, 10),
        qualityScore:   1.0,
        stabilityScore: 1.0,
        reliabilityScore: 1.0,
        potentialEnergy: 1.0,
        kineticEnergy:  0.0,
        entropyContribution: 0.0,
        forceMagnitude: 0.0,
        cpuUsage:       0.0,
        memoryUsage:    0,
        bufferUsage:    0,
        tcpMetrics:     &TCPMetrics{},
        udpMetrics:     &UDPMetrics{},
        wsMetrics:      &WebSocketMetrics{},
        securityMetrics: &SecurityMetrics{},
        lastUpdated:    now,
        version:        1,
    }
}

// NewBandwidthCalculator creates a new bandwidth calculator
func NewBandwidthCalculator(windowSize time.Duration, windowCount int) *BandwidthCalculator {
    calc := &BandwidthCalculator{
        windows:    make([]*BandwidthWindow, 0, windowCount),
        windowSize: windowSize,
    }
    
    calc.currentWindow = &BandwidthWindow{
        startTime: time.Now(),
        endTime:   time.Now().Add(windowSize),
    }
    
    return calc
}

// RecordMessageSent records metrics for a sent message
func (cm *ConnectionMetrics) RecordMessageSent(messageSize uint64, latency time.Duration) {
    atomic.AddUint64(&cm.messagesSent, 1)
    atomic.AddUint64(&cm.bytesSent, messageSize)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.LastActivity = time.Now()
    cm.latencyStats.Add(latency.Seconds())
    cm.messageSizeStats.Add(float64(messageSize))
    cm.bandwidthStats.RecordSent(messageSize)
    
    cm.updateUptime()
    cm.updateQualityMetrics()
    cm.updateVersion()
}

// RecordMessageReceived records metrics for a received message
func (cm *ConnectionMetrics) RecordMessageReceived(messageSize uint64) {
    atomic.AddUint64(&cm.messagesReceived, 1)
    atomic.AddUint64(&cm.bytesReceived, messageSize)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.LastActivity = time.Now()
    cm.messageSizeStats.Add(float64(messageSize))
    cm.bandwidthStats.RecordReceived(messageSize)
    
    cm.updateUptime()
    cm.updateVersion()
}

// RecordSuccess records a successful operation
func (cm *ConnectionMetrics) RecordSuccess() {
    atomic.AddUint64(&cm.successCount, 1)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.updateQualityMetrics()
    cm.updateVersion()
}

// RecordFailure records a failed operation
func (cm *ConnectionMetrics) RecordFailure() {
    atomic.AddUint64(&cm.failureCount, 1)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.updateQualityMetrics()
    cm.updateVersion()
}

// RecordError records an error occurrence
func (cm *ConnectionMetrics) RecordError() {
    atomic.AddUint64(&cm.errorsEncountered, 1)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.updateQualityMetrics()
    cm.updateVersion()
}

// RecordRetry records a retry attempt
func (cm *ConnectionMetrics) RecordRetry() {
    atomic.AddUint64(&cm.retriesAttempted, 1)
    
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.updateVersion()
}

// RecordLatency records a new latency measurement
func (cm *ConnectionMetrics) RecordLatency(latency time.Duration) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.latencyStats.Add(latency.Seconds())
    cm.LastActivity = time.Now()
    
    cm.updateQualityMetrics()
    cm.updateVersion()
}

// RecordRTT records a round-trip time measurement
func (cm *ConnectionMetrics) RecordRTT(rtt time.Duration) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.rttStats.Add(rtt.Seconds())
    cm.updateVersion()
}

// UpdatePhysicsMetrics updates physics-inspired metrics
func (cm *ConnectionMetrics) UpdatePhysicsMetrics(potential, kinetic, entropy, force float64) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.potentialEnergy = potential
    cm.kineticEnergy = kinetic
    cm.entropyContribution = entropy
    cm.forceMagnitude = force
    
    cm.updateVersion()
}

// UpdateResourceUsage updates resource utilization metrics
func (cm *ConnectionMetrics) UpdateResourceUsage(cpu float64, memory, buffer uint64) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    cm.cpuUsage = cpu
    cm.memoryUsage = memory
    cm.bufferUsage = buffer
    
    cm.updateVersion()
}

// GetMetricsSnapshot returns a consistent snapshot of current metrics
func (cm *ConnectionMetrics) GetMetricsSnapshot() *ConnectionMetricsSnapshot {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    
    snapshot := &ConnectionMetricsSnapshot{
        ConnectionID:        cm.ConnectionID,
        PeerID:              cm.PeerID,
        Protocol:            cm.Protocol,
        EstablishedAt:       cm.EstablishedAt,
        LastActivity:        cm.LastActivity,
        Uptime:              cm.Uptime,
        BytesSent:           atomic.LoadUint64(&cm.bytesSent),
        BytesReceived:       atomic.LoadUint64(&cm.bytesReceived),
        MessagesSent:        atomic.LoadUint64(&cm.messagesSent),
        MessagesReceived:    atomic.LoadUint64(&cm.messagesReceived),
        ErrorsEncountered:   atomic.LoadUint64(&cm.errorsEncountered),
        RetriesAttempted:    atomic.LoadUint64(&cm.retriesAttempted),
        SuccessCount:        atomic.LoadUint64(&cm.successCount),
        FailureCount:        atomic.LoadUint64(&cm.failureCount),
        TimeoutCount:        atomic.LoadUint64(&cm.timeoutCount),
        QualityScore:        cm.qualityScore,
        StabilityScore:      cm.stabilityScore,
        ReliabilityScore:    cm.reliabilityScore,
        PotentialEnergy:     cm.potentialEnergy,
        KineticEnergy:       cm.kineticEnergy,
        EntropyContribution: cm.entropyContribution,
        ForceMagnitude:      cm.forceMagnitude,
        CPUUsage:            cm.cpuUsage,
        MemoryUsage:         cm.memoryUsage,
        BufferUsage:         cm.bufferUsage,
        Timestamp:           time.Now(),
        Version:             cm.version,
    }
    
    // Copy statistics data
    if cm.latencyStats != nil {
        snapshot.AverageLatency = time.Duration(cm.latencyStats.Mean() * float64(time.Second))
        snapshot.LatencyVariance = cm.latencyStats.Variance()
        snapshot.MinLatency = time.Duration(cm.latencyStats.Min() * float64(time.Second))
        snapshot.MaxLatency = time.Duration(cm.latencyStats.Max() * float64(time.Second))
        snapshot.LatencyPercentiles = cm.latencyStats.Percentiles([]float64{0.5, 0.95, 0.99})
    }
    
    if cm.rttStats != nil {
        snapshot.AverageRTT = time.Duration(cm.rttStats.Mean() * float64(time.Second))
        snapshot.RTTVariance = cm.rttStats.Variance()
    }
    
    if cm.messageSizeStats != nil {
        snapshot.AverageMessageSize = cm.messageSizeStats.Mean()
        snapshot.MessageSizeVariance = cm.messageSizeStats.Variance()
    }
    
    // Calculate bandwidth
    if cm.bandwidthStats != nil {
        snapshot.BytesPerSecondSent, snapshot.BytesPerSecondReceived = cm.bandwidthStats.GetCurrentBandwidth()
    }
    
    // Calculate rates
    totalOperations := snapshot.SuccessCount + snapshot.FailureCount
    if totalOperations > 0 {
        snapshot.SuccessRate = float64(snapshot.SuccessCount) / float64(totalOperations)
    }
    
    totalMessages := snapshot.MessagesSent + snapshot.MessagesReceived
    if cm.Uptime > 0 {
        snapshot.MessagesPerSecond = float64(totalMessages) / cm.Uptime.Seconds()
    }
    
    return snapshot
}

// ConnectionMetricsSnapshot provides a thread-safe snapshot of metrics
type ConnectionMetricsSnapshot struct {
    ConnectionID        string        `json:"connection_id"`
    PeerID              string        `json:"peer_id"`
    Protocol            string        `json:"protocol"`
    EstablishedAt       time.Time     `json:"established_at"`
    LastActivity        time.Time     `json:"last_activity"`
    Uptime              time.Duration `json:"uptime"`
    
    // Counters
    BytesSent           uint64        `json:"bytes_sent"`
    BytesReceived       uint64        `json:"bytes_received"`
    MessagesSent        uint64        `json:"messages_sent"`
    MessagesReceived    uint64        `json:"messages_received"`
    ErrorsEncountered   uint64        `json:"errors_encountered"`
    RetriesAttempted    uint64        `json:"retries_attempted"`
    SuccessCount        uint64        `json:"success_count"`
    FailureCount        uint64        `json:"failure_count"`
    TimeoutCount        uint64        `json:"timeout_count"`
    
    // Quality metrics
    QualityScore        float64       `json:"quality_score"`
    StabilityScore      float64       `json:"stability_score"`
    ReliabilityScore    float64       `json:"reliability_score"`
    
    // Physics metrics
    PotentialEnergy     float64       `json:"potential_energy"`
    KineticEnergy       float64       `json:"kinetic_energy"`
    EntropyContribution float64       `json:"entropy_contribution"`
    ForceMagnitude      float64       `json:"force_magnitude"`
    
    // Performance statistics
    AverageLatency      time.Duration `json:"average_latency"`
    LatencyVariance     float64       `json:"latency_variance"`
    MinLatency          time.Duration `json:"min_latency"`
    MaxLatency          time.Duration `json:"max_latency"`
    LatencyPercentiles  []float64     `json:"latency_percentiles"`
    
    AverageRTT          time.Duration `json:"average_rtt"`
    RTTVariance         float64       `json:"rtt_variance"`
    
    AverageMessageSize  float64       `json:"average_message_size"`
    MessageSizeVariance float64       `json:"message_size_variance"`
    
    // Bandwidth
    BytesPerSecondSent  float64       `json:"bytes_per_second_sent"`
    BytesPerSecondReceived float64   `json:"bytes_per_second_received"`
    
    // Rates
    SuccessRate         float64       `json:"success_rate"`
    MessagesPerSecond   float64       `json:"messages_per_second"`
    
    // Resource usage
    CPUUsage            float64       `json:"cpu_usage"`
    MemoryUsage         uint64        `json:"memory_usage"`
    BufferUsage         uint64        `json:"buffer_usage"`
    
    // Metadata
    Timestamp           time.Time     `json:"timestamp"`
    Version             uint64        `json:"version"`
}

// CalculateQualityScore computes comprehensive connection quality score
func (cm *ConnectionMetrics) CalculateQualityScore() float64 {
    snapshot := cm.GetMetricsSnapshot()
    
    // Base quality factors
    latencyFactor := 1.0
    if snapshot.AverageLatency > 0 {
        latencyFactor = 1.0 / (1.0 + snapshot.AverageLatency.Seconds())
    }
    
    successFactor := snapshot.SuccessRate
    stabilityFactor := cm.calculateStabilityFactor(snapshot)
    
    // Bandwidth efficiency
    bandwidthEfficiency := 1.0
    if snapshot.BytesPerSecondSent > 0 && snapshot.BytesPerSecondReceived > 0 {
        totalBytes := snapshot.BytesPerSecondSent + snapshot.BytesPerSecondReceived
        if totalBytes > 1000000 { // 1 MB/s threshold
            bandwidthEfficiency = 0.8
        }
    }
    
    // Physics factors
    physicsFactor := (snapshot.PotentialEnergy + (1.0 - snapshot.EntropyContribution)) / 2.0
    
    // Weighted combination
    quality := (latencyFactor * 0.25) +
              (successFactor * 0.20) +
              (stabilityFactor * 0.15) +
              (bandwidthEfficiency * 0.15) +
              (physicsFactor * 0.25)
    
    return utils.Clamp(quality, 0.0, 1.0)
}

// calculateStabilityFactor computes connection stability based on variance and error rates
func (cm *ConnectionMetrics) calculateStabilityFactor(snapshot *ConnectionMetricsSnapshot) float64 {
    // Low latency variance indicates stability
    latencyStability := 1.0
    if snapshot.LatencyVariance > 0 {
        latencyStability = 1.0 / (1.0 + math.Sqrt(snapshot.LatencyVariance))
    }
    
    // Low error rate indicates stability
    errorStability := 1.0
    totalOperations := snapshot.SuccessCount + snapshot.FailureCount + snapshot.ErrorsEncountered
    if totalOperations > 0 {
        errorRate := float64(snapshot.FailureCount+snapshot.ErrorsEncountered) / float64(totalOperations)
        errorStability = 1.0 - errorRate
    }
    
    // Uptime contributes to stability
    uptimeStability := math.Min(snapshot.Uptime.Hours()/24.0, 1.0) // Normalize to 24 hours
    
    return (latencyStability * 0.4) + (errorStability * 0.4) + (uptimeStability * 0.2)
}

// updateUptime updates the connection uptime
func (cm *ConnectionMetrics) updateUptime() {
    cm.Uptime = time.Since(cm.EstablishedAt)
}

// updateQualityMetrics recalculates quality scores
func (cm *ConnectionMetrics) updateQualityMetrics() {
    cm.qualityScore = cm.CalculateQualityScore()
    cm.stabilityScore = cm.calculateStabilityFactor(cm.GetMetricsSnapshot())
    cm.reliabilityScore = cm.calculateReliabilityScore()
}

// calculateReliabilityScore computes connection reliability
func (cm *ConnectionMetrics) calculateReliabilityScore() float64 {
    snapshot := cm.GetMetricsSnapshot()
    
    successRate := snapshot.SuccessRate
    errorRate := float64(snapshot.ErrorsEncountered) / float64(snapshot.MessagesSent+snapshot.MessagesReceived+1)
    
    // Recent activity bonus
    activityBonus := 0.0
    if time.Since(snapshot.LastActivity) < 5*time.Minute {
        activityBonus = 0.1
    }
    
    reliability := (successRate * 0.7) + ((1.0 - errorRate) * 0.2) + activityBonus
    
    return utils.Clamp(reliability, 0.0, 1.0)
}

// GetBandwidthUsage returns current bandwidth usage
func (cm *ConnectionMetrics) GetBandwidthUsage() (sent, received float64) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    
    if cm.bandwidthStats != nil {
        return cm.bandwidthStats.GetCurrentBandwidth()
    }
    return 0, 0
}

// GetLatencyStats returns latency statistics
func (cm *ConnectionMetrics) GetLatencyStats() (avg, min, max time.Duration, variance float64) {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    
    if cm.latencyStats != nil {
        avg = time.Duration(cm.latencyStats.Mean() * float64(time.Second))
        min = time.Duration(cm.latencyStats.Min() * float64(time.Second))
        max = time.Duration(cm.latencyStats.Max() * float64(time.Second))
        variance = cm.latencyStats.Variance()
    }
    return
}

// Reset resets all metrics (useful for connection recycling)
func (cm *ConnectionMetrics) Reset() {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    
    // Reset atomic counters
    atomic.StoreUint64(&cm.bytesSent, 0)
    atomic.StoreUint64(&cm.bytesReceived, 0)
    atomic.StoreUint64(&cm.messagesSent, 0)
    atomic.StoreUint64(&cm.messagesReceived, 0)
    atomic.StoreUint64(&cm.errorsEncountered, 0)
    atomic.StoreUint64(&cm.retriesAttempted, 0)
    atomic.StoreUint64(&cm.successCount, 0)
    atomic.StoreUint64(&cm.failureCount, 0)
    atomic.StoreUint64(&cm.timeoutCount, 0)
    
    // Reset statistics
    if cm.latencyStats != nil {
        cm.latencyStats.Reset()
    }
    if cm.rttStats != nil {
        cm.rttStats.Reset()
    }
    if cm.messageSizeStats != nil {
        cm.messageSizeStats.Reset()
    }
    if cm.bandwidthStats != nil {
        cm.bandwidthStats.Reset()
    }
    
    // Reset quality scores
    cm.qualityScore = 1.0
    cm.stabilityScore = 1.0
    cm.reliabilityScore = 1.0
    
    // Reset physics metrics
    cm.potentialEnergy = 1.0
    cm.kineticEnergy = 0.0
    cm.entropyContribution = 0.0
    cm.forceMagnitude = 0.0
    
    // Update timestamps
    cm.EstablishedAt = time.Now()
    cm.LastActivity = time.Now()
    cm.Uptime = 0
    cm.lastUpdated = time.Now()
    cm.version++
}

// BandwidthCalculator methods

// RecordSent records sent bytes in current window
func (bc *BandwidthCalculator) RecordSent(bytes uint64) {
    bc.mu.Lock()
    defer bc.mu.Unlock()
    
    now := time.Now()
    bc.ensureCurrentWindow(now)
    bc.currentWindow.bytesSent += bytes
    bc.currentWindow.messageCount++
}

// RecordReceived records received bytes in current window
func (bc *BandwidthCalculator) RecordReceived(bytes uint64) {
    bc.mu.Lock()
    defer bc.mu.Unlock()
    
    now := time.Now()
    bc.ensureCurrentWindow(now)
    bc.currentWindow.bytesReceived += bytes
    bc.currentWindow.messageCount++
}

// ensureCurrentWindow ensures we have a valid current window
func (bc *BandwidthCalculator) ensureCurrentWindow(now time.Time) {
    if now.After(bc.currentWindow.endTime) {
        // Move current window to history and create new window
        bc.windows = append(bc.windows, bc.currentWindow)
        
        // Maintain window count limit
        if len(bc.windows) > cap(bc.windows) {
            bc.windows = bc.windows[1:]
        }
        
        bc.currentWindow = &BandwidthWindow{
            startTime: now,
            endTime:   now.Add(bc.windowSize),
        }
    }
}

// GetCurrentBandwidth returns current bandwidth in bytes per second
func (bc *BandwidthCalculator) GetCurrentBandwidth() (sent, received float64) {
    bc.mu.RLock()
    defer bc.mu.RUnlock()
    
    now := time.Now()
    windowDuration := now.Sub(bc.currentWindow.startTime).Seconds()
    
    if windowDuration > 0 {
        sent = float64(bc.currentWindow.bytesSent) / windowDuration
        received = float64(bc.currentWindow.bytesReceived) / windowDuration
    }
    
    return
}

// GetAverageBandwidth returns average bandwidth over all windows
func (bc *BandwidthCalculator) GetAverageBandwidth() (sent, received float64) {
    bc.mu.RLock()
    defer bc.mu.RUnlock()
    
    if len(bc.windows) == 0 {
        return bc.GetCurrentBandwidth()
    }
    
    var totalSent, totalReceived uint64
    var totalDuration float64
    
    for _, window := range bc.windows {
        windowDuration := window.endTime.Sub(window.startTime).Seconds()
        totalSent += window.bytesSent
        totalReceived += window.bytesReceived
        totalDuration += windowDuration
    }
    
    // Include current window
    now := time.Now()
    currentDuration := now.Sub(bc.currentWindow.startTime).Seconds()
    totalSent += bc.currentWindow.bytesSent
    totalReceived += bc.currentWindow.bytesReceived
    totalDuration += currentDuration
    
    if totalDuration > 0 {
        sent = float64(totalSent) / totalDuration
        received = float64(totalReceived) / totalDuration
    }
    
    return
}

// Reset resets all bandwidth statistics
func (bc *BandwidthCalculator) Reset() {
    bc.mu.Lock()
    defer bc.mu.Unlock()
    
    bc.windows = make([]*BandwidthWindow, 0, cap(bc.windows))
    bc.currentWindow = &BandwidthWindow{
        startTime: time.Now(),
        endTime:   time.Now().Add(bc.windowSize),
    }
}

// updateVersion increments the metrics version
func (cm *ConnectionMetrics) updateVersion() {
    cm.version++
    cm.lastUpdated = time.Now()
}

// String returns a string representation of the metrics
func (cm *ConnectionMetrics) String() string {
    snapshot := cm.GetMetricsSnapshot()
    return fmt.Sprintf("Metrics[Conn:%s, Peer:%s, Quality:%.3f, Latency:%v, BW:%.1f/%.1f KB/s]", 
        cm.ConnectionID[:8], cm.PeerID[:8], snapshot.QualityScore, snapshot.AverageLatency,
        snapshot.BytesPerSecondSent/1024, snapshot.BytesPerSecondReceived/1024)
}