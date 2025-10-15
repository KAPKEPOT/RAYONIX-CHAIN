package utils

import (
    "encoding/json"
    "fmt"
    "math"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/models"
)

// MetricsCollector implements a comprehensive, physics-inspired metrics collection and analysis system
type MetricsCollector struct {
    // Configuration
    config        *MetricsConfig
    
    // Core metrics storage
    globalMetrics *GlobalMetrics
    peerMetrics   *ConcurrentMap[string, *PeerMetrics]
    connectionMetrics *ConcurrentMap[string, *ConnectionMetrics]
    messageMetrics *ConcurrentMap[string, *MessageMetrics]
    
    // Physics-inspired metrics
    networkPhysics *NetworkPhysicsModel
    fieldMetrics   *FieldMetrics
    
    // Time series data
    timeSeries    *TimeSeriesDatabase
    rollingStats  *RollingStatisticsManager
    
    // Alerting and monitoring
    alertManager  *AlertManager
    thresholdManager *ThresholdManager
    
    // Export and reporting
    exporters     []MetricsExporter
    reportGenerator *ReportGenerator
    
    // Resource management
    resourceMonitor *ResourceMonitor
    
    // Synchronization
    mu            sync.RWMutex
    startedAt     time.Time
    lastExport    time.Time
    version       uint64
}

// MetricsConfig defines metrics collection configuration
type MetricsConfig struct {
    // Collection intervals
    CollectionInterval time.Duration `json:"collection_interval"`
    RetentionPeriod    time.Duration `json:"retention_period"`
    RollingWindowSize  int           `json:"rolling_window_size"`
    
    // Storage configuration
    MaxTimeSeriesPoints int          `json:"max_time_series_points"`
    MaxPeerMetrics      int          `json:"max_peer_metrics"`
    MaxConnectionMetrics int         `json:"max_connection_metrics"`
    
    // Physics model parameters
    PhysicsUpdateInterval time.Duration `json:"physics_update_interval"`
    FieldResolution     float64       `json:"field_resolution"`
    EntropyCalculationInterval time.Duration `json:"entropy_calculation_interval"`
    
    // Alerting configuration
    EnableAlerts       bool          `json:"enable_alerts"`
    AlertCooldown      time.Duration `json:"alert_cooldown"`
    CriticalThresholds *ThresholdSet `json:"critical_thresholds"`
    
    // Export configuration
    ExportInterval     time.Duration `json:"export_interval"`
    EnablePrometheus   bool          `json:"enable_prometheus"`
    EnableJSONExport   bool          `json:"enable_json_export"`
    EnableInfluxDB     bool          `json:"enable_influxdb"`
}

// GlobalMetrics tracks network-wide metrics
type GlobalMetrics struct {
    // Network topology
    TotalPeers        atomic.Uint64   `json:"total_peers"`
    ActivePeers       atomic.Uint64   `json:"active_peers"`
    ConnectedPeers    atomic.Uint64   `json:"connected_peers"`
    BannedPeers       atomic.Uint64   `json:"banned_peers"`
    
    // Message statistics
    TotalMessages     atomic.Uint64   `json:"total_messages"`
    MessagesPerSecond atomic.Float64  `json:"messages_per_second"`
    AverageMessageSize atomic.Float64 `json:"average_message_size"`
    
    // Bandwidth usage
    BytesSent         atomic.Uint64   `json:"bytes_sent"`
    BytesReceived     atomic.Uint64   `json:"bytes_received"`
    BytesPerSecondSent atomic.Float64 `json:"bytes_per_second_sent"`
    BytesPerSecondReceived atomic.Float64 `json:"bytes_per_second_received"`
    
    // Performance metrics
    AverageLatency    atomic.Float64  `json:"average_latency"` // in seconds
    SuccessRate       atomic.Float64  `json:"success_rate"`
    ErrorRate         atomic.Float64  `json:"error_rate"`
    
    // Resource usage
    MemoryUsage       atomic.Uint64   `json:"memory_usage"`    // in bytes
    CPUUsage          atomic.Float64  `json:"cpu_usage"`       // percentage
    GoroutineCount    atomic.Uint32   `json:"goroutine_count"`
    
    // Physics-inspired metrics
    NetworkEntropy    atomic.Float64  `json:"network_entropy"`
    AveragePotential  atomic.Float64  `json:"average_potential"`
    KineticEnergy     atomic.Float64  `json:"kinetic_energy"`
    FieldStrength     atomic.Float64  `json:"field_strength"`
    
    // Consensus integration
    ValidatorCount    atomic.Uint32   `json:"validator_count"`
    AverageStake      atomic.Float64  `json:"average_stake"`
    ConsensusParticipation atomic.Float64 `json:"consensus_participation"`
    
    // Timestamps
    LastUpdate        atomic.Int64    `json:"last_update"` // Unix nanoseconds
    CollectionTime    atomic.Float64  `json:"collection_time"` // Time to collect metrics
    
    mu                sync.RWMutex    `json:"-"`
}

// PeerMetrics tracks per-peer metrics
type PeerMetrics struct {
    PeerID            string                 `json:"peer_id"`
    Address           string                 `json:"address"`
    
    // Connection statistics
    ConnectionCount   atomic.Uint32          `json:"connection_count"`
    Uptime            atomic.Int64           `json:"uptime"` // nanoseconds
    LastSeen          atomic.Int64           `json:"last_seen"`
    
    // Message statistics
    MessagesSent      atomic.Uint64          `json:"messages_sent"`
    MessagesReceived  atomic.Uint64          `json:"messages_received"`
    MessageSuccessRate atomic.Float64        `json:"message_success_rate"`
    
    // Performance metrics
    AverageLatency    atomic.Float64         `json:"average_latency"`
    ResponseTime      atomic.Float64         `json:"response_time"`
    BandwidthUsage    atomic.Float64         `json:"bandwidth_usage"`
    
    // Reputation and quality
    Reputation        atomic.Int32           `json:"reputation"`
    QualityScore      atomic.Float64         `json:"quality_score"`
    StabilityScore    atomic.Float64         `json:"stability_score"`
    
    // Physics properties
    PotentialEnergy   atomic.Float64         `json:"potential_energy"`
    EntropyContribution atomic.Float64       `json:"entropy_contribution"`
    ForceMagnitude    atomic.Float64         `json:"force_magnitude"`
    
    // Resource usage
    MemoryFootprint   atomic.Uint64          `json:"memory_footprint"`
    CPUContribution   atomic.Float64         `json:"cpu_contribution"`
    
    // Time series data
    LatencyHistory    *RollingStatistics     `json:"-"`
    MessageRateHistory *RollingStatistics    `json:"-"`
    QualityHistory    *RollingStatistics     `json:"-"`
    
    // Metadata
    FirstSeen         time.Time              `json:"first_seen"`
    LastUpdated       atomic.Int64           `json:"last_updated"`
    Version           atomic.Uint64          `json:"version"`
    
    mu                sync.RWMutex           `json:"-"`
}

// ConnectionMetrics tracks per-connection metrics (detailed version)
type ConnectionMetrics struct {
    ConnectionID      string                 `json:"connection_id"`
    PeerID            string                 `json:"peer_id"`
    Protocol          config.ProtocolType    `json:"protocol"`
    
    // Timing information
    EstablishedAt     time.Time              `json:"established_at"`
    LastActivity      atomic.Int64           `json:"last_activity"`
    Uptime            atomic.Int64           `json:"uptime"`
    
    // Traffic statistics
    BytesSent         atomic.Uint64          `json:"bytes_sent"`
    BytesReceived     atomic.Uint64          `json:"bytes_received"`
    MessagesSent      atomic.Uint64          `json:"messages_sent"`
    MessagesReceived  atomic.Uint64          `json:"messages_received"`
    
    // Performance metrics
    CurrentLatency    atomic.Float64         `json:"current_latency"`
    AverageLatency    atomic.Float64         `json:"average_latency"`
    MinLatency        atomic.Float64         `json:"min_latency"`
    MaxLatency        atomic.Float64         `json:"max_latency"`
    Jitter            atomic.Float64         `json:"jitter"`
    
    // Quality metrics
    SuccessRate       atomic.Float64         `json:"success_rate"`
    ErrorCount        atomic.Uint64          `json:"error_count"`
    RetryCount        atomic.Uint64          `json:"retry_count"`
    TimeoutCount      atomic.Uint64          `json:"timeout_count"`
    
    // Physics properties
    ConnectionQuality atomic.Float64         `json:"connection_quality"`
    SignalStrength    atomic.Float64         `json:"signal_strength"`
    NoiseLevel        atomic.Float64         `json:"noise_level"`
    
    // Protocol-specific metrics
    TCPMetrics        *TCPConnectionMetrics  `json:"tcp_metrics,omitempty"`
    UDPMetrics        *UDPConnectionMetrics  `json:"udp_metrics,omitempty"`
    WSMetrics         *WSConnectionMetrics   `json:"ws_metrics,omitempty"`
    
    // Resource usage
    BufferUsage       atomic.Uint64          `json:"buffer_usage"`
    BandwidthLimit    atomic.Uint64          `json:"bandwidth_limit"`
    
    // History
    LatencyHistory    *RollingStatistics     `json:"-"`
    TrafficHistory    *TrafficHistory        `json:"-"`
    
    mu                sync.RWMutex           `json:"-"`
}

// MessageMetrics tracks message-level metrics
type MessageMetrics struct {
    MessageID         string                 `json:"message_id"`
    MessageType       config.MessageType     `json:"message_type"`
    SourcePeer        string                 `json:"source_peer"`
    
    // Delivery tracking
    CreatedAt         time.Time              `json:"created_at"`
    DeliveredAt       atomic.Int64           `json:"delivered_at"`
    DeliveryTime      atomic.Float64         `json:"delivery_time"`
    HopCount          atomic.Uint32          `json:"hop_count"`
    
    // Routing metrics
    RoutingPath       []string               `json:"routing_path"`
    PathEfficiency    atomic.Float64         `json:"path_efficiency"`
    RoutingCost       atomic.Float64         `json:"routing_cost"`
    
    // Physics routing properties
    FieldStrength     atomic.Float64         `json:"field_strength"`
    PotentialUsed     atomic.Float64         `json:"potential_used"`
    EntropyImpact     atomic.Float64         `json:"entropy_impact"`
    
    // Reliability metrics
    DeliveryStatus    config.DeliveryStatus  `json:"delivery_status"`
    RetryAttempts     atomic.Uint32          `json:"retry_attempts"`
    AckReceived       atomic.Bool            `json:"ack_received"`
    
    // Size information
    PayloadSize       uint32                 `json:"payload_size"`
    TotalSize         uint32                 `json:"total_size"`
    CompressionRatio  atomic.Float64         `json:"compression_ratio"`
    
    mu                sync.RWMutex           `json:"-"`
}

// NetworkPhysicsModel implements physics-inspired network analysis
type NetworkPhysicsModel struct {
    // Field properties
    FieldTensor       *FieldTensor           `json:"field_tensor"`
    PotentialField    *PotentialMap          `json:"potential_field"`
    EntropyField      *EntropyMap            `json:"entropy_field"`
    
    // Force calculations
    AttractionForces  *ForceMatrix           `json:"attraction_forces"`
    RepulsionForces   *ForceMatrix           `json:"repulsion_forces"`
    NetForces         *ForceMatrix           `json:"net_forces"`
    
    // Energy calculations
    TotalPotential    atomic.Float64         `json:"total_potential"`
    TotalKinetic      atomic.Float64         `json:"total_kinetic"`
    EnergyEfficiency  atomic.Float64         `json:"energy_efficiency"`
    
    // Evolution parameters
    TimeStep          float64                `json:"time_step"`
    DampingFactor     float64                `json:"damping_factor"`
    CouplingConstants *CouplingConstants     `json:"coupling_constants"`
    
    // Statistical properties
    FieldCoherence    atomic.Float64         `json:"field_coherence"`
    CorrelationLength atomic.Float64         `json:"correlation_length"`
    Criticality       atomic.Float64         `json:"criticality"`
    
    mu                sync.RWMutex           `json:"-"`
}

// FieldMetrics tracks field-theoretic network properties
type FieldMetrics struct {
    // Scalar fields
    NodeDensity       atomic.Float64         `json:"node_density"`
    MessageDensity    atomic.Float64         `json:"message_density"`
    EnergyDensity     atomic.Float64         `json:"energy_density"`
    
    // Vector fields
    InformationFlow   *Vector3D              `json:"information_flow"`
    MomentumFlow      *Vector3D              `json:"momentum_flow"`
    EntropyGradient   *Vector3D              `json:"entropy_gradient"`
    
    // Tensor properties
    StressTensor      *StressTensor          `json:"stress_tensor"`
    StrainTensor      *StrainTensor          `json:"strain_tensor"`
    MetricTensor      *MetricTensor          `json:"metric_tensor"`
    
    // Topological properties
    EulerCharacteristic atomic.Float64       `json:"euler_characteristic"`
    BettiNumbers      []int32                `json:"betti_numbers"`
    HomologyGroups    []*HomologyGroup       `json:"homology_groups"`
    
    // Critical phenomena
    Phase             config.NetworkPhase    `json:"phase"`
    OrderParameter    atomic.Float64         `json:"order_parameter"`
    CriticalExponents *CriticalExponents     `json:"critical_exponents"`
    
    mu                sync.RWMutex           `json:"-"`
}

// TimeSeriesDatabase manages time-series metric data
type TimeSeriesDatabase struct {
    series           *ConcurrentMap[string, *TimeSeries]
    retentionPolicy  *RetentionPolicy
    compression      *TimeSeriesCompression
    queryEngine      *QueryEngine
    
    mu               sync.RWMutex
}

// AlertManager handles metric-based alerting
type AlertManager struct {
    alerts           *ConcurrentMap[string, *Alert]
    rules            *ConcurrentMap[string, *AlertRule]
    cooldowns        *ConcurrentMap[string, time.Time]
    notifiers        []AlertNotifier
    
    mu               sync.RWMutex
}

// ThresholdManager manages dynamic threshold calculations
type ThresholdManager struct {
    thresholds       *ConcurrentMap[string, *DynamicThreshold]
    adaptiveAlgorithms *ConcurrentMap[string, AdaptiveAlgorithm]
    history          *ThresholdHistory
    
    mu               sync.RWMutex
}

// NewMetricsCollector creates a comprehensive metrics collection system
func NewMetricsCollector(cfg *MetricsConfig) *MetricsCollector {
    if cfg == nil {
        cfg = &MetricsConfig{
            CollectionInterval:      time.Second,
            RetentionPeriod:         time.Hour * 24,
            RollingWindowSize:       1000,
            MaxTimeSeriesPoints:     100000,
            MaxPeerMetrics:          10000,
            MaxConnectionMetrics:    50000,
            PhysicsUpdateInterval:   time.Second * 5,
            FieldResolution:         1.0,
            EntropyCalculationInterval: time.Second * 10,
            EnableAlerts:            true,
            AlertCooldown:           time.Minute,
            CriticalThresholds:      NewDefaultThresholdSet(),
            ExportInterval:          time.Minute,
            EnablePrometheus:        true,
            EnableJSONExport:        true,
            EnableInfluxDB:          false,
        }
    }
    
    now := time.Now()
    collector := &MetricsCollector{
        config:            cfg,
        globalMetrics:     NewGlobalMetrics(),
        peerMetrics:       NewConcurrentMap[string, *PeerMetrics](),
        connectionMetrics: NewConcurrentMap[string, *ConnectionMetrics](),
        messageMetrics:    NewConcurrentMap[string, *MessageMetrics](),
        networkPhysics:    NewNetworkPhysicsModel(cfg.FieldResolution),
        fieldMetrics:      NewFieldMetrics(),
        timeSeries:        NewTimeSeriesDatabase(cfg.MaxTimeSeriesPoints),
        rollingStats:      NewRollingStatisticsManager(cfg.RollingWindowSize),
        alertManager:      NewAlertManager(cfg.EnableAlerts, cfg.AlertCooldown),
        thresholdManager:  NewThresholdManager(),
        exporters:         make([]MetricsExporter, 0),
        reportGenerator:   NewReportGenerator(),
        resourceMonitor:   NewResourceMonitor(),
        startedAt:         now,
        lastExport:        now,
        version:           1,
    }
    
    // Initialize exporters
    if cfg.EnablePrometheus {
        collector.exporters = append(collector.exporters, NewPrometheusExporter())
    }
    if cfg.EnableJSONExport {
        collector.exporters = append(collector.exporters, NewJSONExporter())
    }
    
    return collector
}

// NewGlobalMetrics creates a new global metrics instance
func NewGlobalMetrics() *GlobalMetrics {
    return &GlobalMetrics{
        NetworkEntropy:   *atomic.NewFloat64(1.0),
        AveragePotential: *atomic.NewFloat64(0.5),
        KineticEnergy:    *atomic.NewFloat64(0.0),
        FieldStrength:    *atomic.NewFloat64(1.0),
    }
}

// NewNetworkPhysicsModel creates a new network physics model
func NewNetworkPhysicsModel(resolution float64) *NetworkPhysicsModel {
    return &NetworkPhysicsModel{
        FieldTensor:   NewFieldTensor(100, 100, 100, 1000, resolution, 0.1),
        PotentialField: NewPotentialMap(100, 100, 100, resolution),
        EntropyField:  NewEntropyMap(100, 100, 100, resolution),
        AttractionForces: NewForceMatrix(),
        RepulsionForces:  NewForceMatrix(),
        NetForces:       NewForceMatrix(),
        TimeStep:       0.1,
        DampingFactor:  0.01,
        CouplingConstants: &CouplingConstants{
            Alpha: 0.5,
            Beta:  0.3,
            Gamma: 0.2,
        },
    }
}

// NewFieldMetrics creates new field metrics
func NewFieldMetrics() *FieldMetrics {
    return &FieldMetrics{
        InformationFlow: &Vector3D{X: 0, Y: 0, Z: 0},
        MomentumFlow:    &Vector3D{X: 0, Y: 0, Z: 0},
        EntropyGradient: &Vector3D{X: 0, Y: 0, Z: 0},
        StressTensor:    NewStressTensor(),
        StrainTensor:    NewStrainTensor(),
        MetricTensor:    NewMetricTensor(),
        BettiNumbers:    make([]int32, 3),
        HomologyGroups:  make([]*HomologyGroup, 0),
        Phase:           config.PhaseLiquid,
        CriticalExponents: &CriticalExponents{
            Alpha: 0.0,
            Beta:  0.0,
            Gamma: 0.0,
            Delta: 0.0,
        },
    }
}

// Start begins metrics collection and processing
func (mc *MetricsCollector) Start() error {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    // Start background collectors
    go mc.globalMetricsCollector()
    go mc.physicsModelUpdater()
    go mc.alertProcessor()
    go mc.exporter()
    go mc.cleanupWorker()
    
    mc.startedAt = time.Now()
    return nil
}

// Stop gracefully stops metrics collection
func (mc *MetricsCollector) Stop() {
    mc.mu.Lock()
    defer mc.mu.Unlock()
    
    // Final export
    mc.exportMetrics()
    
    // Note: In production, you would implement proper context cancellation
    // for the background goroutines
}

// RecordPeerMetrics records metrics for a peer
func (mc *MetricsCollector) RecordPeerMetrics(peer *models.PeerInfo) {
    metrics := mc.getOrCreatePeerMetrics(peer.NodeID)
    
    metrics.mu.Lock()
    defer metrics.mu.Unlock()
    
    // Update basic information
    metrics.Address = peer.Address
    
    // Update connection statistics
    if peer.ConnectedAt != nil {
        metrics.Uptime.Store(time.Since(*peer.ConnectedAt).Nanoseconds())
    }
    metrics.LastSeen.Store(time.Now().UnixNano())
    
    // Update message statistics
    metrics.MessagesSent.Store(peer.MessagesSent)
    metrics.MessagesReceived.Store(peer.MessagesReceived)
    
    // Update performance metrics
    metrics.AverageLatency.Store(peer.Latency.Seconds())
    
    // Update reputation and quality
    metrics.Reputation.Store(peer.Reputation)
    metrics.QualityScore.Store(peer.CalculateQualityScore())
    
    // Update physics properties
    metrics.PotentialEnergy.Store(peer.PotentialEnergy)
    metrics.EntropyContribution.Store(peer.EntropyContribution)
    if peer.ForceVector != nil {
        metrics.ForceMagnitude.Store(peer.ForceVector.Magnitude)
    }
    
    // Update time series
    metrics.LatencyHistory.Add(peer.Latency.Seconds())
    metrics.QualityHistory.Add(peer.CalculateQualityScore())
    
    // Calculate message success rate
    totalMessages := peer.MessagesSent + peer.MessagesReceived
    if totalMessages > 0 {
        successRate := float64(peer.SuccessfulPings) / float64(peer.SuccessfulPings+peer.FailedPings)
        metrics.MessageSuccessRate.Store(successRate)
    }
    
    metrics.LastUpdated.Store(time.Now().UnixNano())
    metrics.Version.Add(1)
    
    // Update global metrics
    mc.updateGlobalMetricsFromPeer(peer)
}

// RecordConnectionMetrics records metrics for a connection
func (mc *MetricsCollector) RecordConnectionMetrics(connMetrics *models.ConnectionMetrics) {
    snapshot := connMetrics.GetMetricsSnapshot()
    metrics := mc.getOrCreateConnectionMetrics(snapshot.ConnectionID)
    
    metrics.mu.Lock()
    defer metrics.mu.Unlock()
    
    // Update basic information
    metrics.PeerID = snapshot.PeerID
    metrics.Protocol = config.TCP // This should come from the connection
    
    // Update traffic statistics
    metrics.BytesSent.Store(snapshot.BytesSent)
    metrics.BytesReceived.Store(snapshot.BytesReceived)
    metrics.MessagesSent.Store(snapshot.MessagesSent)
    metrics.MessagesReceived.Store(snapshot.MessagesReceived)
    
    // Update performance metrics
    metrics.CurrentLatency.Store(snapshot.AverageLatency.Seconds())
    metrics.AverageLatency.Store(snapshot.AverageLatency.Seconds())
    metrics.MinLatency.Store(snapshot.MinLatency.Seconds())
    metrics.MaxLatency.Store(snapshot.MaxLatency.Seconds())
    
    // Update quality metrics
    metrics.SuccessRate.Store(snapshot.SuccessRate)
    metrics.ErrorCount.Store(snapshot.ErrorsEncountered)
    metrics.RetryCount.Store(snapshot.RetriesAttempted)
    
    // Update physics properties
    metrics.ConnectionQuality.Store(snapshot.QualityScore)
    
    // Update timing
    metrics.LastActivity.Store(time.Now().UnixNano())
    metrics.Uptime.Store(time.Since(metrics.EstablishedAt).Nanoseconds())
    
    // Update history
    metrics.LatencyHistory.Add(snapshot.AverageLatency.Seconds())
    metrics.TrafficHistory.Record(snapshot.BytesSent, snapshot.BytesReceived)
    
    // Update global metrics
    mc.updateGlobalMetricsFromConnection(snapshot)
}

// RecordMessageMetrics records metrics for a message
func (mc *MetricsCollector) RecordMessageMetrics(msg *models.NetworkMessage) {
    metrics := mc.getOrCreateMessageMetrics(msg.Header.GetMessageIDString())
    
    metrics.mu.Lock()
    defer metrics.mu.Unlock()
    
    // Update basic information
    metrics.MessageType = msg.Header.MessageType
    metrics.SourcePeer = msg.GetSourceNode()
    metrics.PayloadSize = msg.Header.PayloadSize
    metrics.TotalSize = msg.Header.TotalSize
    
    // Update delivery tracking
    if !msg.receivedAt.IsZero() {
        metrics.DeliveryTime.Store(msg.receivedAt.Sub(time.Unix(0, msg.Header.Timestamp)).Seconds())
        metrics.DeliveredAt.Store(msg.receivedAt.UnixNano())
    }
    
    // Update routing metrics
    if msg.RoutingInfo != nil {
        metrics.HopCount.Store(uint32(msg.RoutingInfo.CurrentHop))
        metrics.RoutingCost.Store(msg.CalculateRoutingCost())
        metrics.PathEfficiency.Store(1.0 / msg.CalculateRoutingCost())
    }
    
    // Update physics properties
    if msg.PhysicsMetadata != nil {
        metrics.FieldStrength.Store(msg.PhysicsMetadata.FieldStrength)
        metrics.PotentialUsed.Store(msg.PhysicsMetadata.Potential)
        metrics.EntropyImpact.Store(msg.PhysicsMetadata.Entropy)
    }
    
    // Update reliability metrics
    metrics.RetryAttempts.Store(uint32(msg.GetAttempts()))
    metrics.AckReceived.Store(msg.DeliveryInfo != nil && msg.DeliveryInfo.AckReceived)
    
    if msg.IsExpired() {
        metrics.DeliveryStatus = config.DeliveryExpired
    } else if metrics.AckReceived.Load() {
        metrics.DeliveryStatus = config.DeliveryConfirmed
    } else {
        metrics.DeliveryStatus = config.DeliveryInProgress
    }
    
    // Update global metrics
    mc.updateGlobalMetricsFromMessage(msg)
}

// getOrCreatePeerMetrics gets or creates peer metrics
func (mc *MetricsCollector) getOrCreatePeerMetrics(peerID string) *PeerMetrics {
    if metrics, exists := mc.peerMetrics.Get(peerID); exists {
        return metrics
    }
    
    metrics := &PeerMetrics{
        PeerID:           peerID,
        FirstSeen:        time.Now(),
        LatencyHistory:   NewRollingStatistics(1000),
        MessageRateHistory: NewRollingStatistics(1000),
        QualityHistory:   NewRollingStatistics(1000),
    }
    
    mc.peerMetrics.Set(peerID, metrics)
    return metrics
}

// getOrCreateConnectionMetrics gets or creates connection metrics
func (mc *MetricsCollector) getOrCreateConnectionMetrics(connID string) *ConnectionMetrics {
    if metrics, exists := mc.connectionMetrics.Get(connID); exists {
        return metrics
    }
    
    metrics := &ConnectionMetrics{
        ConnectionID:    connID,
        EstablishedAt:   time.Now(),
        LatencyHistory:  NewRollingStatistics(1000),
        TrafficHistory:  NewTrafficHistory(time.Minute, 60),
    }
    
    mc.connectionMetrics.Set(connID, metrics)
    return metrics
}

// getOrCreateMessageMetrics gets or creates message metrics
func (mc *MetricsCollector) getOrCreateMessageMetrics(msgID string) *MessageMetrics {
    if metrics, exists := mc.messageMetrics.Get(msgID); exists {
        return metrics
    }
    
    metrics := &MessageMetrics{
        MessageID: msgID,
        CreatedAt: time.Now(),
    }
    
    mc.messageMetrics.Set(msgID, metrics)
    return metrics
}

// updateGlobalMetricsFromPeer updates global metrics based on peer data
func (mc *MetricsCollector) updateGlobalMetricsFromPeer(peer *models.PeerInfo) {
    mc.globalMetrics.mu.Lock()
    defer mc.globalMetrics.mu.Unlock()
    
    // Update peer counts
    mc.globalMetrics.TotalPeers.Add(1)
    if peer.State == config.Ready {
        mc.globalMetrics.ConnectedPeers.Add(1)
    }
    if peer.IsBanned {
        mc.globalMetrics.BannedPeers.Add(1)
    }
    
    // Update physics metrics
    mc.globalMetrics.AveragePotential.Store(
        (mc.globalMetrics.AveragePotential.Load() + peer.PotentialEnergy) / 2.0,
    )
    
    mc.globalMetrics.LastUpdate.Store(time.Now().UnixNano())
}

// updateGlobalMetricsFromConnection updates global metrics based on connection data
func (mc *MetricsCollector) updateGlobalMetricsFromConnection(snapshot *models.ConnectionMetricsSnapshot) {
    mc.globalMetrics.mu.Lock()
    defer mc.globalMetrics.mu.Unlock()
    
    // Update bandwidth
    mc.globalMetrics.BytesSent.Add(snapshot.BytesSent)
    mc.globalMetrics.BytesReceived.Add(snapshot.BytesReceived)
    
    // Update message counts
    mc.globalMetrics.TotalMessages.Add(snapshot.MessagesSent + snapshot.MessagesReceived)
    
    // Update performance metrics
    currentAvgLatency := mc.globalMetrics.AverageLatency.Load()
    newAvgLatency := (currentAvgLatency + snapshot.AverageLatency.Seconds()) / 2.0
    mc.globalMetrics.AverageLatency.Store(newAvgLatency)
    
    // Update success rate
    totalOps := snapshot.SuccessCount + snapshot.FailureCount
    if totalOps > 0 {
        successRate := float64(snapshot.SuccessCount) / float64(totalOps)
        currentSuccessRate := mc.globalMetrics.SuccessRate.Load()
        newSuccessRate := (currentSuccessRate + successRate) / 2.0
        mc.globalMetrics.SuccessRate.Store(newSuccessRate)
    }
    
    mc.globalMetrics.LastUpdate.Store(time.Now().UnixNano())
}

// updateGlobalMetricsFromMessage updates global metrics based on message data
func (mc *MetricsCollector) updateGlobalMetricsFromMessage(msg *models.NetworkMessage) {
    mc.globalMetrics.mu.Lock()
    defer mc.globalMetrics.mu.Unlock()
    
    // Update message size statistics
    currentAvgSize := mc.globalMetrics.AverageMessageSize.Load()
    messageSize := float64(len(msg.Payload))
    newAvgSize := (currentAvgSize + messageSize) / 2.0
    mc.globalMetrics.AverageMessageSize.Store(newAvgSize)
    
    mc.globalMetrics.LastUpdate.Store(time.Now().UnixNano())
}

// globalMetricsCollector periodically updates derived global metrics
func (mc *MetricsCollector) globalMetricsCollector() {
    ticker := time.NewTicker(mc.config.CollectionInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.updateDerivedGlobalMetrics()
    }
}

// updateDerivedGlobalMetrics calculates derived global metrics
func (mc *MetricsCollector) updateDerivedGlobalMetrics() {
    mc.globalMetrics.mu.Lock()
    defer mc.globalMetrics.mu.Unlock()
    
    now := time.Now()
    
    // Calculate messages per second
    elapsed := now.Sub(time.Unix(0, mc.globalMetrics.LastUpdate.Load())).Seconds()
    if elapsed > 0 {
        totalMessages := mc.globalMetrics.TotalMessages.Load()
        mps := float64(totalMessages) / elapsed
        mc.globalMetrics.MessagesPerSecond.Store(mps)
    }
    
    // Calculate bandwidth rates
    totalBytesSent := mc.globalMetrics.BytesSent.Load()
    totalBytesReceived := mc.globalMetrics.BytesReceived.Load()
    
    if elapsed > 0 {
        sentRate := float64(totalBytesSent) / elapsed
        receivedRate := float64(totalBytesReceived) / elapsed
        mc.globalMetrics.BytesPerSecondSent.Store(sentRate)
        mc.globalMetrics.BytesPerSecondReceived.Store(receivedRate)
    }
    
    // Update resource usage
    mc.updateResourceMetrics()
    
    // Calculate error rate
    successRate := mc.globalMetrics.SuccessRate.Load()
    mc.globalMetrics.ErrorRate.Store(1.0 - successRate)
    
    mc.globalMetrics.LastUpdate.Store(now.UnixNano())
    mc.globalMetrics.CollectionTime.Store(float64(time.Since(now).Nanoseconds()) / 1e9)
}

// updateResourceMetrics updates system resource usage metrics
func (mc *MetricsCollector) updateResourceMetrics() {
    // Get memory usage
    memStats := mc.resourceMonitor.GetMemoryUsage()
    mc.globalMetrics.MemoryUsage.Store(memStats.Used)
    
    // Get CPU usage
    cpuUsage := mc.resourceMonitor.GetCPUUsage()
    mc.globalMetrics.CPUUsage.Store(cpuUsage)
    
    // Get goroutine count
    goroutineCount := mc.resourceMonitor.GetGoroutineCount()
    mc.globalMetrics.GoroutineCount.Store(goroutineCount)
}

// physicsModelUpdater periodically updates the physics model
func (mc *MetricsCollector) physicsModelUpdater() {
    ticker := time.NewTicker(mc.config.PhysicsUpdateInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.updatePhysicsModel()
    }
}

// updatePhysicsModel updates the network physics model
func (mc *MetricsCollector) updatePhysicsModel() {
    mc.networkPhysics.mu.Lock()
    defer mc.networkPhysics.mu.Unlock()
    
    // Update field tensor based on current network state
    mc.updateFieldTensor()
    
    // Calculate forces between nodes
    mc.calculateNetworkForces()
    
    // Update energy calculations
    mc.updateEnergyCalculations()
    
    // Update field metrics
    mc.updateFieldMetrics()
    
    // Calculate network entropy
    mc.calculateNetworkEntropy()
}

// updateFieldTensor updates the field tensor representation
func (mc *MetricsCollector) updateFieldTensor() {
    // This would implement the actual field tensor update logic
    // based on current peer distribution and message flow
    
    // For now, implement a simple update based on peer density
    totalPeers := mc.globalMetrics.TotalPeers.Load()
    connectedPeers := mc.globalMetrics.ConnectedPeers.Load()
    
    if totalPeers > 0 {
        connectivityRatio := float64(connectedPeers) / float64(totalPeers)
        
        // Update field strength based on connectivity
        mc.globalMetrics.FieldStrength.Store(connectivityRatio)
        
        // Update network physics model
        mc.networkPhysics.FieldCoherence.Store(connectivityRatio)
    }
}

// calculateNetworkForces calculates forces between network nodes
func (mc *MetricsCollector) calculateNetworkForces() {
    // This would calculate attraction and repulsion forces
    // between all pairs of peers based on their properties
    
    // Simplified implementation for demonstration
    var totalAttraction, totalRepulsion float64
    peerCount := 0
    
    mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
        attraction := metrics.ForceMagnitude.Load() * metrics.PotentialEnergy.Load()
        repulsion := metrics.EntropyContribution.Load() * (1.0 - metrics.QualityScore.Load())
        
        totalAttraction += attraction
        totalRepulsion += repulsion
        peerCount++
        
        return true
    })
    
    if peerCount > 0 {
        avgAttraction := totalAttraction / float64(peerCount)
        avgRepulsion := totalRepulsion / float64(peerCount)
        
        mc.networkPhysics.TotalPotential.Store(avgAttraction)
        mc.globalMetrics.AveragePotential.Store(avgAttraction)
        mc.globalMetrics.KineticEnergy.Store(avgRepulsion)
    }
}

// updateEnergyCalculations updates energy-related metrics
func (mc *MetricsCollector) updateEnergyCalculations() {
    potential := mc.networkPhysics.TotalPotential.Load()
    kinetic := mc.globalMetrics.KineticEnergy.Load()
    
    // Calculate energy efficiency
    totalEnergy := potential + kinetic
    if totalEnergy > 0 {
        efficiency := potential / totalEnergy
        mc.networkPhysics.EnergyEfficiency.Store(efficiency)
    }
}

// updateFieldMetrics updates field-theoretic metrics
func (mc *MetricsCollector) updateFieldMetrics() {
    // Update scalar field metrics
    totalPeers := float64(mc.globalMetrics.TotalPeers.Load())
    if totalPeers > 0 {
        // Estimate node density (simplified)
        nodeDensity := totalPeers / 1000.0 // Assuming 1000 unit volume
        mc.fieldMetrics.NodeDensity.Store(nodeDensity)
        
        // Estimate message density
        totalMessages := float64(mc.globalMetrics.TotalMessages.Load())
        messageDensity := totalMessages / 1000.0
        mc.fieldMetrics.MessageDensity.Store(messageDensity)
    }
    
    // Update energy density
    potential := mc.networkPhysics.TotalPotential.Load()
    mc.fieldMetrics.EnergyDensity.Store(potential / 1000.0)
}

// calculateNetworkEntropy calculates the network entropy
func (mc *MetricsCollector) calculateNetworkEntropy() {
    // Calculate Shannon entropy based on peer distribution and behavior
    
    var entropy float64
    peerCount := 0
    totalQuality := 0.0
    
    mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
        quality := metrics.QualityScore.Load()
        totalQuality += quality
        peerCount++
        return true
    })
    
    if peerCount > 0 {
        avgQuality := totalQuality / float64(peerCount)
        
        // Calculate entropy based on quality distribution variance
        variance := 0.0
        mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
            deviation := metrics.QualityScore.Load() - avgQuality
            variance += deviation * deviation
            return true
        })
        
        variance /= float64(peerCount)
        entropy = math.Sqrt(variance) // Using standard deviation as entropy proxy
        
        mc.globalMetrics.NetworkEntropy.Store(entropy)
        mc.fieldMetrics.EntropyGradient = &Vector3D{
            X: entropy,
            Y: 0.0, // These would be calculated based on spatial distribution
            Z: 0.0,
        }
    }
}

// alertProcessor processes metric-based alerts
func (mc *MetricsCollector) alertProcessor() {
    ticker := time.NewTicker(mc.config.CollectionInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.checkAlerts()
    }
}

// checkAlerts checks all alert rules and triggers alerts if needed
func (mc *MetricsCollector) checkAlerts() {
    // Check global metric alerts
    mc.checkGlobalMetricAlerts()
    
    // Check peer-specific alerts
    mc.checkPeerAlerts()
    
    // Check connection-specific alerts
    mc.checkConnectionAlerts()
}

// checkGlobalMetricAlerts checks alerts based on global metrics
func (mc *MetricsCollector) checkGlobalMetricAlerts() {
    currentEntropy := mc.globalMetrics.NetworkEntropy.Load()
    if currentEntropy > 0.8 {
        mc.alertManager.TriggerAlert("high_entropy", 
            fmt.Sprintf("Network entropy is high: %.3f", currentEntropy),
            config.AlertLevelWarning)
    }
    
    successRate := mc.globalMetrics.SuccessRate.Load()
    if successRate < 0.5 {
        mc.alertManager.TriggerAlert("low_success_rate",
            fmt.Sprintf("Network success rate is low: %.3f", successRate),
            config.AlertLevelCritical)
    }
    
    connectedPeers := mc.globalMetrics.ConnectedPeers.Load()
    if connectedPeers < 10 {
        mc.alertManager.TriggerAlert("low_connectivity",
            fmt.Sprintf("Low peer connectivity: %d peers", connectedPeers),
            config.AlertLevelWarning)
    }
}

// checkPeerAlerts checks alerts for individual peers
func (mc *MetricsCollector) checkPeerAlerts() {
    mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
        // Check for low reputation
        if metrics.Reputation.Load() < -80 {
            mc.alertManager.TriggerAlert("low_peer_reputation",
                fmt.Sprintf("Peer %s has low reputation: %d", peerID[:8], metrics.Reputation.Load()),
                config.AlertLevelWarning)
        }
        
        // Check for high latency
        if metrics.AverageLatency.Load() > 5.0 { // 5 seconds
            mc.alertManager.TriggerAlert("high_peer_latency",
                fmt.Sprintf("Peer %s has high latency: %.3fs", peerID[:8], metrics.AverageLatency.Load()),
                config.AlertLevelWarning)
        }
        
        return true
    })
}

// checkConnectionAlerts checks alerts for connections
func (mc *MetricsCollector) checkConnectionAlerts() {
    mc.connectionMetrics.Range(func(connID string, metrics *ConnectionMetrics) bool {
        // Check for high error rate
        if metrics.ErrorCount.Load() > 100 {
            mc.alertManager.TriggerAlert("high_connection_errors",
                fmt.Sprintf("Connection %s has high error count: %d", connID[:8], metrics.ErrorCount.Load()),
                config.AlertLevelWarning)
        }
        
        // Check for connection quality
        if metrics.ConnectionQuality.Load() < 0.3 {
            mc.alertManager.TriggerAlert("poor_connection_quality",
                fmt.Sprintf("Connection %s has poor quality: %.3f", connID[:8], metrics.ConnectionQuality.Load()),
                config.AlertLevelWarning)
        }
        
        return true
    })
}

// exporter periodically exports metrics to configured exporters
func (mc *MetricsCollector) exporter() {
    ticker := time.NewTicker(mc.config.ExportInterval)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.exportMetrics()
    }
}

// exportMetrics exports metrics to all configured exporters
func (mc *MetricsCollector) exportMetrics() {
    mc.mu.RLock()
    defer mc.mu.RUnlock()
    
    // Generate export data
    exportData := mc.generateExportData()
    
    // Export to all configured exporters
    for _, exporter := range mc.exporters {
        if err := exporter.Export(exportData); err != nil {
            // Log export error
            fmt.Printf("Metrics export error: %v\n", err)
        }
    }
    
    mc.lastExport = time.Now()
}

// generateExportData generates data for export
func (mc *MetricsCollector) generateExportData() *MetricsExportData {
    return &MetricsExportData{
        Timestamp:       time.Now(),
        GlobalMetrics:   mc.getGlobalMetricsSnapshot(),
        PeerMetrics:     mc.getPeerMetricsSnapshot(),
        ConnectionMetrics: mc.getConnectionMetricsSnapshot(),
        PhysicsMetrics:  mc.getPhysicsMetricsSnapshot(),
        FieldMetrics:    mc.getFieldMetricsSnapshot(),
        Alerts:          mc.alertManager.GetActiveAlerts(),
    }
}

// getGlobalMetricsSnapshot returns a snapshot of global metrics
func (mc *MetricsCollector) getGlobalMetricsSnapshot() *GlobalMetrics {
    mc.globalMetrics.mu.RLock()
    defer mc.globalMetrics.mu.RUnlock()
    
    // Create a copy for safe access
    snapshot := *mc.globalMetrics
    return &snapshot
}

// getPeerMetricsSnapshot returns snapshots of all peer metrics
func (mc *MetricsCollector) getPeerMetricsSnapshot() []*PeerMetrics {
    snapshots := make([]*PeerMetrics, 0)
    
    mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
        metrics.mu.RLock()
        snapshot := *metrics
        metrics.mu.RUnlock()
        snapshots = append(snapshots, &snapshot)
        return true
    })
    
    return snapshots
}

// getConnectionMetricsSnapshot returns snapshots of connection metrics
func (mc *MetricsCollector) getConnectionMetricsSnapshot() []*ConnectionMetrics {
    snapshots := make([]*ConnectionMetrics, 0)
    
    mc.connectionMetrics.Range(func(connID string, metrics *ConnectionMetrics) bool {
        metrics.mu.RLock()
        snapshot := *metrics
        metrics.mu.RUnlock()
        snapshots = append(snapshots, &snapshot)
        return true
    })
    
    return snapshots
}

// getPhysicsMetricsSnapshot returns physics metrics snapshot
func (mc *MetricsCollector) getPhysicsMetricsSnapshot() *NetworkPhysicsModel {
    mc.networkPhysics.mu.RLock()
    defer mc.networkPhysics.mu.RUnlock()
    
    snapshot := *mc.networkPhysics
    return &snapshot
}

// getFieldMetricsSnapshot returns field metrics snapshot
func (mc *MetricsCollector) getFieldMetricsSnapshot() *FieldMetrics {
    mc.fieldMetrics.mu.RLock()
    defer mc.fieldMetrics.mu.RUnlock()
    
    snapshot := *mc.fieldMetrics
    return &snapshot
}

// cleanupWorker periodically cleans up old metrics data
func (mc *MetricsCollector) cleanupWorker() {
    ticker := time.NewTicker(mc.config.RetentionPeriod / 10)
    defer ticker.Stop()
    
    for range ticker.C {
        mc.cleanupOldData()
    }
}

// cleanupOldData removes old metrics data based on retention policy
func (mc *MetricsCollector) cleanupOldData() {
    cutoff := time.Now().Add(-mc.config.RetentionPeriod)
    
    // Clean up old peer metrics
    mc.cleanupOldPeerMetrics(cutoff)
    
    // Clean up old connection metrics
    mc.cleanupOldConnectionMetrics(cutoff)
    
    // Clean up old message metrics
    mc.cleanupOldMessageMetrics(cutoff)
    
    // Clean up time series data
    mc.timeSeries.Cleanup(cutoff)
}

// cleanupOldPeerMetrics removes old peer metrics
func (mc *MetricsCollector) cleanupOldPeerMetrics(cutoff time.Time) {
    var toDelete []string
    
    mc.peerMetrics.Range(func(peerID string, metrics *PeerMetrics) bool {
        lastUpdated := time.Unix(0, metrics.LastUpdated.Load())
        if lastUpdated.Before(cutoff) {
            toDelete = append(toDelete, peerID)
        }
        return true
    })
    
    for _, peerID := range toDelete {
        mc.peerMetrics.Delete(peerID)
    }
}

// cleanupOldConnectionMetrics removes old connection metrics
func (mc *MetricsCollector) cleanupOldConnectionMetrics(cutoff time.Time) {
    var toDelete []string
    
    mc.connectionMetrics.Range(func(connID string, metrics *ConnectionMetrics) bool {
        lastActivity := time.Unix(0, metrics.LastActivity.Load())
        if lastActivity.Before(cutoff) {
            toDelete = append(toDelete, connID)
        }
        return true
    })
    
    for _, connID := range toDelete {
        mc.connectionMetrics.Delete(connID)
    }
}

// cleanupOldMessageMetrics removes old message metrics
func (mc *MetricsCollector) cleanupOldMessageMetrics(cutoff time.Time) {
    var toDelete []string
    
    mc.messageMetrics.Range(func(msgID string, metrics *MessageMetrics) bool {
        if metrics.CreatedAt.Before(cutoff) {
            toDelete = append(toDelete, msgID)
        }
        return true
    })
    
    for _, msgID := range toDelete {
        mc.messageMetrics.Delete(msgID)
    }
}

// GetGlobalMetrics returns the current global metrics
func (mc *MetricsCollector) GetGlobalMetrics() *GlobalMetrics {
    return mc.getGlobalMetricsSnapshot()
}

// GetPeerMetrics returns metrics for a specific peer
func (mc *MetricsCollector) GetPeerMetrics(peerID string) *PeerMetrics {
    if metrics, exists := mc.peerMetrics.Get(peerID); exists {
        metrics.mu.RLock()
        defer metrics.mu.RUnlock()
        snapshot := *metrics
        return &snapshot
    }
    return nil
}

// GetConnectionMetrics returns metrics for a specific connection
func (mc *MetricsCollector) GetConnectionMetrics(connID string) *ConnectionMetrics {
    if metrics, exists := mc.connectionMetrics.Get(connID); exists {
        metrics.mu.RLock()
        defer metrics.mu.RUnlock()
        snapshot := *metrics
        return &snapshot
    }
    return nil
}

// GetNetworkPhysics returns the current network physics model
func (mc *MetricsCollector) GetNetworkPhysics() *NetworkPhysicsModel {
    return mc.getPhysicsMetricsSnapshot()
}

// GetFieldMetrics returns the current field metrics
func (mc *MetricsCollector) GetFieldMetrics() *FieldMetrics {
    return mc.getFieldMetricsSnapshot()
}

// GenerateReport generates a comprehensive metrics report
func (mc *MetricsCollector) GenerateReport() *MetricsReport {
    return mc.reportGenerator.GenerateReport(mc)
}

// String returns a string representation of the metrics collector
func (mc *MetricsCollector) String() string {
    global := mc.GetGlobalMetrics()
    return fmt.Sprintf("MetricsCollector[Peers:%d/%d, Messages:%d, Entropy:%.3f, Quality:%.3f]", 
        global.ConnectedPeers.Load(), global.TotalPeers.Load(),
        global.TotalMessages.Load(), global.NetworkEntropy.Load(),
        global.SuccessRate.Load())
}