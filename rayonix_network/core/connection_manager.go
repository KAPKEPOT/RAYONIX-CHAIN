package core

import (
    "context"
    "fmt"
    "math"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/models"
    "github.com/rayxnetwork/p2p/protocols"
    "github.com/rayxnetwork/p2p/utils"
    "github.com/sirupsen/logrus"
)

// ConnectionManager implements physics-inspired connection topology management
type ConnectionManager struct {
    network    *AdvancedP2PNetwork
    config     *config.NodeConfig
    isRunning  atomic.Bool
    mu         sync.RWMutex

    // Connection state
    activeConnections   map[string]*ConnectionState
    pendingConnections  map[string]*PendingConnection
    connectionPool      *utils.ConnectionPool
    connectionSemaphore *utils.Semaphore

    // Physics model integration
    topologyMatrix   *TopologyMatrix
    connectionForces map[string]*ConnectionForce
    potentialEnergy  float64
    kineticEnergy    float64

    // Performance tracking
    connectionMetrics map[string]*models.ConnectionMetrics
    healthMonitor     *utils.HealthMonitor
    loadBalancer      *utils.LoadBalancer

    // Control channels
    connectQueue    chan *ConnectionRequest
    disconnectQueue chan string
    controlChannel  chan ControlSignal
}

// ConnectionState represents the complete state of a connection
type ConnectionState struct {
    PeerInfo      *models.PeerInfo
    ConnectionID  string
    Protocol      config.ProtocolType
    Handler       protocols.ProtocolHandler
    EstablishedAt time.Time
    LastActivity  time.Time
    
    // Physics model state
    ConnectionForce *ConnectionForce
    Potential       float64
    Stability       float64
    
    // Performance metrics
    Metrics        *models.ConnectionMetrics
    QualityScore   float64
    FailureCount   int
}

// ConnectionForce represents the physics-inspired force acting on a connection
type ConnectionForce struct {
    AttractionMagnitude float64
    RepulsionMagnitude  float64
    NetForce           float64
    Direction          *ForceDirection
    Torque             float64
    LastUpdated        time.Time
}

// ForceDirection represents the directional component of connection forces
type ForceDirection struct {
    Theta    float64 // Angle in radians
    Phi      float64 // Phase angle
    Velocity float64 // Rate of change
}

// TopologyMatrix represents the network connection graph
type TopologyMatrix struct {
    Adjacency    map[string]map[string]float64 // Connection weights
    Distance     map[string]map[string]float64 // Latency-based distances
    Centrality   map[string]float64            // Eigenvector centrality
    Clustering   map[string]float64            // Local clustering coefficients
    mu           sync.RWMutex
}

// ConnectionRequest represents a connection initiation attempt
type ConnectionRequest struct {
    PeerInfo    *models.PeerInfo
    Priority    int
    ForceType   ForceType
    RetryCount  int
    Context     context.Context
    ResultChan  chan<- *ConnectionResult
}

// ConnectionResult represents the outcome of a connection attempt
type ConnectionResult struct {
    Success    bool
    Connection *ConnectionState
    Error      error
    Latency    time.Duration
}

// ControlSignal represents control messages for connection management
type ControlSignal struct {
    Type      ControlSignalType
    PeerID    string
    Parameter interface{}
}

type ControlSignalType int

const (
    SignalRecalculateForces ControlSignalType = iota
    SignalOptimizeTopology
    SignalEvictPeer
    SignalInjectEntropy
    SignalUpdateWeights
)

type ForceType int

const (
    GravitationalForce ForceType = iota
    ElectromagneticForce
    WeakNuclearForce
    StrongNuclearForce
)

// NewConnectionManager creates a new physics-inspired connection manager
func NewConnectionManager(network *AdvancedP2PNetwork) *ConnectionManager {
    cfg := network.config
    
    cm := &ConnectionManager{
        network:    network,
        config:     cfg,
        activeConnections: make(map[string]*ConnectionState),
        pendingConnections: make(map[string]*PendingConnection),
        connectionPool: utils.NewConnectionPool(cfg.MaxConnections),
        connectionSemaphore: utils.NewSemaphore(cfg.MaxConnections / 2), // Allow concurrent connections
        topologyMatrix: &TopologyMatrix{
            Adjacency:  make(map[string]map[string]float64),
            Distance:   make(map[string]map[string]float64),
            Centrality: make(map[string]float64),
            Clustering: make(map[string]float64),
        },
        connectionForces:   make(map[string]*ConnectionForce),
        connectionMetrics:  make(map[string]*models.ConnectionMetrics),
        healthMonitor:      utils.NewHealthMonitor(),
        loadBalancer:       utils.NewLoadBalancer(),
        connectQueue:       make(chan *ConnectionRequest, 1000),
        disconnectQueue:    make(chan string, 1000),
        controlChannel:     make(chan ControlSignal, 100),
    }

    // Initialize topology with self-connection
    cm.topologyMatrix.Adjacency[network.nodeID] = make(map[string]float64)
    cm.topologyMatrix.Distance[network.nodeID] = make(map[string]float64)

    return cm
}

// Start begins the connection management subsystem
func (cm *ConnectionManager) Start() error {
    if cm.isRunning.Swap(true) {
        return fmt.Errorf("connection manager already running")
    }

    logrus.Info("Starting physics-inspired connection manager")

    // Start worker goroutines
    go cm.connectionWorker()
    go cm.forceCalculationWorker()
    go cm.topologyOptimizationWorker()
    go cm.healthMonitoringWorker()

    logrus.Info("Connection manager started successfully")
    return nil
}

// Stop gracefully shuts down the connection manager
func (cm *ConnectionManager) Stop() {
    if !cm.isRunning.Swap(false) {
        return
    }

    logrus.Info("Stopping connection manager")

    // Close control channels
    close(cm.connectQueue)
    close(cm.disconnectQueue)
    close(cm.controlChannel)

    // Close all active connections
    cm.CloseAll()

    logrus.Info("Connection manager stopped")
}

// ConnectToPeer initiates a physics-inspired connection to a peer
func (cm *ConnectionManager) ConnectToPeer(peer *models.PeerInfo) (*ConnectionState, error) {
    if !cm.isRunning.Load() {
        return nil, fmt.Errorf("connection manager not running")
    }

    // Check if already connected
    if state := cm.GetConnectionState(peer.NodeID); state != nil {
        return state, nil
    }

    // Create connection request
    req := &ConnectionRequest{
        PeerInfo:   peer,
        Priority:   cm.calculateConnectionPriority(peer),
        ForceType:  cm.determineForceType(peer),
        RetryCount: 0,
        Context:    context.Background(),
        ResultChan: make(chan *ConnectionResult, 1),
    }

    // Submit to connection queue
    select {
    case cm.connectQueue <- req:
        // Request queued successfully
    case <-time.After(5 * time.Second):
        return nil, fmt.Errorf("connection queue timeout")
    }

    // Wait for result
    select {
    case result := <-req.ResultChan:
        if result.Success {
            return result.Connection, nil
        }
        return nil, result.Error
    case <-time.After(30 * time.Second):
        return nil, fmt.Errorf("connection attempt timeout")
    }
}

// calculateConnectionPriority computes physics-based connection priority
func (cm *ConnectionManager) calculateConnectionPriority(peer *models.PeerInfo) int {
    // Base priority from validator score and stake
    validatorScore := cm.network.validatorScores[peer.NodeID]
    stake := cm.network.stakeDistribution[peer.NodeID]
    
    // Network quality factors
    latencyPriority := int(1000 / (1 + peer.Latency.Seconds()))
    reputationPriority := peer.Reputation
    
    // Physics model integration
    potentialPriority := int(cm.calculateConnectionPotential(peer) * 100)
    forcePriority := int(cm.calculateInitialForce(peer).NetForce * 50)
    
    totalPriority := (latencyPriority + reputationPriority + potentialPriority + forcePriority) / 4
    
    // Clamp to valid range
    return utils.Clamp(totalPriority, 1, 1000)
}

// calculateConnectionPotential estimates the potential energy of a connection
func (cm *ConnectionManager) calculateConnectionPotential(peer *models.PeerInfo) float64 {
    // Use multiple factors to estimate connection quality
    factors := []float64{
        float64(peer.Reputation) / 100.0,
        1.0 / (1.0 + peer.Latency.Seconds()),
        cm.network.validatorScores[peer.NodeID],
        cm.network.stakeDistribution[peer.NodeID],
    }
    
    // Weighted geometric mean
    product := 1.0
    for _, factor := range factors {
        product *= math.Max(0.01, factor) // Avoid zero factors
    }
    
    return math.Pow(product, 1.0/float64(len(factors)))
}

// calculateInitialForce computes the initial force for a connection
func (cm *ConnectionManager) calculateInitialForce(peer *models.PeerInfo) *ConnectionForce {
    // Attraction components
    validatorAttraction := cm.network.validatorScores[peer.NodeID]
    stakeAttraction := cm.network.stakeDistribution[peer.NodeID]
    reputationAttraction := float64(peer.Reputation) / 100.0
    
    totalAttraction := (validatorAttraction + stakeAttraction + reputationAttraction) / 3.0
    
    // Repulsion components
    latencyRepulsion := 1.0 / (1.0 + math.Exp(-peer.Latency.Seconds()))
    entropyRepulsion := cm.network.networkEntropy
    distanceRepulsion := cm.calculateNetworkDistance(peer.NodeID)
    
    totalRepulsion := (latencyRepulsion + entropyRepulsion + distanceRepulsion) / 3.0
    
    // Net force
    netForce := totalAttraction - totalRepulsion
    
    return &ConnectionForce{
        AttractionMagnitude: totalAttraction,
        RepulsionMagnitude:  totalRepulsion,
        NetForce:           netForce,
        Direction: &ForceDirection{
            Theta:    rand.Float64() * 2 * math.Pi,
            Phi:      rand.Float64() * math.Pi,
            Velocity: 0.1,
        },
        Torque:      rand.NormFloat64() * 0.01,
        LastUpdated: time.Now(),
    }
}

// determineForceType selects the appropriate force model for the connection
func (cm *ConnectionManager) determineForceType(peer *models.PeerInfo) ForceType {
    // Use different force models based on connection characteristics
    latency := peer.Latency.Seconds()
    reputation := float64(peer.Reputation)
    
    if latency < 0.1 && reputation > 80 {
        return StrongNuclearForce // High-quality, low-latency connections
    } else if latency < 0.5 && reputation > 60 {
        return ElectromagneticForce // Medium-quality connections
    } else if reputation > 40 {
        return GravitationalForce // Basic connections
    } else {
        return WeakNuclearForce // Experimental or low-trust connections
    }
}

// connectionWorker processes connection requests
func (cm *ConnectionManager) connectionWorker() {
    for req := range cm.connectQueue {
        cm.processConnectionRequest(req)
    }
}

// processConnectionRequest handles individual connection attempts
func (cm *ConnectionManager) processConnectionRequest(req *ConnectionRequest) {
    // Acquire connection semaphore
    if !cm.connectionSemaphore.Acquire(req.Context) {
        req.ResultChan <- &ConnectionResult{
            Success: false,
            Error:   fmt.Errorf("connection limit reached"),
        }
        return
    }
    defer cm.connectionSemaphore.Release()

    // Check if connection already established
    cm.mu.RLock()
    if state, exists := cm.activeConnections[req.PeerInfo.NodeID]; exists {
        cm.mu.RUnlock()
        req.ResultChan <- &ConnectionResult{
            Success:    true,
            Connection: state,
        }
        return
    }
    cm.mu.RUnlock()

    // Attempt connection
    connection, err := cm.establishConnection(req)
    result := &ConnectionResult{
        Success:    err == nil,
        Connection: connection,
        Error:      err,
    }

    if err == nil {
        cm.finalizeConnection(connection, req.ForceType)
    } else {
        cm.handleConnectionFailure(req, err)
    }

    req.ResultChan <- result
}

// establishConnection performs the actual connection establishment
func (cm *ConnectionManager) establishConnection(req *ConnectionRequest) (*ConnectionState, error) {
    startTime := time.Now()
    
    // Select appropriate protocol handler
    handler, err := cm.selectProtocolHandler(req.PeerInfo.Protocol)
    if err != nil {
        return nil, fmt.Errorf("protocol selection failed: %w", err)
    }

    // Perform connection with timeout
    ctx, cancel := context.WithTimeout(req.Context, cm.config.ConnectionTimeout)
    defer cancel()

    // Establish connection through protocol handler
    connectionID, err := handler.Connect(ctx, req.PeerInfo.Address, req.PeerInfo.Port)
    if err != nil {
        return nil, fmt.Errorf("connection failed: %w", err)
    }

    // Perform security handshake
    if err := cm.network.securityManager.PerformHandshake(connectionID, req.PeerInfo); err != nil {
        handler.Disconnect(connectionID)
        return nil, fmt.Errorf("security handshake failed: %w", err)
    }

    // Create connection state
    latency := time.Since(startTime)
    state := &ConnectionState{
        PeerInfo:      req.PeerInfo,
        ConnectionID:  connectionID,
        Protocol:      req.PeerInfo.Protocol,
        Handler:       handler,
        EstablishedAt: time.Now(),
        LastActivity:  time.Now(),
        ConnectionForce: cm.calculateInitialForce(req.PeerInfo),
        Potential:     cm.calculateConnectionPotential(req.PeerInfo),
        Stability:     1.0,
        Metrics:       models.NewConnectionMetrics(),
        QualityScore:  cm.calculateQualityScore(req.PeerInfo, latency),
        FailureCount:  0,
    }

    // Update peer info with measured latency
    state.PeerInfo.Latency = latency
    state.PeerInfo.LastSeen = time.Now()
    state.PeerInfo.State = config.Ready

    return state, nil
}

// selectProtocolHandler chooses the appropriate protocol handler
func (cm *ConnectionManager) selectProtocolHandler(protocol config.ProtocolType) (protocols.ProtocolHandler, error) {
    switch protocol {
    case config.TCP:
        return cm.network.tcpHandler, nil
    case config.UDP:
        return cm.network.discoveryHandler, nil
    case config.WebSocket:
        return cm.network.apiHandler, nil
    default:
        return nil, fmt.Errorf("unsupported protocol: %s", protocol)
    }
}

// calculateQualityScore computes a comprehensive connection quality score
func (cm *ConnectionManager) calculateQualityScore(peer *models.PeerInfo, latency time.Duration) float64 {
    factors := map[string]float64{
        "latency":    1.0 / (1.0 + latency.Seconds()),
        "reputation": float64(peer.Reputation) / 100.0,
        "stake":      cm.network.stakeDistribution[peer.NodeID],
        "validator":  cm.network.validatorScores[peer.NodeID],
        "stability":  float64(peer.ConnectionCount) / float64(peer.ConnectionCount+peer.FailedAttempts+1),
    }

    // Weighted geometric mean
    weights := map[string]float64{
        "latency":    0.3,
        "reputation": 0.25,
        "stake":      0.2,
        "validator":  0.15,
        "stability":  0.1,
    }

    product := 1.0
    totalWeight := 0.0

    for factor, value := range factors {
        weight := weights[factor]
        product *= math.Pow(math.Max(0.01, value), weight)
        totalWeight += weight
    }

    return math.Pow(product, 1.0/totalWeight)
}

// finalizeConnection completes the connection establishment process
func (cm *ConnectionManager) finalizeConnection(state *ConnectionState, forceType ForceType) {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    // Add to active connections
    cm.activeConnections[state.PeerInfo.NodeID] = state
    cm.connectionMetrics[state.PeerInfo.NodeID] = state.Metrics

    // Update topology matrix
    cm.updateTopologyMatrix(state.PeerInfo.NodeID)

    // Initialize connection force
    cm.connectionForces[state.PeerInfo.NodeID] = state.ConnectionForce

    // Update energy calculations
    cm.updateEnergyCalculations()

    logrus.Infof("Connection established with %s (quality: %.3f, force: %.3f)",
        state.PeerInfo.NodeID, state.QualityScore, state.ConnectionForce.NetForce)
}

// handleConnectionFailure processes connection failures
func (cm *ConnectionManager) handleConnectionFailure(req *ConnectionRequest, err error) {
    req.PeerInfo.RecordFailure()
    
    // Update ban manager if necessary
    if req.RetryCount > 3 {
        cm.network.banManager.RecordFailure(req.PeerInfo.Address)
    }

    logrus.Debugf("Connection to %s failed: %v (retry %d)",
        req.PeerInfo.NodeID, err, req.RetryCount)
}

// DisconnectPeer gracefully disconnects from a peer
func (cm *ConnectionManager) DisconnectPeer(peerID string) error {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    state, exists := cm.activeConnections[peerID]
    if !exists {
        return fmt.Errorf("peer not connected: %s", peerID)
    }

    // Perform graceful disconnect
    if err := state.Handler.Disconnect(state.ConnectionID); err != nil {
        logrus.Warnf("Error disconnecting from %s: %v", peerID, err)
    }

    // Clean up connection state
    cm.cleanupConnection(state)

    logrus.Infof("Disconnected from peer: %s", peerID)
    return nil
}

// cleanupConnection removes connection state and updates topology
func (cm *ConnectionManager) cleanupConnection(state *ConnectionState) {
    delete(cm.activeConnections, state.PeerInfo.NodeID)
    delete(cm.connectionMetrics, state.PeerInfo.NodeID)
    delete(cm.connectionForces, state.PeerInfo.NodeID)

    // Update topology matrix
    cm.removeFromTopologyMatrix(state.PeerInfo.NodeID)

    // Recalculate energies
    cm.updateEnergyCalculations()
}

// updateTopologyMatrix adds a peer to the topology graph
func (cm *ConnectionManager) updateTopologyMatrix(peerID string) {
    cm.topologyMatrix.mu.Lock()
    defer cm.topologyMatrix.mu.Unlock()

    // Initialize adjacency row if needed
    if cm.topologyMatrix.Adjacency[peerID] == nil {
        cm.topologyMatrix.Adjacency[peerID] = make(map[string]float64)
    }

    // Set self-connection weight
    cm.topologyMatrix.Adjacency[cm.network.nodeID][peerID] = 1.0
    cm.topologyMatrix.Adjacency[peerID][cm.network.nodeID] = 1.0

    // Calculate initial distances
    cm.updateTopologyDistances()
}

// removeFromTopologyMatrix removes a peer from the topology graph
func (cm *ConnectionManager) removeFromTopologyMatrix(peerID string) {
    cm.topologyMatrix.mu.Lock()
    defer cm.topologyMatrix.mu.Unlock()

    // Remove connections to this peer
    delete(cm.topologyMatrix.Adjacency[cm.network.nodeID], peerID)
    delete(cm.topologyMatrix.Adjacency, peerID)

    // Remove distances
    delete(cm.topologyMatrix.Distance[cm.network.nodeID], peerID)
    delete(cm.topologyMatrix.Distance, peerID)

    // Recalculate centrality
    cm.calculateCentrality()
}

// updateTopologyDistances recalculates distances in the topology graph
func (cm *ConnectionManager) updateTopologyDistances() {
    // Simple distance calculation based on connection quality
    for peerID, state := range cm.activeConnections {
        distance := 1.0 / state.QualityScore
        cm.topologyMatrix.Distance[cm.network.nodeID][peerID] = distance
        cm.topologyMatrix.Distance[peerID][cm.network.nodeID] = distance
    }
}

// calculateCentrality computes eigenvector centrality for the topology
func (cm *ConnectionManager) calculateCentrality() {
    // Simple centrality calculation
    nodeCount := len(cm.topologyMatrix.Adjacency)
    if nodeCount == 0 {
        return
    }

    // Initialize centrality scores
    centrality := make(map[string]float64)
    for node := range cm.topologyMatrix.Adjacency {
        centrality[node] = 1.0 / float64(nodeCount)
    }

    // Power iteration for eigenvector centrality
    for iter := 0; iter < 100; iter++ {
        newCentrality := make(map[string]float64)
        var total float64

        for node := range centrality {
            var sum float64
            for neighbor, weight := range cm.topologyMatrix.Adjacency[node] {
                sum += weight * centrality[neighbor]
            }
            newCentrality[node] = sum
            total += sum
        }

        // Normalize
        for node := range newCentrality {
            centrality[node] = newCentrality[node] / total
        }
    }

    cm.topologyMatrix.Centrality = centrality
}

// forceCalculationWorker continuously updates connection forces
func (cm *ConnectionManager) forceCalculationWorker() {
    ticker := time.NewTicker(500 * time.Millisecond) // 2Hz force updates
    defer ticker.Stop()

    for range ticker.C {
        if !cm.isRunning.Load() {
            return
        }
        cm.recalculateForces()
    }
}

// recalculateForces updates all connection forces based on current state
func (cm *ConnectionManager) recalculateForces() {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    for peerID, state := range cm.activeConnections {
        force := cm.calculateDynamicForce(state)
        cm.connectionForces[peerID] = force
        state.ConnectionForce = force
        
        // Update connection potential
        state.Potential = cm.calculateConnectionPotential(state.PeerInfo)
    }

    cm.updateEnergyCalculations()
}

// calculateDynamicForce computes the current force for a connection
func (cm *ConnectionManager) calculateDynamicForce(state *ConnectionState) *ConnectionForce {
    baseForce := state.ConnectionForce
    
    // Dynamic adjustments based on real-time metrics
    metrics := state.Metrics.GetMetricsSnapshot()
    
    // Stability factor from success rate
    stabilityFactor := metrics.SuccessRate
    
    // Load factor from message rate
    loadFactor := 1.0 / (1.0 + metrics.MessageRate/1000.0)
    
    // Latency factor
    latencyFactor := 1.0 / (1.0 + metrics.GetAverageLatency().Seconds())
    
    // Adjust attraction based on dynamic factors
    dynamicAttraction := baseForce.AttractionMagnitude * stabilityFactor * loadFactor * latencyFactor
    
    // Adjust repulsion based on network entropy
    entropyRepulsion := baseForce.RepulsionMagnitude * cm.network.networkEntropy
    
    netForce := dynamicAttraction - entropyRepulsion
    
    return &ConnectionForce{
        AttractionMagnitude: dynamicAttraction,
        RepulsionMagnitude:  entropyRepulsion,
        NetForce:           netForce,
        Direction:          baseForce.Direction,
        Torque:             baseForce.Torque * stabilityFactor,
        LastUpdated:        time.Now(),
    }
}

// updateEnergyCalculations recalculates system energies
func (cm *ConnectionManager) updateEnergyCalculations() {
    var totalPotential float64
    var totalKinetic float64

    for _, force := range cm.connectionForces {
        // Potential energy from force magnitudes
        potential := math.Abs(force.NetForce)
        totalPotential += potential
        
        // Kinetic energy from force changes
        kinetic := force.Direction.Velocity * force.Direction.Velocity
        totalKinetic += kinetic
    }

    cm.potentialEnergy = totalPotential
    cm.kineticEnergy = totalKinetic
}

// topologyOptimizationWorker continuously optimizes network topology
func (cm *ConnectionManager) topologyOptimizationWorker() {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()

    for range ticker.C {
        if !cm.isRunning.Load() {
            return
        }
        cm.optimizeTopology()
    }
}

// optimizeTopology adjusts connections based on physics model
func (cm *ConnectionManager) optimizeTopology() {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    // Calculate connection scores using physics model
    connectionScores := make(map[string]float64)
    var totalScore float64

    for peerID, state := range cm.activeConnections {
        score := cm.calculateConnectionScore(state)
        connectionScores[peerID] = score
        totalScore += math.Exp(score)
    }

    // Determine connections to maintain or close
    for peerID, score := range connectionScores {
        probability := math.Exp(score) / totalScore
        
        if probability < 0.05 { // Very low probability
            cm.disconnectQueue <- peerID
        } else if probability < 0.2 { // Low probability
            // Reduce connection priority
            cm.adjustConnectionPriority(peerID, -1)
        }
    }

    // Process disconnections
    go cm.processDisconnections()
}

// calculateConnectionScore computes physics-based connection score
func (cm *ConnectionManager) calculateConnectionScore(state *ConnectionState) float64 {
    force := state.ConnectionForce
    metrics := state.Metrics.GetMetricsSnapshot()
    
    // Base score from force model
    forceScore := force.NetForce
    
    // Quality adjustments
    qualityScore := state.QualityScore
    stabilityScore := metrics.SuccessRate
    latencyScore := 1.0 / (1.0 + metrics.GetAverageLatency().Seconds())
    
    // Combined score with weights
    combinedScore := (forceScore * 0.4) + (qualityScore * 0.3) + 
                    (stabilityScore * 0.2) + (latencyScore * 0.1)
    
    // Adjust for network entropy
    entropyAdjusted := combinedScore * (1.0 - cm.network.networkEntropy*0.5)
    
    return entropyAdjusted
}

// processDisconnections handles queued disconnections
func (cm *ConnectionManager) processDisconnections() {
    for {
        select {
        case peerID, ok := <-cm.disconnectQueue:
            if !ok {
                return
            }
            cm.DisconnectPeer(peerID)
        default:
            return
        }
    }
}

// healthMonitoringWorker monitors connection health
func (cm *ConnectionManager) healthMonitoringWorker() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for range ticker.C {
        if !cm.isRunning.Load() {
            return
        }
        cm.checkConnectionHealth()
    }
}

// checkConnectionHealth verifies all active connections
func (cm *ConnectionManager) checkConnectionHealth() {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    currentTime := time.Now()
    var unhealthy []string

    for peerID, state := range cm.activeConnections {
        // Check for stale connections
        if currentTime.Sub(state.LastActivity) > cm.config.ConnectionTimeout {
            unhealthy = append(unhealthy, peerID)
            continue
        }

        // Check connection quality
        if state.QualityScore < 0.3 {
            unhealthy = append(unhealthy, peerID)
            continue
        }

        // Check metrics for degradation
        metrics := state.Metrics.GetMetricsSnapshot()
        if metrics.SuccessRate < 0.5 || metrics.ErrorCount > 10 {
            unhealthy = append(unhealthy, peerID)
        }
    }

    // Queue unhealthy connections for disconnection
    for _, peerID := range unhealthy {
        cm.disconnectQueue <- peerID
    }
}

// GetConnectionState returns the state of a specific connection
func (cm *ConnectionManager) GetConnectionState(peerID string) *ConnectionState {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return cm.activeConnections[peerID]
}

// GetActivePeers returns all active connections
func (cm *ConnectionManager) GetActivePeers() map[string]*models.PeerInfo {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    peers := make(map[string]*models.PeerInfo)
    for peerID, state := range cm.activeConnections {
        peers[peerID] = state.PeerInfo
    }
    return peers
}

// IsConnected checks if connected to a specific peer
func (cm *ConnectionManager) IsConnected(peerID string) bool {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    _, exists := cm.activeConnections[peerID]
    return exists
}

// ConnectionCount returns the number of active connections
func (cm *ConnectionManager) ConnectionCount() int {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return len(cm.activeConnections)
}

// GetTopologyMetrics returns topology analysis metrics
func (cm *ConnectionManager) GetTopologyMetrics() *TopologyMetrics {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    return &TopologyMetrics{
        NodeCount:      len(cm.activeConnections),
        PotentialEnergy: cm.potentialEnergy,
        KineticEnergy:  cm.kineticEnergy,
        AverageQuality: cm.calculateAverageQuality(),
        Centrality:     cm.topologyMatrix.Centrality[cm.network.nodeID],
        ConnectionForces: cm.connectionForces,
        Timestamp:      time.Now(),
    }
}

// calculateAverageQuality computes the average connection quality
func (cm *ConnectionManager) calculateAverageQuality() float64 {
    if len(cm.activeConnections) == 0 {
        return 0
    }

    var total float64
    for _, state := range cm.activeConnections {
        total += state.QualityScore
    }

    return total / float64(len(cm.activeConnections))
}

// calculateNetworkDistance estimates network distance to a peer
func (cm *ConnectionManager) calculateNetworkDistance(peerID string) float64 {
    // Use topology matrix distance if available
    if dist, exists := cm.topologyMatrix.Distance[cm.network.nodeID][peerID]; exists {
        return dist
    }
    
    // Fallback to simple estimation
    return 1.0 // Default distance
}

// adjustConnectionPriority modifies connection priority
func (cm *ConnectionManager) adjustConnectionPriority(peerID string, delta int) {
    // Implementation for dynamic priority adjustment
    // This would affect connection maintenance and resource allocation
}

// CloseAll disconnects all active connections
func (cm *ConnectionManager) CloseAll() {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    for peerID, state := range cm.activeConnections {
        state.Handler.Disconnect(state.ConnectionID)
        delete(cm.activeConnections, peerID)
        delete(cm.connectionMetrics, peerID)
        delete(cm.connectionForces, peerID)
    }

    // Reset topology
    cm.topologyMatrix.Adjacency = make(map[string]map[string]float64)
    cm.topologyMatrix.Distance = make(map[string]map[string]float64)
    cm.topologyMatrix.Centrality = make(map[string]float64)

    cm.potentialEnergy = 0
    cm.kineticEnergy = 0
}

// TopologyMetrics represents topology analysis results
type TopologyMetrics struct {
    NodeCount       int
    PotentialEnergy float64
    KineticEnergy   float64
    AverageQuality  float64
    Centrality      float64
    ConnectionForces map[string]*ConnectionForce
    Timestamp       time.Time
}