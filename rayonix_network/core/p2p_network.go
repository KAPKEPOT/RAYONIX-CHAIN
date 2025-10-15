package core

import (
    "context"
    "crypto/ecdsa"
    "fmt"
    "math"
    "math/rand"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/models"
    "github.com/rayxnetwork/p2p/protocols"
    "github.com/rayxnetwork/p2p/utils"
    "github.com/sirupsen/logrus"
)

// ConsensusCallback defines the interface for consensus network interactions
type ConsensusCallback interface {
    OnBlockProposal(block *models.BlockData) error
    OnValidatorScoreUpdate(validator string, score float64) error
    OnNetworkMetrics(metrics *models.NetworkMetrics) error
    GetValidatorScores() map[string]float64
    GetStakeDistribution() map[string]float64
}

// AdvancedP2PNetwork orchestrates the physics-inspired P2P substrate
type AdvancedP2PNetwork struct {
    nodeID      string
    privateKey  *ecdsa.PrivateKey
    config      *config.NodeConfig
    isRunning   atomic.Bool
    startTime   time.Time

    // Core subsystems
    connectionManager *ConnectionManager
    peerDiscovery    *PeerDiscovery
    messageProcessor *MessageProcessor
    securityManager  *SecurityManager

    // Protocol handlers
    tcpHandler      *protocols.TCPHandler
    discoveryHandler *protocols.DiscoveryHandler
    apiHandler      *protocols.APIHandler

    // Utilities
    rateLimiter     *utils.RateLimiter
    banManager      *utils.BanManager
    metricsCollector *utils.MetricsCollector
    dht             *utils.DHT

    // Physics-inspired state
    peerPotential   map[string]float64 // Potential energy for each peer connection
    networkEntropy  float64           // Current network entropy state
    connectionField *VectorField      // Dynamic connection force field

    // Consensus coupling
    consensusCallback ConsensusCallback
    validatorScores   map[string]float64
    stakeDistribution map[string]float64

    // Concurrency control
    mu            sync.RWMutex
    ctx           context.Context
    cancel        context.CancelFunc
    workerGroup   sync.WaitGroup
    messageQueue  chan *models.NetworkMessage
    controlQueue  chan ControlMessage

    // Performance tracking
    messageLatency   *utils.RollingStatistics
    connectionHealth *utils.HealthMonitor
}

// VectorField represents the physics-inspired connection force field
type VectorField struct {
    forces    map[string]*ForceVector
    mu        sync.RWMutex
    fieldType FieldType
}

type ForceVector struct {
    Attraction float64   `json:"attraction"`
    Repulsion  float64   `json:"repulsion"`
    Direction  []float64 `json:"direction"` // Normalized direction vector
    Magnitude  float64   `json:"magnitude"`
}

type FieldType int

const (
    Gravitational FieldType = iota
    Electromagnetic
    Thermodynamic
)

type ControlMessage struct {
    Type    ControlMessageType
    Payload interface{}
}

type ControlMessageType int

const (
    TopologyUpdate ControlMessageType = iota
    EntropyInjection
    ForceRecalculation
    ConsensusSync
    PeerEviction
)

// NewAdvancedP2PNetwork creates a new physics-inspired P2P network instance
func NewAdvancedP2PNetwork(cfg *config.NodeConfig, privKey *ecdsa.PrivateKey, consensusCallback ConsensusCallback) (*AdvancedP2PNetwork, error) {
    if cfg == nil {
        cfg = config.DefaultConfig()
    }

    ctx, cancel := context.WithCancel(context.Background())

    network := &AdvancedP2PNetwork{
        nodeID:            generateNodeID(privKey),
        privateKey:        privKey,
        config:            cfg,
        consensusCallback: consensusCallback,
        validatorScores:   make(map[string]float64),
        stakeDistribution: make(map[string]float64),
        peerPotential:     make(map[string]float64),
        networkEntropy:    1.0, // Initial entropy state
        connectionField: &VectorField{
            forces:    make(map[string]*ForceVector),
            fieldType: Electromagnetic,
        },
        ctx:          ctx,
        cancel:       cancel,
        messageQueue: make(chan *models.NetworkMessage, 10000),
        controlQueue: make(chan ControlMessage, 1000),
        messageLatency: utils.NewRollingStatistics(1000),
        connectionHealth: utils.NewHealthMonitor(),
    }

    // Initialize utilities
    network.rateLimiter = utils.NewRateLimiter(cfg.RateLimitPerPeer)
    network.banManager = utils.NewBanManager(cfg.BanThreshold, cfg.BanDuration)
    network.metricsCollector = utils.NewMetricsCollector()
    network.dht = utils.NewDHT(network.nodeID, cfg.DHTBootstrapNodes)

    // Initialize core subsystems
    network.securityManager = NewSecurityManager(network, privKey)
    network.connectionManager = NewConnectionManager(network)
    network.peerDiscovery = NewPeerDiscovery(network)
    network.messageProcessor = NewMessageProcessor(network)

    // Initialize protocol handlers
    network.tcpHandler = protocols.NewTCPHandler(network, cfg)
    network.discoveryHandler = protocols.NewDiscoveryHandler(network, cfg)
    network.apiHandler = protocols.NewAPIHandler(network, cfg)

    return network, nil
}

// Start initializes and starts all network subsystems
func (n *AdvancedP2PNetwork) Start() error {
    if n.isRunning.Swap(true) {
        return fmt.Errorf("network is already running")
    }

    n.startTime = time.Now()
    logrus.Info("Starting Advanced P2P Network with physics-inspired substrate")

    // Initialize security layer
    if err := n.securityManager.Initialize(); err != nil {
        return fmt.Errorf("security manager initialization failed: %w", err)
    }

    // Start protocol handlers
    if err := n.tcpHandler.Start(); err != nil {
        return fmt.Errorf("TCP handler start failed: %w", err)
    }

    if err := n.discoveryHandler.Start(); err != nil {
        return fmt.Errorf("discovery handler start failed: %w", err)
    }

    if err := n.apiHandler.Start(); err != nil {
        return fmt.Errorf("API handler start failed: %w", err)
    }

    // Start worker goroutines
    n.startWorkers()

    // Bootstrap network topology
    if err := n.peerDiscovery.Bootstrap(); err != nil {
        logrus.Warnf("Initial bootstrap failed: %v", err)
    }

    // Initialize physics model
    n.initializePhysicsModel()

    logrus.Infof("Advanced P2P Network started successfully. NodeID: %s", n.nodeID)
    return nil
}

// Stop gracefully shuts down the network
func (n *AdvancedP2PNetwork) Stop() error {
    if !n.isRunning.Swap(false) {
        return fmt.Errorf("network is not running")
    }

    logrus.Info("Shutting down Advanced P2P Network")

    // Cancel context to signal shutdown
    n.cancel()

    // Stop protocol handlers
    n.tcpHandler.Stop()
    n.discoveryHandler.Stop()
    n.apiHandler.Stop()

    // Close all connections
    n.connectionManager.CloseAll()

    // Wait for workers to complete
    n.workerGroup.Wait()

    // Close channels
    close(n.messageQueue)
    close(n.controlQueue)

    logrus.Info("Advanced P2P Network stopped successfully")
    return nil
}

// startWorkers initiates all background processing goroutines
func (n *AdvancedP2PNetwork) startWorkers() {
    // Message processing worker
    n.workerGroup.Add(1)
    go n.messageWorker()

    // Physics model worker
    n.workerGroup.Add(1)
    go n.physicsWorker()

    // Topology optimization worker
    n.workerGroup.Add(1)
    go n.topologyWorker()

    // Consensus sync worker
    n.workerGroup.Add(1)
    go n.consensusSyncWorker()

    // Metrics collection worker
    n.workerGroup.Add(1)
    go n.metricsWorker()
}

// messageWorker processes incoming network messages
func (n *AdvancedP2PNetwork) messageWorker() {
    defer n.workerGroup.Done()

    for {
        select {
        case <-n.ctx.Done():
            return
        case msg, ok := <-n.messageQueue:
            if !ok {
                return
            }
            n.processIncomingMessage(msg)
        }
    }
}

// physicsWorker maintains the physics-inspired network model
func (n *AdvancedP2PNetwork) physicsWorker() {
    defer n.workerGroup.Done()

    ticker := time.NewTicker(100 * time.Millisecond) // 10Hz physics update
    defer ticker.Stop()

    for {
        select {
        case <-n.ctx.Done():
            return
        case <-ticker.C:
            n.updatePhysicsModel()
        case controlMsg := <-n.controlQueue:
            n.handleControlMessage(controlMsg)
        }
    }
}

// topologyWorker optimizes network topology based on physics model
func (n *AdvancedP2PNetwork) topologyWorker() {
    defer n.workerGroup.Done()

    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-n.ctx.Done():
            return
        case <-ticker.C:
            n.optimizeTopology()
        }
    }
}

// consensusSyncWorker synchronizes with consensus layer
func (n *AdvancedP2PNetwork) consensusSyncWorker() {
    defer n.workerGroup.Done()

    ticker := time.NewTicker(2 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-n.ctx.Done():
            return
        case <-ticker.C:
            n.syncWithConsensus()
        }
    }
}

// metricsWorker collects and reports network metrics
func (n *AdvancedP2PNetwork) metricsWorker() {
    defer n.workerGroup.Done()

    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-n.ctx.Done():
            return
        case <-ticker.C:
            n.collectMetrics()
        }
    }
}

// initializePhysicsModel sets up the initial physics-inspired network state
func (n *AdvancedP2PNetwork) initializePhysicsModel() {
    n.mu.Lock()
    defer n.mu.Unlock()

    // Initialize with maximum entropy
    n.networkEntropy = 1.0

    // Set initial potentials based on configuration
    for peerID := range n.connectionManager.GetActivePeers() {
        n.peerPotential[peerID] = n.calculateInitialPotential(peerID)
    }

    logrus.Info("Physics model initialized")
}

// updatePhysicsModel evolves the network according to physics principles
func (n *AdvancedP2PNetwork) updatePhysicsModel() {
    n.mu.Lock()
    defer n.mu.Unlock()

    // Update force field based on current state
    n.updateForceField()

    // Evolve potentials according to field equations
    n.evolvePotentials()

    // Update network entropy
    n.updateEntropy()

    // Apply stochastic fluctuations
    n.applyStochasticForces()
}

// updateForceField recalculates the connection force field
func (n *AdvancedP2PNetwork) updateForceField() {
    activePeers := n.connectionManager.GetActivePeers()
    
    for peerID, peerInfo := range activePeers {
        force := n.calculatePeerForce(peerID, peerInfo)
        n.connectionField.mu.Lock()
        n.connectionField.forces[peerID] = force
        n.connectionField.mu.Unlock()
    }
}

// calculatePeerForce computes the force vector for a peer connection
func (n *AdvancedP2PNetwork) calculatePeerForce(peerID string, peerInfo *models.PeerInfo) *ForceVector {
    // Base attraction from validator score and stake
    validatorScore := n.validatorScores[peerID]
    stake := n.stakeDistribution[peerID]
    
    // Network quality factors
    latencyWeight := 1.0 / (1.0 + peerInfo.Latency.Seconds())
    reliabilityWeight := peerInfo.Reputation / 100.0
    
    // Calculate attraction force (gravitational analogy)
    attraction := validatorScore * stake * latencyWeight * reliabilityWeight
    
    // Calculate repulsion force (entropic pressure)
    repulsion := n.networkEntropy * (1.0 - reliabilityWeight)
    
    // Net force magnitude
    magnitude := attraction - repulsion
    
    // Direction vector (simplified 2D for now)
    direction := []float64{
        rand.Float64()*2 - 1, // Random direction component
        rand.Float64()*2 - 1,
    }
    
    // Normalize direction
    norm := math.Sqrt(direction[0]*direction[0] + direction[1]*direction[1])
    if norm > 0 {
        direction[0] /= norm
        direction[1] /= norm
    }
    
    return &ForceVector{
        Attraction: attraction,
        Repulsion:  repulsion,
        Direction:  direction,
        Magnitude:  magnitude,
    }
}

// evolvePotentials updates peer potentials based on force field
func (n *AdvancedP2PNetwork) evolvePotentials() {
    deltaTime := 0.1 // Fixed time step for numerical integration
    
    for peerID, force := range n.connectionField.forces {
        currentPotential := n.peerPotential[peerID]
        
        // Euler integration of potential evolution
        // dP/dt = -∇V + η (stochastic term)
        potentialDerivative := -force.Magnitude
        
        // Add small stochastic fluctuation
        stochasticTerm := rand.NormFloat64() * 0.1 * n.networkEntropy
        
        newPotential := currentPotential + deltaTime*(potentialDerivative+stochasticTerm)
        
        // Clamp potential to reasonable bounds
        n.peerPotential[peerID] = math.Max(-10.0, math.Min(10.0, newPotential))
    }
}

// updateEntropy recalculates network entropy
func (n *AdvancedP2PNetwork) updateEntropy() {
    totalPeers := len(n.peerPotential)
    if totalPeers == 0 {
        n.networkEntropy = 1.0
        return
    }
    
    // Calculate entropy based on potential distribution
    var entropy float64
    for _, potential := range n.peerPotential {
        probability := math.Exp(-potential) / float64(totalPeers)
        if probability > 0 {
            entropy -= probability * math.Log(probability)
        }
    }
    
    // Normalize entropy to [0,1] range
    maxEntropy := math.Log(float64(totalPeers))
    if maxEntropy > 0 {
        n.networkEntropy = entropy / maxEntropy
    } else {
        n.networkEntropy = 0
    }
}

// applyStochasticForces injects random fluctuations into the system
func (n *AdvancedP2PNetwork) applyStochasticForces() {
    // Wiener process increments for SDE
    for peerID := range n.peerPotential {
        wienerIncrement := rand.NormFloat64() * math.Sqrt(0.1) // sqrt(dt)
        n.peerPotential[peerID] += 0.1 * wienerIncrement // σ * dW
    }
}

// optimizeTopology adjusts connections based on physics model
func (n *AdvancedP2PNetwork) optimizeTopology() {
    n.mu.RLock()
    defer n.mu.RUnlock()

    activePeers := n.connectionManager.GetActivePeers()
    
    // Calculate connection scores using softmax distribution
    connectionScores := make(map[string]float64)
    var totalWeight float64
    
    for peerID, peerInfo := range activePeers {
        score := n.calculateConnectionScore(peerID, peerInfo)
        connectionScores[peerID] = score
        totalWeight += math.Exp(score)
    }
    
    // Determine which connections to maintain
    for peerID, score := range connectionScores {
        probability := math.Exp(score) / totalWeight
        
        // Connection decision based on physics model
        if probability < 0.1 { // Low probability connection
            n.connectionManager.DisconnectPeer(peerID)
        }
    }
    
    // Attempt new connections based on force field
    n.attemptNewConnections()
}

// calculateConnectionScore computes the softmax-weighted connection score
func (n *AdvancedP2PNetwork) calculateConnectionScore(peerID string, peerInfo *models.PeerInfo) float64 {
    // Base score from physics model
    potential := n.peerPotential[peerID]
    force := n.connectionField.forces[peerID]
    
    if force == nil {
        return 0
    }
    
    // Temperature parameter for softmax (higher = more random)
    temperature := 1.0 / (1.0 + n.networkEntropy)
    
    // Combined score
    score := (force.Attraction - force.Repulsion + potential) / temperature
    
    return score
}

// attemptNewConnections tries to establish new connections based on physics model
func (n *AdvancedP2PNetwork) attemptNewConnections() {
    discoveredPeers := n.peerDiscovery.GetDiscoveredPeers()
    
    for _, peer := range discoveredPeers {
        if n.shouldConnectToPeer(peer) {
            n.connectionManager.ConnectToPeer(peer)
        }
    }
}

// shouldConnectToPeer determines if connection should be attempted
func (n *AdvancedP2PNetwork) shouldConnectToPeer(peer *models.PeerInfo) bool {
    // Check if already connected
    if n.connectionManager.IsConnected(peer.NodeID) {
        return false
    }
    
    // Check ban status
    if n.banManager.IsBanned(peer.Address) {
        return false
    }
    
    // Physics-based decision
    simulatedPotential := n.simulateConnectionPotential(peer)
    return simulatedPotential > 0.5 // Connection threshold
}

// simulateConnectionPotential estimates potential for new connection
func (n *AdvancedP2PNetwork) simulateConnectionPotential(peer *models.PeerInfo) float64 {
    // Use available information to estimate connection quality
    validatorScore := n.validatorScores[peer.NodeID]
    stake := n.stakeDistribution[peer.NodeID]
    
    // Estimate based on peer reputation and network state
    basePotential := (validatorScore + stake) / 2.0
    entropyAdjustment := 1.0 - n.networkEntropy // Lower entropy favors connections
    
    return basePotential * entropyAdjustment
}

// processIncomingMessage handles incoming network messages with physics-aware routing
func (n *AdvancedP2PNetwork) processIncomingMessage(msg *models.NetworkMessage) {
    start := time.Now()
    
    // Update message latency statistics
    latency := time.Since(msg.Timestamp)
    n.messageLatency.Add(latency.Seconds())
    
    // Process message through security layer
    if err := n.securityManager.VerifyMessage(msg); err != nil {
        logrus.Warnf("Message verification failed: %v", err)
        return
    }
    
    // Route to appropriate handler with physics-aware prioritization
    n.routeMessageWithEntropy(msg)
    
    // Update metrics
    n.metricsCollector.RecordMessageProcessed(latency)
}

// routeMessageWithEntropy uses entropy-weighted routing decisions
func (n *AdvancedP2PNetwork) routeMessageWithEntropy(msg *models.NetworkMessage) {
    // Determine routing strategy based on message type and network state
    switch msg.MessageType {
    case config.Block, config.Transaction, config.Consensus:
        n.routeCriticalMessage(msg)
    case config.Gossip, config.PeerList:
        n.routeWithStochasticForwarding(msg)
    default:
        n.messageProcessor.ProcessMessage(msg)
    }
}

// routeCriticalMessage routes high-priority messages with reliability
func (n *AdvancedP2PNetwork) routeCriticalMessage(msg *models.NetworkMessage) {
    // Use most reliable paths for critical messages
    reliablePeers := n.selectReliablePeersForRouting()
    n.messageProcessor.ProcessMessageWithRoute(msg, reliablePeers)
}

// routeWithStochasticForwarding uses entropy-driven message propagation
func (n *AdvancedP2PNetwork) routeWithStochasticForwarding(msg *models.NetworkMessage) {
    // Softmax-weighted peer selection for gossip
    peers := n.connectionManager.GetActivePeers()
    forwardProbabilities := n.calculateForwardProbabilities(peers)
    
    for peerID, probability := range forwardProbabilities {
        if rand.Float64() < probability {
            n.messageProcessor.ForwardMessage(msg, peerID)
        }
    }
}

// calculateForwardProbabilities computes softmax forwarding probabilities
func (n *AdvancedP2PNetwork) calculateForwardProbabilities(peers map[string]*models.PeerInfo) map[string]float64 {
    scores := make(map[string]float64)
    var total float64
    
    for peerID, peer := range peers {
        score := n.calculateForwardingScore(peer)
        expScore := math.Exp(score / n.networkEntropy) // Temperature = entropy
        scores[peerID] = expScore
        total += expScore
    }
    
    // Normalize to probabilities
    probabilities := make(map[string]float64)
    for peerID, score := range scores {
        probabilities[peerID] = score / total
    }
    
    return probabilities
}

// calculateForwardingScore determines forwarding score for a peer
func (n *AdvancedP2PNetwork) calculateForwardingScore(peer *models.PeerInfo) float64 {
    return float64(peer.Reputation)/100.0 + 
           (1.0 / (1.0 + peer.Latency.Seconds())) +
           n.validatorScores[peer.NodeID]
}

// selectReliablePeersForRouting chooses peers for reliable message delivery
func (n *AdvancedP2PNetwork) selectReliablePeersForRouting() []string {
    peers := n.connectionManager.GetActivePeers()
    var reliablePeers []string
    
    for peerID, peer := range peers {
        if peer.Reputation >= 80 && peer.Latency < time.Second {
            reliablePeers = append(reliablePeers, peerID)
        }
    }
    
    return reliablePeers
}

// syncWithConsensus updates network state from consensus layer
func (n *AdvancedP2PNetwork) syncWithConsensus() {
    if n.consensusCallback == nil {
        return
    }
    
    // Get latest validator scores and stake distribution
    n.validatorScores = n.consensusCallback.GetValidatorScores()
    n.stakeDistribution = n.consensusCallback.GetStakeDistribution()
    
    // Send network metrics to consensus
    metrics := n.collectNetworkMetrics()
    n.consensusCallback.OnNetworkMetrics(metrics)
}

// collectMetrics gathers and reports system metrics
func (n *AdvancedP2PNetwork) collectMetrics() {
    metrics := n.collectNetworkMetrics()
    n.metricsCollector.ExportMetrics(metrics)
}

// collectNetworkMetrics comp comprehensive network metrics
func (n *AdvancedP2PNetwork) collectNetworkMetrics() *models.NetworkMetrics {
    n.mu.RLock()
    defer n.mu.RUnlock()
    
    return &models.NetworkMetrics{
        NodeID:              n.nodeID,
        Uptime:              time.Since(n.startTime),
        ActiveConnections:   n.connectionManager.ConnectionCount(),
        NetworkEntropy:      n.networkEntropy,
        AverageLatency:      time.Duration(n.messageLatency.Mean() * float64(time.Second)),
        MessageThroughput:   n.metricsCollector.GetMessageRate(),
        PeerPotentials:      n.peerPotential,
        ValidatorScores:     n.validatorScores,
        StakeDistribution:   n.stakeDistribution,
        Timestamp:           time.Now(),
    }
}

// handleControlMessage processes control plane messages
func (n *AdvancedP2PNetwork) handleControlMessage(msg ControlMessage) {
    switch msg.Type {
    case TopologyUpdate:
        n.optimizeTopology()
    case EntropyInjection:
        n.injectEntropy(msg.Payload.(float64))
    case ForceRecalculation:
        n.updateForceField()
    case ConsensusSync:
        n.syncWithConsensus()
    case PeerEviction:
        n.connectionManager.DisconnectPeer(msg.Payload.(string))
    }
}

// injectEntropy artificially increases network entropy
func (n *AdvancedP2PNetwork) injectEntropy(amount float64) {
    n.mu.Lock()
    defer n.mu.Unlock()
    
    n.networkEntropy = math.Min(1.0, n.networkEntropy+amount)
    
    // Apply entropy injection to peer potentials
    for peerID := range n.peerPotential {
        n.peerPotential[peerID] += rand.NormFloat64() * amount * 0.1
    }
}

// calculateInitialPotential computes initial potential for a peer
func (n *AdvancedP2PNetwork) calculateInitialPotential(peerID string) float64 {
    // Base potential from available information
    validatorScore := n.validatorScores[peerID]
    stake := n.stakeDistribution[peerID]
    
    return (validatorScore + stake) / 2.0
}

// generateNodeID creates a unique node identifier from private key
func generateNodeID(privKey *ecdsa.PrivateKey) string {
    pubKey := &privKey.PublicKey
    // Use compressed public key hash as node ID
    pubKeyBytes := append(pubKey.X.Bytes(), pubKey.Y.Bytes()...)
    hash := utils.SHA3Hash(pubKeyBytes)
    return fmt.Sprintf("node_%x", hash[:16])
}

// GetNetworkState returns current physics-inspired network state
func (n *AdvancedP2PNetwork) GetNetworkState() *models.NetworkState {
    n.mu.RLock()
    defer n.mu.RUnlock()
    
    return &models.NetworkState{
        NodeID:            n.nodeID,
        NetworkEntropy:    n.networkEntropy,
        PeerPotentials:    n.peerPotential,
        ConnectionForces:  n.connectionField.forces,
        ValidatorScores:   n.validatorScores,
        ActiveConnections: n.connectionManager.ConnectionCount(),
        Uptime:           time.Since(n.startTime),
        Timestamp:        time.Now(),
    }
}

// BroadcastMessage sends message with physics-aware propagation
func (n *AdvancedP2PNetwork) BroadcastMessage(msg *models.NetworkMessage) error {
    return n.messageProcessor.BroadcastWithEntropy(msg, n.networkEntropy)
}

// SendMessageToPeer delivers message to specific peer with reliability
func (n *AdvancedP2PNetwork) SendMessageToPeer(peerID string, msg *models.NetworkMessage) error {
    return n.messageProcessor.SendToPeer(peerID, msg)
}

// GetActivePeers returns currently connected peers
func (n *AdvancedP2PNetwork) GetActivePeers() map[string]*models.PeerInfo {
    return n.connectionManager.GetActivePeers()
}

// IsConnected checks if connected to specific peer
func (n *AdvancedP2PNetwork) IsConnected(peerID string) bool {
    return n.connectionManager.IsConnected(peerID)
}

// GetNetworkMetrics returns current network performance metrics
func (n *AdvancedP2PNetwork) GetNetworkMetrics() *models.NetworkMetrics {
    return n.collectNetworkMetrics()
}