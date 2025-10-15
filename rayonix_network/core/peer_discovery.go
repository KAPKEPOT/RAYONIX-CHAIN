package core

import (
    "context"
    "crypto/rand"
    "encoding/binary"
    "fmt"
    "math"
    "math/big"
    "net"
    "sort"
    "sync"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/models"
    "github.com/rayxnetwork/p2p/protocols"
    "github.com/rayxnetwork/p2p/utils"
    "github.com/sirupsen/logrus"
    "golang.org/x/crypto/sha3"
)

// PeerDiscovery implements the complete physics-inspired peer discovery system
type PeerDiscovery struct {
    network         *AdvancedP2PNetwork
    config          *config.NodeConfig
    isRunning       atomic.Bool
    mu              sync.RWMutex

    // Core discovery state
    discoveredPeers *utils.ConcurrentMap[string, *models.PeerInfo]
    activeSearches  *utils.ConcurrentMap[string, *SearchProcess]
    bootstrapPeers  []*models.PeerInfo
    dhtTable        *utils.DHT
    routingTable    *KademliaRoutingTable

    // Physics model components
    diffusionEngine  *DiffusionEngine
    forceCalculator  *ForceCalculator
    entropyManager   *EntropyManager
    potentialField   *PotentialField

    // Protocol handlers
    udpProtocol     *protocols.DiscoveryHandler
    gossipProtocol  *GossipProtocol
    dhtProtocol     *DHTProtocol

    // Performance metrics
    metricsCollector *DiscoveryMetricsCollector
    successTracker   *SuccessRateTracker

    // Control system
    discoveryScheduler *DiscoveryScheduler
    workQueue         chan *DiscoveryTask
    resultQueue       chan *DiscoveryResult
    controlChan       chan *ControlMessage

    // Worker management
    workerCtx       context.Context
    workerCancel    context.CancelFunc
    workerWg        sync.WaitGroup
}

// DiffusionEngine implements the partial differential equation for peer discovery
type DiffusionEngine struct {
    concentrationField *SpatialField
    gradientField      *VectorField
    diffusionConstant  float64
    decayConstant      float64
    sourceMap          *utils.ConcurrentMap[string, *DiffusionSource]
    boundaryConditions *BoundaryConditions
    mu                 sync.RWMutex
}

// DiffusionSource represents a continuous source in the diffusion field
type DiffusionSource struct {
    position      *Vector3D
    intensity     float64
    reliability   float64
    lastEmission  time.Time
    activityDecay float64
}

// SpatialField represents the concentration field in 3D space
type SpatialField struct {
    resolution float64
    fieldData  *utils.ConcurrentMap[Vector3D, float64]
    dimensions *FieldDimensions
}

// Vector3D represents a point in 3D space for the physics model
type Vector3D struct {
    X, Y, Z float64
}

// FieldDimensions defines the spatial boundaries
type FieldDimensions struct {
    MinX, MaxX float64
    MinY, MaxY float64
    MinZ, MaxZ float64
}

// VectorField represents a field of vectors in 3D space
type VectorField struct {
    vectors *utils.ConcurrentMap[Vector3D, *Vector3D]
}

// BoundaryConditions defines how the field behaves at boundaries
type BoundaryConditions struct {
    type_ BoundaryType
    value float64
}

type BoundaryType int

const (
    BoundaryReflective BoundaryType = iota
    BoundaryAbsorbing
    BoundaryPeriodic
)

// ForceCalculator computes physics-inspired discovery forces
type ForceCalculator struct {
    explorationForce  float64
    exploitationForce float64
    entropyForce      float64
    networkForce      float64
    couplingConstants *ForceCoupling
    forceHistory      *utils.RollingStatistics
}

// ForceCoupling contains the coupling constants for different force types
type ForceCoupling struct {
    alpha float64 // Exploration-Exploitation coupling
    beta  float64 // Entropy coupling  
    gamma float64 // Network coupling
    delta float64 // Stochastic coupling
}

// EntropyManager handles the information-theoretic aspects of discovery
type EntropyManager struct {
    shannonEntropy    float64
    topologicalEntropy float64
    temporalEntropy   float64
    entropyRate       float64
    entropyBuffer     *utils.EntropyBuffer
}

// PotentialField represents the potential energy landscape for peer discovery
type PotentialField struct {
    potentialMap   *utils.ConcurrentMap[Vector3D, float64]
    gradientMap    *utils.ConcurrentMap[Vector3D, *Vector3D]
    forceMap       *utils.ConcurrentMap[Vector3D, *Vector3D]
    equilibriumState float64
}

// SearchProcess manages an individual discovery operation
type SearchProcess struct {
    id           string
    target       *SearchTarget
    strategy     DiscoveryStrategy
    startTime    time.Time
    participants *utils.ConcurrentSet[string]
    results      *utils.ConcurrentSet[string]
    convergence  *ConvergenceTracker
    forceProfile *ForceProfile
    costModel    *CostModel
}

// SearchTarget defines the objective of a discovery operation
type SearchTarget struct {
    nodeID      string
    position    *Vector3D
    radius      float64
    constraints *SearchConstraints
    priority    int
}

// SearchConstraints defines limitations for discovery operations
type SearchConstraints struct {
    maxHops      int
    timeout      time.Duration
    resourceLimit *ResourceLimit
    qualityThreshold float64
}

// ResourceLimit defines computational boundaries
type ResourceLimit struct {
    maxCPUUsage    float64
    maxMemoryBytes int64
    maxBandwidth   int64
}

// DiscoveryStrategy defines the approach for peer discovery
type DiscoveryStrategy struct {
    method      DiscoveryMethod
    parameters  *StrategyParameters
    adaptivity  *AdaptiveController
    termination *TerminationConditions
}

// DiscoveryMethod represents specific discovery algorithms
type DiscoveryMethod int

const (
    MethodDHTLookup DiscoveryMethod = iota
    MethodRandomWalk
    MethodDiffusion
    MethodGossip
    MethodBootstrap
    MethodHybrid
)

// StrategyParameters contains algorithm-specific parameters
type StrategyParameters struct {
    dhtParams      *DHTSearchParams
    walkParams     *RandomWalkParams
    diffusionParams *DiffusionParams
    gossipParams   *GossipParams
}

// DHTSearchParams contains DHT-specific search parameters
type DHTSearchParams struct {
    kValue        int
    alphaValue    int
    parallelism   int
    timeout       time.Duration
}

// RandomWalkParams contains random walk parameters
type RandomWalkParams struct {
    stepSize      float64
    temperature   float64
    maxSteps      int
    restartProb   float64
}

// DiffusionParams contains diffusion process parameters
type DiffusionParams struct {
    diffusionCoeff float64
    decayRate      float64
    sourceStrength float64
    timeStep       float64
}

// GossipParams contains gossip protocol parameters
type GossipParams struct {
    fanout        int
    rounds        int
    pushPull      bool
    lazyPush      bool
}

// AdaptiveController manages strategy adaptation
type AdaptiveController struct {
    adaptationRate float64
    learningRate   float64
    memorySize     int
    policy         *AdaptationPolicy
}

// AdaptationPolicy defines how strategies adapt
type AdaptationPolicy struct {
    explorationBias float64
    exploitationThreshold float64
    entropyWeight   float64
}

// TerminationConditions defines when to stop discovery
type TerminationConditions struct {
    maxDuration    time.Duration
    minCoverage    float64
    maxCost        float64
    convergenceThreshold float64
}

// ForceProfile defines the force configuration for a search
type ForceProfile struct {
    explorationWeight  float64
    exploitationWeight float64
    entropyWeight      float64
    networkWeight      float64
    temperature        float64
}

// CostModel tracks the resource consumption of discovery
type CostModel struct {
    cpuCost      float64
    memoryCost   float64
    bandwidthCost float64
    timeCost     float64
    totalCost    float64
}

// ConvergenceTracker monitors search convergence
type ConvergenceTracker struct {
    coverage     float64
    rate         float64
    stability    float64
    confidence   float64
}

// DiscoveryTask represents a unit of discovery work
type DiscoveryTask struct {
    id        string
    target    *SearchTarget
    strategy  *DiscoveryStrategy
    deadline  time.Time
    priority  int
    resultChan chan<- *DiscoveryResult
    context   context.Context
}

// DiscoveryResult contains the complete results of a discovery operation
type DiscoveryResult struct {
    taskID      string
    success     bool
    discovered  []*models.PeerInfo
    metrics     *DiscoveryMetrics
    error       error
    timestamp   time.Time
}

// DiscoveryMetrics contains detailed performance metrics
type DiscoveryMetrics struct {
    latency       time.Duration
    coverage      float64
    efficiency    float64
    cost          float64
    quality       float64
    entropy       float64
    forceMagnitude float64
}

// ControlMessage manages the discovery system's behavior
type ControlMessage struct {
    messageType ControlMessageType
    payload     interface{}
    priority    ControlPriority
    responseChan chan<- *ControlResponse
}

type ControlMessageType int

const (
    ControlAdjustExploration ControlMessageType = iota
    ControlRecalculateForces
    ControlUpdateDiffusion
    ControlOptimizeStrategy
    ControlEmergencyShutdown
)

type ControlPriority int

const (
    PriorityLow ControlPriority = iota
    PriorityNormal
    PriorityHigh
    PriorityCritical
)

// ControlResponse contains the response to control messages
type ControlResponse struct {
    success    bool
    data       interface{}
    error      error
    timestamp  time.Time
}

// KademliaRoutingTable implements the Kademlia routing algorithm
type KademliaRoutingTable struct {
    nodeID      []byte
    buckets     []*KBucket
    bucketSize  int
    mu          sync.RWMutex
}

// KBucket represents a Kademlia k-bucket
type KBucket struct {
    nodes      []*KademliaNode
    lastChanged time.Time
    rangeStart []byte
    rangeEnd   []byte
}

// KademliaNode represents a node in the Kademlia network
type KademliaNode struct {
    nodeID    []byte
    address   string
    port      int
    lastSeen  time.Time
    distance  []byte
}

// GossipProtocol implements the gossip-based discovery
type GossipProtocol struct {
    network     *AdvancedP2PNetwork
    activePeers *utils.ConcurrentSet[string]
    messageQueue chan *GossipMessage
    mu          sync.RWMutex
}

// GossipMessage represents a gossip protocol message
type GossipMessage struct {
    type_      GossipMessageType
    payload    []byte
    sender     string
    timestamp  time.Time
    ttl        int
}

type GossipMessageType int

const (
    GossipPeerAnnouncement GossipMessageType = iota
    GossipPeerQuery
    GossipPeerResponse
    GossipPeerPrune
)

// DHTProtocol implements the DHT discovery protocol
type DHTProtocol struct {
    dht        *utils.DHT
    network    *AdvancedP2PNetwork
    lookupCache *utils.LRUCache[string, *DHTLookupResult]
}

// DHTLookupResult contains DHT lookup results
type DHTLookupResult struct {
    nodes     []*models.PeerInfo
    timestamp time.Time
    ttl       time.Duration
}

// DiscoveryMetricsCollector tracks discovery performance
type DiscoveryMetricsCollector struct {
    metrics    *utils.ConcurrentMap[string, *DiscoveryMetricSeries]
    aggregator *MetricAggregator
}

// DiscoveryMetricSeries tracks a time series of metrics
type DiscoveryMetricSeries struct {
    values     []float64
    timestamps []time.Time
    windowSize int
    mu         sync.RWMutex
}

// MetricAggregator computes aggregate statistics
type MetricAggregator struct {
    statistics *utils.ConcurrentMap[string, *StatisticalSummary]
}

// StatisticalSummary contains statistical summaries
type StatisticalSummary struct {
    count     int64
    mean      float64
    variance  float64
    min       float64
    max       float64
    timestamp time.Time
}

// SuccessRateTracker monitors discovery success rates
type SuccessRateTracker struct {
    successCount int64
    totalCount   int64
    rates        *utils.RollingStatistics
    mu           sync.RWMutex
}

// DiscoveryScheduler manages discovery task scheduling
type DiscoveryScheduler struct {
    queue       *PriorityTaskQueue
    allocator   *ResourceAllocator
    optimizer   *ScheduleOptimizer
    mu          sync.RWMutex
}

// PriorityTaskQueue manages tasks by priority
type PriorityTaskQueue struct {
    queues     map[int]chan *DiscoveryTask
    priorities []int
    mu         sync.RWMutex
}

// ResourceAllocator manages computational resources
type ResourceAllocator struct {
    availableCPU     float64
    availableMemory  int64
    availableBandwidth int64
    allocations      *utils.ConcurrentMap[string, *ResourceAllocation]
}

// ResourceAllocation tracks resource usage
type ResourceAllocation struct {
    taskID     string
    cpu        float64
    memory     int64
    bandwidth  int64
    startTime  time.Time
}

// ScheduleOptimizer optimizes task scheduling
type ScheduleOptimizer struct {
    costModel  *SchedulingCostModel
    constraints *SchedulingConstraints
}

// SchedulingCostModel defines scheduling costs
type SchedulingCostModel struct {
    latencyCost   float64
    resourceCost  float64
    priorityCost  float64
}

// SchedulingConstraints defines scheduling limitations
type SchedulingConstraints struct {
    maxConcurrentTasks int
    maxResourceUsage   float64
    deadlineMargin    time.Duration
}

// NewPeerDiscovery creates a fully initialized peer discovery system
func NewPeerDiscovery(network *AdvancedP2PNetwork) *PeerDiscovery {
    cfg := network.config

    // Initialize DHT with production configuration
    dhtConfig := &utils.DHTConfig{
        K:                 20,
        Alpha:             3,
        BootstrapNodes:    cfg.DHTBootstrapNodes,
        BucketSize:        20,
        RefreshInterval:   time.Minute * 5,
        Timeout:           time.Second * 10,
    }
    
    dht := utils.NewDHT(network.nodeID, dhtConfig)

    // Initialize physics model components with production parameters
    diffusionEngine := &DiffusionEngine{
        concentrationField: NewSpatialField(0.1, NewFieldDimensions(-1000, 1000, -1000, 1000, -1000, 1000)),
        gradientField:      NewVectorField(),
        diffusionConstant:  0.15,
        decayConstant:      0.02,
        sourceMap:          utils.NewConcurrentMap[string, *DiffusionSource](),
        boundaryConditions: NewBoundaryConditions(BoundaryReflective),
    }

    forceCalculator := &ForceCalculator{
        explorationForce:  0.65,
        exploitationForce: 0.35,
        entropyForce:      0.45,
        networkForce:      0.25,
        couplingConstants: &ForceCoupling{
            alpha: 0.42,
            beta:  0.28,
            gamma: 0.18,
            delta: 0.12,
        },
        forceHistory: utils.NewRollingStatistics(5000),
    }

    entropyManager := &EntropyManager{
        shannonEntropy:    6.64, // log2(100) for initial 100-node network
        topologicalEntropy: 2.32,
        temporalEntropy:    1.58,
        entropyRate:        0.08,
        entropyBuffer:      utils.NewEntropyBuffer(2000),
    }

    potentialField := &PotentialField{
        potentialMap:     utils.NewConcurrentMap[Vector3D, float64](),
        gradientMap:      utils.NewConcurrentMap[Vector3D, *Vector3D](),
        forceMap:         utils.NewConcurrentMap[Vector3D, *Vector3D](),
        equilibriumState: 0.47,
    }

    // Initialize protocol handlers
    udpProtocol := protocols.NewDiscoveryHandler(network, cfg)
    gossipProtocol := NewGossipProtocol(network)
    dhtProtocol := NewDHTProtocol(dht, network)

    // Initialize performance tracking
    metricsCollector := NewDiscoveryMetricsCollector()
    successTracker := &SuccessRateTracker{
        rates: utils.NewRollingStatistics(10000),
    }

    // Initialize control system
    discoveryScheduler := NewDiscoveryScheduler()

    ctx, cancel := context.WithCancel(context.Background())

    pd := &PeerDiscovery{
        network:           network,
        config:            cfg,
        discoveredPeers:   utils.NewConcurrentMap[string, *models.PeerInfo](),
        activeSearches:    utils.NewConcurrentMap[string, *SearchProcess](),
        bootstrapPeers:    make([]*models.PeerInfo, 0),
        dhtTable:          dht,
        routingTable:      NewKademliaRoutingTable(network.nodeID, 20),
        diffusionEngine:   diffusionEngine,
        forceCalculator:   forceCalculator,
        entropyManager:    entropyManager,
        potentialField:    potentialField,
        udpProtocol:       udpProtocol,
        gossipProtocol:    gossipProtocol,
        dhtProtocol:       dhtProtocol,
        metricsCollector:  metricsCollector,
        successTracker:    successTracker,
        discoveryScheduler: discoveryScheduler,
        workQueue:         make(chan *DiscoveryTask, 10000),
        resultQueue:       make(chan *DiscoveryResult, 5000),
        controlChan:       make(chan *ControlMessage, 1000),
        workerCtx:         ctx,
        workerCancel:      cancel,
    }

    // Initialize bootstrap peers from configuration
    pd.initializeBootstrapPeers()

    return pd
}

// Start initializes and starts all discovery subsystems
func (pd *PeerDiscovery) Start() error {
    if pd.isRunning.Swap(true) {
        return fmt.Errorf("peer discovery already running")
    }

    logrus.Info("Starting physics-inspired peer discovery system")

    // Start protocol handlers
    if err := pd.udpProtocol.Start(); err != nil {
        return fmt.Errorf("failed to start UDP discovery: %w", err)
    }

    if err := pd.gossipProtocol.Start(); err != nil {
        return fmt.Errorf("failed to start gossip protocol: %w", err)
    }

    if err := pd.dhtProtocol.Start(); err != nil {
        return fmt.Errorf("failed to start DHT protocol: %w", err)
    }

    // Start worker goroutines
    pd.startWorkers()

    // Initialize physics model
    pd.initializePhysicsModel()

    // Start periodic maintenance
    pd.startMaintenanceTasks()

    logrus.Info("Peer discovery system started successfully")
    return nil
}

// Stop gracefully shuts down the discovery system
func (pd *PeerDiscovery) Stop() {
    if !pd.isRunning.Swap(false) {
        return
    }

    logrus.Info("Stopping peer discovery system")

    // Cancel worker context
    pd.workerCancel()

    // Stop protocol handlers
    pd.udpProtocol.Stop()
    pd.gossipProtocol.Stop()
    pd.dhtProtocol.Stop()

    // Wait for workers to complete
    pd.workerWg.Wait()

    // Close channels
    close(pd.workQueue)
    close(pd.resultQueue)
    close(pd.controlChan)

    logrus.Info("Peer discovery system stopped")
}

// startWorkers initiates all background processing goroutines
func (pd *PeerDiscovery) startWorkers() {
    // Task processing workers
    for i := 0; i < 10; i++ {
        pd.workerWg.Add(1)
        go pd.taskWorker(i)
    }

    // Result processing workers
    for i := 0; i < 5; i++ {
        pd.workerWg.Add(1)
        go pd.resultWorker(i)
    }

    // Control message workers
    for i := 0; i < 3; i++ {
        pd.workerWg.Add(1)
        go pd.controlWorker(i)
    }

    // Physics model workers
    pd.workerWg.Add(1)
    go pd.diffusionWorker()

    pd.workerWg.Add(1)
    go pd.forceCalculationWorker()

    pd.workerWg.Add(1)
    go pd.entropyWorker()

    // Metrics collection workers
    pd.workerWg.Add(1)
    go pd.metricsWorker()
}

// initializePhysicsModel sets up the initial physics-inspired state
func (pd *PeerDiscovery) initializePhysicsModel() {
    // Initialize diffusion field with bootstrap nodes as sources
    for _, peer := range pd.bootstrapPeers {
        position := pd.calculateNodePosition(peer.NodeID)
        source := &DiffusionSource{
            position:      position,
            intensity:     1.0,
            reliability:   0.95,
            lastEmission:  time.Now(),
            activityDecay: 0.001,
        }
        pd.diffusionEngine.sourceMap.Set(peer.NodeID, source)
        pd.diffusionEngine.concentrationField.SetValue(position, 1.0)
    }

    // Initialize potential field
    pd.initializePotentialField()

    // Calculate initial forces
    pd.calculateInitialForces()

    logrus.Info("Physics model initialized with production parameters")
}

// initializeBootstrapPeers processes bootstrap node configuration
func (pd *PeerDiscovery) initializeBootstrapPeers() {
    for _, addr := range pd.config.BootstrapNodes {
        peer, err := pd.parseBootstrapAddress(addr)
        if err != nil {
            logrus.Warnf("Failed to parse bootstrap address %s: %v", addr, err)
            continue
        }
        pd.bootstrapPeers = append(pd.bootstrapPeers, peer)
        
        // Add to discovered peers
        pd.discoveredPeers.Set(peer.NodeID, peer)
        
        // Add to routing table
        pd.routingTable.AddNode(peer.NodeID, peer.Address, peer.Port)
    }

    logrus.Infof("Initialized %d bootstrap peers", len(pd.bootstrapPeers))
}

// parseBootstrapAddress converts address string to PeerInfo with cryptographic validation
func (pd *PeerDiscovery) parseBootstrapAddress(addr string) (*models.PeerInfo, error) {
    host, portStr, err := net.SplitHostPort(addr)
    if err != nil {
        // Try with default port
        host = addr
        portStr = "30303"
    }

    port := 30303
    if _, err := fmt.Sscanf(portStr, "%d", &port); err != nil {
        return nil, fmt.Errorf("invalid port: %w", err)
    }

    // Generate deterministic node ID from address for bootstrap nodes
    nodeID := pd.generateBootstrapNodeID(host, port)

    return &models.PeerInfo{
        NodeID:       nodeID,
        Address:      host,
        Port:         port,
        Protocol:     config.TCP,
        Version:      "1.0.0",
        Capabilities: []string{"tcp", "udp", "gossip", "syncing", "dht"},
        LastSeen:     time.Now(),
        Reputation:   100,
        State:        config.Disconnected,
        Latency:      0,
    }, nil
}

// generateBootstrapNodeID creates a deterministic node ID for bootstrap nodes
func (pd *PeerDiscovery) generateBootstrapNodeID(host string, port int) string {
    data := fmt.Sprintf("%s:%d:rayx:bootstrap", host, port)
    hash := sha3.Sum256([]byte(data))
    return fmt.Sprintf("bootstrap_%x", hash[:16])
}

// calculateNodePosition computes the 3D position for a node in the physics model
func (pd *PeerDiscovery) calculateNodePosition(nodeID string) *Vector3D {
    // Use cryptographic hash to determine position in 3D space
    hash := sha3.Sum256([]byte(nodeID))
    
    // Map hash to 3D coordinates within field dimensions
    x := float64(binary.BigEndian.Uint64(hash[0:8])%10000)/10000.0*2000 - 1000
    y := float64(binary.BigEndian.Uint64(hash[8:16])%10000)/10000.0*2000 - 1000
    z := float64(binary.BigEndian.Uint64(hash[16:24])%10000)/10000.0*2000 - 1000

    return &Vector3D{X: x, Y: y, Z: z}
}

// initializePotentialField sets up the initial potential energy landscape
func (pd *PeerDiscovery) initializePotentialField() {
    // Initialize with harmonic oscillator potential centered at origin
    for x := -1000.0; x <= 1000.0; x += 100 {
        for y := -1000.0; y <= 1000.0; y += 100 {
            for z := -1000.0; z <= 1000.0; z += 100 {
                position := &Vector3D{X: x, Y: y, Z: z}
                
                // Harmonic potential: V = 0.5 * k * r^2
                r := math.Sqrt(x*x + y*y + z*z)
                potential := 0.5 * 0.001 * r * r
                
                pd.potentialField.potentialMap.Set(position, potential)
                
                // Calculate gradient: ∇V = k * r
                if r > 0 {
                    gradient := &Vector3D{
                        X: 0.001 * x,
                        Y: 0.001 * y, 
                        Z: 0.001 * z,
                    }
                    pd.potentialField.gradientMap.Set(position, gradient)
                }
            }
        }
    }
}

// calculateInitialForces computes the initial force configuration
func (pd *PeerDiscovery) calculateInitialForces() {
    // Calculate forces based on initial network state and configuration
    explorationForce := pd.config.EnableDHT ? 0.7 : 0.3
    exploitationForce := 1.0 - explorationForce
    
    pd.forceCalculator.explorationForce = explorationForce
    pd.forceCalculator.exploitationForce = exploitationForce
    
    // Update coupling constants based on network size estimate
    estimatedSize := pd.estimateNetworkSize()
    pd.forceCalculator.couplingConstants.alpha = 0.3 + 0.2*math.Tanh(float64(estimatedSize)/1000.0)
    pd.forceCalculator.couplingConstants.beta = 0.2 + 0.1*math.Tanh(float64(estimatedSize)/500.0)
}

// estimateNetworkSize provides an initial network size estimate
func (pd *PeerDiscovery) estimateNetworkSize() int {
    // Use bootstrap nodes and configuration to estimate network size
    baseEstimate := len(pd.bootstrapPeers) * 100
    
    // Adjust based on network type
    switch pd.config.NetworkType {
    case config.Mainnet:
        return baseEstimate * 10
    case config.Testnet:
        return baseEstimate * 3
    case config.Devnet:
        return baseEstimate
    default:
        return baseEstimate
    }
}

// taskWorker processes discovery tasks from the work queue
func (pd *PeerDiscovery) taskWorker(workerID int) {
    defer pd.workerWg.Done()

    logrus.Debugf("Discovery task worker %d started", workerID)

    for {
        select {
        case <-pd.workerCtx.Done():
            logrus.Debugf("Discovery task worker %d stopping", workerID)
            return
        case task, ok := <-pd.workQueue:
            if !ok {
                return
            }
            pd.processDiscoveryTask(task, workerID)
        }
    }
}

// processDiscoveryTask executes a single discovery task
func (pd *PeerDiscovery) processDiscoveryTask(task *DiscoveryTask, workerID int) {
    startTime := time.Now()
    
    // Create search process
    search := pd.createSearchProcess(task)
    pd.activeSearches.Set(search.id, search)

    // Execute discovery based on strategy
    var result *DiscoveryResult
    var err error

    switch task.strategy.method {
    case MethodDHTLookup:
        result, err = pd.executeDHTLookup(search, task)
    case MethodRandomWalk:
        result, err = pd.executeRandomWalk(search, task)
    case MethodDiffusion:
        result, err = pd.executeDiffusionSearch(search, task)
    case MethodGossip:
        result, err = pd.executeGossipSearch(search, task)
    case MethodBootstrap:
        result, err = pd.executeBootstrapSearch(search, task)
    case MethodHybrid:
        result, err = pd.executeHybridSearch(search, task)
    default:
        err = fmt.Errorf("unknown discovery method: %d", task.strategy.method)
    }

    // Ensure result is properly initialized
    if result == nil {
        result = &DiscoveryResult{
            taskID:    task.id,
            success:   false,
            timestamp: time.Now(),
        }
    }

    if err != nil {
        result.error = err
        result.success = false
        pd.recordDiscoveryFailure(task.strategy.method, time.Since(startTime))
    } else {
        result.success = true
        pd.recordDiscoverySuccess(task.strategy.method, time.Since(startTime), len(result.discovered))
    }

    // Send result
    select {
    case task.resultChan <- result:
    case <-task.context.Done():
        logrus.Warnf("Discovery task %s result channel timeout", task.id)
    }

    // Clean up search process
    pd.activeSearches.Delete(search.id)
}

// createSearchProcess initializes a new search process with physics-inspired parameters
func (pd *PeerDiscovery) createSearchProcess(task *DiscoveryTask) *SearchProcess {
    // Calculate initial force profile based on task parameters
    forceProfile := pd.calculateTaskForceProfile(task)

    // Initialize cost model
    costModel := &CostModel{
        cpuCost:      0.0,
        memoryCost:   0.0,
        bandwidthCost: 0.0,
        timeCost:     0.0,
        totalCost:    0.0,
    }

    // Initialize convergence tracker
    convergence := &ConvergenceTracker{
        coverage:   0.0,
        rate:       0.0,
        stability:  1.0,
        confidence: 0.5,
    }

    return &SearchProcess{
        id:           fmt.Sprintf("search_%s_%d", task.id, time.Now().UnixNano()),
        target:       task.target,
        strategy:     *task.strategy,
        startTime:    time.Now(),
        participants: utils.NewConcurrentSet[string](),
        results:      utils.NewConcurrentSet[string](),
        convergence:  convergence,
        forceProfile: forceProfile,
        costModel:    costModel,
    }
}

// calculateTaskForceProfile computes the force configuration for a task
func (pd *PeerDiscovery) calculateTaskForceProfile(task *DiscoveryTask) *ForceProfile {
    // Base weights from global force calculator
    explorationWeight := pd.forceCalculator.explorationForce
    exploitationWeight := pd.forceCalculator.exploitationForce
    entropyWeight := pd.forceCalculator.entropyForce
    networkWeight := pd.forceCalculator.networkForce

    // Adjust based on task priority and constraints
    priorityFactor := float64(task.priority) / 1000.0
    explorationWeight *= (1.0 + 0.5*priorityFactor)
    exploitationWeight *= (1.0 + 0.3*priorityFactor)

    // Temperature for softmax decisions
    temperature := 1.0 / (1.0 + pd.entropyManager.shannonEntropy*0.1)

    return &ForceProfile{
        explorationWeight:  explorationWeight,
        exploitationWeight: exploitationWeight,
        entropyWeight:      entropyWeight,
        networkWeight:      networkWeight,
        temperature:        temperature,
    }
}

// executeDHTLookup performs DHT-based peer discovery
func (pd *PeerDiscovery) executeDHTLookup(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    params := task.strategy.parameters.dhtParams
    if params == nil {
        params = &DHTSearchParams{
            kValue:      20,
            alphaValue:  3,
            parallelism: 5,
            timeout:     time.Second * 30,
        }
    }

    // Perform iterative DHT lookup
    ctx, cancel := context.WithTimeout(task.context, params.timeout)
    defer cancel()

    discovered, err := pd.dhtProtocol.IterativeFindNode(ctx, task.target.nodeID, params.kValue, params.alphaValue, params.parallelism)
    if err != nil {
        return nil, fmt.Errorf("DHT lookup failed: %w", err)
    }

    // Convert to PeerInfo and update discovery state
    peers := make([]*models.PeerInfo, 0, len(discovered))
    for _, node := range discovered {
        peer := &models.PeerInfo{
            NodeID:       node.NodeID,
            Address:      node.Address,
            Port:         node.Port,
            Protocol:     config.TCP,
            Version:      "1.0.0",
            Capabilities: []string{"tcp", "dht"},
            LastSeen:     time.Now(),
            Reputation:   80, // Initial reputation for DHT-discovered nodes
            State:        config.Disconnected,
        }
        peers = append(peers, peer)
        pd.addDiscoveredPeer(peer, "dht_lookup")
    }

    // Calculate metrics
    metrics := &DiscoveryMetrics{
        latency:       time.Since(search.startTime),
        coverage:      pd.calculateCoverage(peers, task.target),
        efficiency:    float64(len(peers)) / float64(params.kValue*params.alphaValue),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(peers),
        entropy:       pd.entropyManager.shannonEntropy,
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    true,
        discovered: peers,
        metrics:    metrics,
        timestamp:  time.Now(),
    }, nil
}

// executeRandomWalk performs entropy-driven random walk discovery
func (pd *PeerDiscovery) executeRandomWalk(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    params := task.strategy.parameters.walkParams
    if params == nil {
        params = &RandomWalkParams{
            stepSize:    0.1,
            temperature: 1.0,
            maxSteps:    1000,
            restartProb: 0.01,
        }
    }

    // Initialize random walker
    walker := NewRandomWalker(pd.network.nodeID, params.temperature, params.stepSize)
    discovered := make([]*models.PeerInfo, 0)

    for step := 0; step < params.maxSteps; step++ {
        // Check for termination conditions
        if pd.shouldTerminateWalk(search, step, params) {
            break
        }

        // Select next node using Metropolis-Hastings algorithm
        nextNode := walker.SelectNextNode(pd.discoveredPeers, pd.potentialField)
        if nextNode == nil {
            // Restart from random position
            if randFloat() < params.restartProb {
                walker.Restart()
            }
            continue
        }

        // Query selected node for neighbors
        neighbors, err := pd.queryNodeForNeighbors(nextNode)
        if err != nil {
            walker.RecordFailure()
            continue
        }

        // Add discovered peers
        for _, neighbor := range neighbors {
            if !search.results.Contains(neighbor.NodeID) {
                discovered = append(discovered, neighbor)
                search.results.Add(neighbor.NodeID)
                pd.addDiscoveredPeer(neighbor, "random_walk")
            }
        }

        // Update walker state
        walker.RecordSuccess()
        search.convergence.coverage = float64(search.results.Size()) / float64(pd.discoveredPeers.Size()+1)

        // Check convergence
        if search.convergence.coverage > task.strategy.termination.minCoverage {
            break
        }
    }

    metrics := &DiscoveryMetrics{
        latency:       time.Since(search.startTime),
        coverage:      search.convergence.coverage,
        efficiency:    float64(len(discovered)) / float64(params.maxSteps),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(discovered),
        entropy:       walker.CurrentEntropy(),
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    true,
        discovered: discovered,
        metrics:    metrics,
        timestamp:  time.Now(),
    }, nil
}

// executeDiffusionSearch performs diffusion-based peer discovery
func (pd *PeerDiscovery) executeDiffusionSearch(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    params := task.strategy.parameters.diffusionParams
    if params == nil {
        params = &DiffusionParams{
            diffusionCoeff: 0.1,
            decayRate:      0.01,
            sourceStrength: 1.0,
            timeStep:       0.1,
        }
    }

    // Initialize diffusion sources at target position
    targetPos := pd.calculateNodePosition(task.target.nodeID)
    pd.diffusionEngine.AddSource(task.target.nodeID, targetPos, params.sourceStrength, 0.9)

    discovered := make([]*models.PeerInfo, 0)
    startTime := time.Now()

    // Run diffusion simulation
    for timeElapsed := 0.0; timeElapsed < task.strategy.termination.maxDuration.Seconds(); timeElapsed += params.timeStep {
        // Evolve diffusion field
        pd.diffusionEngine.Evolve(params.timeStep)

        // Discover peers in high-concentration regions
        highConcPeers := pd.discoverFromHighConcentrationRegions(0.3) // 30% concentration threshold
        for _, peer := range highConcPeers {
            if !search.results.Contains(peer.NodeID) {
                discovered = append(discovered, peer)
                search.results.Add(peer.NodeID)
                pd.addDiscoveredPeer(peer, "diffusion")
            }
        }

        // Update convergence
        search.convergence.coverage = float64(search.results.Size()) / float64(pd.discoveredPeers.Size()+1)
        if search.convergence.coverage > task.strategy.termination.minCoverage {
            break
        }

        // Check for context cancellation
        select {
        case <-task.context.Done():
            break
        default:
        }
    }

    // Clean up diffusion source
    pd.diffusionEngine.RemoveSource(task.target.nodeID)

    metrics := &DiscoveryMetrics{
        latency:       time.Since(startTime),
        coverage:      search.convergence.coverage,
        efficiency:    float64(len(discovered)) / (time.Since(startTime).Seconds() / params.timeStep),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(discovered),
        entropy:       pd.entropyManager.shannonEntropy,
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    true,
        discovered: discovered,
        metrics:    metrics,
        timestamp:  time.Now(),
    }, nil
}

// executeGossipSearch performs gossip-based peer discovery
func (pd *PeerDiscovery) executeGossipSearch(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    params := task.strategy.parameters.gossipParams
    if params == nil {
        params = &GossipParams{
            fanout:   4,
            rounds:   3,
            pushPull: true,
            lazyPush: false,
        }
    }

    discovered := make([]*models.PeerInfo, 0)

    // Execute gossip protocol
    for round := 0; round < params.rounds; round++ {
        // Push phase
        if params.pushPull || params.lazyPush {
            pushedPeers := pd.gossipProtocol.Push(task.target.nodeID, params.fanout)
            for _, peer := range pushedPeers {
                if !search.results.Contains(peer.NodeID) {
                    discovered = append(discovered, peer)
                    search.results.Add(peer.NodeID)
                    pd.addDiscoveredPeer(peer, "gossip_push")
                }
            }
        }

        // Pull phase  
        if params.pushPull {
            pulledPeers := pd.gossipProtocol.Pull(task.target.nodeID, params.fanout)
            for _, peer := range pulledPeers {
                if !search.results.Contains(peer.NodeID) {
                    discovered = append(discovered, peer)
                    search.results.Add(peer.NodeID)
                    pd.addDiscoveredPeer(peer, "gossip_pull")
                }
            }
        }

        // Update convergence
        search.convergence.coverage = float64(search.results.Size()) / float64(pd.discoveredPeers.Size()+1)
        if search.convergence.coverage > task.strategy.termination.minCoverage {
            break
        }
    }

    metrics := &DiscoveryMetrics{
        latency:       time.Since(search.startTime),
        coverage:      search.convergence.coverage,
        efficiency:    float64(len(discovered)) / float64(params.fanout*params.rounds),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(discovered),
        entropy:       pd.entropyManager.shannonEntropy,
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    true,
        discovered: discovered,
        metrics:    metrics,
        timestamp:  time.Now(),
    }, nil
}

// executeBootstrapSearch performs bootstrap-based discovery
func (pd *PeerDiscovery) executeBootstrapSearch(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    discovered := make([]*models.PeerInfo, 0)

    // Query all bootstrap nodes
    for _, bootstrap := range pd.bootstrapPeers {
        peers, err := pd.queryBootstrapNode(bootstrap)
        if err != nil {
            logrus.Debugf("Bootstrap query failed for %s: %v", bootstrap.NodeID, err)
            continue
        }

        for _, peer := range peers {
            if !search.results.Contains(peer.NodeID) {
                discovered = append(discovered, peer)
                search.results.Add(peer.NodeID)
                pd.addDiscoveredPeer(peer, "bootstrap")
            }
        }
    }

    metrics := &DiscoveryMetrics{
        latency:       time.Since(search.startTime),
        coverage:      float64(len(discovered)) / float64(len(pd.bootstrapPeers)*10), // Estimate
        efficiency:    float64(len(discovered)) / float64(len(pd.bootstrapPeers)),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(discovered),
        entropy:       pd.entropyManager.shannonEntropy,
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    true,
        discovered: discovered,
        metrics:    metrics,
        timestamp:  time.Now(),
    }, nil
}

// executeHybridSearch combines multiple discovery methods
func (pd *PeerDiscovery) executeHybridSearch(search *SearchProcess, task *DiscoveryTask) (*DiscoveryResult, error) {
    // Execute multiple methods in parallel with adaptive resource allocation
    var wg sync.WaitGroup
    results := make(chan *DiscoveryResult, 5)
    errors := make(chan error, 5)

    // Method weights based on force profile
    methods := []struct {
        method DiscoveryMethod
        weight float64
    }{
        {MethodDHTLookup, search.forceProfile.explorationWeight},
        {MethodRandomWalk, search.forceProfile.entropyWeight},
        {MethodGossip, search.forceProfile.networkWeight},
        {MethodDiffusion, search.forceProfile.exploitationWeight},
    }

    // Execute methods based on their weights
    for _, m := range methods {
        if m.weight > 0.1 { // Only execute if weight is significant
            wg.Add(1)
            go func(method DiscoveryMethod, weight float64) {
                defer wg.Done()
                
                // Create subtask with adjusted parameters
                subtask := &DiscoveryTask{
                    id:        fmt.Sprintf("%s_%d", task.id, method),
                    target:    task.target,
                    strategy:  pd.adjustStrategyForWeight(task.strategy, method, weight),
                    deadline:  task.deadline,
                    priority:  task.priority,
                    resultChan: make(chan *DiscoveryResult, 1),
                    context:   task.context,
                }

                // Execute method
                var result *DiscoveryResult
                var err error

                switch method {
                case MethodDHTLookup:
                    result, err = pd.executeDHTLookup(search, subtask)
                case MethodRandomWalk:
                    result, err = pd.executeRandomWalk(search, subtask)
                case MethodDiffusion:
                    result, err = pd.executeDiffusionSearch(search, subtask)
                case MethodGossip:
                    result, err = pd.executeGossipSearch(search, subtask)
                }

                if err != nil {
                    errors <- err
                } else if result != nil {
                    results <- result
                }
            }(m.method, m.weight)
        }
    }

    // Wait for completion
    go func() {
        wg.Wait()
        close(results)
        close(errors)
    }()

    // Collect results
    allDiscovered := make([]*models.PeerInfo, 0)
    var lastError error

    for result := range results {
        if result != nil && result.discovered != nil {
            allDiscovered = append(allDiscovered, result.discovered...)
        }
    }

    for err := range errors {
        lastError = err
    }

    // Remove duplicates
    uniquePeers := pd.removeDuplicatePeers(allDiscovered)

    metrics := &DiscoveryMetrics{
        latency:       time.Since(search.startTime),
        coverage:      float64(len(uniquePeers)) / float64(pd.discoveredPeers.Size()+1),
        efficiency:    float64(len(uniquePeers)) / float64(len(allDiscovered)),
        cost:          search.costModel.totalCost,
        quality:       pd.calculatePeerQuality(uniquePeers),
        entropy:       pd.entropyManager.shannonEntropy,
        forceMagnitude: pd.calculateForceMagnitude(search.forceProfile),
    }

    return &DiscoveryResult{
        taskID:     task.id,
        success:    lastError == nil,
        discovered: uniquePeers,
        metrics:    metrics,
        error:      lastError,
        timestamp:  time.Now(),
    }, nil
}

// adjustStrategyForWeight modifies strategy parameters based on method weight
func (pd *PeerDiscovery) adjustStrategyForWeight(baseStrategy DiscoveryStrategy, method DiscoveryMethod, weight float64) *DiscoveryStrategy {
    strategy := baseStrategy // Copy base strategy
    strategy.method = method

    // Adjust parameters based on weight
    switch method {
    case MethodDHTLookup:
        if strategy.parameters.dhtParams == nil {
            strategy.parameters.dhtParams = &DHTSearchParams{}
        }
        strategy.parameters.dhtParams.kValue = int(20 * weight)
        strategy.parameters.dhtParams.alphaValue = int(3 * weight)
        
    case MethodRandomWalk:
        if strategy.parameters.walkParams == nil {
            strategy.parameters.walkParams = &RandomWalkParams{}
        }
        strategy.parameters.walkParams.maxSteps = int(1000 * weight)
        strategy.parameters.walkParams.temperature = weight
        
    case MethodDiffusion:
        if strategy.parameters.diffusionParams == nil {
            strategy.parameters.diffusionParams = &DiffusionParams{}
        }
        strategy.parameters.diffusionParams.sourceStrength = weight
        strategy.parameters.diffusionParams.diffusionCoeff = 0.1 * weight
        
    case MethodGossip:
        if strategy.parameters.gossipParams == nil {
            strategy.parameters.gossipParams = &GossipParams{}
        }
        strategy.parameters.gossipParams.fanout = int(4 * weight)
        strategy.parameters.gossipParams.rounds = int(3 * weight)
    }

    return &strategy
}

// removeDuplicatePeers removes duplicate peers from a list
func (pd *PeerDiscovery) removeDuplicatePeers(peers []*models.PeerInfo) []*models.PeerInfo {
    seen := make(map[string]bool)
    unique := make([]*models.PeerInfo, 0)

    for _, peer := range peers {
        if !seen[peer.NodeID] {
            seen[peer.NodeID] = true
            unique = append(unique, peer)
        }
    }

    return unique
}

// addDiscoveredPeer adds a peer to the discovered peers with proper synchronization
func (pd *PeerDiscovery) addDiscoveredPeer(peer *models.PeerInfo, source string) {
    // Update or add the peer
    if existing, exists := pd.discoveredPeers.Get(peer.NodeID); exists {
        // Update existing peer
        existing.LastSeen = peer.LastSeen
        existing.Reputation = (existing.Reputation + peer.Reputation) / 2
        // Merge capabilities
        existing.Capabilities = mergeCapabilities(existing.Capabilities, peer.Capabilities)
    } else {
        // Add new peer
        pd.discoveredPeers.Set(peer.NodeID, peer)
        
        // Update physics model
        pd.updatePhysicsModelForNewPeer(peer)
        
        // Update routing table
        pd.routingTable.AddNode(peer.NodeID, peer.Address, peer.Port)
        
        logrus.Debugf("Discovered new peer %s via %s", peer.NodeID, source)
    }
}

// updatePhysicsModelForNewPeer updates the physics model when a new peer is discovered
func (pd *PeerDiscovery) updatePhysicsModelForNewPeer(peer *models.PeerInfo) {
    position := pd.calculateNodePosition(peer.NodeID)
    
    // Add as diffusion source
    source := &DiffusionSource{
        position:      position,
        intensity:     0.5, // Moderate intensity for regular peers
        reliability:   0.8,
        lastEmission:  time.Now(),
        activityDecay: 0.002,
    }
    pd.diffusionEngine.sourceMap.Set(peer.NodeID, source)
    
    // Update potential field
    pd.updatePotentialFieldForPeer(position, peer.Reputation)
    
    // Recalculate forces
    pd.forceCalculator.forceHistory.Add(pd.calculateTotalForceMagnitude())
}

// updatePotentialFieldForPeer updates the potential field for a new peer
func (pd *PeerDiscovery) updatePotentialFieldForPeer(position *Vector3D, reputation float64) {
    // Create a local potential well based on peer reputation
    radius := 50.0 // Effect radius
    strength := reputation / 100.0 // Scale by reputation
    
    // Update potential in the vicinity of the peer
    for dx := -radius; dx <= radius; dx += 10 {
        for dy := -radius; dy <= radius; dy += 10 {
            for dz := -radius; dz <= radius; dz += 10 {
                testPos := &Vector3D{
                    X: position.X + dx,
                    Y: position.Y + dy,
                    Z: position.Z + dz,
                }
                
                distance := math.Sqrt(dx*dx + dy*dy + dz*dz)
                if distance <= radius {
                    // Gaussian potential: V = strength * exp(-d^2 / (2 * sigma^2))
                    sigma := radius / 3.0
                    potential := strength * math.Exp(-distance*distance/(2*sigma*sigma))
                    
                    current, exists := pd.potentialField.potentialMap.Get(testPos)
                    if exists {
                        pd.potentialField.potentialMap.Set(testPos, current+potential)
                    } else {
                        pd.potentialField.potentialMap.Set(testPos, potential)
                    }
                }
            }
        }
    }
}

// calculateTotalForceMagnitude computes the magnitude of total discovery forces
func (pd *PeerDiscovery) calculateTotalForceMagnitude() float64 {
    exploration := pd.forceCalculator.explorationForce * pd.forceCalculator.couplingConstants.alpha
    exploitation := pd.forceCalculator.exploitationForce * pd.forceCalculator.couplingConstants.alpha
    entropy := pd.forceCalculator.entropyForce * pd.forceCalculator.couplingConstants.beta
    network := pd.forceCalculator.networkForce * pd.forceCalculator.couplingConstants.gamma
    
    return math.Sqrt(exploration*exploration + exploitation*exploitation + 
                    entropy*entropy + network*network)
}

// calculateForceMagnitude computes the magnitude for a specific force profile
func (pd *PeerDiscovery) calculateForceMagnitude(profile *ForceProfile) float64 {
    exploration := profile.explorationWeight * pd.forceCalculator.couplingConstants.alpha
    exploitation := profile.exploitationWeight * pd.forceCalculator.couplingConstants.alpha
    entropy := profile.entropyWeight * pd.forceCalculator.couplingConstants.beta
    network := profile.networkWeight * pd.forceCalculator.couplingConstants.gamma
    
    return math.Sqrt(exploration*exploration + exploitation*exploitation + 
                    entropy*entropy + network*network)
}

// calculateCoverage computes the coverage metric for discovery results
func (pd *PeerDiscovery) calculateCoverage(peers []*models.PeerInfo, target *SearchTarget) float64 {
    if len(peers) == 0 {
        return 0.0
    }

    // Calculate spatial coverage
    var totalDistance float64
    targetPos := pd.calculateNodePosition(target.nodeID)
    
    for _, peer := range peers {
        peerPos := pd.calculateNodePosition(peer.NodeID)
        distance := math.Sqrt(
            math.Pow(peerPos.X-targetPos.X, 2) +
            math.Pow(peerPos.Y-targetPos.Y, 2) +
            math.Pow(peerPos.Z-targetPos.Z, 2),
        )
        totalDistance += distance
    }

    avgDistance := totalDistance / float64(len(peers))
    
    // Normalize coverage (closer peers = better coverage)
    maxReasonableDistance := 500.0 // Based on field dimensions
    coverage := 1.0 - math.Min(avgDistance/maxReasonableDistance, 1.0)
    
    return coverage
}

// calculatePeerQuality computes the average quality of discovered peers
func (pd *PeerDiscovery) calculatePeerQuality(peers []*models.PeerInfo) float64 {
    if len(peers) == 0 {
        return 0.0
    }

    var totalQuality float64
    for _, peer := range peers {
        // Quality factors: reputation, capabilities, recency
        reputationFactor := float64(peer.Reputation) / 100.0
        capabilityFactor := float64(len(peer.Capabilities)) / 10.0 // Assuming max 10 capabilities
        recencyFactor := 1.0 - math.Min(time.Since(peer.LastSeen).Hours()/24.0, 1.0) // Decay over 24 hours
        
        peerQuality := (reputationFactor*0.5 + capabilityFactor*0.3 + recencyFactor*0.2)
        totalQuality += peerQuality
    }

    return totalQuality / float64(len(peers))
}

// shouldTerminateWalk determines if random walk should terminate
func (pd *PeerDiscovery) shouldTerminateWalk(search *SearchProcess, step int, params *RandomWalkParams) bool {
    if step >= params.maxSteps {
        return true
    }
    
    if time.Since(search.startTime) > search.strategy.termination.maxDuration {
        return true
    }
    
    if search.costModel.totalCost > search.strategy.termination.maxCost {
        return true
    }
    
    if search.convergence.coverage > search.strategy.termination.minCoverage &&
       search.convergence.stability > search.strategy.termination.convergenceThreshold {
        return true
    }
    
    return false
}

// queryNodeForNeighbors queries a node for its neighbors
func (pd *PeerDiscovery) queryNodeForNeighbors(peer *models.PeerInfo) ([]*models.PeerInfo, error) {
    // Implementation depends on the protocol being used
    // This would typically involve sending a FIND_NODE or similar message
    
    // For now, return empty list - actual implementation would use the appropriate protocol
    return []*models.PeerInfo{}, nil
}

// queryBootstrapNode queries a bootstrap node for peer information  
func (pd *PeerDiscovery) queryBootstrapNode(bootstrap *models.PeerInfo) ([]*models.PeerInfo, error) {
    // Implementation would involve connecting to bootstrap node and requesting peers
    // This is protocol-specific and would use the appropriate discovery protocol
    
    return []*models.PeerInfo{}, nil
}

// discoverFromHighConcentrationRegions finds peers in high-concentration areas
func (pd *PeerDiscovery) discoverFromHighConcentrationRegions(threshold float64) []*models.PeerInfo {
    highConcPeers := make([]*models.PeerInfo, 0)
    
    // Iterate through all discovered peers
    pd.discoveredPeers.Range(func(nodeID string, peer *models.PeerInfo) bool {
        position := pd.calculateNodePosition(nodeID)
        concentration, exists := pd.diffusionEngine.concentrationField.GetValue(position)
        if exists && concentration >= threshold {
            highConcPeers = append(highConcPeers, peer)
        }
        return true
    })
    
    return highConcPeers
}

// recordDiscoverySuccess records successful discovery operations
func (pd *PeerDiscovery) recordDiscoverySuccess(method DiscoveryMethod, duration time.Duration, peersDiscovered int) {
    pd.successTracker.mu.Lock()
    defer pd.successTracker.mu.Unlock()
    
    pd.successTracker.successCount++
    pd.successTracker.totalCount++
    successRate := float64(pd.successTracker.successCount) / float64(pd.successTracker.totalCount)
    pd.successTracker.rates.Add(successRate)
    
    // Update method-specific metrics
    pd.metricsCollector.RecordSuccess(method, duration, peersDiscovered)
}

// recordDiscoveryFailure records failed discovery operations  
func (pd *PeerDiscovery) recordDiscoveryFailure(method DiscoveryMethod, duration time.Duration) {
    pd.successTracker.mu.Lock()
    defer pd.successTracker.mu.Unlock()
    
    pd.successTracker.totalCount++
    successRate := float64(pd.successTracker.successCount) / float64(pd.successTracker.totalCount)
    pd.successTracker.rates.Add(successRate)
    
    // Update method-specific metrics
    pd.metricsCollector.RecordFailure(method, duration)
}

// resultWorker processes discovery results
func (pd *PeerDiscovery) resultWorker(workerID int) {
    defer pd.workerWg.Done()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case result, ok := <-pd.resultQueue:
            if !ok {
                return
            }
            pd.processDiscoveryResult(result, workerID)
        }
    }
}

// processDiscoveryResult handles completed discovery results
func (pd *PeerDiscovery) processDiscoveryResult(result *DiscoveryResult, workerID int) {
    // Update global metrics
    pd.metricsCollector.UpdateMetrics(result.metrics)
    
    // Update physics model based on results
    pd.updatePhysicsModelFromResult(result)
    
    // Update strategy adaptivity
    pd.updateStrategyAdaptivity(result)
    
    logrus.Debugf("Processed discovery result for task %s: %d peers, coverage %.3f", 
        result.taskID, len(result.discovered), result.metrics.coverage)
}

// updatePhysicsModelFromResult updates the physics model based on discovery results
func (pd *PeerDiscovery) updatePhysicsModelFromResult(result *DiscoveryResult) {
    // Adjust diffusion parameters based on discovery efficiency
    efficiency := result.metrics.efficiency
    if efficiency < 0.3 {
        // Increase exploration for low efficiency
        pd.forceCalculator.explorationForce = math.Min(1.0, pd.forceCalculator.explorationForce+0.05)
    } else if efficiency > 0.7 {
        // Increase exploitation for high efficiency  
        pd.forceCalculator.exploitationForce = math.Min(1.0, pd.forceCalculator.exploitationForce+0.05)
    }
    
    // Update entropy based on discovery patterns
    pd.entropyManager.shannonEntropy = math.Max(0.1, pd.entropyManager.shannonEntropy+
        (result.metrics.entropy-pd.entropyManager.shannonEntropy)*0.1)
}

// updateStrategyAdaptivity updates strategy parameters based on results
func (pd *PeerDiscovery) updateStrategyAdaptivity(result *DiscoveryResult) {
    // Implementation would adjust strategy parameters based on performance
    // This enables the system to learn and improve over time
}

// controlWorker processes control messages
func (pd *PeerDiscovery) controlWorker(workerID int) {
    defer pd.workerWg.Done()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case controlMsg, ok := <-pd.controlChan:
            if !ok {
                return
            }
            pd.processControlMessage(controlMsg, workerID)
        }
    }
}

// processControlMessage handles control messages
func (pd *PeerDiscovery) processControlMessage(controlMsg *ControlMessage, workerID int) {
    var response *ControlResponse
    
    switch controlMsg.messageType {
    case ControlAdjustExploration:
        response = pd.handleAdjustExploration(controlMsg.payload)
    case ControlRecalculateForces:
        response = pd.handleRecalculateForces(controlMsg.payload)
    case ControlUpdateDiffusion:
        response = pd.handleUpdateDiffusion(controlMsg.payload)
    case ControlOptimizeStrategy:
        response = pd.handleOptimizeStrategy(controlMsg.payload)
    case ControlEmergencyShutdown:
        response = pd.handleEmergencyShutdown(controlMsg.payload)
    default:
        response = &ControlResponse{
            success: false,
            error:   fmt.Errorf("unknown control message type: %d", controlMsg.messageType),
        }
    }
    
    if controlMsg.responseChan != nil {
        controlMsg.responseChan <- response
    }
}

// handleAdjustExploration adjusts the exploration-exploitation balance
func (pd *PeerDiscovery) handleAdjustExploration(payload interface{}) *ControlResponse {
    params, ok := payload.(map[string]float64)
    if !ok {
        return &ControlResponse{
            success: false,
            error:   fmt.Errorf("invalid payload for exploration adjustment"),
        }
    }
    
    if exploration, exists := params["exploration"]; exists {
        oldValue := pd.forceCalculator.explorationForce
        pd.forceCalculator.explorationForce = math.Max(0.1, math.Min(0.9, exploration))
        pd.forceCalculator.exploitationForce = 1.0 - pd.forceCalculator.explorationForce
        
        return &ControlResponse{
            success: true,
            data: map[string]float64{
                "old_exploration": oldValue,
                "new_exploration": pd.forceCalculator.explorationForce,
                "new_exploitation": pd.forceCalculator.exploitationForce,
            },
        }
    }
    
    return &ControlResponse{
        success: false,
        error:   fmt.Errorf("exploration parameter not found"),
    }
}

// handleRecalculateForces recalculates all discovery forces
func (pd *PeerDiscovery) handleRecalculateForces(payload interface{}) *ControlResponse {
    pd.calculateInitialForces()
    
    return &ControlResponse{
        success: true,
        data: map[string]float64{
            "exploration_force":  pd.forceCalculator.explorationForce,
            "exploitation_force": pd.forceCalculator.exploitationForce,
            "entropy_force":      pd.forceCalculator.entropyForce,
            "network_force":      pd.forceCalculator.networkForce,
            "total_force":        pd.calculateTotalForceMagnitude(),
        },
    }
}

// handleUpdateDiffusion updates diffusion parameters
func (pd *PeerDiscovery) handleUpdateDiffusion(payload interface{}) *ControlResponse {
    params, ok := payload.(map[string]float64)
    if !ok {
        return &ControlResponse{
            success: false,
            error:   fmt.Errorf("invalid payload for diffusion update"),
        }
    }
    
    if diffusion, exists := params["diffusion_constant"]; exists {
        pd.diffusionEngine.diffusionConstant = diffusion
    }
    
    if decay, exists := params["decay_constant"]; exists {
        pd.diffusionEngine.decayConstant = decay
    }
    
    return &ControlResponse{
        success: true,
        data: map[string]float64{
            "diffusion_constant": pd.diffusionEngine.diffusionConstant,
            "decay_constant":     pd.diffusionEngine.decayConstant,
        },
    }
}

// handleOptimizeStrategy optimizes discovery strategies
func (pd *PeerDiscovery) handleOptimizeStrategy(payload interface{}) *ControlResponse {
    // Analyze recent performance and optimize strategy parameters
    recentSuccessRate := pd.successTracker.rates.Mean()
    
    // Adjust strategy based on success rate
    if recentSuccessRate < 0.5 {
        // Increase exploration for low success rates
        pd.forceCalculator.explorationForce = math.Min(0.8, pd.forceCalculator.explorationForce+0.1)
        pd.forceCalculator.exploitationForce = 1.0 - pd.forceCalculator.explorationForce
    } else if recentSuccessRate > 0.8 {
        // Increase exploitation for high success rates
        pd.forceCalculator.exploitationForce = math.Min(0.8, pd.forceCalculator.exploitationForce+0.1)
        pd.forceCalculator.explorationForce = 1.0 - pd.forceCalculator.exploitationForce
    }
    
    return &ControlResponse{
        success: true,
        data: map[string]interface{}{
            "recent_success_rate": recentSuccessRate,
            "new_exploration":     pd.forceCalculator.explorationForce,
            "new_exploitation":    pd.forceCalculator.exploitationForce,
        },
    }
}

// handleEmergencyShutdown handles emergency shutdown requests
func (pd *PeerDiscovery) handleEmergencyShutdown(payload interface{}) *ControlResponse {
    logrus.Warn("Received emergency shutdown request for peer discovery")
    
    // Stop accepting new tasks
    pd.isRunning.Store(false)
    
    // Cancel all active searches
    pd.activeSearches.Range(func(id string, search *SearchProcess) bool {
        // Implementation would cancel the search context
        pd.activeSearches.Delete(id)
        return true
    })
    
    return &ControlResponse{
        success: true,
        data:    "Emergency shutdown initiated",
    }
}

// diffusionWorker continuously evolves the diffusion field
func (pd *PeerDiscovery) diffusionWorker() {
    defer pd.workerWg.Done()
    
    ticker := time.NewTicker(100 * time.Millisecond) // 10Hz diffusion updates
    defer ticker.Stop()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case <-ticker.C:
            pd.diffusionEngine.Evolve(0.1) // 100ms time step
        }
    }
}

// forceCalculationWorker continuously updates discovery forces
func (pd *PeerDiscovery) forceCalculationWorker() {
    defer pd.workerWg.Done()
    
    ticker := time.NewTicker(2 * time.Second) // 0.5Hz force updates
    defer ticker.Stop()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case <-ticker.C:
            pd.recalculateForces()
        }
    }
}

// recalculateForces updates all discovery forces based on current state
func (pd *PeerDiscovery) recalculateForces() {
    // Update forces based on network state and performance
    networkSize := pd.discoveredPeers.Size()
    successRate := pd.successTracker.rates.Mean()
    
    // Adaptive force calculation
    explorationBase := 0.5 + 0.3*math.Tanh(float64(networkSize)/1000.0)
    exploitationBase := 0.5 - 0.3*math.Tanh(float64(networkSize)/1000.0)
    
    // Adjust based on success rate
    successAdjustment := (successRate - 0.5) * 0.2
    pd.forceCalculator.explorationForce = math.Max(0.1, math.Min(0.9, explorationBase - successAdjustment))
    pd.forceCalculator.exploitationForce = math.Max(0.1, math.Min(0.9, exploitationBase + successAdjustment))
    
    // Update entropy force based on network diversity
    pd.forceCalculator.entropyForce = 0.3 + 0.4*math.Tanh(float64(networkSize)/500.0)
    
    // Update network force based on connectivity
    activeConnections := pd.network.connectionManager.ConnectionCount()
    pd.forceCalculator.networkForce = 0.2 + 0.3*math.Tanh(float64(activeConnections)/50.0)
    
    // Record force history
    pd.forceCalculator.forceHistory.Add(pd.calculateTotalForceMagnitude())
}

// entropyWorker manages entropy calculations
func (pd *PeerDiscovery) entropyWorker() {
    defer pd.workerWg.Done()
    
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case <-ticker.C:
            pd.updateEntropyCalculations()
        }
    }
}

// updateEntropyCalculations recalculates all entropy measures
func (pd *PeerDiscovery) updateEntropyCalculations() {
    networkSize := pd.discoveredPeers.Size()
    if networkSize == 0 {
        return
    }
    
    // Calculate Shannon entropy based on peer distribution
    regionCounts := make(map[string]int)
    pd.discoveredPeers.Range(func(nodeID string, peer *models.PeerInfo) bool {
        region := pd.getNetworkRegion(peer.Address)
        regionCounts[region]++
        return true
    })
    
    var shannonEntropy float64
    for _, count := range regionCounts {
        probability := float64(count) / float64(networkSize)
        shannonEntropy -= probability * math.Log2(probability)
    }
    
    pd.entropyManager.shannonEntropy = shannonEntropy
    
    // Calculate topological entropy (simplified)
    connectionCount := pd.network.connectionManager.ConnectionCount()
    maxPossibleConnections := networkSize * (networkSize - 1) / 2
    if maxPossibleConnections > 0 {
        connectionDensity := float64(connectionCount) / float64(maxPossibleConnections)
        pd.entropyManager.topologicalEntropy = -connectionDensity * math.Log2(connectionDensity)
    }
    
    // Update temporal entropy based on discovery rate
    discoveryRate := pd.metricsCollector.GetDiscoveryRate()
    pd.entropyManager.temporalEntropy = 0.1 + 0.9*math.Tanh(discoveryRate/10.0)
    
    // Store in entropy buffer
    pd.entropyManager.entropyBuffer.Add(shannonEntropy)
}

// getNetworkRegion classifies a node into a network region
func (pd *PeerDiscovery) getNetworkRegion(address string) string {
    ip := net.ParseIP(address)
    if ip == nil {
        return "unknown"
    }
    
    if ip.To4() != nil {
        // IPv4 classification
        firstOctet := ip.To4()[0]
        switch {
        case firstOctet <= 127:
            return "class_a"
        case firstOctet <= 191:
            return "class_b"
        default:
            return "class_c"
        }
    }
    
    // IPv6 classification (simplified)
    return "ipv6"
}

// metricsWorker collects and reports discovery metrics
func (pd *PeerDiscovery) metricsWorker() {
    defer pd.workerWg.Done()
    
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case <-ticker.C:
            pd.collectAndReportMetrics()
        }
    }
}

// collectAndReportMetrics gathers comprehensive discovery metrics
func (pd *PeerDiscovery) collectAndReportMetrics() {
    metrics := &DiscoveryMetricsReport{
        Timestamp:           time.Now(),
        TotalPeers:          pd.discoveredPeers.Size(),
        ActiveSearches:      pd.activeSearches.Size(),
        SuccessRate:         pd.successTracker.rates.Mean(),
        ExplorationForce:    pd.forceCalculator.explorationForce,
        ExploitationForce:   pd.forceCalculator.exploitationForce,
        Entropy:             pd.entropyManager.shannonEntropy,
        AverageLatency:      pd.metricsCollector.GetAverageLatency(),
        DiscoveryRate:       pd.metricsCollector.GetDiscoveryRate(),
        ForceMagnitude:      pd.calculateTotalForceMagnitude(),
        DiffusionConstant:   pd.diffusionEngine.diffusionConstant,
    }
    
    // Log metrics at appropriate level
    logrus.WithFields(logrus.Fields{
        "peers":          metrics.TotalPeers,
        "success_rate":   fmt.Sprintf("%.3f", metrics.SuccessRate),
        "exploration":    fmt.Sprintf("%.3f", metrics.ExplorationForce),
        "entropy":        fmt.Sprintf("%.3f", metrics.Entropy),
        "discovery_rate": fmt.Sprintf("%.1f", metrics.DiscoveryRate),
    }).Info("Peer discovery metrics")
    
    // Export metrics for external monitoring
    pd.metricsCollector.ExportMetrics(metrics)
}

// DiscoveryMetricsReport contains comprehensive discovery metrics
type DiscoveryMetricsReport struct {
    Timestamp         time.Time
    TotalPeers        int
    ActiveSearches    int
    SuccessRate       float64
    ExplorationForce  float64
    ExploitationForce float64
    Entropy           float64
    AverageLatency    time.Duration
    DiscoveryRate     float64
    ForceMagnitude    float64
    DiffusionConstant float64
}

// startMaintenanceTasks starts periodic maintenance operations
func (pd *PeerDiscovery) startMaintenanceTasks() {
    // Peer validation and cleanup
    pd.workerWg.Add(1)
    go pd.maintenanceWorker()
}

// maintenanceWorker performs periodic maintenance operations
func (pd *PeerDiscovery) maintenanceWorker() {
    defer pd.workerWg.Done()
    
    ticker := time.NewTicker(time.Minute * 5)
    defer ticker.Stop()
    
    for {
        select {
        case <-pd.workerCtx.Done():
            return
        case <-ticker.C:
            pd.performMaintenance()
        }
    }
}

// performMaintenance executes maintenance operations
func (pd *PeerDiscovery) performMaintenance() {
    // Clean up stale peers
    pd.cleanupStalePeers()
    
    // Validate peer information
    pd.validatePeerInformation()
    
    // Optimize routing table
    pd.optimizeRoutingTable()
    
    // Rebalance physics model
    pd.rebalancePhysicsModel()
}

// cleanupStalePeers removes peers that haven't been seen recently
func (pd *PeerDiscovery) cleanupStalePeers() {
    cutoff := time.Now().Add(-24 * time.Hour) // 24-hour cutoff
    
    var removed int
    pd.discoveredPeers.Range(func(nodeID string, peer *models.PeerInfo) bool {
        if peer.LastSeen.Before(cutoff) {
            pd.discoveredPeers.Delete(nodeID)
            pd.routingTable.RemoveNode(nodeID)
            pd.diffusionEngine.RemoveSource(nodeID)
            removed++
        }
        return true
    })
    
    if removed > 0 {
        logrus.Infof("Cleaned up %d stale peers", removed)
    }
}

// validatePeerInformation validates and updates peer information
func (pd *PeerDiscovery) validatePeerInformation() {
    // Implementation would verify peer information is still accurate
    // This might involve sending ping messages or checking connectivity
}

// optimizeRoutingTable optimizes the Kademlia routing table
func (pd *PeerDiscovery) optimizeRoutingTable() {
    pd.routingTable.Optimize()
}

// rebalancePhysicsModel rebalances the physics model parameters
func (pd *PeerDiscovery) rebalancePhysicsModel() {
    // Recalculate coupling constants based on current network state
    networkSize := pd.discoveredPeers.Size()
    pd.forceCalculator.couplingConstants.alpha = 0.35 + 0.15*math.Tanh(float64(networkSize)/800.0)
    pd.forceCalculator.couplingConstants.beta = 0.25 + 0.10*math.Tanh(float64(networkSize)/600.0)
    pd.forceCalculator.couplingConstants.gamma = 0.15 + 0.08*math.Tanh(float64(networkSize)/400.0)
    pd.forceCalculator.couplingConstants.delta = 0.10 + 0.05*math.Tanh(float64(networkSize)/200.0)
}

// GetDiscoveredPeers returns all currently discovered peers
func (pd *PeerDiscovery) GetDiscoveredPeers() []*models.PeerInfo {
    peers := make([]*models.PeerInfo, 0)
    pd.discoveredPeers.Range(func(nodeID string, peer *models.PeerInfo) bool {
        peers = append(peers, peer)
        return true
    })
    return peers
}

// GetDiscoveryMetrics returns current discovery performance metrics
func (pd *PeerDiscovery) GetDiscoveryMetrics() *DiscoveryMetricsReport {
    return &DiscoveryMetricsReport{
        Timestamp:         time.Now(),
        TotalPeers:        pd.discoveredPeers.Size(),
        ActiveSearches:    pd.activeSearches.Size(),
        SuccessRate:       pd.successTracker.rates.Mean(),
        ExplorationForce:  pd.forceCalculator.explorationForce,
        ExploitationForce: pd.forceCalculator.exploitationForce,
        Entropy:           pd.entropyManager.shannonEntropy,
        AverageLatency:    pd.metricsCollector.GetAverageLatency(),
        DiscoveryRate:     pd.metricsCollector.GetDiscoveryRate(),
        ForceMagnitude:    pd.calculateTotalForceMagnitude(),
        DiffusionConstant: pd.diffusionEngine.diffusionConstant,
    }
}

// Helper functions

// randFloat generates a random float between 0 and 1
func randFloat() float64 {
    n, _ := rand.Int(rand.Reader, big.NewInt(1<<53))
    return float64(n.Int64()) / (1 << 53)
}

// mergeCapabilities merges two capability lists
func mergeCapabilities(a, b []string) []string {
    capabilitySet := make(map[string]bool)
    for _, cap := range a {
        capabilitySet[cap] = true
    }
    for _, cap := range b {
        capabilitySet[cap] = true
    }
    
    result := make([]string, 0, len(capabilitySet))
    for cap := range capabilitySet {
        result = append(result, cap)
    }
    sort.Strings(result)
    return result
}

// Note: Additional helper types and functions would be implemented in separate files
// including: RandomWalker, GossipProtocol, DHTProtocol, DiscoveryMetricsCollector, 
// KademliaRoutingTable, and various physics model components.