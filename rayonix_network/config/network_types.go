package config

import "time"

// PeerState represents the connection state of a peer in the physics-inspired network
type PeerState int

const (
	Disconnected PeerState = iota
	Connecting
	Connected
	Ready
	Disconnecting
	Failed
)

// String returns the string representation of PeerState
func (s PeerState) String() string {
	switch s {
	case Disconnected:
		return "disconnected"
	case Connecting:
		return "connecting"
	case Connected:
		return "connected"
	case Ready:
		return "ready"
	case Disconnecting:
		return "disconnecting"
	case Failed:
		return "failed"
	default:
		return "unknown"
	}
}

// DiscoveryAlgorithm defines the algorithm used for physics-inspired peer discovery
type DiscoveryAlgorithm int

const (
	KademliaDiscovery DiscoveryAlgorithm = iota
	RandomWalkDiscovery
	DiffusionDiscovery
	HybridDiscovery
)

// RoutingAlgorithm defines the algorithm used for entropy-driven message routing
type RoutingAlgorithm int

const (
	FloodRouting RoutingAlgorithm = iota
	GossipRouting
	DHTRouting
	PhysicsRouting
)

// SecurityLevel defines the security level for cryptographic identity
type SecurityLevel int

const (
	SecurityNone SecurityLevel = iota
	SecurityLow
	SecurityMedium
	SecurityHigh
)

// CompressionType defines the compression algorithm for efficient message propagation
type CompressionType int

const (
	CompressionNone CompressionType = iota
	CompressionSnappy
	CompressionGzip
	CompressionZstd
)

// NetworkEvent represents events in the physics-inspired network layer
type NetworkEvent struct {
	Type      NetworkEventType
	PeerID    string
	Timestamp time.Time
	Data      interface{}
}

// NetworkEventType defines types of network events for monitoring
type NetworkEventType int

const (
	PeerConnected NetworkEventType = iota
	PeerDisconnected
	MessageReceived
	MessageSent
	ConnectionFailed
	DiscoveryComplete
	SecurityViolation
	TopologyUpdated
	EntropyChanged
)

// HandshakeData contains data exchanged during cryptographic handshake
type HandshakeData struct {
	Version      string
	NetworkID    string
	NodeID       string
	ListenPort   int
	Capabilities []string
	Timestamp    int64
	Nonce        uint64
	PublicKey    []byte
}

// Capability represents a node's capabilities for protocol negotiation
type Capability struct {
	Name    string
	Version string
	Support bool
}

// NetworkStats contains physics-inspired network statistics
type NetworkStats struct {
	TotalPeers          int
	ActivePeers         int
	MessagesSent        int64
	MessagesReceived    int64
	BytesSent           int64
	BytesReceived       int64
	ConnectionAttempts  int
	FailedConnections   int
	AverageLatency      time.Duration
	Uptime              time.Duration
	NetworkEntropy      float64
	PotentialEnergy     float64
	KineticEnergy       float64
}

// PhysicsParameters contains parameters for the physics-inspired model
type PhysicsParameters struct {
	EntropyDecayRate   float64
	ForceGain         float64
	PotentialDecay    float64
	StochasticFactor  float64
	Temperature       float64
	FieldStrength     float64
	WaveSpeed         float64
	DampingFactor     float64
}

// RateLimitPolicy defines rate limiting policies for DDoS protection
type RateLimitPolicy struct {
	Type         RateLimitType
	Limit        int
	Burst        int
	Interval     time.Duration
	ApplyPerPeer bool
}

// RateLimitType defines types of rate limiting in the network
type RateLimitType int

const (
	RateLimitMessage RateLimitType = iota
	RateLimitConnection
	RateLimitBandwidth
	RateLimitDiscovery
)

// BanPolicy defines banning policies for misbehaving peers
type BanPolicy struct {
	Duration  time.Duration
	Threshold int
	AutoUnban bool
	Reason    string
}

// DHTConfig contains Kademlia DHT-specific configuration
type DHTConfig struct {
	BucketSize      int
	Alpha           int
	BootstrapNodes  []string
	RefreshInterval time.Duration
	StorageSize     int
}

// ProtocolConfig contains hybrid transport protocol configuration
type ProtocolConfig struct {
	EnableTCP       bool
	EnableUDP       bool
	EnableWebSocket bool
	TCPConfig       TCPConfig
	UDPConfig       UDPConfig
	WSConfig        WebSocketConfig
}

// TCPConfig contains TCP-specific configuration for validator channels
type TCPConfig struct {
	KeepAlive       bool
	NoDelay         bool
	ReadBufferSize  int
	WriteBufferSize int
}

// UDPConfig contains UDP-specific configuration for discovery beacons
type UDPConfig struct {
	ReadBufferSize  int
	WriteBufferSize int
	MaxPacketSize   int
}

// WebSocketConfig contains WebSocket-specific configuration for external APIs
type WebSocketConfig struct {
	ReadBufferSize    int
	WriteBufferSize   int
	EnableCompression bool
	CheckOrigin       bool
}

// SecurityConfig contains security-specific configuration
type SecurityConfig struct {
	EnableEncryption    bool
	EnableSigning       bool
	KeyRotationInterval time.Duration
	SessionTimeout      time.Duration
	MaxSessions         int
}

// MessagePriority defines message priority levels for entropy-driven routing
type MessagePriority int

const (
	PriorityLow MessagePriority = iota
	PriorityNormal
	PriorityHigh
	PriorityCritical
)

// String returns the string representation of MessagePriority
func (p MessagePriority) String() string {
	switch p {
	case PriorityLow:
		return "low"
	case PriorityNormal:
		return "normal"
	case PriorityHigh:
		return "high"
	case PriorityCritical:
		return "critical"
	default:
		return "unknown"
	}
}