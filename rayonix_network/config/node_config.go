package config

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"time"
)

// ProtocolType defines the network protocol type for physics-inspired transport
type ProtocolType int

const (
	TCP ProtocolType = iota
	UDP
	WebSocket
)

// String returns the string representation of ProtocolType
func (p ProtocolType) String() string {
	switch p {
	case TCP:
		return "tcp"
	case UDP:
		return "udp"
	case WebSocket:
		return "websocket"
	default:
		return "unknown"
	}
}

// MessageType defines the types of network messages for consensus coupling
type MessageType int

const (
	Handshake MessageType = iota
	Ping
	Pong
	PeerList
	Block
	Transaction
	Consensus
	Gossip
	SyncRequest
	SyncResponse
)

// String returns the string representation of MessageType
func (m MessageType) String() string {
	switch m {
	case Handshake:
		return "handshake"
	case Ping:
		return "ping"
	case Pong:
		return "pong"
	case PeerList:
		return "peerlist"
	case Block:
		return "block"
	case Transaction:
		return "transaction"
	case Consensus:
		return "consensus"
	case Gossip:
		return "gossip"
	case SyncRequest:
		return "syncrequest"
	case SyncResponse:
		return "syncresponse"
	default:
		return "unknown"
	}
}

// NodeConfig contains all configuration parameters for physics-inspired P2P node
type NodeConfig struct {
	// Node identification
	NodeID    string `json:"node_id"`
	NetworkID string `json:"network_id"`
	Version   string `json:"version"`

	// Network addresses for hybrid transport
	ListenIP   string `json:"listen_ip"`
	ListenPort int    `json:"listen_port"`
	UDPPort    int    `json:"udp_port"`
	APIPort    int    `json:"api_port"`

	// Connection limits for topology optimization
	MaxConnections    int           `json:"max_connections"`
	MaxIncomingConns  int           `json:"max_incoming_connections"`
	MaxOutgoingConns  int           `json:"max_outgoing_connections"`
	ConnectionTimeout time.Duration `json:"connection_timeout"`

	// Message handling for entropy-driven gossip
	MaxMessageSize    int           `json:"max_message_size"`
	MessageTimeout    time.Duration `json:"message_timeout"`
	HandshakeTimeout  time.Duration `json:"handshake_timeout"`
	KeepAliveInterval time.Duration `json:"keepalive_interval"`

	// Peer discovery for diffusion processes
	BootstrapPeers    []string      `json:"bootstrap_peers"`
	DiscoveryInterval time.Duration `json:"discovery_interval"`
	PeerCacheSize     int           `json:"peer_cache_size"`
	MinPeers          int           `json:"min_peers"`
	MaxPeers          int           `json:"max_peers"`

	// Security for cryptographic identity
	EnableEncryption  bool          `json:"enable_encryption"`
	EnableSigning     bool          `json:"enable_signing"`
	SessionKeyTimeout time.Duration `json:"session_key_timeout"`
	BanDuration       time.Duration `json:"ban_duration"`
	BanThreshold      int           `json:"ban_threshold"`

	// Rate limiting for DDoS resilience
	RateLimitPerPeer  int           `json:"rate_limit_per_peer"`
	RateLimitBurst    int           `json:"rate_limit_burst"`
	RateLimitInterval time.Duration `json:"rate_limit_interval"`

	// Physics model parameters for dynamic adaptation
	NetworkEntropyDecay float64 `json:"network_entropy_decay"`
	ConnectionForceGain float64 `json:"connection_force_gain"`
	PotentialDecayRate  float64 `json:"potential_decay_rate"`
	StochasticForce     float64 `json:"stochastic_force"`
	SoftmaxTemperature  float64 `json:"softmax_temperature"`

	// DHT configuration for Kademlia discovery
	DHTBootstrapNodes []string `json:"dht_bootstrap_nodes"`
	DHTBucketSize     int      `json:"dht_bucket_size"`
	DHTAlpha          int      `json:"dht_alpha"`

	// File paths for persistence
	DataDirectory   string `json:"data_directory"`
	KeyFile         string `json:"key_file"`
	PeerDatabase    string `json:"peer_database"`
	MetricsDatabase string `json:"metrics_database"`
}

// DefaultConfig returns the default configuration for physics-inspired P2P network
func DefaultConfig() *NodeConfig {
	return &NodeConfig{
		NodeID:               "",
		NetworkID:           "mainnet",
		Version:             "1.0.0",
		ListenIP:            "0.0.0.0",
		ListenPort:          30303,
		UDPPort:             30304,
		APIPort:             8080,
		MaxConnections:      200,
		MaxIncomingConns:    100,
		MaxOutgoingConns:    100,
		ConnectionTimeout:   30 * time.Second,
		MaxMessageSize:      10 * 1024 * 1024, // 10MB
		MessageTimeout:      30 * time.Second,
		HandshakeTimeout:    10 * time.Second,
		KeepAliveInterval:   30 * time.Second,
		BootstrapPeers:      []string{},
		DiscoveryInterval:   60 * time.Second,
		PeerCacheSize:       10000,
		MinPeers:            10,
		MaxPeers:            200,
		EnableEncryption:    true,
		EnableSigning:       true,
		SessionKeyTimeout:   24 * time.Hour,
		BanDuration:         24 * time.Hour,
		BanThreshold:        10,
		RateLimitPerPeer:    1000,
		RateLimitBurst:      100,
		RateLimitInterval:   time.Second,
		NetworkEntropyDecay: 0.01,
		ConnectionForceGain: 0.1,
		PotentialDecayRate:  0.05,
		StochasticForce:     0.01,
		SoftmaxTemperature:  1.0,
		DHTBootstrapNodes:   []string{},
		DHTBucketSize:       20,
		DHTAlpha:            3,
		DataDirectory:       "./data",
		KeyFile:            "node.key",
		PeerDatabase:       "peers.db",
		MetricsDatabase:    "metrics.db",
	}
}

// Validate validates the configuration parameters
func (c *NodeConfig) Validate() error {
	if c.ListenPort <= 0 || c.ListenPort > 65535 {
		return fmt.Errorf("invalid listen port: %d", c.ListenPort)
	}

	if c.UDPPort <= 0 || c.UDPPort > 65535 {
		return fmt.Errorf("invalid UDP port: %d", c.UDPPort)
	}

	if c.APIPort <= 0 || c.APIPort > 65535 {
		return fmt.Errorf("invalid API port: %d", c.APIPort)
	}

	if c.MaxConnections <= 0 {
		return fmt.Errorf("max connections must be positive")
	}

	if c.MaxMessageSize <= 0 {
		return fmt.Errorf("max message size must be positive")
	}

	// Validate physics parameters
	if c.NetworkEntropyDecay < 0 || c.NetworkEntropyDecay > 1 {
		return fmt.Errorf("network entropy decay must be between 0 and 1")
	}

	if c.SoftmaxTemperature <= 0 {
		return fmt.Errorf("softmax temperature must be positive")
	}

	// Validate IP address
	if ip := net.ParseIP(c.ListenIP); ip == nil {
		return fmt.Errorf("invalid listen IP: %s", c.ListenIP)
	}

	return nil
}

// LoadConfig loads configuration from a file
func LoadConfig(filename string) (*NodeConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config NodeConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("config validation failed: %w", err)
	}

	return &config, nil
}

// SaveConfig saves configuration to a file
func (c *NodeConfig) SaveConfig(filename string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	// Create directory if it doesn't exist
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// GetDataPath returns the full path for a data file
func (c *NodeConfig) GetDataPath(filename string) string {
	return filepath.Join(c.DataDirectory, filename)
}

// GetPeerDatabasePath returns the path to the peer database
func (c *NodeConfig) GetPeerDatabasePath() string {
	return c.GetDataPath(c.PeerDatabase)
}

// GetMetricsDatabasePath returns the path to the metrics database
func (c *NodeConfig) GetMetricsDatabasePath() string {
	return c.GetDataPath(c.MetricsDatabase)
}

// GetKeyFilePath returns the path to the node key file
func (c *NodeConfig) GetKeyFilePath() string {
	return c.GetDataPath(c.KeyFile)
}

// InitializeDataDirectory creates the data directory structure
func (c *NodeConfig) InitializeDataDirectory() error {
	dirs := []string{
		c.DataDirectory,
		filepath.Join(c.DataDirectory, "logs"),
		filepath.Join(c.DataDirectory, "keys"),
		filepath.Join(c.DataDirectory, "db"),
		filepath.Join(c.DataDirectory, "cache"),
	}

	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}