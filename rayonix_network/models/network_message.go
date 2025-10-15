package models

import (
    "bytes"
    "crypto/sha256"
    "encoding/binary"
    "encoding/json"
    "fmt"
    "io"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/utils"
    "github.com/vmihailenco/msgpack/v5"
    "golang.org/x/crypto/sha3"
)

// NetworkMessage represents a complete network message with physics-inspired routing metadata
type NetworkMessage struct {
    // Header section (fixed layout for zero-copy parsing)
    Header *MessageHeader `json:"header" msgpack:"header"`
    
    // Payload section
    Payload []byte `json:"payload" msgpack:"payload"`
    
    // Routing and delivery metadata
    RoutingInfo *MessageRouting `json:"routing_info" msgpack:"routing_info"`
    DeliveryInfo *DeliveryMetadata `json:"delivery_info" msgpack:"delivery_info"`
    
    // Physics-inspired routing properties
    PhysicsMetadata *PhysicsMetadata `json:"physics_metadata" msgpack:"physics_metadata"`
    
    // Security and validation
    Signature []byte `json:"signature" msgpack:"signature"`
    Validation *ValidationInfo `json:"validation" msgpack:"validation"`
    
    // Internal processing state
    receivedAt  time.Time `json:"-" msgpack:"-"`
    processedAt time.Time `json:"-" msgpack:"-"`
    priority    int       `json:"-" msgpack:"-"`
    attempts    uint32    `json:"-" msgpack:"-"`
}

// MessageHeader contains fixed-size message header for efficient parsing
type MessageHeader struct {
    // Magic number and version
    Magic        [4]byte           `json:"magic" msgpack:"magic"`
    Version      uint16            `json:"version" msgpack:"version"`
    Flags        uint16            `json:"flags" msgpack:"flags"`
    
    // Message identification
    MessageID    [32]byte          `json:"message_id" msgpack:"message_id"`
    MessageType  config.MessageType `json:"message_type" msgpack:"message_type"`
    Priority     uint8             `json:"priority" msgpack:"priority"`
    
    // Size and timing
    PayloadSize  uint32            `json:"payload_size" msgpack:"payload_size"`
    TotalSize    uint32            `json:"total_size" msgpack:"total_size"`
    Timestamp    int64             `json:"timestamp" msgpack:"timestamp"`
    TTL          uint32            `json:"ttl" msgpack:"ttl"`
    
    // Source and destination
    SourceNode   [32]byte          `json:"source_node" msgpack:"source_node"`
    TargetNode   [32]byte          `json:"target_node" msgpack:"target_node"`
    
    // Cryptographic nonce
    Nonce        [24]byte          `json:"nonce" msgpack:"nonce"`
    
    // Checksum for integrity verification
    Checksum     [32]byte          `json:"checksum" msgpack:"checksum"`
}

// MessageRouting contains dynamic routing information
type MessageRouting struct {
    // Path information
    Hops         []HopInfo         `json:"hops" msgpack:"hops"`
    MaxHops      uint8             `json:"max_hops" msgpack:"max_hops"`
    CurrentHop   uint8             `json:"current_hop" msgpack:"current_hop"`
    
    // Routing strategy
    RoutingType  config.RoutingType `json:"routing_type" msgpack:"routing_type"`
    RoutingFlags uint16            `json:"routing_flags" msgpack:"routing_flags"`
    
    // Quality of service
    QoS          config.QoSLevel   `json:"qos" msgpack:"qos"`
    Reliability  uint8             `json:"reliability" msgpack:"reliability"`
    
    // Forwarding constraints
    ExcludeNodes [][32]byte        `json:"exclude_nodes" msgpack:"exclude_nodes"`
    IncludeNodes [][32]byte        `json:"include_nodes" msgpack:"include_nodes"`
    
    // Cost metrics
    PathCost     float64           `json:"path_cost" msgpack:"path_cost"`
    EnergyCost   float64           `json:"energy_cost" msgpack:"energy_cost"`
}

// HopInfo represents a single hop in message routing path
type HopInfo struct {
    NodeID      [32]byte `json:"node_id" msgpack:"node_id"`
    Address     string   `json:"address" msgpack:"address"`
    Port        uint16   `json:"port" msgpack:"port"`
    Timestamp   int64    `json:"timestamp" msgpack:"timestamp"`
    Latency     int64    `json:"latency" msgpack:"latency"`
    Success     bool     `json:"success" msgpack:"success"`
}

// DeliveryMetadata tracks message delivery progress
type DeliveryMetadata struct {
    // Acknowledgment tracking
    RequiresAck  bool              `json:"requires_ack" msgpack:"requires_ack"`
    AckReceived  bool              `json:"ack_received" msgpack:"ack_received"`
    AckNodes     [][32]byte        `json:"ack_nodes" msgpack:"ack_nodes"`
    
    // Retry and timeout configuration
    MaxRetries   uint8             `json:"max_retries" msgpack:"max_retries"`
    RetryCount   uint8             `json:"retry_count" msgpack:"retry_count"`
    Timeout      int64             `json:"timeout" msgpack:"timeout"`
    
    // Delivery guarantees
    Guarantee    config.DeliveryGuarantee `json:"guarantee" msgpack:"guarantee"`
    Persistence  config.PersistenceLevel `json:"persistence" msgpack:"persistence"`
    
    // Sequence tracking
    Sequence     uint64            `json:"sequence" msgpack:"sequence"`
    ExpectedAck  uint64            `json:"expected_ack" msgpack:"expected_ack"`
}

// PhysicsMetadata contains physics-inspired routing properties
type PhysicsMetadata struct {
    // Field properties
    FieldStrength float64 `json:"field_strength" msgpack:"field_strength"`
    Potential     float64 `json:"potential" msgpack:"potential"`
    Entropy       float64 `json:"entropy" msgpack:"entropy"`
    
    // Force vectors
    ForceVector   *ForceVector `json:"force_vector" msgpack:"force_vector"`
    Gradient      *Vector3D    `json:"gradient" msgpack:"gradient"`
    
    // Wave properties
    WaveAmplitude float64 `json:"wave_amplitude" msgpack:"wave_amplitude"`
    WaveFrequency float64 `json:"wave_frequency" msgpack:"wave_frequency"`
    WavePhase     float64 `json:"wave_phase" msgpack:"wave_phase"`
    
    // Quantum properties
    Superposition float64 `json:"superposition" msgpack:"superposition"`
    Coherence     float64 `json:"coherence" msgpack:"coherence"`
    Entanglement  string  `json:"entanglement" msgpack:"entanglement"`
}

// Vector3D represents a 3D vector for physics calculations
type Vector3D struct {
    X float64 `json:"x" msgpack:"x"`
    Y float64 `json:"y" msgpack:"y"`
    Z float64 `json:"z" msgpack:"z"`
}

// ValidationInfo contains message validation data
type ValidationInfo struct {
    // Cryptographic validation
    HashValid    bool   `json:"hash_valid" msgpack:"hash_valid"`
    SigValid     bool   `json:"sig_valid" msgpack:"sig_valid"`
    NonceValid   bool   `json:"nonce_valid" msgpack:"nonce_valid"`
    
    // Structural validation
    SizeValid    bool   `json:"size_valid" msgpack:"size_valid"`
    FormatValid  bool   `json:"format_valid" msgpack:"format_valid"`
    
    // Policy validation
    PolicyValid  bool   `json:"policy_valid" msgpack:"policy_valid"`
    Allowed      bool   `json:"allowed" msgpack:"allowed"`
    
    // Timestamp validation
    TimeValid    bool   `json:"time_valid" msgpack:"time_valid"`
    NotExpired   bool   `json:"not_expired" msgpack:"not_expired"`
    
    // Validation errors
    Errors       []string `json:"errors" msgpack:"errors"`
}

// NewNetworkMessage creates a new network message with proper initialization
func NewNetworkMessage(messageType config.MessageType, sourceNode, targetNode string, payload []byte) *NetworkMessage {
    now := time.Now().UnixNano()
    messageID := generateMessageID(sourceNode, now)
    
    msg := &NetworkMessage{
        Header: &MessageHeader{
            Magic:        [4]byte{'R', 'A', 'Y', 'X'},
            Version:      1,
            Flags:        0,
            MessageID:    messageID,
            MessageType:  messageType,
            Priority:     uint8(config.PriorityNormal),
            PayloadSize:  uint32(len(payload)),
            TotalSize:    uint32(len(payload)) + 256, // Header + routing overhead
            Timestamp:    now,
            TTL:          3600, // 1 hour default TTL
        },
        Payload: payload,
        RoutingInfo: &MessageRouting{
            Hops:        make([]HopInfo, 0),
            MaxHops:     10,
            CurrentHop:  0,
            RoutingType: config.RoutingGossip,
            QoS:         config.QoSNormal,
            Reliability: 80,
            PathCost:    1.0,
            EnergyCost:  0.0,
        },
        DeliveryInfo: &DeliveryMetadata{
            RequiresAck: false,
            MaxRetries:  3,
            Timeout:     int64(30 * time.Second),
            Guarantee:   config.GuaranteeBestEffort,
            Persistence: config.PersistenceVolatile,
        },
        PhysicsMetadata: &PhysicsMetadata{
            FieldStrength: 1.0,
            Potential:     0.5,
            Entropy:       0.3,
            ForceVector: &ForceVector{
                Magnitude:  0.0,
                DirectionX: 0.0,
                DirectionY: 0.0,
                DirectionZ: 0.0,
                Attraction: 0.5,
                Repulsion:  0.5,
                LastUpdated: time.Now(),
            },
            Gradient: &Vector3D{X: 0.0, Y: 0.0, Z: 0.0},
            WaveAmplitude: 1.0,
            WaveFrequency: 1.0,
            WavePhase:     0.0,
            Superposition: 1.0,
            Coherence:     1.0,
        },
        Validation: &ValidationInfo{
            HashValid:   false,
            SigValid:    false,
            NonceValid:  false,
            SizeValid:   false,
            FormatValid: false,
            PolicyValid: false,
            Allowed:     false,
            TimeValid:   false,
            NotExpired:  false,
            Errors:      make([]string, 0),
        },
        receivedAt:  time.Now(),
        priority:    int(config.PriorityNormal),
        attempts:    0,
    }
    
    // Set source and target nodes
    copy(msg.Header.SourceNode[:], utils.SHA3Hash([]byte(sourceNode))[:32])
    if targetNode != "" {
        copy(msg.Header.TargetNode[:], utils.SHA3Hash([]byte(targetNode))[:32])
    }
    
    // Generate nonce
    nonce := generateNonce()
    copy(msg.Header.Nonce[:], nonce[:])
    
    return msg
}

// generateMessageID creates a unique message ID
func generateMessageID(sourceNode string, timestamp int64) [32]byte {
    data := fmt.Sprintf("%s:%d:%d", sourceNode, timestamp, utils.RandomUint64())
    hash := sha3.Sum256([]byte(data))
    return hash
}

// generateNonce creates a cryptographic nonce
func generateNonce() [24]byte {
    var nonce [24]byte
    randomData := utils.GenerateRandomBytes(24)
    copy(nonce[:], randomData)
    return nonce
}

// CalculateChecksum computes the message checksum
func (m *NetworkMessage) CalculateChecksum() [32]byte {
    // Combine header fields (excluding checksum itself)
    var data []byte
    data = append(data, m.Header.Magic[:]...)
    data = append(data, utils.Uint16ToBytes(m.Header.Version)...)
    data = append(data, utils.Uint16ToBytes(m.Header.Flags)...)
    data = append(data, m.Header.MessageID[:]...)
    data = append(data, byte(m.Header.MessageType))
    data = append(data, m.Header.Priority)
    data = append(data, utils.Uint32ToBytes(m.Header.PayloadSize)...)
    data = append(data, utils.Uint32ToBytes(m.Header.TotalSize)...)
    data = append(data, utils.Int64ToBytes(m.Header.Timestamp)...)
    data = append(data, utils.Uint32ToBytes(m.Header.TTL)...)
    data = append(data, m.Header.SourceNode[:]...)
    data = append(data, m.Header.TargetNode[:]...)
    data = append(data, m.Header.Nonce[:]...)
    
    // Include payload
    data = append(data, m.Payload...)
    
    return sha3.Sum256(data)
}

// UpdateChecksum updates the message checksum
func (m *NetworkMessage) UpdateChecksum() {
    checksum := m.CalculateChecksum()
    copy(m.Header.Checksum[:], checksum[:])
}

// VerifyChecksum verifies message integrity
func (m *NetworkMessage) VerifyChecksum() bool {
    expected := m.CalculateChecksum()
    return bytes.Equal(m.Header.Checksum[:], expected[:])
}

// Serialize serializes the message to binary format
func (m *NetworkMessage) Serialize() ([]byte, error) {
    // Update checksum before serialization
    m.UpdateChecksum()
    
    // Use MessagePack for efficient serialization
    return msgpack.Marshal(m)
}

// Deserialize deserializes binary data to NetworkMessage
func Deserialize(data []byte) (*NetworkMessage, error) {
    var msg NetworkMessage
    if err := msgpack.Unmarshal(data, &msg); err != nil {
        return nil, fmt.Errorf("failed to deserialize message: %w", err)
    }
    
    // Verify checksum
    if !msg.VerifyChecksum() {
        return nil, fmt.Errorf("message checksum verification failed")
    }
    
    msg.receivedAt = time.Now()
    return &msg, nil
}

// AddHop adds a hop to the message routing path
func (m *NetworkMessage) AddHop(nodeID, address string, port uint16, latency time.Duration, success bool) {
    if m.RoutingInfo == nil {
        m.RoutingInfo = &MessageRouting{
            Hops: make([]HopInfo, 0),
        }
    }
    
    hop := HopInfo{
        Address:   address,
        Port:      port,
        Timestamp: time.Now().UnixNano(),
        Latency:   latency.Nanoseconds(),
        Success:   success,
    }
    
    copy(hop.NodeID[:], utils.SHA3Hash([]byte(nodeID))[:32])
    
    m.RoutingInfo.Hops = append(m.RoutingInfo.Hops, hop)
    m.RoutingInfo.CurrentHop++
}

// GetSourceNode returns the source node ID as string
func (m *NetworkMessage) GetSourceNode() string {
    return utils.BytesToHex(m.Header.SourceNode[:])
}

// GetTargetNode returns the target node ID as string
func (m *NetworkMessage) GetTargetNode() string {
    return utils.BytesToHex(m.Header.TargetNode[:])
}

// IsExpired checks if the message has expired based on TTL
func (m *NetworkMessage) IsExpired() bool {
    messageTime := time.Unix(0, m.Header.Timestamp)
    expirationTime := messageTime.Add(time.Duration(m.Header.TTL) * time.Second)
    return time.Now().After(expirationTime)
}

// GetAge returns the age of the message
func (m *NetworkMessage) GetAge() time.Duration {
    messageTime := time.Unix(0, m.Header.Timestamp)
    return time.Since(messageTime)
}

// ShouldRetry determines if the message should be retried
func (m *NetworkMessage) ShouldRetry() bool {
    if m.DeliveryInfo == nil {
        return false
    }
    
    return m.attempts < uint32(m.DeliveryInfo.MaxRetries) && !m.IsExpired()
}

// RecordAttempt records a delivery attempt
func (m *NetworkMessage) RecordAttempt() {
    m.attempts++
}

// GetAttempts returns the number of delivery attempts
func (m *NetworkMessage) GetAttempts() uint32 {
    return m.attempts
}

// SetPriority sets the message processing priority
func (m *NetworkMessage) SetPriority(priority config.MessagePriority) {
    m.priority = int(priority)
    m.Header.Priority = uint8(priority)
}

// GetPriority returns the message processing priority
func (m *NetworkMessage) GetPriority() config.MessagePriority {
    return config.MessagePriority(m.priority)
}

// CalculateRoutingCost computes the current routing cost
func (m *NetworkMessage) CalculateRoutingCost() float64 {
    if m.RoutingInfo == nil {
        return 1.0
    }
    
    baseCost := m.RoutingInfo.PathCost
    hopPenalty := float64(m.RoutingInfo.CurrentHop) * 0.1
    energyCost := m.RoutingInfo.EnergyCost
    
    return baseCost + hopPenalty + energyCost
}

// UpdatePhysicsProperties updates physics metadata based on current state
func (m *NetworkMessage) UpdatePhysicsProperties(networkEntropy, fieldStrength float64) {
    if m.PhysicsMetadata == nil {
        m.PhysicsMetadata = &PhysicsMetadata{}
    }
    
    // Update entropy based on network state and message age
    ageFactor := 1.0 - math.Min(m.GetAge().Seconds()/3600.0, 1.0)
    m.PhysicsMetadata.Entropy = networkEntropy * ageFactor
    
    // Update field strength
    m.PhysicsMetadata.FieldStrength = fieldStrength * (1.0 - float64(m.RoutingInfo.CurrentHop)/10.0)
    
    // Update potential based on routing cost
    routingCost := m.CalculateRoutingCost()
    m.PhysicsMetadata.Potential = 1.0 / (1.0 + routingCost)
}

// ValidateStructure performs structural validation of the message
func (m *NetworkMessage) ValidateStructure() error {
    if m.Header == nil {
        return fmt.Errorf("message header is nil")
    }
    
    // Check magic number
    expectedMagic := [4]byte{'R', 'A', 'Y', 'X'}
    if m.Header.Magic != expectedMagic {
        return fmt.Errorf("invalid magic number")
    }
    
    // Check version
    if m.Header.Version != 1 {
        return fmt.Errorf("unsupported message version: %d", m.Header.Version)
    }
    
    // Check payload size
    if uint32(len(m.Payload)) != m.Header.PayloadSize {
        return fmt.Errorf("payload size mismatch: expected %d, got %d", 
            m.Header.PayloadSize, len(m.Payload))
    }
    
    // Check TTL
    if m.Header.TTL == 0 {
        return fmt.Errorf("message TTL is zero")
    }
    
    // Check timestamp (not in future and not too far in past)
    messageTime := time.Unix(0, m.Header.Timestamp)
    now := time.Now()
    
    if messageTime.After(now.Add(5 * time.Minute)) {
        return fmt.Errorf("message timestamp is in the future")
    }
    
    if messageTime.Before(now.Add(-24 * time.Hour)) {
        return fmt.Errorf("message timestamp is too far in the past")
    }
    
    return nil
}

// Clone creates a deep copy of the message for forwarding
func (m *NetworkMessage) Clone() *NetworkMessage {
    clone := &NetworkMessage{
        Header: &MessageHeader{},
        Payload: make([]byte, len(m.Payload)),
        receivedAt:  m.receivedAt,
        processedAt: m.processedAt,
        priority:    m.priority,
        attempts:    m.attempts,
    }
    
    // Copy header
    *clone.Header = *m.Header
    
    // Copy payload
    copy(clone.Payload, m.Payload)
    
    // Copy routing info
    if m.RoutingInfo != nil {
        clone.RoutingInfo = &MessageRouting{
            Hops:         make([]HopInfo, len(m.RoutingInfo.Hops)),
            MaxHops:      m.RoutingInfo.MaxHops,
            CurrentHop:   m.RoutingInfo.CurrentHop,
            RoutingType:  m.RoutingInfo.RoutingType,
            RoutingFlags: m.RoutingInfo.RoutingFlags,
            QoS:          m.RoutingInfo.QoS,
            Reliability:  m.RoutingInfo.Reliability,
            PathCost:     m.RoutingInfo.PathCost,
            EnergyCost:   m.RoutingInfo.EnergyCost,
        }
        copy(clone.RoutingInfo.Hops, m.RoutingInfo.Hops)
        
        // Copy exclude/include nodes
        if len(m.RoutingInfo.ExcludeNodes) > 0 {
            clone.RoutingInfo.ExcludeNodes = make([][32]byte, len(m.RoutingInfo.ExcludeNodes))
            copy(clone.RoutingInfo.ExcludeNodes, m.RoutingInfo.ExcludeNodes)
        }
        
        if len(m.RoutingInfo.IncludeNodes) > 0 {
            clone.RoutingInfo.IncludeNodes = make([][32]byte, len(m.RoutingInfo.IncludeNodes))
            copy(clone.RoutingInfo.IncludeNodes, m.RoutingInfo.IncludeNodes)
        }
    }
    
    // Copy delivery info
    if m.DeliveryInfo != nil {
        clone.DeliveryInfo = &DeliveryMetadata{
            RequiresAck:  m.DeliveryInfo.RequiresAck,
            AckReceived:  m.DeliveryInfo.AckReceived,
            MaxRetries:   m.DeliveryInfo.MaxRetries,
            RetryCount:   m.DeliveryInfo.RetryCount,
            Timeout:      m.DeliveryInfo.Timeout,
            Guarantee:    m.DeliveryInfo.Guarantee,
            Persistence:  m.DeliveryInfo.Persistence,
            Sequence:     m.DeliveryInfo.Sequence,
            ExpectedAck:  m.DeliveryInfo.ExpectedAck,
        }
        
        if len(m.DeliveryInfo.AckNodes) > 0 {
            clone.DeliveryInfo.AckNodes = make([][32]byte, len(m.DeliveryInfo.AckNodes))
            copy(clone.DeliveryInfo.AckNodes, m.DeliveryInfo.AckNodes)
        }
    }
    
    // Copy physics metadata
    if m.PhysicsMetadata != nil {
        clone.PhysicsMetadata = &PhysicsMetadata{
            FieldStrength: m.PhysicsMetadata.FieldStrength,
            Potential:     m.PhysicsMetadata.Potential,
            Entropy:       m.PhysicsMetadata.Entropy,
            WaveAmplitude: m.PhysicsMetadata.WaveAmplitude,
            WaveFrequency: m.PhysicsMetadata.WaveFrequency,
            WavePhase:     m.PhysicsMetadata.WavePhase,
            Superposition: m.PhysicsMetadata.Superposition,
            Coherence:     m.PhysicsMetadata.Coherence,
            Entanglement:  m.PhysicsMetadata.Entanglement,
        }
        
        if m.PhysicsMetadata.ForceVector != nil {
            clone.PhysicsMetadata.ForceVector = &ForceVector{
                Magnitude:   m.PhysicsMetadata.ForceVector.Magnitude,
                DirectionX:  m.PhysicsMetadata.ForceVector.DirectionX,
                DirectionY:  m.PhysicsMetadata.ForceVector.DirectionY,
                DirectionZ:  m.PhysicsMetadata.ForceVector.DirectionZ,
                Attraction:  m.PhysicsMetadata.ForceVector.Attraction,
                Repulsion:   m.PhysicsMetadata.ForceVector.Repulsion,
                LastUpdated: m.PhysicsMetadata.ForceVector.LastUpdated,
            }
        }
        
        if m.PhysicsMetadata.Gradient != nil {
            clone.PhysicsMetadata.Gradient = &Vector3D{
                X: m.PhysicsMetadata.Gradient.X,
                Y: m.PhysicsMetadata.Gradient.Y,
                Z: m.PhysicsMetadata.Gradient.Z,
            }
        }
    }
    
    // Copy validation info
    if m.Validation != nil {
        clone.Validation = &ValidationInfo{
            HashValid:   m.Validation.HashValid,
            SigValid:    m.Validation.SigValid,
            NonceValid:  m.Validation.NonceValid,
            SizeValid:   m.Validation.SizeValid,
            FormatValid: m.Validation.FormatValid,
            PolicyValid: m.Validation.PolicyValid,
            Allowed:     m.Validation.Allowed,
            TimeValid:   m.Validation.TimeValid,
            NotExpired:  m.Validation.NotExpired,
            Errors:      make([]string, len(m.Validation.Errors)),
        }
        copy(clone.Validation.Errors, m.Validation.Errors)
    }
    
    // Copy signature
    if len(m.Signature) > 0 {
        clone.Signature = make([]byte, len(m.Signature))
        copy(clone.Signature, m.Signature)
    }
    
    return clone
}

// String returns a string representation of the message
func (m *NetworkMessage) String() string {
    return fmt.Sprintf("Message[ID:%s, Type:%s, Src:%s, Dst:%s, Size:%d, Hops:%d]", 
        utils.BytesToHex(m.Header.MessageID[:8]),
        m.Header.MessageType,
        utils.BytesToHex(m.Header.SourceNode[:8]),
        utils.BytesToHex(m.Header.TargetNode[:8]),
        len(m.Payload),
        m.RoutingInfo.CurrentHop)
}

// GetSize returns the total size of the message in bytes
func (m *NetworkMessage) GetSize() int {
    size := 256 // Header size
    
    size += len(m.Payload)
    
    if m.RoutingInfo != nil {
        size += len(m.RoutingInfo.Hops) * 64
        size += len(m.RoutingInfo.ExcludeNodes) * 32
        size += len(m.RoutingInfo.IncludeNodes) * 32
    }
    
    if len(m.Signature) > 0 {
        size += len(m.Signature)
    }
    
    return size
}