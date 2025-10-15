package models

import (
    "encoding/json"
    "fmt"
    "net"
    "sync/atomic"
    "time"

    "github.com/rayxnetwork/p2p/config"
    "github.com/rayxnetwork/p2p/utils"
)

// PeerInfo represents comprehensive information about a network peer with physics-inspired properties
type PeerInfo struct {
    // Core identification and network properties
    NodeID           string                `json:"node_id" msgpack:"node_id"`
    PublicKey        []byte                `json:"public_key" msgpack:"public_key"`
    Address          string                `json:"address" msgpack:"address"`
    Port             int                   `json:"port" msgpack:"port"`
    Protocol         config.ProtocolType   `json:"protocol" msgpack:"protocol"`
    Transport        config.TransportType  `json:"transport" msgpack:"transport"`
    
    // Network characteristics and capabilities
    Version          string                `json:"version" msgpack:"version"`
    Capabilities     []string              `json:"capabilities" msgpack:"capabilities"`
    UserAgent        string                `json:"user_agent" msgpack:"user_agent"`
    NetworkID        uint32                `json:"network_id" msgpack:"network_id"`
    
    // State management with atomic operations
    State            config.ConnectionState `json:"state" msgpack:"state"`
    LastSeen         time.Time             `json:"last_seen" msgpack:"last_seen"`
    FirstSeen        time.Time             `json:"first_seen" msgpack:"first_seen"`
    ConnectedAt      *time.Time            `json:"connected_at" msgpack:"connected_at"`
    DisconnectedAt   *time.Time            `json:"disconnected_at" msgpack:"disconnected_at"`
    
    // Performance metrics (atomic for concurrent access)
    Reputation       int32                 `json:"reputation" msgpack:"reputation"`
    Latency          time.Duration         `json:"latency" msgpack:"latency"`
    ResponseTime     time.Duration         `json:"response_time" msgpack:"response_time"`
    Uptime           time.Duration         `json:"uptime" msgpack:"uptime"`
    
    // Connection statistics
    ConnectionCount  uint32                `json:"connection_count" msgpack:"connection_count"`
    FailedAttempts   uint32                `json:"failed_attempts" msgpack:"failed_attempts"`
    SuccessfulPings  uint32                `json:"successful_pings" msgpack:"successful_pings"`
    FailedPings      uint32                `json:"failed_pings" msgpack:"failed_pings"`
    
    // Bandwidth and message statistics
    BytesSent        uint64                `json:"bytes_sent" msgpack:"bytes_sent"`
    BytesReceived    uint64                `json:"bytes_received" msgpack:"bytes_received"`
    MessagesSent     uint64                `json:"messages_sent" msgpack:"messages_sent"`
    MessagesReceived uint64                `json:"messages_received" msgpack:"messages_received"`
    
    // Physics-inspired properties
    PotentialEnergy  float64               `json:"potential_energy" msgpack:"potential_energy"`
    KineticEnergy    float64               `json:"kinetic_energy" msgpack:"kinetic_energy"`
    EntropyContribution float64           `json:"entropy_contribution" msgpack:"entropy_contribution"`
    ForceVector      *ForceVector          `json:"force_vector" msgpack:"force_vector"`
    
    // Consensus integration
    ValidatorScore   float64               `json:"validator_score" msgpack:"validator_score"`
    StakeAmount      float64               `json:"stake_amount" msgpack:"stake_amount"`
    VotingPower      float64               `json:"voting_power" msgpack:"voting_power"`
    
    // Security and validation
    IsBootstrap      bool                  `json:"is_bootstrap" msgpack:"is_bootstrap"`
    IsValidator      bool                  `json:"is_validator" msgpack:"is_validator"`
    IsBanned         bool                  `json:"is_banned" msgpack:"is_banned"`
    BanReason        string                `json:"ban_reason" msgpack:"ban_reason"`
    BanExpiry        *time.Time            `json:"ban_expiry" msgpack:"ban_expiry"`
    
    // Geographic and network location
    CountryCode      string                `json:"country_code" msgpack:"country_code"`
    Region           string                `json:"region" msgpack:"region"`
    ASN              uint32                `json:"asn" msgpack:"asn"`
    Coordinates      *GeoCoordinates       `json:"coordinates" msgpack:"coordinates"`
    
    // Resource constraints
    MaxConnections   uint32                `json:"max_connections" msgpack:"max_connections"`
    RateLimit        uint64                `json:"rate_limit" msgpack:"rate_limit"`
    MemoryUsage      uint64                `json:"memory_usage" msgpack:"memory_usage"`
    
    // Synchronization primitives
    mu               utils.SpinLock        `json:"-" msgpack:"-"`
    lastUpdated      time.Time             `json:"last_updated" msgpack:"last_updated"`
    version          uint64                `json:"version" msgpack:"version"`
}

// ForceVector represents physics-inspired force acting on peer connection
type ForceVector struct {
    Magnitude    float64   `json:"magnitude" msgpack:"magnitude"`
    DirectionX   float64   `json:"direction_x" msgpack:"direction_x"`
    DirectionY   float64   `json:"direction_y" msgpack:"direction_y"`
    DirectionZ   float64   `json:"direction_z" msgpack:"direction_z"`
    Attraction   float64   `json:"attraction" msgpack:"attraction"`
    Repulsion    float64   `json:"repulsion" msgpack:"repulsion"`
    LastUpdated  time.Time `json:"last_updated" msgpack:"last_updated"`
}

// GeoCoordinates represents geographic location
type GeoCoordinates struct {
    Latitude  float64 `json:"latitude" msgpack:"latitude"`
    Longitude float64 `json:"longitude" msgpack:"longitude"`
    Accuracy  float64 `json:"accuracy" msgpack:"accuracy"` // Accuracy in kilometers
}

// NewPeerInfo creates a new PeerInfo instance with proper initialization
func NewPeerInfo(nodeID, address string, port int, protocol config.ProtocolType) *PeerInfo {
    now := time.Now()
    return &PeerInfo{
        NodeID:          nodeID,
        Address:         address,
        Port:            port,
        Protocol:        protocol,
        Version:         "1.0.0",
        Capabilities:    []string{"tcp", "udp", "gossip", "syncing"},
        UserAgent:       "RayX-P2P/1.0.0",
        NetworkID:       1,
        State:           config.Disconnected,
        FirstSeen:       now,
        LastSeen:        now,
        Reputation:      50, // Neutral starting reputation
        PotentialEnergy: 1.0,
        KineticEnergy:   0.0,
        EntropyContribution: 0.5,
        ForceVector: &ForceVector{
            Magnitude:   0.0,
            DirectionX:  0.0,
            DirectionY:  0.0,
            DirectionZ:  0.0,
            Attraction:  0.5,
            Repulsion:   0.5,
            LastUpdated: now,
        },
        ValidatorScore: 0.0,
        StakeAmount:    0.0,
        VotingPower:    0.0,
        IsBootstrap:    false,
        IsValidator:    false,
        IsBanned:       false,
        MaxConnections: 50,
        RateLimit:      1000,
        lastUpdated:    now,
        version:        1,
    }
}

// Validate performs comprehensive validation of peer information
func (p *PeerInfo) Validate() error {
    p.mu.Lock()
    defer p.mu.Unlock()

    if p.NodeID == "" {
        return fmt.Errorf("node ID cannot be empty")
    }
    
    if len(p.NodeID) > 256 {
        return fmt.Errorf("node ID too long: %d characters", len(p.NodeID))
    }
    
    if p.Address == "" {
        return fmt.Errorf("address cannot be empty")
    }
    
    // Validate IP address format
    if ip := net.ParseIP(p.Address); ip == nil {
        // Check if it's a hostname
        if _, err := net.ResolveIPAddr("ip", p.Address); err != nil {
            return fmt.Errorf("invalid address format: %s", p.Address)
        }
    }
    
    if p.Port <= 0 || p.Port > 65535 {
        return fmt.Errorf("invalid port number: %d", p.Port)
    }
    
    if p.Version == "" {
        return fmt.Errorf("version cannot be empty")
    }
    
    if p.Reputation < -100 || p.Reputation > 100 {
        return fmt.Errorf("reputation out of range: %d", p.Reputation)
    }
    
    if p.Latency < 0 {
        return fmt.Errorf("latency cannot be negative: %v", p.Latency)
    }
    
    if p.PotentialEnergy < 0 {
        return fmt.Errorf("potential energy cannot be negative: %f", p.PotentialEnergy)
    }
    
    return nil
}

// UpdateReputation atomically updates peer reputation with bounds checking
func (p *PeerInfo) UpdateReputation(delta int32) int32 {
    newRep := atomic.AddInt32(&p.Reputation, delta)
    
    // Clamp reputation to valid range [-100, 100]
    if newRep < -100 {
        atomic.StoreInt32(&p.Reputation, -100)
        return -100
    } else if newRep > 100 {
        atomic.StoreInt32(&p.Reputation, 100)
        return 100
    }
    
    return newRep
}

// RecordSuccess updates metrics for successful interaction
func (p *PeerInfo) RecordSuccess(latency time.Duration, bytesSent, bytesReceived uint64) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    p.LastSeen = time.Now()
    p.Latency = latency
    atomic.AddUint32(&p.SuccessfulPings, 1)
    atomic.AddUint64(&p.BytesSent, bytesSent)
    atomic.AddUint64(&p.BytesReceived, bytesReceived)
    
    // Positive reputation adjustment for success
    p.UpdateReputation(1)
    
    p.updateVersion()
}

// RecordFailure updates metrics for failed interaction
func (p *PeerInfo) RecordFailure(reason string) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    atomic.AddUint32(&p.FailedAttempts, 1)
    atomic.AddUint32(&p.FailedPings, 1)
    
    // Negative reputation adjustment for failure
    p.UpdateReputation(-5)
    
    p.updateVersion()
}

// UpdatePhysicsState updates physics-inspired properties
func (p *PeerInfo) UpdatePhysicsState(potential, kinetic, entropy float64, force *ForceVector) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    p.PotentialEnergy = potential
    p.KineticEnergy = kinetic
    p.EntropyContribution = entropy
    
    if force != nil {
        p.ForceVector = force
        p.ForceVector.LastUpdated = time.Now()
    }
    
    p.updateVersion()
}

// UpdateConsensusData updates validator-specific information
func (p *PeerInfo) UpdateConsensusData(score, stake, power float64, isValidator bool) {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    p.ValidatorScore = score
    p.StakeAmount = stake
    p.VotingPower = power
    p.IsValidator = isValidator
    
    p.updateVersion()
}

// CalculateQualityScore computes comprehensive peer quality score
func (p *PeerInfo) CalculateQualityScore() float64 {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    // Base quality factors
    reputationFactor := float64(p.Reputation+100) / 200.0 // Normalize to [0,1]
    
    latencyFactor := 1.0
    if p.Latency > 0 {
        latencyFactor = 1.0 / (1.0 + p.Latency.Seconds())
    }
    
    successRate := float64(p.SuccessfulPings+1) / float64(p.SuccessfulPings+p.FailedPings+1)
    
    // Physics factors
    physicsFactor := (p.PotentialEnergy + (1.0 - p.EntropyContribution)) / 2.0
    
    // Consensus factors
    consensusFactor := (p.ValidatorScore + p.StakeAmount) / 2.0
    
    // Weighted combination
    quality := (reputationFactor * 0.25) +
              (latencyFactor * 0.20) +
              (successRate * 0.20) +
              (physicsFactor * 0.15) +
              (consensusFactor * 0.20)
    
    return utils.Clamp(quality, 0.0, 1.0)
}

// ShouldEvict determines if peer should be evicted based on multiple factors
func (p *PeerInfo) ShouldEvict() bool {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    // Check reputation threshold
    if p.Reputation <= -80 {
        return true
    }
    
    // Check failure rate
    totalPings := p.SuccessfulPings + p.FailedPings
    if totalPings > 10 && float64(p.FailedPings)/float64(totalPings) > 0.7 {
        return true
    }
    
    // Check if banned
    if p.IsBanned {
        if p.BanExpiry == nil || p.BanExpiry.After(time.Now()) {
            return true
        }
    }
    
    // Check physics stability
    if p.EntropyContribution > 0.9 && p.PotentialEnergy < 0.1 {
        return true
    }
    
    return false
}

// GetConnectionWeight calculates physics-inspired connection weight
func (p *PeerInfo) GetConnectionWeight() float64 {
    quality := p.CalculateQualityScore()
    
    // Enhanced with physics model
    physicsWeight := p.ForceVector.Attraction - p.ForceVector.Repulsion
    normalizedPhysics := (physicsWeight + 1.0) / 2.0 // Normalize to [0,1]
    
    // Combine quality and physics
    combined := (quality * 0.7) + (normalizedPhysics * 0.3)
    
    return utils.Clamp(combined, 0.0, 1.0)
}

// Clone creates a deep copy of PeerInfo
func (p *PeerInfo) Clone() *PeerInfo {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    clone := &PeerInfo{
        NodeID:             p.NodeID,
        Address:            p.Address,
        Port:               p.Port,
        Protocol:           p.Protocol,
        Transport:          p.Transport,
        Version:            p.Version,
        UserAgent:          p.UserAgent,
        NetworkID:          p.NetworkID,
        State:              p.State,
        LastSeen:           p.LastSeen,
        FirstSeen:          p.FirstSeen,
        Reputation:         p.Reputation,
        Latency:            p.Latency,
        ResponseTime:       p.ResponseTime,
        Uptime:             p.Uptime,
        ConnectionCount:    p.ConnectionCount,
        FailedAttempts:     p.FailedAttempts,
        SuccessfulPings:    p.SuccessfulPings,
        FailedPings:        p.FailedPings,
        BytesSent:          p.BytesSent,
        BytesReceived:      p.BytesReceived,
        MessagesSent:       p.MessagesSent,
        MessagesReceived:   p.MessagesReceived,
        PotentialEnergy:    p.PotentialEnergy,
        KineticEnergy:      p.KineticEnergy,
        EntropyContribution: p.EntropyContribution,
        ValidatorScore:     p.ValidatorScore,
        StakeAmount:        p.StakeAmount,
        VotingPower:        p.VotingPower,
        IsBootstrap:        p.IsBootstrap,
        IsValidator:        p.IsValidator,
        IsBanned:           p.IsBanned,
        BanReason:          p.BanReason,
        CountryCode:        p.CountryCode,
        Region:             p.Region,
        ASN:                p.ASN,
        MaxConnections:     p.MaxConnections,
        RateLimit:          p.RateLimit,
        MemoryUsage:        p.MemoryUsage,
        lastUpdated:        p.lastUpdated,
        version:            p.version,
    }
    
    // Deep copy slices
    clone.Capabilities = make([]string, len(p.Capabilities))
    copy(clone.Capabilities, p.Capabilities)
    
    clone.PublicKey = make([]byte, len(p.PublicKey))
    copy(clone.PublicKey, p.PublicKey)
    
    // Deep copy pointers
    if p.ConnectedAt != nil {
        connectedAt := *p.ConnectedAt
        clone.ConnectedAt = &connectedAt
    }
    
    if p.DisconnectedAt != nil {
        disconnectedAt := *p.DisconnectedAt
        clone.DisconnectedAt = &disconnectedAt
    }
    
    if p.BanExpiry != nil {
        banExpiry := *p.BanExpiry
        clone.BanExpiry = &banExpiry
    }
    
    if p.Coordinates != nil {
        clone.Coordinates = &GeoCoordinates{
            Latitude:  p.Coordinates.Latitude,
            Longitude: p.Coordinates.Longitude,
            Accuracy:  p.Coordinates.Accuracy,
        }
    }
    
    if p.ForceVector != nil {
        clone.ForceVector = &ForceVector{
            Magnitude:   p.ForceVector.Magnitude,
            DirectionX:  p.ForceVector.DirectionX,
            DirectionY:  p.ForceVector.DirectionY,
            DirectionZ:  p.ForceVector.DirectionZ,
            Attraction:  p.ForceVector.Attraction,
            Repulsion:   p.ForceVector.Repulsion,
            LastUpdated: p.ForceVector.LastUpdated,
        }
    }
    
    return clone
}

// MarshalJSON implements custom JSON marshaling
func (p *PeerInfo) MarshalJSON() ([]byte, error) {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    type Alias PeerInfo
    return json.Marshal(&struct {
        *Alias
        LastSeen    int64 `json:"last_seen"`
        FirstSeen   int64 `json:"first_seen"`
        Latency     int64 `json:"latency"`
        ResponseTime int64 `json:"response_time"`
        Uptime      int64 `json:"uptime"`
    }{
        Alias:       (*Alias)(p),
        LastSeen:    p.LastSeen.UnixNano(),
        FirstSeen:   p.FirstSeen.UnixNano(),
        Latency:     p.Latency.Nanoseconds(),
        ResponseTime: p.ResponseTime.Nanoseconds(),
        Uptime:      p.Uptime.Nanoseconds(),
    })
}

// UnmarshalJSON implements custom JSON unmarshaling
func (p *PeerInfo) UnmarshalJSON(data []byte) error {
    p.mu.Lock()
    defer p.mu.Unlock()
    
    type Alias PeerInfo
    aux := &struct {
        *Alias
        LastSeen    int64 `json:"last_seen"`
        FirstSeen   int64 `json:"first_seen"`
        Latency     int64 `json:"latency"`
        ResponseTime int64 `json:"response_time"`
        Uptime      int64 `json:"uptime"`
    }{
        Alias: (*Alias)(p),
    }
    
    if err := json.Unmarshal(data, &aux); err != nil {
        return err
    }
    
    p.LastSeen = time.Unix(0, aux.LastSeen)
    p.FirstSeen = time.Unix(0, aux.FirstSeen)
    p.Latency = time.Duration(aux.Latency)
    p.ResponseTime = time.Duration(aux.ResponseTime)
    p.Uptime = time.Duration(aux.Uptime)
    
    p.lastUpdated = time.Now()
    p.version++
    
    return nil
}

// GetAge returns the age of peer information
func (p *PeerInfo) GetAge() time.Duration {
    p.mu.RLock()
    defer p.mu.RUnlock()
    return time.Since(p.lastUpdated)
}

// IsStale checks if peer information is stale
func (p *PeerInfo) IsStale(maxAge time.Duration) bool {
    return p.GetAge() > maxAge
}

// updateVersion increments the version and updates timestamp
func (p *PeerInfo) updateVersion() {
    p.version++
    p.lastUpdated = time.Now()
}

// GetVersion returns the current version of peer information
func (p *PeerInfo) GetVersion() uint64 {
    p.mu.RLock()
    defer p.mu.RUnlock()
    return p.version
}

// String returns a string representation of the peer
func (p *PeerInfo) String() string {
    p.mu.RLock()
    defer p.mu.RUnlock()
    
    return fmt.Sprintf("Peer[%s@%s:%d, Rep:%d, State:%s, Quality:%.3f]", 
        p.NodeID[:8], p.Address, p.Port, p.Reputation, p.State, p.CalculateQualityScore())
}