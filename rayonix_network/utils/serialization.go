package utils

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"reflect"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/golang/snappy"
	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
	"github.com/rayxnetwork/p2p/config"
	"github.com/rayxnetwork/p2p/models"
	"golang.org/x/crypto/sha3"
)

// ZeroCopySerializer implements high-performance, zero-copy binary serialization for physics-inspired network
type ZeroCopySerializer struct {
	headerPool     *sync.Pool
	bufferPool     *BufferPool
	compressor     *MultiCompressor
	checksumTable  *crc32.Table
	stats          *SerializationStats
	config         *SerializationConfig
	mu             sync.RWMutex
	messageIDSeq   atomic.Uint64
}

// SerializationConfig contains comprehensive serialization configuration
type SerializationConfig struct {
	UseZeroCopy          bool
	EnableCompression    bool
	CompressionAlgorithm config.CompressionType
	CompressionLevel     int
	CompressionThreshold int
	ChecksumEnabled      bool
	MaxMessageSize       int
	BufferPoolSize       int
	PreAllocateBuffers   bool
	ValidationStrict     bool
	EnableEncryption     bool
	EncryptionKey        []byte
}

// MessageHeader represents the optimized binary header structure
type MessageHeader struct {
	Magic           [4]byte
	Version         uint16
	MessageType     uint8
	Flags           uint8
	PayloadSize     uint32
	CompressedSize  uint32
	Checksum        uint32
	Timestamp       int64
	Sequence        uint64
	SourceNodeHash  [16]byte
	Reserved        [8]byte
}

// SerializationStats tracks serialization performance metrics
type SerializationStats struct {
	MessagesSerialized   atomic.Int64
	MessagesDeserialized atomic.Int64
	BytesSerialized      atomic.Int64
	BytesDeserialized    atomic.Int64
	CompressionRatio     atomic.Value
	AverageLatency       atomic.Value
	ErrorCount           atomic.Int64
	CacheHits            atomic.Int64
	CacheMisses          atomic.Int64
}

// MultiCompressor provides multiple compression algorithm support
type MultiCompressor struct {
	snappyEncoder *snappy.Encoder
	snappyDecoder *snappy.Decoder
	zstdEncoders  *sync.Pool
	zstdDecoders  *sync.Pool
	lz4Encoders   *sync.Pool
	lz4Decoders   *sync.Pool
	config        *CompressionConfig
}

// CompressionConfig contains compression algorithm configurations
type CompressionConfig struct {
	DefaultAlgorithm config.CompressionType
	SnappyLevel      int
	ZstdLevel        int
	LZ4Level         int
	Threshold        int
}

// NewZeroCopySerializer creates a new high-performance serializer
func NewZeroCopySerializer(cfg *SerializationConfig) *ZeroCopySerializer {
	if cfg == nil {
		cfg = &SerializationConfig{
			UseZeroCopy:          true,
			EnableCompression:    true,
			CompressionAlgorithm: config.CompressionZstd,
			CompressionLevel:     3,
			CompressionThreshold: 512,
			ChecksumEnabled:      true,
			MaxMessageSize:       16 * 1024 * 1024,
			BufferPoolSize:       2048,
			PreAllocateBuffers:   true,
			ValidationStrict:     true,
			EnableEncryption:     false,
		}
	}

	compressor := NewMultiCompressor(&CompressionConfig{
		DefaultAlgorithm: cfg.CompressionAlgorithm,
		SnappyLevel:      cfg.CompressionLevel,
		ZstdLevel:        cfg.CompressionLevel,
		LZ4Level:         cfg.CompressionLevel,
		Threshold:        cfg.CompressionThreshold,
	})

	serializer := &ZeroCopySerializer{
		headerPool: &sync.Pool{
			New: func() interface{} {
				return &MessageHeader{}
			},
		},
		bufferPool:    NewBufferPool(16384, cfg.BufferPoolSize),
		compressor:    compressor,
		checksumTable: crc32.MakeTable(crc32.Castagnoli),
		stats:         &SerializationStats{},
		config:        cfg,
	}

	serializer.stats.CompressionRatio.Store(float64(1.0))
	serializer.stats.AverageLatency.Store(time.Duration(0))

	return serializer
}

// SerializeMessage performs zero-copy serialization of network messages
func (zs *ZeroCopySerializer) SerializeMessage(msg *models.NetworkMessage) ([]byte, error) {
	startTime := time.Now()

	if err := zs.validateMessage(msg); err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("message validation failed: %w", err)
	}

	header := zs.headerPool.Get().(*MessageHeader)
	defer zs.headerPool.Put(header)

	zs.initializeHeader(header, msg)

	payloadData, compressed, err := zs.processPayload(msg.Payload, header)
	if err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("payload processing failed: %w", err)
	}

	if compressed {
		header.Flags |= 0x02
		header.CompressedSize = uint32(len(payloadData))
	} else {
		header.CompressedSize = 0
	}

	buffer := zs.bufferPool.Get()
	defer zs.bufferPool.Put(buffer)

	if err := zs.serializeHeaderOptimized(buffer, header); err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("header serialization failed: %w", err)
	}

	if _, err := buffer.Write(payloadData); err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("payload write failed: %w", err)
	}

	if zs.config.ChecksumEnabled {
		checksum := zs.calculateChecksumOptimized(buffer.Bytes())
		header.Checksum = checksum
		zs.updateChecksumInPlace(buffer.Bytes(), checksum)
	}

	result := make([]byte, buffer.Len())
	copy(result, buffer.Bytes())

	zs.updateStats(len(result), time.Since(startTime), compressed, len(msg.Payload), len(payloadData))

	return result, nil
}

// DeserializeMessage performs zero-copy deserialization of network messages
func (zs *ZeroCopySerializer) DeserializeMessage(data []byte) (*models.NetworkMessage, error) {
	startTime := time.Now()

	if len(data) < 48 {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("message too short: %d bytes", len(data))
	}

	header, err := zs.deserializeHeaderSafe(data)
	if err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("header deserialization failed: %w", err)
	}

	if err := zs.validateHeader(header); err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("header validation failed: %w", err)
	}

	if zs.config.ChecksumEnabled {
		if err := zs.verifyChecksum(data, header.Checksum); err != nil {
			zs.stats.ErrorCount.Add(1)
			return nil, fmt.Errorf("checksum verification failed: %w", err)
		}
	}

	payloadData, err := zs.extractPayload(data, header)
	if err != nil {
		zs.stats.ErrorCount.Add(1)
		return nil, fmt.Errorf("payload extraction failed: %w", err)
	}

	if header.Flags&0x02 != 0 {
		algo := config.CompressionType((header.Flags >> 2) & 0x03)
		decompressed, err := zs.compressor.Decompress(payloadData, algo)
		if err != nil {
			zs.stats.ErrorCount.Add(1)
			return nil, fmt.Errorf("payload decompression failed: %w", err)
		}
		payloadData = decompressed
	}

	message := &models.NetworkMessage{
		MessageID:   zs.generateMessageID(header),
		MessageType: config.MessageType(header.MessageType),
		Payload:     payloadData,
		Timestamp:   time.Unix(0, header.Timestamp),
		Sequence:    header.Sequence,
		SourceNode:  zs.extractSourceNode(header),
		Priority:    zs.extractPriority(header),
	}

	zs.stats.MessagesDeserialized.Add(1)
	zs.stats.BytesDeserialized.Add(int64(len(data)))

	return message, nil
}

// SerializePeerInfo serializes peer information for efficient gossip propagation
func (zs *ZeroCopySerializer) SerializePeerInfo(peer *models.PeerInfo) ([]byte, error) {
	buffer := zs.bufferPool.Get()
	defer zs.bufferPool.Put(buffer)

	// Write node ID length and data
	nodeIDBytes := []byte(peer.NodeID)
	if err := binary.Write(buffer, binary.BigEndian, uint16(len(nodeIDBytes))); err != nil {
		return nil, err
	}
	if _, err := buffer.Write(nodeIDBytes); err != nil {
		return nil, err
	}

	// Write address
	addressBytes := []byte(peer.Address)
	if err := binary.Write(buffer, binary.BigEndian, uint16(len(addressBytes))); err != nil {
		return nil, err
	}
	if _, err := buffer.Write(addressBytes); err != nil {
		return nil, err
	}

	// Write port
	if err := binary.Write(buffer, binary.BigEndian, uint16(peer.Port)); err != nil {
		return nil, err
	}

	// Write protocol
	if err := binary.Write(buffer, binary.BigEndian, uint8(peer.Protocol)); err != nil {
		return nil, err
	}

	// Write reputation
	if err := binary.Write(buffer, binary.BigEndian, uint8(peer.Reputation)); err != nil {
		return nil, err
	}

	// Write latency (nanoseconds)
	if err := binary.Write(buffer, binary.BigEndian, int64(peer.Latency)); err != nil {
		return nil, err
	}

	// Write connection count
	if err := binary.Write(buffer, binary.BigEndian, uint16(peer.ConnectionCount)); err != nil {
		return nil, err
	}

	// Write failed attempts
	if err := binary.Write(buffer, binary.BigEndian, uint16(peer.FailedAttempts)); err != nil {
		return nil, err
	}

	// Write last seen timestamp
	if err := binary.Write(buffer, binary.BigEndian, peer.LastSeen.UnixNano()); err != nil {
		return nil, err
	}

	// Write state
	if err := binary.Write(buffer, binary.BigEndian, uint8(peer.State)); err != nil {
		return nil, err
	}

	// Write capabilities count and each capability
	capabilities := peer.Capabilities
	if err := binary.Write(buffer, binary.BigEndian, uint16(len(capabilities))); err != nil {
		return nil, err
	}
	for _, cap := range capabilities {
		capBytes := []byte(cap)
		if err := binary.Write(buffer, binary.BigEndian, uint16(len(capBytes))); err != nil {
			return nil, err
		}
		if _, err := buffer.Write(capBytes); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

// DeserializePeerInfo deserializes peer information from binary data
func (zs *ZeroCopySerializer) DeserializePeerInfo(data []byte) (*models.PeerInfo, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("peer info data too short")
	}

	buffer := bytes.NewReader(data)
	peer := &models.PeerInfo{}

	// Read node ID
	var nodeIDLen uint16
	if err := binary.Read(buffer, binary.BigEndian, &nodeIDLen); err != nil {
		return nil, err
	}
	nodeIDBytes := make([]byte, nodeIDLen)
	if _, err := buffer.Read(nodeIDBytes); err != nil {
		return nil, err
	}
	peer.NodeID = string(nodeIDBytes)

	// Read address
	var addrLen uint16
	if err := binary.Read(buffer, binary.BigEndian, &addrLen); err != nil {
		return nil, err
	}
	addrBytes := make([]byte, addrLen)
	if _, err := buffer.Read(addrBytes); err != nil {
		return nil, err
	}
	peer.Address = string(addrBytes)

	// Read port
	var port uint16
	if err := binary.Read(buffer, binary.BigEndian, &port); err != nil {
		return nil, err
	}
	peer.Port = int(port)

	// Read protocol
	var protocol uint8
	if err := binary.Read(buffer, binary.BigEndian, &protocol); err != nil {
		return nil, err
	}
	peer.Protocol = config.ProtocolType(protocol)

	// Read reputation
	var reputation uint8
	if err := binary.Read(buffer, binary.BigEndian, &reputation); err != nil {
		return nil, err
	}
	peer.Reputation = int(reputation)

	// Read latency
	var latency int64
	if err := binary.Read(buffer, binary.BigEndian, &latency); err != nil {
		return nil, err
	}
	peer.Latency = time.Duration(latency)

	// Read connection count
	var connCount uint16
	if err := binary.Read(buffer, binary.BigEndian, &connCount); err != nil {
		return nil, err
	}
	peer.ConnectionCount = int(connCount)

	// Read failed attempts
	var failedAttempts uint16
	if err := binary.Read(buffer, binary.BigEndian, &failedAttempts); err != nil {
		return nil, err
	}
	peer.FailedAttempts = int(failedAttempts)

	// Read last seen
	var lastSeen int64
	if err := binary.Read(buffer, binary.BigEndian, &lastSeen); err != nil {
		return nil, err
	}
	peer.LastSeen = time.Unix(0, lastSeen)

	// Read state
	var state uint8
	if err := binary.Read(buffer, binary.BigEndian, &state); err != nil {
		return nil, err
	}
	peer.State = config.PeerState(state)

	// Read capabilities
	var capCount uint16
	if err := binary.Read(buffer, binary.BigEndian, &capCount); err != nil {
		return nil, err
	}
	peer.Capabilities = make([]string, capCount)
	for i := 0; i < int(capCount); i++ {
		var capLen uint16
		if err := binary.Read(buffer, binary.BigEndian, &capLen); err != nil {
			return nil, err
		}
		capBytes := make([]byte, capLen)
		if _, err := buffer.Read(capBytes); err != nil {
			return nil, err
		}
		peer.Capabilities[i] = string(capBytes)
	}

	return peer, nil
}

// SerializeValidatorScores serializes validator scores for consensus coupling
func (zs *ZeroCopySerializer) SerializeValidatorScores(scores map[string]float64) ([]byte, error) {
	buffer := zs.bufferPool.Get()
	defer zs.bufferPool.Put(buffer)

	// Write count of validators
	if err := binary.Write(buffer, binary.BigEndian, uint32(len(scores))); err != nil {
		return nil, err
	}

	// Write each validator score with delta encoding
	previousScore := float64(0.0)
	for nodeID, score := range scores {
		// Write node ID
		nodeIDBytes := []byte(nodeID)
		if err := binary.Write(buffer, binary.BigEndian, uint16(len(nodeIDBytes))); err != nil {
			return nil, err
		}
		if _, err := buffer.Write(nodeIDBytes); err != nil {
			return nil, err
		}

		// Write score delta for compression
		delta := score - previousScore
		if err := binary.Write(buffer, binary.BigEndian, delta); err != nil {
			return nil, err
		}
		previousScore = score
	}

	return buffer.Bytes(), nil
}

// DeserializeValidatorScores deserializes validator scores
func (zs *ZeroCopySerializer) DeserializeValidatorScores(data []byte) (map[string]float64, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("validator scores data too short")
	}

	buffer := bytes.NewReader(data)
	scores := make(map[string]float64)

	// Read count
	var count uint32
	if err := binary.Read(buffer, binary.BigEndian, &count); err != nil {
		return nil, err
	}

	// Read each validator score with delta decoding
	previousScore := float64(0.0)
	for i := 0; i < int(count); i++ {
		// Read node ID
		var nodeIDLen uint16
		if err := binary.Read(buffer, binary.BigEndian, &nodeIDLen); err != nil {
			return nil, err
		}
		nodeIDBytes := make([]byte, nodeIDLen)
		if _, err := buffer.Read(nodeIDBytes); err != nil {
			return nil, err
		}
		nodeID := string(nodeIDBytes)

		// Read score delta
		var delta float64
		if err := binary.Read(buffer, binary.BigEndian, &delta); err != nil {
			return nil, err
		}

		score := previousScore + delta
		scores[nodeID] = score
		previousScore = score
	}

	return scores, nil
}

// SerializeNetworkMetrics serializes network metrics for monitoring
func (zs *ZeroCopySerializer) SerializeNetworkMetrics(metrics *models.NetworkMetrics) ([]byte, error) {
	buffer := zs.bufferPool.Get()
	defer zs.bufferPool.Put(buffer)

	// Write node ID
	nodeIDBytes := []byte(metrics.NodeID)
	if err := binary.Write(buffer, binary.BigEndian, uint16(len(nodeIDBytes))); err != nil {
		return nil, err
	}
	if _, err := buffer.Write(nodeIDBytes); err != nil {
		return nil, err
	}

	// Write uptime
	if err := binary.Write(buffer, binary.BigEndian, int64(metrics.Uptime)); err != nil {
		return nil, err
	}

	// Write active connections
	if err := binary.Write(buffer, binary.BigEndian, uint32(metrics.ActiveConnections)); err != nil {
		return nil, err
	}

	// Write network entropy
	if err := binary.Write(buffer, binary.BigEndian, metrics.NetworkEntropy); err != nil {
		return nil, err
	}

	// Write average latency
	if err := binary.Write(buffer, binary.BigEndian, int64(metrics.AverageLatency)); err != nil {
		return nil, err
	}

	// Write message throughput
	if err := binary.Write(buffer, binary.BigEndian, metrics.MessageThroughput); err != nil {
		return nil, err
	}

	// Write timestamp
	if err := binary.Write(buffer, binary.BigEndian, metrics.Timestamp.UnixNano()); err != nil {
		return nil, err
	}

	// Write peer potentials count
	if err := binary.Write(buffer, binary.BigEndian, uint32(len(metrics.PeerPotentials))); err != nil {
		return nil, err
	}

	// Write each peer potential
	for peerID, potential := range metrics.PeerPotentials {
		peerIDBytes := []byte(peerID)
		if err := binary.Write(buffer, binary.BigEndian, uint16(len(peerIDBytes))); err != nil {
			return nil, err
		}
		if _, err := buffer.Write(peerIDBytes); err != nil {
			return nil, err
		}
		if err := binary.Write(buffer, binary.BigEndian, potential); err != nil {
			return nil, err
		}
	}

	// Write validator scores count
	if err := binary.Write(buffer, binary.BigEndian, uint32(len(metrics.ValidatorScores))); err != nil {
		return nil, err
	}

	// Write each validator score
	for validator, score := range metrics.ValidatorScores {
		validatorBytes := []byte(validator)
		if err := binary.Write(buffer, binary.BigEndian, uint16(len(validatorBytes))); err != nil {
			return nil, err
		}
		if _, err := buffer.Write(validatorBytes); err != nil {
			return nil, err
		}
		if err := binary.Write(buffer, binary.BigEndian, score); err != nil {
			return nil, err
		}
	}

	return buffer.Bytes(), nil
}

// DeserializeNetworkMetrics deserializes network metrics
func (zs *ZeroCopySerializer) DeserializeNetworkMetrics(data []byte) (*models.NetworkMetrics, error) {
	if len(data) < 2 {
		return nil, fmt.Errorf("network metrics data too short")
	}

	buffer := bytes.NewReader(data)
	metrics := &models.NetworkMetrics{
		PeerPotentials:  make(map[string]float64),
		ValidatorScores: make(map[string]float64),
	}

	// Read node ID
	var nodeIDLen uint16
	if err := binary.Read(buffer, binary.BigEndian, &nodeIDLen); err != nil {
		return nil, err
	}
	nodeIDBytes := make([]byte, nodeIDLen)
	if _, err := buffer.Read(nodeIDBytes); err != nil {
		return nil, err
	}
	metrics.NodeID = string(nodeIDBytes)

	// Read uptime
	var uptime int64
	if err := binary.Read(buffer, binary.BigEndian, &uptime); err != nil {
		return nil, err
	}
	metrics.Uptime = time.Duration(uptime)

	// Read active connections
	var activeConns uint32
	if err := binary.Read(buffer, binary.BigEndian, &activeConns); err != nil {
		return nil, err
	}
	metrics.ActiveConnections = int(activeConns)

	// Read network entropy
	if err := binary.Read(buffer, binary.BigEndian, &metrics.NetworkEntropy); err != nil {
		return nil, err
	}

	// Read average latency
	var avgLatency int64
	if err := binary.Read(buffer, binary.BigEndian, &avgLatency); err != nil {
		return nil, err
	}
	metrics.AverageLatency = time.Duration(avgLatency)

	// Read message throughput
	if err := binary.Read(buffer, binary.BigEndian, &metrics.MessageThroughput); err != nil {
		return nil, err
	}

	// Read timestamp
	var timestamp int64
	if err := binary.Read(buffer, binary.BigEndian, &timestamp); err != nil {
		return nil, err
	}
	metrics.Timestamp = time.Unix(0, timestamp)

	// Read peer potentials count
	var potentialsCount uint32
	if err := binary.Read(buffer, binary.BigEndian, &potentialsCount); err != nil {
		return nil, err
	}

	// Read each peer potential
	for i := 0; i < int(potentialsCount); i++ {
		var peerIDLen uint16
		if err := binary.Read(buffer, binary.BigEndian, &peerIDLen); err != nil {
			return nil, err
		}
		peerIDBytes := make([]byte, peerIDLen)
		if _, err := buffer.Read(peerIDBytes); err != nil {
			return nil, err
		}
		var potential float64
		if err := binary.Read(buffer, binary.BigEndian, &potential); err != nil {
			return nil, err
		}
		metrics.PeerPotentials[string(peerIDBytes)] = potential
	}

	// Read validator scores count
	var scoresCount uint32
	if err := binary.Read(buffer, binary.BigEndian, &scoresCount); err != nil {
		return nil, err
	}

	// Read each validator score
	for i := 0; i < int(scoresCount); i++ {
		var validatorLen uint16
		if err := binary.Read(buffer, binary.BigEndian, &validatorLen); err != nil {
			return nil, err
		}
		validatorBytes := make([]byte, validatorLen)
		if _, err := buffer.Read(validatorBytes); err != nil {
			return nil, err
		}
		var score float64
		if err := binary.Read(buffer, binary.BigEndian, &score); err != nil {
			return nil, err
		}
		metrics.ValidatorScores[string(validatorBytes)] = score
	}

	return metrics, nil
}

// initializeHeader initializes the message header with optimized values
func (zs *ZeroCopySerializer) initializeHeader(header *MessageHeader, msg *models.NetworkMessage) {
	header.Magic = [4]byte{'R', 'A', 'Y', 'X'}
	header.Version = 2
	header.MessageType = uint8(msg.MessageType)
	header.Flags = zs.calculateFlags(msg)
	header.PayloadSize = uint32(len(msg.Payload))
	header.Timestamp = msg.Timestamp.UnixNano()
	header.Sequence = msg.Sequence
	
	// Generate source node hash
	if msg.SourceNode != "" {
		hash := sha3.Sum128([]byte(msg.SourceNode))
		copy(header.SourceNodeHash[:], hash[:])
	}
	
	// Set compression algorithm in flags
	compAlgo := zs.config.CompressionAlgorithm
	header.Flags |= uint8(compAlgo) << 2
	
	// Set priority in reserved field
	binary.BigEndian.PutUint16(header.Reserved[0:2], uint16(msg.Priority))
}

// calculateFlags calculates optimal flags for the message
func (zs *ZeroCopySerializer) calculateFlags(msg *models.NetworkMessage) uint8 {
	var flags uint8
	
	if zs.config.ChecksumEnabled {
		flags |= 0x01
	}
	
	if zs.config.EnableCompression && len(msg.Payload) > zs.config.CompressionThreshold {
		flags |= 0x02
	}
	
	switch msg.MessageType {
	case config.Consensus, config.Block:
		flags |= 0x04
	case config.Transaction:
		flags |= 0x08
	}
	
	return flags
}

// processPayload handles payload compression and optimization
func (zs *ZeroCopySerializer) processPayload(payload []byte, header *MessageHeader) ([]byte, bool, error) {
	if len(payload) == 0 {
		return payload, false, nil
	}

	shouldCompress := zs.config.EnableCompression && 
		len(payload) > zs.config.CompressionThreshold &&
		(header.Flags&0x02 != 0)

	if !shouldCompress {
		return payload, false, nil
	}

	compressed, err := zs.compressor.Compress(payload, zs.config.CompressionAlgorithm)
	if err != nil {
		return nil, false, fmt.Errorf("compression failed: %w", err)
	}

	compressionRatio := float64(len(payload)) / float64(len(compressed))
	if compressionRatio > 1.1 {
		return compressed, true, nil
	}

	return payload, false, nil
}

// serializeHeaderOptimized performs optimized header serialization
func (zs *ZeroCopySerializer) serializeHeaderOptimized(w io.Writer, header *MessageHeader) error {
	bytes := make([]byte, 48)
	
	copy(bytes[0:4], header.Magic[:])
	binary.BigEndian.PutUint16(bytes[4:6], header.Version)
	bytes[6] = header.MessageType
	bytes[7] = header.Flags
	binary.BigEndian.PutUint32(bytes[8:12], header.PayloadSize)
	binary.BigEndian.PutUint32(bytes[12:16], header.CompressedSize)
	binary.BigEndian.PutUint32(bytes[16:20], header.Checksum)
	binary.BigEndian.PutUint64(bytes[20:28], uint64(header.Timestamp))
	binary.BigEndian.PutUint64(bytes[28:36], header.Sequence)
	copy(bytes[36:52], header.SourceNodeHash[:])
	copy(bytes[52:60], header.Reserved[:])
	
	_, err := w.Write(bytes)
	return err
}

// deserializeHeaderSafe performs safe header deserialization with bounds checking
func (zs *ZeroCopySerializer) deserializeHeaderSafe(data []byte) (*MessageHeader, error) {
	if len(data) < 48 {
		return nil, fmt.Errorf("insufficient data for header")
	}

	header := &MessageHeader{}
	
	copy(header.Magic[:], data[0:4])
	header.Version = binary.BigEndian.Uint16(data[4:6])
	header.MessageType = data[6]
	header.Flags = data[7]
	header.PayloadSize = binary.BigEndian.Uint32(data[8:12])
	header.CompressedSize = binary.BigEndian.Uint32(data[12:16])
	header.Checksum = binary.BigEndian.Uint32(data[16:20])
	header.Timestamp = int64(binary.BigEndian.Uint64(data[20:28]))
	header.Sequence = binary.BigEndian.Uint64(data[28:36])
	copy(header.SourceNodeHash[:], data[36:52])
	copy(header.Reserved[:], data[52:60])
	
	return header, nil
}

// extractPayload safely extracts payload with comprehensive bounds checking
func (zs *ZeroCopySerializer) extractPayload(data []byte, header *MessageHeader) ([]byte, error) {
	headerSize := 48
	payloadStart := headerSize
	
	payloadSize := header.PayloadSize
	if header.Flags&0x02 != 0 && header.CompressedSize > 0 {
		payloadSize = header.CompressedSize
	}
	
	payloadEnd := payloadStart + int(payloadSize)
	if payloadEnd > len(data) {
		return nil, fmt.Errorf("payload exceeds message bounds: %d > %d", payloadEnd, len(data))
	}
	
	if payloadSize > uint32(zs.config.MaxMessageSize) {
		return nil, fmt.Errorf("payload size %d exceeds maximum %d", payloadSize, zs.config.MaxMessageSize)
	}
	
	return data[payloadStart:payloadEnd], nil
}

// calculateChecksumOptimized performs optimized checksum calculation
func (zs *ZeroCopySerializer) calculateChecksumOptimized(data []byte) uint32 {
	return crc32.Checksum(data, zs.checksumTable)
}

// updateChecksumInPlace updates checksum in serialized data without re-serialization
func (zs *ZeroCopySerializer) updateChecksumInPlace(data []byte, checksum uint32) {
	if len(data) >= 20 {
		binary.BigEndian.PutUint32(data[16:20], checksum)
	}
}

// verifyChecksum verifies message checksum
func (zs *ZeroCopySerializer) verifyChecksum(data []byte, expectedChecksum uint32) error {
	verifyData := make([]byte, len(data))
	copy(verifyData, data)
	
	if len(verifyData) >= 20 {
		binary.BigEndian.PutUint32(verifyData[16:20], 0)
	}
	
	actualChecksum := zs.calculateChecksumOptimized(verifyData)
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch: expected %08x, got %08x", expectedChecksum, actualChecksum)
	}
	
	return nil
}

// validateMessage performs comprehensive message validation
func (zs *ZeroCopySerializer) validateMessage(msg *models.NetworkMessage) error {
	if msg == nil {
		return fmt.Errorf("message is nil")
	}
	
	if len(msg.Payload) > zs.config.MaxMessageSize {
		return fmt.Errorf("payload size %d exceeds maximum %d", len(msg.Payload), zs.config.MaxMessageSize)
	}
	
	if msg.Timestamp.IsZero() {
		return fmt.Errorf("invalid timestamp")
	}
	
	return nil
}

// validateHeader performs comprehensive header validation
func (zs *ZeroCopySerializer) validateHeader(header *MessageHeader) error {
	expectedMagic := [4]byte{'R', 'A', 'Y', 'X'}
	if header.Magic != expectedMagic {
		return fmt.Errorf("invalid magic number")
	}
	
	if header.Version != 1 && header.Version != 2 {
		return fmt.Errorf("unsupported version: %d", header.Version)
	}
	
	if header.MessageType > uint8(config.SyncResponse) {
		return fmt.Errorf("invalid message type: %d", header.MessageType)
	}
	
	if header.PayloadSize > uint32(zs.config.MaxMessageSize) {
		return fmt.Errorf("payload size %d exceeds maximum", header.PayloadSize)
	}
	
	if header.CompressedSize > uint32(zs.config.MaxMessageSize) {
		return fmt.Errorf("compressed size %d exceeds maximum", header.CompressedSize)
	}
	
	maxFuture := time.Now().Add(24 * time.Hour)
	if time.Unix(0, header.Timestamp).After(maxFuture) {
		return fmt.Errorf("timestamp too far in future")
	}
	
	return nil
}

// generateMessageID creates a unique message identifier
func (zs *ZeroCopySerializer) generateMessageID(header *MessageHeader) string {
	seq := zs.messageIDSeq.Add(1)
	return fmt.Sprintf("msg_%x_%d_%d_%d", header.Checksum, header.Sequence, header.Timestamp, seq)
}

// extractSourceNode extracts source node information from header
func (zs *ZeroCopySerializer) extractSourceNode(header *MessageHeader) string {
	// In a real implementation, you would maintain a mapping of hashes to node IDs
	// For now, return empty - the actual node ID would be resolved elsewhere
	return ""
}

// extractPriority extracts message priority from header
func (zs *ZeroCopySerializer) extractPriority(header *MessageHeader) int {
	if len(header.Reserved) >= 2 {
		return int(binary.BigEndian.Uint16(header.Reserved[0:2]))
	}
	return 1
}

// updateStats updates serialization statistics
func (zs *ZeroCopySerializer) updateStats(messageSize int, latency time.Duration, compressed bool, originalSize, compressedSize int) {
	zs.stats.MessagesSerialized.Add(1)
	zs.stats.BytesSerialized.Add(int64(messageSize))
	
	if compressed && originalSize > 0 {
		ratio := float64(originalSize) / float64(compressedSize)
		zs.stats.CompressionRatio.Store(ratio)
	}
	
	currentAvg := zs.stats.AverageLatency.Load().(time.Duration)
	if currentAvg == 0 {
		zs.stats.AverageLatency.Store(latency)
	} else {
		newAvg := time.Duration(0.9*float64(currentAvg) + 0.1*float64(latency))
		zs.stats.AverageLatency.Store(newAvg)
	}
}

// GetStats returns current serialization statistics
func (zs *ZeroCopySerializer) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"messages_serialized":   zs.stats.MessagesSerialized.Load(),
		"messages_deserialized": zs.stats.MessagesDeserialized.Load(),
		"bytes_serialized":      zs.stats.BytesSerialized.Load(),
		"bytes_deserialized":    zs.stats.BytesDeserialized.Load(),
		"compression_ratio":     zs.stats.CompressionRatio.Load(),
		"average_latency":       zs.stats.AverageLatency.Load(),
		"error_count":           zs.stats.ErrorCount.Load(),
		"cache_hits":            zs.stats.CacheHits.Load(),
		"cache_misses":          zs.stats.CacheMisses.Load(),
	}
}

// NewMultiCompressor creates a new multi-algorithm compressor
func NewMultiCompressor(cfg *CompressionConfig) *MultiCompressor {
	zstdLevel := zstd.SpeedDefault
	switch cfg.ZstdLevel {
	case 1:
		zstdLevel = zstd.SpeedFastest
	case 2:
		zstdLevel = zstd.SpeedDefault
	case 3:
		zstdLevel = zstd.SpeedBetterCompression
	case 4:
		zstdLevel = zstd.SpeedBestCompression
	}

	return &MultiCompressor{
		snappyEncoder: snappy.NewBufferedWriter(nil),
		snappyDecoder: snappy.NewReader(nil),
		zstdEncoders: &sync.Pool{
			New: func() interface{} {
				enc, _ := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstdLevel))
				return enc
			},
		},
		zstdDecoders: &sync.Pool{
			New: func() interface{} {
				dec, _ := zstd.NewReader(nil)
				return dec
			},
		},
		lz4Encoders: &sync.Pool{
			New: func() interface{} {
				return lz4.NewWriter(nil)
			},
		},
		lz4Decoders: &sync.Pool{
			New: func() interface{} {
				return lz4.NewReader(nil)
			},
		},
		config: cfg,
	}
}

// Compress compresses data using specified algorithm
func (mc *MultiCompressor) Compress(data []byte, algo config.CompressionType) ([]byte, error) {
	switch algo {
	case config.CompressionSnappy:
		var buf bytes.Buffer
		mc.snappyEncoder.Reset(&buf)
		if _, err := mc.snappyEncoder.Write(data); err != nil {
			return nil, err
		}
		if err := mc.snappyEncoder.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil

	case config.CompressionZstd:
		encoder := mc.zstdEncoders.Get().(*zstd.Encoder)
		defer mc.zstdEncoders.Put(encoder)
		return encoder.EncodeAll(data, make([]byte, 0, len(data))), nil

	case config.CompressionLZ4:
		var buf bytes.Buffer
		writer := mc.lz4Encoders.Get().(*lz4.Writer)
		defer mc.lz4Encoders.Put(writer)
		writer.Reset(&buf)
		if _, err := writer.Write(data); err != nil {
			return nil, err
		}
		if err := writer.Close(); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil

	default:
		return data, nil
	}
}

// Decompress decompresses data using algorithm detection
func (mc *MultiCompressor) Decompress(data []byte, algo config.CompressionType) ([]byte, error) {
	switch algo {
	case config.CompressionSnappy:
		return snappy.Decode(nil, data)

	case config.CompressionZstd:
		decoder := mc.zstdDecoders.Get().(*zstd.Decoder)
		defer mc.zstdDecoders.Put(decoder)
		return decoder.DecodeAll(data, nil)

	case config.CompressionLZ4:
		reader := mc.lz4Decoders.Get().(*lz4.Reader)
		defer mc.lz4Decoders.Put(reader)
		reader.Reset(bytes.NewReader(data))
		var buf bytes.Buffer
		if _, err := buf.ReadFrom(reader); err != nil {
			return nil, err
		}
		return buf.Bytes(), nil

	default:
		return data, nil
	}
}