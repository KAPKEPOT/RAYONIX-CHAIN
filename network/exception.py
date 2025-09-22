class NetworkError(Exception):
    """Base network error"""
    pass

class ConnectionError(NetworkError):
    """Connection-related errors"""
    pass

class HandshakeError(NetworkError):
    """Handshake-related errors"""
    pass

class MessageError(NetworkError):
    """Message-related errors"""
    pass

class PeerBannedError(NetworkError):
    """Peer is banned"""
    pass

class RateLimitError(NetworkError):
    """Rate limit exceeded"""
    pass

class ProtocolError(NetworkError):
    """Protocol-specific errors"""
    pass

class SecurityError(NetworkError):
    """Security-related errors"""
    pass

class SerializationError(NetworkError):
    """Message serialization/deserialization errors"""
    pass

class DHTError(NetworkError):
    """DHT-related errors"""
    pass

class NATError(NetworkError):
    """NAT traversal errors"""
    pass

class ConfigurationError(NetworkError):
    """Configuration errors"""
    pass

class TimeoutError(NetworkError):
    """Operation timeout errors"""
    pass

class ResourceExhaustedError(NetworkError):
    """Resource exhaustion errors"""
    pass