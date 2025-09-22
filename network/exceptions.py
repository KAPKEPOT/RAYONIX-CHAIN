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