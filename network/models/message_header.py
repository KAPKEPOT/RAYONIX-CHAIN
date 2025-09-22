from dataclasses import dataclass

@dataclass
class MessageHeader:
    """Network message header structure"""
    magic: bytes = b'RAYX'  # Network magic number
    command: bytes = b'\x00' * 12  # 12-byte command name
    length: int = 0  # Payload length
    checksum: bytes = b'\x00' * 4  # First 4 bytes of sha256(sha256(payload))