import zlib
import lz4.frame  
from typing import Optional

class CompressionManager:
    """Enhanced compression for blockchain data"""
    
    def __init__(self, algorithm: str = "zlib"):
        self.algorithm = algorithm
    
    def compress_data(self, data: bytes) -> bytes:
        """Enhanced compression with algorithm selection"""
        try:
            if self.algorithm == "lz4":
                return lz4.frame.compress(data)
            else:  # zlib default
                return zlib.compress(data, level=6)
        except Exception as e:
            raise Exception(f"Compression error: {e}")
    
    def decompress_data(self, data: bytes) -> bytes:
        """Enhanced decompression with algorithm detection"""
        try:
            # Try LZ4 first (has magic number)
            if len(data) >= 4 and data[:4] == b'\x04\x22\x4D\x18':
                return lz4.frame.decompress(data)
            else:
                return zlib.decompress(data)
        except Exception as e:
            raise Exception(f"Decompression error: {e}")

# Backward compatibility
def compress_data(data: bytes) -> bytes:
    return CompressionManager().compress_data(data)

def decompress_data(data: bytes) -> bytes:
    return CompressionManager().decompress_data(data)