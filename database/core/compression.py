import zlib
import lz4.frame
import snappy
import zstandard as zstd
from typing import Optional
import logging

from ..utils.exceptions import CompressionError

logger = logging.getLogger(__name__)

class ZlibCompression:
    """Zlib compression implementation"""
    
    def __init__(self, level: Optional[int] = 6):
        self.level = level
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using zlib"""
        try:
            return zlib.compress(data, self.level)
        except Exception as e:
            raise CompressionError(f"Zlib compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress zlib data"""
        try:
            return zlib.decompress(data)
        except Exception as e:
            raise CompressionError(f"Zlib decompression failed: {e}")

class LZ4Compression:
    """LZ4 compression implementation"""
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using LZ4"""
        try:
            return lz4.frame.compress(data)
        except Exception as e:
            raise CompressionError(f"LZ4 compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress LZ4 data"""
        try:
            return lz4.frame.decompress(data)
        except Exception as e:
            raise CompressionError(f"LZ4 decompression failed: {e}")

class SnappyCompression:
    """Snappy compression implementation"""
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using Snappy"""
        try:
            return snappy.compress(data)
        except Exception as e:
            raise CompressionError(f"Snappy compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress Snappy data"""
        try:
            return snappy.decompress(data)
        except Exception as e:
            raise CompressionError(f"Snappy decompression failed: {e}")

class ZstdCompression:
    """Zstandard compression implementation"""
    
    def __init__(self, level: Optional[int] = 3):
        self.level = level
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using Zstandard"""
        try:
            return self.compressor.compress(data)
        except Exception as e:
            raise CompressionError(f"Zstd compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress Zstandard data"""
        try:
            return self.decompressor.decompress(data)
        except Exception as e:
            raise CompressionError(f"Zstd decompression failed: {e}")