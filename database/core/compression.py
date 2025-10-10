import zlib
import lz4.frame
import snappy
import zstandard as zstd
import bz2
import lzma
from typing import Optional, Dict, Any, Callable
from enum import Enum, auto
import logging
import threading
from dataclasses import dataclass
from contextlib import contextmanager
import time

from ..utils.exceptions import CompressionError

logger = logging.getLogger(__name__)

class CompressionAlgorithm(Enum):
    """Supported compression algorithms"""
    ZLIB = auto()
    LZ4 = auto()
    SNAPPY = auto()
    ZSTD = auto()
    BZ2 = auto()
    LZMA = auto()
    NONE = auto()

@dataclass
class CompressionStats:
    """Compression statistics"""
    algorithm: CompressionAlgorithm
    compressed_size: int = 0
    original_size: int = 0
    compression_ratio: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    total_compressions: int = 0
    total_decompressions: int = 0

class ZlibCompression:
    """Fixed Zlib compression implementation"""
    
    def __init__(self, level: Optional[int] = 6, strategy: int = zlib.Z_DEFAULT_STRATEGY,
                 mem_level: int = 8, wbits: int = 15):
        """
        Initialize Zlib compressor with proper parameter validation
        """
        if not -1 <= level <= 9:  # -1 is default, 0-9 are compression levels
            raise CompressionError(f"Invalid compression level: {level}. Must be between -1 and 9")
        if not 1 <= mem_level <= 9:
            raise CompressionError(f"Invalid memory level: {mem_level}. Must be between 1-9")
        
        self.level = level
        self.strategy = strategy
        self.mem_level = mem_level
        self.wbits = wbits
        self.stats = CompressionStats(CompressionAlgorithm.ZLIB)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using zlib with proper parameter handling"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                # FIXED: zlib.compress only takes data and level parameters
                # Create compressor object for advanced parameters
                compressor = zlib.compressobj(
                    level=self.level,
                    method=zlib.DEFLATED,
                    wbits=self.wbits,
                    memLevel=self.mem_level,
                    strategy=self.strategy
                )
                
                compressed = compressor.compress(data) + compressor.flush()
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"Zlib compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except zlib.error as e:
            raise CompressionError(f"Zlib compression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Zlib compression unexpected error: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress zlib data with proper error handling"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                # Create decompressor object for advanced parameters
                decompressor = zlib.decompressobj(self.wbits)
                decompressed = decompressor.decompress(data) + decompressor.flush()
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except zlib.error as e:
            raise CompressionError(f"Zlib decompression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Zlib decompression unexpected error: {e}")

class LZ4Compression:
    """LZ4 compression implementation with high-speed compression"""
    
    def __init__(self, compression_level: int = 1, block_size: int = lz4.frame.BLOCKSIZE_MAX4MB,
                 content_checksum: bool = True, block_checksum: bool = False,
                 auto_flush: bool = True):
        """
        Initialize LZ4 compressor
        
        Args:
            compression_level: Compression level (0-16, 0=fastest, 16=best)
            block_size: Block size for compression
            content_checksum: Whether to include content checksum
            block_checksum: Whether to include block checksum
            auto_flush: Whether to auto-flush buffers
        """
        self.compression_level = compression_level
        self.block_size = block_size
        self.content_checksum = content_checksum
        self.block_checksum = block_checksum
        self.auto_flush = auto_flush
        self.stats = CompressionStats(CompressionAlgorithm.LZ4)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using LZ4 with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                compressed = lz4.frame.compress(
                    data,
                    compression_level=self.compression_level,
                    block_size=self.block_size,
                    content_checksum=self.content_checksum,
                    block_checksum=self.block_checksum,
                    auto_flush=self.auto_flush
                )
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"LZ4 compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except lz4.frame.LZ4FrameError as e:
            raise CompressionError(f"LZ4 compression failed: {e}")
        except Exception as e:
            raise CompressionError(f"LZ4 compression unexpected error: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress LZ4 data with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                decompressed = lz4.frame.decompress(data)
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except lz4.frame.LZ4FrameError as e:
            raise CompressionError(f"LZ4 decompression failed: {e}")
        except Exception as e:
            raise CompressionError(f"LZ4 decompression unexpected error: {e}")
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.LZ4)

class SnappyCompression:
    """Snappy compression implementation with very fast compression/decompression"""
    
    def __init__(self, use_crc: bool = True):
        """
        Initialize Snappy compressor
        
        Args:
            use_crc: Whether to use CRC32 checksums
        """
        self.use_crc = use_crc
        self.stats = CompressionStats(CompressionAlgorithm.SNAPPY)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using Snappy with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                compressed = snappy.compress(data)
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"Snappy compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except snappy.SnappyError as e:
            raise CompressionError(f"Snappy compression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Snappy compression unexpected error: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress Snappy data with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                decompressed = snappy.decompress(data)
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except snappy.SnappyError as e:
            raise CompressionError(f"Snappy decompression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Snappy decompression unexpected error: {e}")
    
    def validate(self, data: bytes) -> bool:
        """Validate if data can be decompressed by Snappy"""
        try:
            snappy.decompress(data)
            return True
        except:
            return False
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.SNAPPY)

class ZstdCompression:
    """Zstandard compression implementation with high compression ratios"""
    
    def __init__(self, level: Optional[int] = 3, threads: int = 0,
                 checksum: bool = True, long_distance: bool = False):
        """
        Initialize Zstandard compressor
        
        Args:
            level: Compression level (1-22, 1=fastest, 22=best)
            threads: Number of threads to use (0=auto-detect)
            checksum: Whether to include checksums
            long_distance: Enable long distance matching for better compression
        """
        if not 1 <= level <= 22:
            raise CompressionError(f"Invalid compression level: {level}. Must be between 1-22")
        
        self.level = level
        self.threads = threads
        self.checksum = checksum
        self.long_distance = long_distance
        
        # Initialize compressor and decompressor
        self.compressor = zstd.ZstdCompressor(
            level=level,
            threads=threads,
            checksum=checksum,
            long_distance=long_distance
        )
        self.decompressor = zstd.ZstdDecompressor()
        
        self.stats = CompressionStats(CompressionAlgorithm.ZSTD)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using Zstandard with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                compressed = self.compressor.compress(data)
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"Zstd compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except zstd.ZstdError as e:
            raise CompressionError(f"Zstd compression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Zstd compression unexpected error: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress Zstandard data with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                decompressed = self.decompressor.decompress(data)
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except zstd.ZstdError as e:
            raise CompressionError(f"Zstd decompression failed: {e}")
        except Exception as e:
            raise CompressionError(f"Zstd decompression unexpected error: {e}")
    
    def compress_stream(self, data: bytes) -> bytes:
        """Compress using streaming API for large data"""
        return self.compressor.compress(data)
    
    def decompress_stream(self, data: bytes) -> bytes:
        """Decompress using streaming API for large data"""
        return self.decompressor.decompress(data)
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.ZSTD)

class BZ2Compression:
    """BZ2 compression implementation with good compression ratios"""
    
    def __init__(self, level: int = 9):
        """
        Initialize BZ2 compressor
        
        Args:
            level: Compression level (1-9, 1=fastest, 9=best)
        """
        if not 1 <= level <= 9:
            raise CompressionError(f"Invalid compression level: {level}. Must be between 1-9")
        
        self.level = level
        self.stats = CompressionStats(CompressionAlgorithm.BZ2)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using BZ2 with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                compressed = bz2.compress(data, self.level)
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"BZ2 compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except Exception as e:
            raise CompressionError(f"BZ2 compression failed: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress BZ2 data with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                decompressed = bz2.decompress(data)
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except Exception as e:
            raise CompressionError(f"BZ2 decompression failed: {e}")
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.BZ2)

class LZMACompression:
    """LZMA compression implementation with very high compression ratios"""
    
    def __init__(self, level: int = 6, format: str = 'xz', check: int = lzma.CHECK_CRC64,
                 preset: Optional[int] = None, filters: Optional[list] = None):
        """
        Initialize LZMA compressor
        
        Args:
            level: Compression level (0-9, 0=fastest, 9=best)
            format: Compression format ('xz' or 'lzma')
            check: Integrity check type
            preset: Compression preset
            filters: Custom filter chain
        """
        if not 0 <= level <= 9:
            raise CompressionError(f"Invalid compression level: {level}. Must be between 0-9")
        
        self.level = level
        self.format = format
        self.check = check
        self.preset = preset
        self.filters = filters
        self.stats = CompressionStats(CompressionAlgorithm.LZMA)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Compress data using LZMA with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                compressed = lzma.compress(
                    data,
                    format=self.format,
                    check=self.check,
                    preset=self.preset,
                    filters=self.filters
                )
                
                # Update statistics
                self.stats.compressed_size += len(compressed)
                self.stats.original_size += len(data)
                self.stats.compression_time += time.perf_counter() - start_time
                self.stats.total_compressions += 1
                self.stats.compression_ratio = (
                    self.stats.compressed_size / self.stats.original_size 
                    if self.stats.original_size > 0 else 0.0
                )
                
                logger.debug(f"LZMA compression: {len(data)} -> {len(compressed)} bytes "
                           f"(ratio: {len(compressed)/len(data):.2f})")
                
                return compressed
                
        except lzma.LZMAError as e:
            raise CompressionError(f"LZMA compression failed: {e}")
        except Exception as e:
            raise CompressionError(f"LZMA compression unexpected error: {e}")
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress LZMA data with statistics"""
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                decompressed = lzma.decompress(data)
                
                # Update statistics
                self.stats.decompression_time += time.perf_counter() - start_time
                self.stats.total_decompressions += 1
                
                return decompressed
                
        except lzma.LZMAError as e:
            raise CompressionError(f"LZMA decompression failed: {e}")
        except Exception as e:
            raise CompressionError(f"LZMA decompression unexpected error: {e}")
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.LZMA)

class NoCompression:
    """No compression (pass-through) implementation"""
    
    def __init__(self):
        self.stats = CompressionStats(CompressionAlgorithm.NONE)
        self._lock = threading.RLock()
    
    def compress(self, data: bytes) -> bytes:
        """Return data as-is (no compression)"""
        if not data:
            return b''
        
        with self._lock:
            self.stats.compressed_size += len(data)
            self.stats.original_size += len(data)
            self.stats.total_compressions += 1
            self.stats.compression_ratio = 1.0
            
            return data
    
    def decompress(self, data: bytes) -> bytes:
        """Return data as-is (no decompression)"""
        if not data:
            return b''
        
        with self._lock:
            self.stats.total_decompressions += 1
            return data
    
    def get_stats(self) -> CompressionStats:
        """Get compression statistics"""
        return self.stats
    
    def reset_stats(self):
        """Reset statistics"""
        with self._lock:
            self.stats = CompressionStats(CompressionAlgorithm.NONE)

class CompressionManager:
    """Manager for multiple compression algorithms with auto-selection"""
    
    def __init__(self):
        self.compressors: Dict[CompressionAlgorithm, Any] = {}
        self.default_algorithm = CompressionAlgorithm.ZSTD
        self._lock = threading.RLock()
    
    def register_compressor(self, algorithm: CompressionAlgorithm, compressor: Any):
        """Register a compressor instance"""
        with self._lock:
            self.compressors[algorithm] = compressor
    
    def get_compressor(self, algorithm: CompressionAlgorithm) -> Any:
        """Get compressor for specified algorithm"""
        with self._lock:
            if algorithm not in self.compressors:
                self._create_compressor(algorithm)
            return self.compressors[algorithm]
    
    def _create_compressor(self, algorithm: CompressionAlgorithm):
        """Create a compressor for the specified algorithm"""
        compressors = {
            CompressionAlgorithm.ZLIB: ZlibCompression(),
            CompressionAlgorithm.LZ4: LZ4Compression(),
            CompressionAlgorithm.SNAPPY: SnappyCompression(),
            CompressionAlgorithm.ZSTD: ZstdCompression(),
            CompressionAlgorithm.BZ2: BZ2Compression(),
            CompressionAlgorithm.LZMA: LZMACompression(),
            CompressionAlgorithm.NONE: NoCompression(),
        }
        
        if algorithm not in compressors:
            raise CompressionError(f"Unsupported compression algorithm: {algorithm}")
        
        self.compressors[algorithm] = compressors[algorithm]
    
    def compress(self, data: bytes, algorithm: Optional[CompressionAlgorithm] = None) -> bytes:
        """Compress data using specified algorithm or default"""
        if algorithm is None:
            algorithm = self.default_algorithm
        
        compressor = self.get_compressor(algorithm)
        return compressor.compress(data)
    
    def decompress(self, data: bytes, algorithm: CompressionAlgorithm) -> bytes:
        """Decompress data using specified algorithm"""
        compressor = self.get_compressor(algorithm)
        return compressor.decompress(data)
    
    def auto_compress(self, data: bytes, target_ratio: float = 0.8) -> tuple[bytes, CompressionAlgorithm]:
        """Automatically select best compression algorithm based on data characteristics"""
        if len(data) < 100:  # Too small to benefit from compression
            return data, CompressionAlgorithm.NONE
        
        best_compressed = data
        best_algorithm = CompressionAlgorithm.NONE
        best_ratio = 1.0
        
        # Test different algorithms
        test_algorithms = [
            CompressionAlgorithm.SNAPPY,  # Fastest
            CompressionAlgorithm.LZ4,     # Very fast
            CompressionAlgorithm.ZSTD,    # Good balance
            CompressionAlgorithm.ZLIB,    # Good general purpose
        ]
        
        for algorithm in test_algorithms:
            try:
                compressor = self.get_compressor(algorithm)
                compressed = compressor.compress(data)
                ratio = len(compressed) / len(data)
                
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_compressed = compressed
                    best_algorithm = algorithm
                
                # If we've reached our target ratio, stop testing
                if ratio <= target_ratio:
                    break
                    
            except CompressionError:
                continue  # Skip algorithms that fail
        
        logger.info(f"Auto-selected {best_algorithm.name} with ratio {best_ratio:.2f}")
        return best_compressed, best_algorithm
    
    def get_all_stats(self) -> Dict[CompressionAlgorithm, CompressionStats]:
        """Get statistics for all compressors"""
        with self._lock:
            return {algo: compressor.get_stats() for algo, compressor in self.compressors.items()}
    
    def reset_all_stats(self):
        """Reset statistics for all compressors"""
        with self._lock:
            for compressor in self.compressors.values():
                compressor.reset_stats()

# Global compression manager instance
_compression_manager = CompressionManager()

def get_compression_manager() -> CompressionManager:
    """Get the global compression manager instance"""
    return _compression_manager