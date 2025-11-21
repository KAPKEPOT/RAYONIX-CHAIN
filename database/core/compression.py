# database/core/compression.py

import zstandard as zstd
import threading
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

from database.utils.exceptions import CompressionError

logger = logging.getLogger(__name__)

class CompressionLevel(Enum):
    """Zstandard compression levels optimized for different use cases"""
    FAST = 1          # Fastest compression, lower ratio
    BALANCED = 3      # Perfect balance for blockchain data
    DEFAULT = 6       # Good compression with reasonable speed  
    HIGH = 12         # Better compression, slower
    MAXIMUM = 19      # Maximum compression, very slow
    ULTRA = 22        # Ultra compression, extremely slow

@dataclass
class CompressionStats:
    """Comprehensive compression statistics"""
    total_compressions: int = 0
    total_decompressions: int = 0
    total_compressed_bytes: int = 0
    total_original_bytes: int = 0
    total_compression_time: float = 0.0
    total_decompression_time: float = 0.0
    compression_errors: int = 0
    decompression_errors: int = 0
    
    @property
    def compression_ratio(self) -> float:
        """Calculate overall compression ratio"""
        if self.total_original_bytes == 0:
            return 0.0
        return self.total_compressed_bytes / self.total_original_bytes
    
    @property
    def average_compression_speed(self) -> float:
        """Calculate average compression speed in MB/s"""
        if self.total_compression_time == 0:
            return 0.0
        return (self.total_original_bytes / (1024 * 1024)) / self.total_compression_time
    
    @property
    def average_decompression_speed(self) -> float:
        """Calculate average decompression speed in MB/s"""
        if self.total_decompression_time == 0:
            return 0.0
        return (self.total_compressed_bytes / (1024 * 1024)) / self.total_decompression_time

class ZstdCompression:
    def __init__(
        self,
        level: CompressionLevel = CompressionLevel.BALANCED,
        threads: int = 0,
        enable_checksum: bool = True,
        enable_content_size: bool = True,
        compression_dict: Optional[bytes] = None,
        max_window_size: int = 8 * 1024 * 1024,  # 8MB window for blockchain blocks
        target_block_size: int = 128 * 1024,      # 128KB blocks optimal for blockchain
        strategy: str = "default"
    ):
        self.level = level
        self.threads = threads
        self.enable_checksum = enable_checksum
        self.enable_content_size = enable_content_size
        self.compression_dict = compression_dict
        self.max_window_size = max_window_size
        self.target_block_size = target_block_size
        
        # Compression parameters
        self.compression_params = {
            "level": level.value,
            "threads": threads,
            "checksum": enable_checksum,
            "write_checksum": enable_checksum,
            "write_content_size": enable_content_size,
        }
        
        # Add strategy if specified
        if strategy != "default":
            self.compression_params["strategy"] = self._get_strategy_constant(strategy)
        
        # Initialize compressors with different configurations
        self._initialize_compressors()
        
        # Statistics and state
        self.stats = CompressionStats()
        self._lock = threading.RLock()
        self._active_operations = 0
        
        logger.info(f"Zstandard compression initialized with level {level.name} "
                   f"(threads={threads}, checksum={enable_checksum})")
    
    def _initialize_compressors(self):
        """Initialize Zstandard compressors and decompressors"""
        try:
            # Main compressor with specified parameters
            self.compressor = zstd.ZstdCompressor(**self.compression_params)
            
            # Fast compressor for small data
            self.fast_compressor = zstd.ZstdCompressor(
                level=CompressionLevel.FAST.value,
                threads=self.threads,
                checksum=self.enable_checksum
            )
            
            # High-compression compressor for archival data
            self.high_compressor = zstd.ZstdCompressor(
                level=CompressionLevel.HIGH.value,
                threads=self.threads,
                checksum=self.enable_checksum
            )
            
            # Decompressor with window size limit for security
            self.decompressor = zstd.ZstdDecompressor(
                max_window_size=self.max_window_size
            )
            
            # Streaming decompressor for large data
            self.stream_decompressor = zstd.ZstdDecompressor(
                max_window_size=self.max_window_size
            )
            
        except Exception as e:
            raise CompressionError(f"Failed to initialize Zstandard compressors: {e}")
    
    def _get_strategy_constant(self, strategy: str) -> int:
        """Get Zstandard strategy constant from string"""
        strategies = {
            "default": zstd.ZSTD_strategy.ZSTD_STRATEGY_FAST,
            "fast": zstd.ZSTD_strategy.ZSTD_STRATEGY_FAST,
            "dfast": zstd.ZSTD_strategy.ZSTD_STRATEGY_DFAST,
            "greedy": zstd.ZSTD_strategy.ZSTD_STRATEGY_GREEDY,
            "lazy": zstd.ZSTD_strategy.ZSTD_STRATEGY_LAZY,
            "lazy2": zstd.ZSTD_strategy.ZSTD_STRATEGY_LAZY2,
            "btlazy2": zstd.ZSTD_strategy.ZSTD_STRATEGY_BTLAZY2,
            "btopt": zstd.ZSTD_strategy.ZSTD_STRATEGY_BTOPT,
            "btultra": zstd.ZSTD_strategy.ZSTD_STRATEGY_BTULTRA,
        }
        return strategies.get(strategy, zstd.ZSTD_strategy.ZSTD_STRATEGY_FAST)
    
    def compress(self, data: bytes, level: Optional[CompressionLevel] = None) -> bytes:
        """
        Compress data using Zstandard with comprehensive error handling and statistics
        
        Args:
            data: Data to compress
            level: Override compression level (optional)
            
        Returns:
            Compressed data
        """
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        with self._lock:
            self._active_operations += 1
        
        try:
            # Choose compressor based on level and data size
            compressor = self._select_compressor(data, level)
            
            # Perform compression
            compressed_data = compressor.compress(data)
            
            # Update statistics
            with self._lock:
                self.stats.total_compressions += 1
                self.stats.total_compressed_bytes += len(compressed_data)
                self.stats.total_original_bytes += len(data)
                self.stats.total_compression_time += time.perf_counter() - start_time
            
            logger.debug(f"ZSTD compression: {len(data)} → {len(compressed_data)} bytes "
                        f"(ratio: {len(compressed_data)/len(data):.3f})")
            
            return compressed_data
            
        except zstd.ZstdError as e:
            with self._lock:
                self.stats.compression_errors += 1
            raise CompressionError(f"Zstandard compression failed: {e}")
        except Exception as e:
            with self._lock:
                self.stats.compression_errors += 1
            raise CompressionError(f"Unexpected compression error: {e}")
        finally:
            with self._lock:
                self._active_operations -= 1
    
    def _select_compressor(self, data: bytes, level: Optional[CompressionLevel]) -> zstd.ZstdCompressor:
        """Select appropriate compressor based on data characteristics"""
        # Use specified level if provided
        if level is not None:
            if level == CompressionLevel.FAST:
                return self.fast_compressor
            elif level == CompressionLevel.HIGH:
                return self.high_compressor
            else:
                return self.compressor
        
        # Auto-select based on data size and characteristics
        data_size = len(data)
        
        if data_size < 1024:  # Very small data
            return self.fast_compressor
        elif data_size > 10 * 1024 * 1024:  # Large data (10MB+)
            return self.high_compressor
        else:  # Normal blockchain data (1KB - 10MB)
            return self.compressor
    
    def decompress(self, data: bytes, max_output_size: Optional[int] = None) -> bytes:
        """
        Decompress Zstandard data with comprehensive error handling and security limits
        
        Args:
            data: Compressed data to decompress
            max_output_size: Maximum allowed output size for security
            
        Returns:
            Decompressed data
        """
        if not data:
            return b''
        
        start_time = time.perf_counter()
        
        with self._lock:
            self._active_operations += 1
        
        try:
            # Decompress the data
            decompressed_data = self.decompressor.decompress(data)
            
            # Security check: verify output size if limit specified
            if max_output_size and len(decompressed_data) > max_output_size:
                raise CompressionError(
                    f"Decompressed data too large: {len(decompressed_data)} > {max_output_size}"
                )
            
            # Update statistics
            with self._lock:
                self.stats.total_decompressions += 1
                self.stats.total_decompression_time += time.perf_counter() - start_time
            
            logger.debug(f"ZSTD decompression: {len(data)} → {len(decompressed_data)} bytes")
            
            return decompressed_data
            
        except zstd.ZstdError as e:
            with self._lock:
                self.stats.decompression_errors += 1
            raise CompressionError(f"Zstandard decompression failed: {e}")
        except Exception as e:
            with self._lock:
                self.stats.decompression_errors += 1
            raise CompressionError(f"Unexpected decompression error: {e}")
        finally:
            with self._lock:
                self._active_operations -= 1
    
    def compress_stream(self, data: bytes) -> bytes:
        """Compress data using streaming API (better for very large data)"""
        if not data:
            return b''
        
        try:
            return self.compressor.compress(data)
        except zstd.ZstdError as e:
            raise CompressionError(f"Zstandard stream compression failed: {e}")
    
    def decompress_stream(self, data: bytes, max_output_size: Optional[int] = None) -> bytes:
        """Decompress data using streaming API"""
        if not data:
            return b''
        
        try:
            decompressed = self.stream_decompressor.decompress(data)
            
            if max_output_size and len(decompressed) > max_output_size:
                raise CompressionError(f"Decompressed stream too large: {len(decompressed)}")
                
            return decompressed
        except zstd.ZstdError as e:
            raise CompressionError(f"Zstandard stream decompression failed: {e}")
    
    def create_compression_context(self):
        """Create a compression context for advanced usage"""
        try:
            return zstd.ZstdCompressor(**self.compression_params)
        except zstd.ZstdError as e:
            raise CompressionError(f"Failed to create compression context: {e}")
    
    def create_decompression_context(self) -> zstd.ZstdDecompressionContext:
        """Create a decompression context for advanced usage"""
        try:
            return zstd.ZstdDecompressor(max_window_size=self.max_window_size)
        except zstd.ZstdError as e:
            raise CompressionError(f"Failed to create decompression context: {e}")
    
    def get_frame_info(self, data: bytes) -> Dict[str, Any]:
        """Get information about compressed frame"""
        try:
            frame_info = {}
            
            # Get basic frame information
            if hasattr(zstd, 'frame_header'):
                header = zstd.frame_header(data)
                frame_info.update({
                    "content_size": header.content_size,
                    "window_size": header.window_size,
                    "dict_id": header.dict_id,
                    "has_checksum": header.has_checksum,
                })
            
            # Get compression parameters
            if hasattr(zstd, 'frame_params'):
                params = zstd.frame_parameters(data)
                frame_info.update({
                    "compressed_size": len(data),
                    "content_size": params.content_size,
                    "window_size": params.window_size,
                    "dict_id": params.dict_id,
                    "checksum": params.has_checksum,
                })
            
            return frame_info
            
        except zstd.ZstdError as e:
            raise CompressionError(f"Failed to get frame info: {e}")
    
    def validate_compressed_data(self, data: bytes) -> bool:
        """Validate if data is properly compressed with Zstandard"""
        try:
            # Try to get frame information
            self.get_frame_info(data)
            return True
        except (CompressionError, zstd.ZstdError):
            return False
    
    def get_compression_stats(self) -> CompressionStats:
        """Get comprehensive compression statistics"""
        with self._lock:
            return CompressionStats(
                total_compressions=self.stats.total_compressions,
                total_decompressions=self.stats.total_decompressions,
                total_compressed_bytes=self.stats.total_compressed_bytes,
                total_original_bytes=self.stats.total_original_bytes,
                total_compression_time=self.stats.total_compression_time,
                total_decompression_time=self.stats.total_decompression_time,
                compression_errors=self.stats.compression_errors,
                decompression_errors=self.stats.decompression_errors,
            )
    
    def reset_stats(self):
        """Reset compression statistics"""
        with self._lock:
            self.stats = CompressionStats()
    
    def get_operation_count(self) -> int:
        """Get number of currently active operations"""
        with self._lock:
            return self._active_operations
    
    @contextmanager
    def compression_context(self):
        """Context manager for compression operations"""
        yield self.compressor
    
    @contextmanager
    def decompression_context(self):
        """Context manager for decompression operations"""
        yield self.decompressor

# Global Zstandard compression instance with blockchain-optimized settings
COMPRESSOR = ZstdCompression(
    level=CompressionLevel.BALANCED,
    threads=0,  # Auto-detect CPU cores
    enable_checksum=True,  # Critical for data integrity
    enable_content_size=True,
    max_window_size=64 * 1024 * 1024,  # 64MB max for large blockchain blocks
    target_block_size=256 * 1024,  # 256KB blocks optimal for blockchain
    strategy="btopt"  # Binary tree optimal strategy for structured data
)

# Convenience functions using global instance
def compress(data: bytes, level: Optional[CompressionLevel] = None) -> bytes:
    """Compress data using global blockchain-optimized compressor"""
    return COMPRESSOR.compress(data, level)

def decompress(data: bytes, max_output_size: Optional[int] = None) -> bytes:
    """Decompress data using global blockchain-optimized compressor"""
    return COMPRESSOR.decompress(data, max_output_size)

def get_compression_stats() -> CompressionStats:
    """Get statistics from global compressor"""
    return COMPRESSOR.get_compression_stats()

def validate_compressed_data(data: bytes) -> bool:
    """Validate compressed data using global compressor"""
    return COMPRESSOR.validate_compressed_data(data)