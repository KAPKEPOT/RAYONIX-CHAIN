import zlib

def compress_data(data: bytes) -> bytes:
    """Compress data using zlib"""
    try:
        return zlib.compress(data)
    except Exception as e:
        raise Exception(f"Compression error: {e}")

def decompress_data(data: bytes) -> bytes:
    """Decompress data using zlib"""
    try:
        return zlib.decompress(data)
    except Exception as e:
        raise Exception(f"Decompression error: {e}")