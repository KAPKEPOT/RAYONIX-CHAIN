from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Any, Dict, Union
from datetime import datetime

class DatabaseType(Enum):
    PLYVEL = auto()
    MEMORY = auto()

class CompressionType(Enum):
    NONE = auto()
    ZLIB = auto()
    LZ4 = auto()
    SNAPPY = auto()
    ZSTD = auto()

class EncryptionType(Enum):
    NONE = auto()
    FERNET = auto()
    AES256 = auto()
    CHACHA20 = auto()

class IndexType(Enum):
    BTREE = auto()
    HASH = auto()
    BLOOM = auto()
    LSM = auto()
    COMPOUND = auto()

class SerializationType(Enum):
    JSON = auto()
    MSGPACK = auto()
    PROTOBUF = auto()
    AVRO = auto()

@dataclass
class DatabaseConfig:
    db_type: DatabaseType = DatabaseType.PLYVEL
    create_if_missing: bool = True
    error_if_exists: bool = False
    paranoid_checks: bool = False
    write_buffer_size: int = 4 * 1024 * 1024  # 4MB
    max_open_files: int = 1000
    block_size: int = 4096
    cache_size: int = 8 * 1024 * 1024  # 8MB
    max_cache_size: int = 10000
    cache_ttl: int = 300  # 5 minutes
    compression: CompressionType = CompressionType.ZLIB
    compression_level: int = 6
    encryption: EncryptionType = EncryptionType.NONE
    encryption_key: Optional[bytes] = None
    serialization: SerializationType = SerializationType.JSON
    bloom_filter_size: int = 1000000
    bloom_filter_error_rate: float = 0.01
    memtable_size: int = 10000
    max_sstables: int = 10
    
    # Merkle Integrity Settings
    merkle_integrity: bool = True
    merkle_tree_depth: int = 256
    merkle_hash_algorithm: str = 'sha256'
    merkle_verify_on_read: bool = True
    merkle_verify_on_write: bool = True
    merkle_auto_recover: bool = True
    merkle_proof_format: str = 'binary'
    integrity_check_interval: int = 3600

@dataclass
class IndexConfig:
    index_type: IndexType = IndexType.BTREE
    unique: bool = False
    sparse: bool = False
    fields: Optional[List[str]] = None
    hash_function: str = 'md5'  # 'md5' or 'sha256'
    bloom_filter_size: int = 1000000
    bloom_filter_error_rate: float = 0.01
    memtable_size: int = 10000
    max_sstables: int = 10

@dataclass
class BatchOperation:
    op_type: str  # 'put' or 'delete'
    key: bytes
    value: Optional[Any] = None
    ttl: Optional[int] = None
    index_updates: Optional[Dict[str, Any]] = None