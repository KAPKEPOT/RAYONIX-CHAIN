# config/config_schema.py - Configuration schema validation

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from enum import Enum

class NetworkType(str, Enum):
    MAINNET = "mainnet"
    TESTNET = "testnet"
    REGTEST = "regtest"

class DatabaseEngine(str, Enum):
    PLYVEL = "plyvel"
    MEMORY = "memory"
    
class ConsensusType(str, Enum):
    POS = "pos"
    POW = "pow"
    DPOS = "dpos"

class NetworkConfigSchema(BaseModel):
    network_type: NetworkType = Field(default=NetworkType.TESTNET)
    network_id: int = Field(default=1, ge=1)
    enabled: bool = Field(default=True)
    listen_ip: str = Field(default="0.0.0.0")
    listen_port: int = Field(default=9333, ge=1024, le=65535)
    max_connections: int = Field(default=50, ge=1, le=1000)
    bootstrap_nodes: list = Field(default_factory=list)
    enable_encryption: bool = Field(default=True)
    enable_compression: bool = Field(default=True)
    enable_dht: bool = Field(default=False)
    connection_timeout: int = Field(default=30, ge=1, le=300)
    message_timeout: int = Field(default=10, ge=1, le=60)

class DatabaseConfigSchema(BaseModel):
    db_path: str = Field(default="./rayonix_data")
    db_engine: DatabaseEngine = Field(default=DatabaseEngine.SQLITE)
    connection_string: str = Field(default="")
    max_connections: int = Field(default=10, ge=1, le=100)
    connection_timeout: int = Field(default=30, ge=1, le=300)

class APIConfigSchema(BaseModel):
    enabled: bool = Field(default=True)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8545, ge=1024, le=65535)
    enable_cors: bool = Field(default=True)
    cors_origins: list = Field(default_factory=lambda: ["*"])

class ConsensusConfigSchema(BaseModel):
    consensus_type: ConsensusType = Field(default=ConsensusType.POS)
    min_stake: int = Field(default=1000, ge=0)
    max_stake: int = Field(default=1000000, ge=0)
    block_time: int = Field(default=10, ge=1, le=300)
    difficulty_adjustment_interval: int = Field(default=2016, ge=1)
    reward_halving_interval: int = Field(default=210000, ge=1)

class GasConfigSchema(BaseModel):
    base_gas_price: int = Field(default=1000000000, ge=0)
    min_gas_price: int = Field(default=500000000, ge=0)
    max_gas_price: int = Field(default=10000000000, ge=0)
    adjustment_factor: float = Field(default=1.125, ge=1.0, le=2.0)
    target_utilization: float = Field(default=0.5, ge=0.1, le=0.9)

class LoggingConfigSchema(BaseModel):
    level: str = Field(default="INFO")
    file: str = Field(default="rayonix_node.log")
    max_size: int = Field(default=10485760, ge=0)  # 10MB
    backup_count: int = Field(default=5, ge=0)
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

class ConfigSchema(BaseModel):
    network: NetworkConfigSchema = Field(default_factory=NetworkConfigSchema)
    database: DatabaseConfigSchema = Field(default_factory=DatabaseConfigSchema)
    api: APIConfigSchema = Field(default_factory=APIConfigSchema)
    consensus: ConsensusConfigSchema = Field(default_factory=ConsensusConfigSchema)
    gas: GasConfigSchema = Field(default_factory=GasConfigSchema)
    logging: LoggingConfigSchema = Field(default_factory=LoggingConfigSchema)

def validate_config(config_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate configuration dictionary against schema"""
    try:
        validated_config = ConfigSchema(**config_dict)
        return validated_config.dict()
    except Exception as e:
        print(f"Configuration validation error: {e}")
        return None