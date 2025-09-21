# models.py
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum, auto

@dataclass
class BlockHeader:
    version: int
    height: int
    previous_hash: str
    merkle_root: str
    timestamp: int
    difficulty: int
    nonce: int
    validator: str
    signature: Optional[str] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Block:
    header: BlockHeader
    transactions: List[Any]  # Use Any to avoid circular imports
    hash: str
    chainwork: int
    size: int
    received_time: float = field(default_factory=time.time)