#sync_config.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SyncMode(Enum):
    FULL = "full"           # Download & validate all blocks from genesis
    FAST = "fast"           # Trust recent checkpoints, validate from there
    LIGHT = "light"         # Download headers only, request data on demand
    SNAPSHOT = "snapshot"   # Download recent state snapshot

@dataclass
class SyncConfig:
    mode: SyncMode = SyncMode.FULL
    batch_size: int = 100
    max_parallel_downloads: int = 5
    verify_blocks: bool = True
    preserve_bandwidth: bool = False
    target_peer_count: int = 8
    timeout_per_block: int = 30