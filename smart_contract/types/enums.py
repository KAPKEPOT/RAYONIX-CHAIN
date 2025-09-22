# smart_contract/types/enums.py
from enum import Enum, auto

class ContractType(Enum):
    """Types of smart contracts"""
    STANDARD = auto()
    TOKEN = auto()
    EXCHANGE = auto()
    GOVERNANCE = auto()
    NFT = auto()
    DEFI = auto()
    BRIDGE = auto()
    ORACLE = auto()
    GAMING = auto()
    CUSTOM = auto()

class ContractState(Enum):
    """States of smart contract lifecycle"""
    DEPLOYING = auto()
    ACTIVE = auto()
    PAUSED = auto()
    UPGRADING = auto()
    DESTROYED = auto()
    ERROR = auto()

class ContractSecurityLevel(Enum):
    """Security levels for contracts"""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()