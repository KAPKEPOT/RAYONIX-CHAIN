# rayonix_wallet/core/types.py
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from ecdsa.curves import Curve
from ecdsa import SECP256k1
from rayonix_wallet.utils.secure import SecureString

class WalletType(Enum):
    """Types of wallets"""
    HD = auto()
    NON_HD = auto()
    MULTISIG = auto()
    WATCH_ONLY = auto()
    HARDWARE = auto()
    SMART_CONTRACT = auto()

class KeyDerivation(Enum):
    """Key derivation standards"""
    BIP32 = auto()
    BIP39 = auto()
    BIP44 = auto()
    BIP49 = auto()
    BIP84 = auto()
    ELECTRUM = auto()

class AddressType(Enum):
    """Cryptocurrency address types"""
    P2PKH = auto()
    P2SH = auto()
    P2WPKH = auto()
    P2WSH = auto()
    P2TR = auto()
    BECH32 = auto()
    ETHEREUM = auto()
    RAYONIX = auto()
    CONTRACT = auto()

@dataclass
class SecureKeyPair:
    """Cryptographic key pair with secure memory management"""
    _private_key: SecureString
    public_key: bytes
    chain_code: Optional[bytes] = None
    depth: int = 0
    index: int = 0
    parent_fingerprint: bytes = b'\x00\x00\x00\x00'
    curve: Curve = field(default_factory=lambda: SECP256k1)
    
    @property
    def private_key(self) -> bytes:
        return self._private_key.get_value()
    
    def wipe(self):
        """Securely wipe private key from memory"""
        self._private_key.wipe()
        
    def __del__(self):
        self.wipe()

@dataclass
class Transaction:
    """Wallet transaction"""
    txid: str
    amount: int
    fee: int
    confirmations: int
    timestamp: int
    block_height: Optional[int]
    from_address: str
    to_address: str
    status: str
    direction: str
    memo: Optional[str] = None
    exchange_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AddressInfo:
    """Address information"""
    address: str
    index: int
    derivation_path: str
    balance: int
    received: int
    sent: int
    tx_count: int
    is_used: bool
    is_change: bool
    labels: List[str] = field(default_factory=list)

@dataclass
class WalletBalance:
    """Wallet balance information"""
    total: int
    confirmed: int
    unconfirmed: int
    locked: int
    available: int
    by_address: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, int] = field(default_factory=dict)
    offline_mode: bool = False
    last_online_update: Optional[float] = None
    data_freshness: Optional[str] = None
    confidence_level: Optional[str] = None
    warning: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    offline_estimated: bool = False
    estimation_confidence: Optional[str] = None
    reconstruction_confidence: Optional[str] = None

@dataclass
class WalletState:
    """Wallet state and statistics"""
    sync_height: int
    last_updated: float
    tx_count: int
    addresses_generated: int
    addresses_used: int
    total_received: int
    total_sent: int
    security_score: int