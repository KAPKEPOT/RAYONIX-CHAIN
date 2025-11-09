from dataclasses import dataclass, field
from typing import Dict, Optional
from rayonix_wallet.core.wallet_types import WalletType, KeyDerivation, AddressType

@dataclass
class WalletConfig:
    """Wallet configuration"""
    wallet_type: WalletType = WalletType.HD
    key_derivation: KeyDerivation = KeyDerivation.BIP44
    address_type: AddressType = AddressType.RAYONIX
    encryption: bool = True
    compression: bool = True
    passphrase: Optional[str] = None
    network: str = "mainnet"
    account_index: int = 0
    change_index: int = 0
    gap_limit: int = 20
    auto_backup: bool = True
    backup_interval: int = 86400
    price_alerts: bool = False
    transaction_fees: Dict[str, int] = field(default_factory=lambda: {
        "low": 1, "medium": 2, "high": 5
    })
    db_path: str = "wallet.db"
    sync_interval: int = 300