#rayonix_wallet/core/config.py
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
    coin_type: str = field(default_factory=lambda: "1180") 
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
    
    def __post_init__(self):
        """Initialize derived properties after object creation"""
        # Ensure enums are preserved and not converted to strings
        self._validate_enum_types()
        
        if not hasattr(self, 'coin_type') or self.coin_type is None:
            if self.network == "mainnet":
                self.coin_type = "1180"  # RAYONIX mainnet coin type
            else:
                self.coin_type = "1"    # Testnet coin type
    
    def _validate_enum_types(self):
        """Ensure all enum fields remain as enum types"""
        # If any enum field was converted to string (from serialization), convert it back
        if isinstance(self.wallet_type, str):
            self.wallet_type = WalletType[self.wallet_type.upper()]
        
        if isinstance(self.key_derivation, str):
            self.key_derivation = KeyDerivation[self.key_derivation.upper()]
            
        if isinstance(self.address_type, str):
            self.address_type = AddressType[self.address_type.upper()]