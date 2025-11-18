#rayonix_wallet/core/wallet_factory.py
import logging
from typing import Optional, Tuple
from pathlib import Path

from rayonix_wallet.core.wallet import RayonixWallet
from config.config_manager import ConfigManager
from rayonix_wallet.core.wallet_types import WalletType, AddressType

logger = logging.getLogger(__name__)

class WalletFactory:
    """Factory for creating and managing wallet instances"""
    
    @staticmethod
    def create_new_wallet(
        wallet_type: WalletType = WalletType.HD,
        address_type: AddressType = AddressType.RAYONIX,
        config_manager: Optional[ConfigManager] = None,
        passphrase: str = ""
    ) -> Tuple[RayonixWallet, str]:
        """
        Create a new wallet and return both wallet instance and mnemonic
        Network is automatically taken from config_manager
        """
        try:
            # Use ConfigManager to get wallet configuration
            if config_manager is None:
                config_manager = ConfigManager()
            
            # Get wallet config synchronized with node network
            config = config_manager.get_wallet_config()
            
            # Override specific settings if provided
            config.wallet_type = wallet_type
            config.address_type = address_type
            # Network is automatically set from config_manager via get_wallet_config()
            
            # VALIDATE: Ensure we're using enums, not strings
            if not isinstance(config.address_type, AddressType):
                raise ValueError(f"address_type must be AddressType enum, got {type(config.address_type)}")
            
            if not isinstance(config.wallet_type, WalletType):
                raise ValueError(f"wallet_type must be WalletType enum, got {type(config.wallet_type)}")
            
            logger.debug(f"Creating wallet for network: {config.network}")
            
            # Create wallet instance
            wallet = RayonixWallet(config)
            
            # Generate and initialize with new mnemonic
            mnemonic = wallet.initialize_new_wallet(passphrase)
            
            logger.info(f"Created new {wallet_type.name} wallet for {config.network} network")
            return wallet, mnemonic
            
        except Exception as e:
            logger.error(f"Failed to create new wallet: {e}")
            raise
    
    @staticmethod
    def create_wallet_from_mnemonic(
        mnemonic: str,
        passphrase: str = "",
        wallet_type: WalletType = WalletType.HD,
        address_type: AddressType = AddressType.RAYONIX,
        config_manager: Optional[ConfigManager] = None
    ) -> RayonixWallet:
        """
        Create wallet from existing mnemonic phrase
        
        Args:
            mnemonic: BIP39 mnemonic phrase
            passphrase: Optional passphrase
            wallet_type: Wallet type
            address_type: Address format type
            config_manager: Optional config manager instance (provides network)
            
        Returns:
            Wallet instance
        """
        try:
            # Use ConfigManager to get wallet configuration
            if config_manager is None:
                config_manager = ConfigManager()
            
            config = config_manager.get_wallet_config()
            config.wallet_type = wallet_type
            config.address_type = address_type
            # Network comes from config_manager
            
            wallet = RayonixWallet(config)
            
            if wallet.create_from_mnemonic(mnemonic, passphrase):
                logger.info(f"Successfully created wallet from mnemonic for {config.network} network")
                return wallet
            else:
                raise ValueError("Failed to create wallet from mnemonic")
                
        except Exception as e:
            logger.error(f"Failed to create wallet from mnemonic: {e}")
            raise
    
    @staticmethod
    def create_wallet_from_private_key(
        private_key: str,
        wallet_type: WalletType = WalletType.NON_HD,
        address_type: AddressType = AddressType.RAYONIX,
        config_manager: Optional[ConfigManager] = None
    ) -> RayonixWallet:
        """
        Create wallet from private key
        
        Args:
            private_key: Private key in hex format
            wallet_type: Wallet type
            address_type: Address format type
            config_manager: Optional config manager instance (provides network)
            
        Returns:
            Wallet instance
        """
        try:
            # Use ConfigManager to get wallet configuration
            if config_manager is None:
                config_manager = ConfigManager()
            
            config = config_manager.get_wallet_config()
            config.wallet_type = wallet_type
            config.address_type = address_type
            # Network comes from config_manager
            
            wallet = RayonixWallet(config)
            
            if wallet.create_from_private_key(private_key, wallet_type):
                logger.info(f"Successfully created wallet from private key for {config.network} network")
                return wallet
            else:
                raise ValueError("Failed to create wallet from private key")
                
        except Exception as e:
            logger.error(f"Failed to create wallet from private key: {e}")
            raise
    
    @staticmethod
    def load_wallet_from_file(
        wallet_path: str,
        passphrase: str = "",
        config_manager: Optional[ConfigManager] = None
    ) -> RayonixWallet:
        """
        Load wallet from encrypted file
        
        Args:
            wallet_path: Path to wallet file
            passphrase: Decryption passphrase
            config_manager: Optional config manager instance
            
        Returns:
            Wallet instance
        """
        try:
            # Use ConfigManager to get wallet configuration
            if config_manager is None:
                config_manager = ConfigManager()
            
            config = config_manager.get_wallet_config()
            wallet = RayonixWallet(config)
            
            if wallet.restore(wallet_path, passphrase):
                logger.info(f"Successfully loaded wallet from {wallet_path} for {config.network} network")
                return wallet
            else:
                raise ValueError("Failed to load wallet from file")
                
        except Exception as e:
            logger.error(f"Failed to load wallet from file: {e}")
            raise

# Backward compatibility function (updated to use config manager)
def create_new_wallet(
    wallet_type: WalletType = WalletType.HD,
    address_type: AddressType = AddressType.RAYONIX
) -> Tuple[RayonixWallet, str]:
    """
    Backward compatibility function - creates new wallet and returns instance with mnemonic
    Network is automatically determined from config manager
    """
    return WalletFactory.create_new_wallet(wallet_type, address_type)